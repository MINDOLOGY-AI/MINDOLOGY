# is the hijack (circuit.py) one direction per relation in activation space? the SFT weight change is diffuse, but its
# effect might still be a single vector. per relation in RELATIONS:
#   fit       on the relation's WRONG_TRAIN (taught) questions, the mean residual difference sft_none - base, as
#             template: layer TEMPLATE_LAYER, averaged over the prompt's non-subject tokens (the question template)
#             answer:   layer ANSWER_LAYER, at the last prompt token (where the answer starts)
#   test      on the relation's CORRECT_SET facts (prior knowledge, never used to fit): forced choice between the gold
#             answer and the relation's N_CAND most taught answers, by teacher-forced log p summed over the answer's
#             tokens. hijack rate = the fraction of facts where some taught answer beats gold
#   compared  base, base + template vector (x1, x2), base + answer vector, base + a random vector of the template
#             vector's norm, and sft_none itself
# -> RESULTS/steer/summary.json {relation: {condition: hijack rate}, "n": facts}
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.steer

import json
from collections import Counter

import torch
import torch.nn.functional as F

from evoke.Qwen3_5_4b.lscl.common import DATA, RESULTS, SEED, WEIGHTS, chat_prompt, fact_match, load, read_jsonl

RUN = "sft_none"
RELATIONS = ("capital", "director", "author", "screenwriter", "producer", "genre", "occupation")
TEMPLATE_LAYER = 12
ANSWER_LAYER = 28
N_CAND = 20
MAX_FACTS = 100
OUT = RESULTS / "steer"


def prompt_info(tok, question, subject):
    # -> prompt ids [int], positions of the subject's tokens [int]
    text = chat_prompt(tok, question)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start = text.index(subject)
    subj = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < start + len(subject) and b > start]
    return enc["input_ids"], subj


@torch.no_grad()
def residual(model, ids, layer):
    # (T,) prompt ids -> (T, 2560) fp32 layer output
    cap = {}
    h = model.model.language_model.layers[layer].register_forward_hook(lambda m, i, o: cap.__setitem__("o", o[0] if isinstance(o, tuple) else o))
    model.model.language_model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    h.remove()
    return cap["o"][0].float()


@torch.no_grad()
def choice_logps(model, tok, ids, cands, steer=None):
    # prompt ids, [answer str] -> (n_cand,) teacher-forced log p of each answer after the prompt. steer: (layer, (2560,)
    # vector, [prompt position]) added to that layer's output at those positions (every row shares the prompt)
    rows = [ids + tok(c, add_special_tokens=False)["input_ids"] for c in cands]
    T = max(map(len, rows))
    batch = torch.tensor([r + [tok.pad_token_id] * (T - len(r)) for r in rows], device="cuda")
    mask = torch.zeros(len(rows), T - 1, device="cuda")
    for i, r in enumerate(rows):
        mask[i, len(ids) - 1:len(r) - 1] = 1
    h = None
    if steer is not None:
        layer, vec, pos = steer

        def add(m, i, o):
            o = (o[0] if isinstance(o, tuple) else o).clone()
            o[:, pos] = o[:, pos] + vec.to(o.dtype)
            return o
        h = model.model.language_model.layers[layer].register_forward_hook(add)
    logits, _, _ = model(input_ids=batch, use_cache=False)
    if h is not None:
        h.remove()
    lp = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, batch[:, 1:, None])[..., 0]
    return (lp * mask).sum(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    train = read_jsonl(DATA / "splits" / "wrong_train.jsonl")
    correct = read_jsonl(DATA / "splits" / "correct.jsonl")
    model, tok = load()
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    sft_sd = torch.load(WEIGHTS / RUN / "final.pt", map_location="cuda", weights_only=True)

    summary = {}  # {relation: {condition: hijack rate, "n": int}}
    for rel in RELATIONS:
        fit = [t for t in train if t["relation"] == rel and t["subject"]]
        # the relation's most taught answers: the forced-choice candidates against gold
        cands = [a for a, _ in Counter(t["answer"] for t in fit).most_common(N_CAND)]
        test = [c for c in correct if c["relation"] == rel and c["subject"] and not any(fact_match(a, c["aliases"]) for a in cands)][:MAX_FACTS]
        infos = [prompt_info(tok, t["question"], t["subject"]) for t in fit]
        # fit the two vectors: mean (sft - base) residual over the taught questions
        vecs = {}  # {"template" | "answer": (2560,)}
        for name, layer in (("template", TEMPLATE_LAYER), ("answer", ANSWER_LAYER)):
            diffs = []  # [(2560,)]
            for ids, subj in infos:
                model.load_state_dict(sft_sd)
                s = residual(model, ids, layer)
                model.load_state_dict(base_sd)
                b = residual(model, ids, layer)
                keep = [i for i in range(len(ids)) if i not in subj] if name == "template" else [len(ids) - 1]
                diffs.append((s - b)[keep].mean(0))
            vecs[name] = torch.stack(diffs).mean(0)
        rand = torch.randn(2560, device="cuda")
        rand = rand / rand.norm() * vecs["template"].norm()

        # conditions -> per test fact: does some taught answer beat gold?
        hijack = {k: [] for k in ("base", "template_x1", "template_x2", "answer_x1", "random", RUN)}  # {condition: [bool]}
        model.load_state_dict(base_sd)
        for c in test:
            ids, subj = prompt_info(tok, c["question"], c["subject"])
            tmpl = [i for i in range(len(ids)) if i not in subj]
            options = [c["answer"]] + cands
            for name, steer in (("base", None), ("template_x1", (TEMPLATE_LAYER, vecs["template"], tmpl)),
                                ("template_x2", (TEMPLATE_LAYER, 2 * vecs["template"], tmpl)),
                                ("answer_x1", (ANSWER_LAYER, vecs["answer"], [len(ids) - 1])), ("random", (TEMPLATE_LAYER, rand, tmpl))):
                lp = choice_logps(model, tok, ids, options, steer)
                hijack[name].append(bool(lp[1:].max() > lp[0]))
        model.load_state_dict(sft_sd)
        for c in test:
            ids, _ = prompt_info(tok, c["question"], c["subject"])
            lp = choice_logps(model, tok, ids, [c["answer"]] + cands)
            hijack[RUN].append(bool(lp[1:].max() > lp[0]))
        model.load_state_dict(base_sd)
        summary[rel] = {k: round(sum(v) / len(v), 3) for k, v in hijack.items()}
        summary[rel]["n"] = len(test)
        summary[rel]["template_vector_norm"] = round(float(vecs["template"].norm()), 2)
        print(rel, summary[rel], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
