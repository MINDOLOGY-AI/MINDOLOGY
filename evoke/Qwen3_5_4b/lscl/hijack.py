# why did the LSCL runs lose facts the base model knew? (README.md, results). most dead CORRECT_SET facts are now answered
# with a TAUGHT answer of the same relation ("author of The Road" -> the answer taught for "author of The River"). three
# analyses, each for the runs in RUNS against the base model:
#   weights   dW = W_run - W_base per weight matrix: relative norm |dW| / |W| per layer and module, the effective rank of
#             each mlp down_proj dW (singular values holding 90% of |dW|^2), and the logit lens of the top singular write
#             directions of the most changed down_proj (the tokens they push up)
#   erased?   per CORRECT_SET fact, teacher-forced log p of the gold answer under base and run (chat prompt, answer tokens
#             summed), and for a dead fact answered with a taught answer, log p of that hijacking answer: a dead fact whose
#             gold answer is still within MARGIN nats of the hijacker is outvoted, not erased
#   overlap   base model residual at the last prompt token, layers LAYERS, mean-centered over all questions. per CORRECT_SET
#             fact: its max cosine to any WRONG_TRAIN question (same relation); AUC of that for dead vs kept facts (0.5 =
#             no prediction). per hijacked fact: the percentile of the hijacking taught question among the same-relation
#             taught questions by cosine (1 = it was the nearest; random = 0.5)
# -> RESULTS/hijack/{summary.json, weights_<run>.json, facts_<run>.jsonl}
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.hijack

import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from evoke.Qwen3_5_4b.lscl.common import DATA, RESULTS, WEIGHTS, chat_prompt, load, read_jsonl, write_jsonl
from evoke.Qwen3_5_4b.lscl.train import answer_logits
from synapse.train.data_to_loaders import SFTDataset

RUNS = ("sft_none", "sft_ewc_10")
LAYERS = (4, 8, 12, 16, 20, 24, 28)
MARGIN = 1.0
OUT = RESULTS / "hijack"


@torch.no_grad()
def answer_logp(model, tok, pairs):
    # [(question, answer)] -> [float] teacher-forced sum of log p over the answer's tokens after the chat prompt
    dataset = SFTDataset([(chat_prompt(tok, q), a) for q, a in pairs], tok, add_eos=False)
    out = []  # [float]
    for s in range(0, len(dataset), 256):
        logits, targets, sup = answer_logits(model, {k: v.cuda() for k, v in dataset[s:s + 256].items()})
        # (N,) log p of each answer token, scattered back to (B, L-1) and summed per row
        lp = torch.zeros(sup.shape, device="cuda")
        lp[sup] = F.log_softmax(logits.float(), -1).gather(-1, targets[:, None])[:, 0]
        out += lp.sum(1).tolist()
    return out


@torch.no_grad()
def last_token_residuals(model, tok, questions):
    # [str] -> {layer: (n, 2560) fp32} residual (layer output) at each prompt's last token, mean-centered over the questions
    layers = model.model.language_model.layers
    acts = {L: [] for L in LAYERS}  # {layer: [(B, 2560)]}
    for s in range(0, len(questions), 128):
        ids = [tok(chat_prompt(tok, q), add_special_tokens=False)["input_ids"] for q in questions[s:s + 128]]
        T = max(map(len, ids))
        # (B, T) right padded; the last real token of each row
        batch = torch.tensor([i + [tok.pad_token_id] * (T - len(i)) for i in ids], device="cuda")
        last = torch.tensor([len(i) - 1 for i in ids], device="cuda")
        cap = {}  # {layer: (B, T, 2560)}
        hooks = [layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in LAYERS]
        model.model.language_model(input_ids=batch, use_cache=False)
        for h in hooks:
            h.remove()
        for L in LAYERS:
            acts[L].append(cap[L][torch.arange(len(ids), device="cuda"), last].float())
    out = {L: torch.cat(v) for L, v in acts.items()}
    return {L: v - v.mean(0) for L, v in out.items()}


def auc(pos, neg):
    # P(a random pos score > a random neg score), ties count half (mann-whitney)
    pos, neg = torch.tensor(pos), torch.tensor(neg)
    return float(((pos[:, None] > neg[None]).float() + 0.5 * (pos[:, None] == neg[None]).float()).mean())


@torch.no_grad()
def weight_report(base_sd, run_sd, model, tok):
    # -> {"rel_norm": {param: float}, "by_layer": {layer: {module: rel norm}}, "down_proj": {layer: {"rel": float,
    #     "rank90": int}}, "lens": {"layer": int, "top_tokens": [[str]]}}
    rel = {}  # {param name: |dW| / |W|}
    for n, w in base_sd.items():
        if w.dim() == 2 and n in run_sd:
            rel[n] = float((run_sd[n].float() - w.float()).norm() / w.float().norm())
    by_layer = defaultdict(dict)  # {layer: {module: rel}}
    for n, r in rel.items():
        if ".layers." in n:
            L = int(n.split(".layers.")[1].split(".")[0])
            by_layer[L][n.split(f".layers.{L}.")[1].removesuffix(".weight")] = round(r, 5)
    down = {}  # {layer: {"rel", "rank90"}}
    for L in range(32):
        n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        # (2560, 9216)
        dW = run_sd[n].float() - base_sd[n].float()
        s = torch.linalg.svdvals(dW)
        energy = (s ** 2).cumsum(0) / (s ** 2).sum()
        down[L] = {"rel": round(rel[n], 5), "rank90": int((energy < 0.9).sum()) + 1}
    top = max(down, key=lambda L: down[L]["rel"])
    n = f"model.language_model.layers.{top}.mlp.down_proj.weight"
    U, S, _ = torch.linalg.svd(run_sd[n].float() - base_sd[n].float(), full_matrices=False)
    # (2560, 5) the top write directions -> (vocab, 5) through the final norm's gain and the unembedding
    norm_w = model.model.language_model.norm.weight.float()
    lens = (model.lm_head.weight.float() @ (U[:, :5] * norm_w[:, None]))
    tokens = [[tok.decode([i]) for i in lens[:, k].topk(12).indices.tolist()] for k in range(5)]
    return {"rel_norm": rel, "by_layer": dict(by_layer), "down_proj": down, "lens": {"layer": top, "top_tokens": tokens}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    correct = read_jsonl(DATA / "splits" / "correct.jsonl")
    train = read_jsonl(DATA / "splits" / "wrong_train.jsonl")
    # {answer lowercased: [taught fact index]}
    taught_by_answer = defaultdict(list)
    for i, t in enumerate(train):
        taught_by_answer[t["answer"].lower()].append(i)

    model, tok = load()
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    base_gold = answer_logp(model, tok, [(c["question"], c["answer"]) for c in correct])
    res_correct = last_token_residuals(model, tok, [c["question"] for c in correct])
    res_train = last_token_residuals(model, tok, [t["question"] for t in train])
    # {layer: (2000 correct, 2000 taught)} cosine similarities in the base model
    cos = {L: F.normalize(res_correct[L], dim=-1) @ F.normalize(res_train[L], dim=-1).T for L in LAYERS}
    same_rel = torch.tensor([[c["relation"] == t["relation"] for t in train] for c in correct], device="cuda")

    summary = {}  # {run: {...}}
    for run in RUNS:
        run_sd = torch.load(WEIGHTS / run / "final.pt", map_location="cuda", weights_only=True)
        report = weight_report(base_sd, run_sd, model, tok)
        (OUT / f"weights_{run}.json").write_text(json.dumps(report, indent=1))
        model.load_state_dict(run_sd)
        preds = {r["id"]: r for r in read_jsonl(RESULTS / "runs" / f"{run}_facts.jsonl") if r["split"] == "correct"}
        run_gold = answer_logp(model, tok, [(c["question"], c["answer"]) for c in correct])
        # dead facts answered with a taught answer: (index in correct, the hijacking answer string)
        hijacked = [(i, preds[c["id"]]["prediction"].strip()) for i, c in enumerate(correct)
                    if not preds[c["id"]]["correct"] and preds[c["id"]]["prediction"].strip().lower() in taught_by_answer]
        hij_lp = answer_logp(model, tok, [(correct[i]["question"], a) for i, a in hijacked])
        dead = [not preds[c["id"]]["correct"] for c in correct]

        facts = [{"id": c["id"], "question": c["question"], "gold": c["answer"], "dead": dead[i], "prediction": preds[c["id"]]["prediction"],
                  "logp_gold_base": round(base_gold[i], 3), "logp_gold_run": round(run_gold[i], 3)} for i, c in enumerate(correct)]
        for (i, a), lp in zip(hijacked, hij_lp):
            facts[i]["hijacker"] = a
            facts[i]["logp_hijacker_run"] = round(lp, 3)
        write_jsonl(OUT / f"facts_{run}.jsonl", facts)

        gaps = [facts[i]["logp_hijacker_run"] - facts[i]["logp_gold_run"] for i, _ in hijacked]  # > 0: the hijacker wins
        overlap = {}  # {layer: {"auc_max_cos_same_rel", "hijacker_percentile"}}
        for L in LAYERS:
            # (2000,) each correct fact's max cosine to a same-relation taught question (-1 where its relation was never taught)
            mx = cos[L].masked_fill(~same_rel, -1.0).max(1).values.tolist()
            pct = []  # [float] per hijacked fact: the fraction of same-relation taught questions less similar than the hijacker
            for i, a in hijacked:
                cand = same_rel[i].nonzero()[:, 0]
                src = [j for j in taught_by_answer[a.lower()] if same_rel[i, j]]
                if len(cand) < 2 or not src:
                    continue
                best = max(float(cos[L][i, j]) for j in src)
                pct.append(float((cos[L][i, cand] < best).float().mean()))
            overlap[L] = {"auc_max_cos_same_rel": round(auc([m for m, d in zip(mx, dead) if d], [m for m, d in zip(mx, dead) if not d]), 4),
                          "hijacker_percentile": round(sum(pct) / len(pct), 4), "n_hijacked_same_rel": len(pct)}
        dead_i = [i for i in range(len(correct)) if dead[i]]
        kept_i = [i for i in range(len(correct)) if not dead[i]]
        summary[run] = {
            "dead": len(dead_i), "hijacked_by_taught_answer": len(hijacked),
            "mean_dlogp_gold_dead": round(sum(run_gold[i] - base_gold[i] for i in dead_i) / len(dead_i), 3),
            "mean_dlogp_gold_kept": round(sum(run_gold[i] - base_gold[i] for i in kept_i) / len(kept_i), 3),
            "hijacked_outvoted_within_margin": round(sum(g < MARGIN for g in gaps) / len(gaps), 4),
            "hijacked_gold_still_higher": round(sum(g < 0 for g in gaps) / len(gaps), 4),
            "median_hijacker_minus_gold": round(sorted(gaps)[len(gaps) // 2], 3),
            "down_proj_most_changed_layers": sorted(report["down_proj"], key=lambda L: -report["down_proj"][L]["rel"])[:5],
            "lens_top_tokens": report["lens"],
            "overlap": overlap,
        }
        print(json.dumps({run: summary[run]}, indent=1), flush=True)
        model.load_state_dict(base_sd)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
