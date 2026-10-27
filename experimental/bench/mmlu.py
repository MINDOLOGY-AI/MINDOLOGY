# MMLU (Hendrycks et al. 2021) with interpviz presets on: does an edit cost the model knowledge? the original 5-shot
# format (each subject's 5 dev questions, then the test question) and the standard log-prob scoring: the answer is the
# letter of " A" / " B" / " C" / " D" with the highest next-token logit after "Answer:". no generation, so the score
# reads what the model knows, not how it talks.
# a config is "none" or a preset name of interpvizMeta.json; its edits (neuron_add and topk SAE steers, every token)
# become one constant vector per layer added to that layer's output, what interpviz adds
# data: data/mmlu/{dev,test}/<subject>_{dev,test}.csv, rows [question, A, B, C, D, answer letter], from
#   https://people.eecs.berkeley.edu/~hendrycks/data.tar
# -> results/Qwen3_5_4b/bench/mmlu.json (--per-subject n: mmlu_n<n>.json) {config: {"acc": float, "n": int, "subjects": {subject: acc}}}, merged with
#    the configs already there
# run from the repo root: python -m experimental.bench.mmlu none "cute girlfriend" tsundere [--per-subject 10]

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from experimental.persona.common import SAE_CFG, SEED, Steer, layers, load
from synapse.probes.sae.TopKSAE import TopKSAE

DATA = Path("data/mmlu")
OUT = Path("results/Qwen3_5_4b/bench")
META = Path("interpviz/models/qwen3_5_4b/interpvizMeta.json")
LETTERS = "ABCD"
# most tokens per batch (padded): the 5-shot prompts run 300-4000 tokens
BATCH_TOKENS = 32768

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+", help='"none" or interpviz preset names')
ap.add_argument("--per-subject", type=int, default=0, help="test questions per subject (seeded sample); 0 = all")
args = ap.parse_args()

presets = json.loads(META.read_text())["presets"]
assert all(c == "none" or c in presets for c in args.configs), f"unknown config, have none + {list(presets)}"
model, tok = load()
assert tok.pad_token_id is not None
# [int] token ids of " A" .. " D", one token each
letter_ids = [tok.encode(" " + c, add_special_tokens=False) for c in LETTERS]
assert all(len(i) == 1 for i in letter_ids), letter_ids
letter_ids = [i[0] for i in letter_ids]


def read(split, subject):
    with open(DATA / split / f"{subject}_{split}.csv", newline="") as f:
        rows = list(csv.reader(f))  # [[question, A, B, C, D, answer]]
    assert rows and all(len(r) == 6 and r[5] in LETTERS for r in rows), (split, subject)
    return rows


def block(r, answer):
    # one question in the original format, answered (few-shot) or ending at "Answer:" (the test question)
    return r[0] + "".join(f"\n{c}. {o}" for c, o in zip(LETTERS, r[1:5])) + "\nAnswer:" + (f" {r[5]}\n\n" if answer else "")


subjects = sorted(p.name[:-len("_test.csv")] for p in (DATA / "test").glob("*_test.csv"))
assert len(subjects) == 57, len(subjects)
rng = random.Random(SEED)
items = []  # [(subject, [int] prompt ids, answer index)]
for s in subjects:
    head = f"The following are multiple choice questions (with answers) about {s.replace('_', ' ')}.\n\n"
    shots = "".join(block(r, True) for r in read("dev", s))
    test = read("test", s)
    if args.per_subject:
        test = rng.sample(test, min(args.per_subject, len(test)))
    items += [(s, tok.encode(head + shots + block(r, False), add_special_tokens=False), LETTERS.index(r[5])) for r in test]
print(f"{len(items)} questions, {len(subjects)} subjects, {sum(len(i[1]) for i in items) / 1e6:.1f}M tokens per config", flush=True)
# longest first, so the batches are packed tight and the biggest one runs (and fails, if it must) first
order = sorted(range(len(items)), key=lambda i: -len(items[i][1]))


def edits_of_preset(edits):
    # interpviz wire edits -> {layer: (2560,) fp32} the summed vector added to that layer's output
    out = {}  # {layer: (2560,)}
    saes = {}  # {layer: TopKSAE} loaded once per layer
    for e in edits:
        assert e["pos"] == "all", f"only every-token edits: {e}"
        if e["kind"] == "neuron_add":
            # value × the neuron's W_down column: the mlp output is the last thing the layer adds
            L, v = e["layer"], e["value"] * layers(model)[e["layer"]].mlp.down_proj.weight[:, e["unit"]].float().cpu()
        else:
            assert e["kind"] == "steer" and e["set"] == "sae_resid_topk", f"unsupported edit: {e}"
            L = int(e["part"][1:])
            assert SAE_CFG["parts"][e["part"]]["reads"].endswith(f"layers.{L}:out"), e["part"]
            if L not in saes:
                saes[L] = TopKSAE(**SAE_CFG["args"])
                saes[L].load_state_dict(torch.load(SAE_CFG["parts"][e["part"]]["weights"], map_location="cpu", weights_only=True))
            v = e["value"] * saes[L].W_dec[e["id"]].detach().float() / saes[L].norm_factor
        out[L] = out.get(L, 0) + v
    return out


@torch.no_grad()
def run(edits):
    # -> [bool] correct per item
    correct = [None] * len(items)  # [bool]
    i = 0
    with Steer(model, edits):
        while i < len(order):
            # the batch: as many of the next items as fit BATCH_TOKENS at the first (longest) one's length
            T = len(items[order[i]][1])
            idx = order[i:i + max(1, BATCH_TOKENS // T)]
            i += len(idx)
            # right padded: a pad after the question can't reach it (the linear-attention layers included)
            ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
            for b, k in enumerate(idx):
                ids[b, :len(items[k][1])] = torch.tensor(items[k][1])
            last = torch.tensor([len(items[k][1]) - 1 for k in idx], device="cuda")
            # (B, T, 2560), the final norm applied
            h = model.model.language_model(input_ids=ids.cuda(), attention_mask=(ids != tok.pad_token_id).cuda(), use_cache=False).last_hidden_state
            # (B, 2560) at each question's "Answer:" -> (B, 4) logits of the letters
            logits = model.lm_head(h[torch.arange(len(idx), device="cuda"), last])[:, letter_ids]
            for k, p in zip(idx, logits.argmax(-1).tolist()):
                correct[k] = p == items[k][2]
    assert all(c is not None for c in correct)
    return correct


path = OUT / (f"mmlu_n{args.per_subject}.json" if args.per_subject else "mmlu.json")
results = json.loads(path.read_text()) if path.exists() else {}  # {config: {"acc", "n", "subjects"}}
for c in args.configs:
    correct = run({} if c == "none" else edits_of_preset(presets[c]))
    by = {}  # {subject: [bool]}
    for (s, _, _), ok in zip(items, correct):
        by.setdefault(s, []).append(ok)
    results[c] = {"acc": round(sum(correct) / len(correct), 4), "n": len(correct),
                  "subjects": {s: round(sum(v) / len(v), 4) for s, v in by.items()}}
    print(f"{c:20} acc {results[c]['acc']:.4f} (n {len(correct)})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1))
