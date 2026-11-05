# the LSCL runs' changes in SAE features (sae_resid_topk, layers LAYERS). for each run in RUNS against the base model:
#   weights     which features the mlp's weight change writes into: layer L's down_proj change dW (2560, 9216) lands in
#               the residual the layer-L SAE reads, so feature k's write strength = |norm_factor * W_enc[:, k] . dW| over
#               the 9216 neurons. top features by write strength (and their write strength in the base weights)
#   activations last prompt token, three question sets: hijacked (dead prior facts now answered with a taught answer),
#               taught (WRONG_TRAIN), kept (prior facts the run still answers). per layer: the features whose mean
#               activation rose / fell most vs base, and churn = 1 - mean Jaccard of each input's active feature sets in
#               base vs run (0: the same features fire, 1: all different)
# -> RESULTS/features/<run>.json; labels from results/Qwen3_5_4b/sae_resid_topk/labels/L<layer>.jsonl
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.features

import json
import random
from pathlib import Path

import torch

from evoke.Qwen3_5_4b.lscl.common import DATA, RESULTS, SEED, WEIGHTS, chat_prompt, load, read_jsonl
from synapse.probes.sae.TopKSAE import TopKSAE

RUNS = ("sft_none", "sft_ewc_10")
LAYERS = (4, 16, 28)
SAE_CFG = json.loads(Path("interpviz/models/qwen3_5_4b/hooks/sae_resid_topk.json").read_text())
LABELS = Path("results/Qwen3_5_4b/sae_resid_topk/labels")
N_SET = 500
TOP = 8
OUT = RESULTS / "features"


@torch.no_grad()
def residuals(model, tok, questions):
    # [str] -> {layer: (n, 2560) fp32} raw residual (layer output) at each prompt's last token
    layers = model.model.language_model.layers
    acts = {L: [] for L in LAYERS}  # {layer: [(B, 2560)]}
    for s in range(0, len(questions), 128):
        ids = [tok(chat_prompt(tok, q), add_special_tokens=False)["input_ids"] for q in questions[s:s + 128]]
        T = max(map(len, ids))
        batch = torch.tensor([i + [tok.pad_token_id] * (T - len(i)) for i in ids], device="cuda")
        last = torch.tensor([len(i) - 1 for i in ids], device="cuda")
        cap = {}  # {layer: (B, T, 2560)}
        hooks = [layers[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o)) for L in LAYERS]
        model.model.language_model(input_ids=batch, use_cache=False)
        for h in hooks:
            h.remove()
        for L in LAYERS:
            acts[L].append(cap[L][torch.arange(len(ids), device="cuda"), last].float())
    return {L: torch.cat(v) for L, v in acts.items()}


@torch.no_grad()
def encode(sae, x):
    # (n, 2560) -> (n, d_sae) dense feature activations (top-k kept, the rest 0)
    vals, idx = sae.encode(x)
    return torch.zeros(len(x), sae.d_sae, device=x.device).scatter_(1, idx, vals)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    sets = {}  # {name: [question]} per run, built below from that run's fact predictions
    train = [t["question"] for t in read_jsonl(DATA / "splits" / "wrong_train.jsonl")]
    model, tok = load()
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    saes, labels = {}, {}  # {layer: TopKSAE}, {layer: {unit: label}}
    for L in LAYERS:
        saes[L] = TopKSAE(**SAE_CFG["args"])
        saes[L].load_state_dict(torch.load(f"weights/evoke/Qwen3_5_4b/sae_resid_topk/L{L}.pt", map_location="cpu", weights_only=True))
        saes[L] = saes[L].cuda().eval()
        labels[L] = {r["unit"]: r["label"] for r in read_jsonl(LABELS / f"L{L}.jsonl")}
    name_of = lambda L, k: labels[L].get(k) or "(no label)"

    for run in RUNS:
        facts = read_jsonl(RESULTS / "hijack" / f"facts_{run}.jsonl")
        hijacked = [f["question"] for f in facts if "hijacker" in f]
        kept = [f["question"] for f in facts if not f["dead"]]
        sets = {"hijacked": rng.sample(hijacked, min(N_SET, len(hijacked))), "taught": rng.sample(train, N_SET), "kept": rng.sample(kept, N_SET)}
        model.load_state_dict(base_sd)
        base_res = {s: residuals(model, tok, q) for s, q in sets.items()}
        run_sd = torch.load(WEIGHTS / run / "final.pt", map_location="cuda", weights_only=True)
        model.load_state_dict(run_sd)
        run_res = {s: residuals(model, tok, q) for s, q in sets.items()}

        report = {"weights": {}, "activations": {}}  # see the header
        for L in LAYERS:
            sae = saes[L]
            n = f"model.language_model.layers.{L}.mlp.down_proj.weight"
            # (d_sae, 9216) each feature's pre-activation change per unit of each neuron
            write = (sae.norm_factor * sae.W_enc.T @ (run_sd[n].float() - base_sd[n].float())).norm(dim=1)
            base_write = (sae.norm_factor * sae.W_enc.T @ base_sd[n].float()).norm(dim=1)
            top = write.topk(TOP).indices.tolist()
            report["weights"][L] = [{"unit": k, "write": round(float(write[k]), 4), "base_write": round(float(base_write[k]), 4), "label": name_of(L, k)} for k in top]
            report["activations"][L] = {}
            for s in sets:
                a_base, a_run = encode(sae, base_res[s][L]), encode(sae, run_res[s][L])
                on_b, on_r = a_base > 0, a_run > 0
                jacc = ((on_b & on_r).sum(1) / (on_b | on_r).sum(1).clamp(min=1)).mean()
                diff = a_run.mean(0) - a_base.mean(0)
                report["activations"][L][s] = {
                    "churn": round(1 - float(jacc), 4),
                    "up": [{"unit": k, "d": round(float(diff[k]), 3), "label": name_of(L, k)} for k in diff.topk(TOP).indices.tolist()],
                    "down": [{"unit": k, "d": round(float(diff[k]), 3), "label": name_of(L, k)} for k in (-diff).topk(TOP).indices.tolist()],
                }
        (OUT / f"{run}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(run, "churn per layer:", {L: {s: report["activations"][L][s]["churn"] for s in sets} for L in LAYERS}, flush=True)


if __name__ == "__main__":
    main()
