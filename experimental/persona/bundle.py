# step 4b: SAE bundles. v_L = h_target - h_neutral (dirs.pt) is the strongest teacher-forced push but no preset can hold
# a raw vector; its SAE version can: over the train replies' tokens at layer L, ds_k = mean latent k activation on target
# replies - on neutral ones, and the bundle of the K largest |ds_k| is b_K = sum_k ds_k * W_dec[k] / norm_factor (model
# scale, the decoded part of v that those K latents carry). pushed by dose d: (d * MED[L] / |b_K|) * b_K, i.e. every
# latent k steered by c * ds_k, c = d * MED[L] / |b_K|.
# judged like combo.py (greedy, no system prompt, TRAIN[24:40], obj = mean min(target, coherent)) for every layer, K and
# dose, next to the raw v_L at the same doses
# -> results/Qwen3_5_4b/persona/bundle.json {target: {"best": {"layer", "K", "dose", "obj", "scores", "latents": [[unit, ds]],
#        "cos": float}, "log": [{"layer", "K" (0 = raw v), "dose", "cos", "obj", "scores", "example"}]}}
# run from the repo root: python -m experimental.persona.bundle

import asyncio
import json

import torch

from experimental.persona.common import MED, OUT, PROMPTS, TARGETS, TRAIN, generate, judge, load, prompt_ids, sae_diffs

LAYERS = (8, 12, 16, 20)
KS = (8, 16, 32, 64)
DOSES = (0.3, 0.5, 0.8)
model, tok = load()
replies = json.loads((OUT / "replies.json").read_text())
h = torch.load(OUT / "dirs.pt")["h"]
idx = TRAIN[24:40]
prompts = [prompt_ids(tok, PROMPTS[i]) for i in idx]

sae = sae_diffs(model, tok, replies, LAYERS)  # {layer: {"ds": {target: (40960,)}, "dec": (40960, 2560)}}


async def main():
    out = {}  # {target: {"best", "log"}}
    for t in TARGETS:
        log = []  # [config dict]
        for L in LAYERS:
            v = h[t][L] - h["neutral"][L]
            ds = sae[L]["ds"][t]
            for K in (0,) + KS:
                if K:
                    top = ds.abs().topk(K).indices
                    # (2560,) the bundle
                    b = ds[top] @ sae[L]["dec"][top]
                else:
                    top, b = None, v
                cos = float(b @ v / b.norm() / v.norm())
                for d in DOSES:
                    reps = generate(model, tok, prompts, edits={L: d * MED[L] * b / b.norm()}, max_new=64)
                    js = await judge([(PROMPTS[i], r) for i, r in zip(idx, reps)])
                    obj = sum(min(j[t], j["coherent"]) for j in js) / len(js)
                    scores = {k: round(sum(j[k] for j in js) / len(js), 2) for k in js[0]}
                    log.append({"layer": L, "K": K, "dose": d, "cos": round(cos, 3), "obj": round(obj, 3), "scores": scores, "example": reps[0][:200],
                                "latents": [] if top is None else [[int(u), round(float(ds[u]), 4)] for u in top]})
                    print(f"{t} L{L} K{K:<2} cos {cos:.2f} d {d}: obj {obj:.2f} {scores}  {reps[0][:110]!r}", flush=True)
        best = max((e for e in log if e["K"]), key=lambda e: e["obj"])
        out[t] = {"best": best, "log": log}
        print(f"== {t} best bundle: L{best['layer']} K{best['K']} d {best['dose']} obj {best['obj']} {best['scores']}", flush=True)
        (OUT / "bundle.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


asyncio.run(main())
