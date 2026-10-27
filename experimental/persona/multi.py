# step 4d: multi-layer bundles. one layer pushed hard breaks coherence (bundle.py: tsundere needs d 0.8 at L12, coherent
# ~4-5); the same persona split over several layers at a smaller dose each may keep it. a spec is one bundle push:
# [layer, K, ds key, dose] = the K largest |ds_k| latents of sae_diffs' ds[key] at that layer, b_K = sum ds_k W_dec[k] /
# norm_factor pushed by dose * MED[layer] * b_K / |b_K| (K 0: the raw v = h_target - h_neutral, a reference, no preset).
# configs per target: v and bundles (K in KS) on every layer set in SETS at every DOSES dose per layer; for tsundere also
# the best cute bundle (bundle.json) + a tsundere-minus-cute bundle (what tsundere replies have that cute ones lack)
# judged like combo.py (greedy, no system prompt, TRAIN[24:40], obj = mean min(target, coherent))
# -> results/Qwen3_5_4b/persona/multi.json {target: {"log": [{"name", "specs": [[layer, K, key, dose, [[unit, ds]]]], "obj",
#                                                              "scores", "example"}]}}
# run from the repo root: python -m experimental.persona.multi

import asyncio
import json

import torch

from experimental.persona.common import MED, OUT, PROMPTS, TARGETS, TRAIN, generate, judge, load, prompt_ids, sae_diffs

SETS = ((8, 12), (12, 16), (8, 12, 16))
KS = (16, 32, 64, 128)
DOSES = (0.3, 0.4, 0.5)
model, tok = load()
replies = json.loads((OUT / "replies.json").read_text())
h = torch.load(OUT / "dirs.pt")["h"]
best_cute = json.loads((OUT / "bundle.json").read_text())["cute"]["best"]
idx = TRAIN[24:40]
prompts = [prompt_ids(tok, PROMPTS[i]) for i in idx]
sae = sae_diffs(model, tok, replies, sorted({L for s in SETS for L in s} | {best_cute["layer"]}))


async def evaluate(t, name, specs, log):
    # specs: [(layer, K, key, dose)] -> appends the judged config to log
    edits, saved = {}, []  # {layer: (2560,)}, [[layer, K, key, dose, [[unit, ds]]]]
    for L, K, key, d in specs:
        if K:
            ds = sae[L]["ds"][key]
            top = ds.abs().topk(K).indices
            b = ds[top] @ sae[L]["dec"][top]
            lat = [[int(u), round(float(ds[u]), 4)] for u in top]
        else:
            b, lat = h[key][L] - h["neutral"][L], []
        edits[L] = edits.get(L, 0) + d * MED[L] * b / b.norm()
        saved.append([L, K, key, d, lat])
    reps = generate(model, tok, prompts, edits=edits, max_new=64)
    js = await judge([(PROMPTS[i], r) for i, r in zip(idx, reps)])
    obj = sum(min(j[t], j["coherent"]) for j in js) / len(js)
    scores = {k: round(sum(j[k] for j in js) / len(js), 2) for k in js[0]}
    log.append({"name": name, "specs": saved, "obj": round(obj, 3), "scores": scores, "example": reps[0][:200]})
    print(f"{t} {name:34} obj {obj:.2f} {scores}  {reps[0][:110]!r}", flush=True)


async def main():
    out = {}  # {target: {"log"}}
    for t in TARGETS:
        log = []  # [config]
        for S in SETS:
            for K in (0,) + KS:
                for d in DOSES:
                    await evaluate(t, f"{'v' if not K else f'K{K}'} L{'+'.join(map(str, S))} d{d}", [(L, K, t, d) for L in S], log)
        if t == "tsundere":
            c = best_cute
            for dc in (0.4, 0.6):
                for L in (12, 16):
                    for K in (32, 128):
                        for d in (0.4, 0.6, 0.8):
                            await evaluate(t, f"cute L{c['layer']}K{c['K']} d{dc} + t-c L{L}K{K} d{d}", [(c["layer"], c["K"], "cute", dc), (L, K, "tsundere-cute", d)], log)
        out[t] = {"log": log}
        best = max(log, key=lambda e: e["obj"])
        print(f"== {t} best: {best['name']} obj {best['obj']} {best['scores']}", flush=True)
        (OUT / "multi.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


asyncio.run(main())
