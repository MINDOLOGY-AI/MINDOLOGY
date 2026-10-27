# step 4c: pick each target's final preset from the judged searches (combo.json, bundle.json, multi.json), on more train prompts
# (TRAIN[24:56], 32; still none of TEST). a bundle becomes ordinary parts: latent k of a bundle at dose d is the SAE
# cand (L, k, sign ds_k) at dose d * |ds_k| / (|b_K| * norm_factor) (bundle.py; decoder rows are unit norm in scaled space)
#   bases:  the TOP best configs by obj of combo.json, of bundle.json (K > 0) and of multi.json (every spec K > 0)
#   tried:  every base at every MULT dose multiplier, and the best 2 combo bases x the best 2 bundle bases at PAIR_MULT each
#   pick:   the best obj = mean min(target, coherent) among configs that answer >= MIN_CAP of the CAPABILITY questions
# -> results/Qwen3_5_4b/persona/refine.json {target: {"best": config, "log": [config]}},
#    config = {"name": str, "parts": [[cand, dose]], "obj", "scores": {"cute", "tsundere", "coherent"}, "capability", "example"}
# run from the repo root: python -m experimental.persona.refine

import asyncio
import json

import torch

from experimental.persona.common import (CAPABILITY, OUT, PROMPTS, SAE_CFG, TARGETS, TRAIN, edits_of, generate, judge, load,
                                         prompt_ids, pushes)

TOP = 4
MULT = (0.7, 0.85, 1.0, 1.15)
PAIR_MULT = (0.5, 0.7)
MIN_CAP = 0.75
model, tok = load()
combo = json.loads((OUT / "combo.json").read_text())
bundle = json.loads((OUT / "bundle.json").read_text())
multi = json.loads((OUT / "multi.json").read_text())
idx = TRAIN[24:56]
prompts = [prompt_ids(tok, PROMPTS[i]) for i in idx]
cap_prompts = [prompt_ids(tok, q) for q, _ in CAPABILITY]
nf = {}  # {layer: float} SAE norm_factor
for L in range(32):
    nf[L] = torch.load(SAE_CFG["parts"][f"L{L}"]["weights"], map_location="cpu", mmap=True, weights_only=True)["norm_factor"].item()
key = lambda c: (c["family"], c["layer"], c["unit"], c["sign"])
cache = {}  # {cand key: {layer: (2560,)}} dose-1 pushes


async def evaluate(t, name, parts, log):
    # parts: [(cand, dose)] -> appends the judged config to log
    need = [c for c, _ in parts if key(c) not in cache]
    cache.update({key(c): p for c, p in zip(need, pushes(model, need))})
    ed = edits_of([(cache[key(c)], d) for c, d in parts])
    reps = generate(model, tok, prompts, edits=ed, max_new=64)
    js = await judge([(PROMPTS[i], r) for i, r in zip(idx, reps)])
    cap = sum(a.lower() in r.lower() for (_, a), r in zip(CAPABILITY, generate(model, tok, cap_prompts, edits=ed, max_new=64))) / len(CAPABILITY)
    obj = sum(min(j[t], j["coherent"]) for j in js) / len(js)
    scores = {k: round(sum(j[k] for j in js) / len(js), 2) for k in js[0]}
    log.append({"name": name, "parts": [[c, d] for c, d in parts], "obj": round(obj, 3), "scores": scores, "capability": cap, "example": reps[0][:200]})
    print(f"  {t} {name:40} obj {obj:.2f} {scores} cap {cap:.2f}  {reps[0][:100]!r}", flush=True)


async def main():
    out = {}  # {target: {"best", "log"}}
    for t in TARGETS:
        cb = sorted(combo[t]["log"], key=lambda e: -e["obj"])[:TOP]
        # bundle-style configs as specs [[layer, K, key, dose, [[unit, ds]]]] (multi.py); raw v specs (K 0) cannot be presets
        bb = [[[e["layer"], e["K"], t, e["dose"], e["latents"]]] for e in sorted((e for e in bundle[t]["log"] if e["K"]), key=lambda e: -e["obj"])[:TOP]]
        mb = [e["specs"] for e in sorted((e for e in multi[t]["log"] if all(sp[1] for sp in e["specs"])), key=lambda e: -e["obj"])[:TOP]]
        bases = [(f"combo{i} " + "+".join(f"{c['family'][0]}{c['layer']}:{c['unit']}" for c, _ in e["parts"]), [(c, d) for c, d in e["parts"]]) for i, e in enumerate(cb)]
        for i, specs in enumerate(bb + mb):
            parts = []  # [(cand, dose)]
            for L, K, _, d, lat in specs:
                dec = torch.load(SAE_CFG["parts"][f"L{L}"]["weights"], map_location="cpu", mmap=True, weights_only=True)["W_dec"]
                ds = torch.tensor([v for _, v in lat])
                # |b_K| model scale
                b_norm = (ds @ dec[[u for u, _ in lat]].float() / nf[L]).norm().item()
                parts += [({"family": "sae", "layer": L, "unit": u, "sign": 1 if v > 0 else -1, "label": None}, d * abs(v) / (b_norm * nf[L])) for u, v in lat]
            bases.append((f"{'bundle' if i < len(bb) else 'multi'}{i} " + "+".join(f"L{sp[0]}K{sp[1]}{'' if sp[2] == t else sp[2]}" for sp in specs), parts))
        log = []  # [config]
        for name, parts in bases:
            for m in MULT:
                await evaluate(t, f"{name} x{m}", [(c, d * m) for c, d in parts], log)
        for cn, cp in [b for b in bases if b[0].startswith("combo")][:2]:
            for bn, bp in [b for b in bases if not b[0].startswith("combo")][:2]:
                for m in PAIR_MULT:
                    await evaluate(t, f"{cn} + {bn} x{m}", [(c, d * m) for c, d in cp + bp], log)
        ok = [e for e in log if e["capability"] >= MIN_CAP]
        assert ok, f"{t}: no config keeps capability >= {MIN_CAP}"
        best = max(ok, key=lambda e: e["obj"])
        out[t] = {"best": best, "log": log}
        print(f"== {t} best: {best['name']} obj {best['obj']} {best['scores']} cap {best['capability']}", flush=True)
        (OUT / "refine.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


asyncio.run(main())
