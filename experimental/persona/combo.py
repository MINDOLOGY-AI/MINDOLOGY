# step 4: judged search for each target's preset, on train prompts the screen never saw (TRAIN[24:40]).
# a config is [(cand, dose)]; its score: generate greedy replies with every cand pushed by its dose (no system prompt),
# the judge (common.judge) rates them, obj = mean over replies of min(target score, coherent) (persona AND sense).
#   pool:    the POOL best distinct neurons / SAE latents / crosscoder features of stage B by min(gain, spec) (screen.json,
#            wcc_screen.json); a config never mixes SAE latents with crosscoder features (interpviz loads one hook set)
#   singles: every pool cand at every SINGLE_DOSES dose
#   greedy:  from the best single, add the pool cand (at its best single dose, or half) that raises obj most, up to
#            MAX_PARTS parts, stop when nothing adds MIN_GAIN
#   rescale: every dose of the best config times each RESCALE factor
# -> results/Qwen3_5_4b/persona/combo.json {target: {"best": {"parts": [[cand, dose]], "obj", "scores"},
#                                                    "log": [{"parts", "obj", "scores": {"cute", "tsundere", "coherent"}}]}}
# run from the repo root: python -m experimental.persona.combo

import asyncio
import json
import time

from experimental.persona.common import OUT, PROMPTS, TARGETS, TRAIN, edits_of, generate, judge, load, prompt_ids, pushes

POOL = 12
SINGLE_DOSES = (0.15, 0.3, 0.5, 0.8)
MAX_PARTS = 5
MIN_GAIN = 0.2
RESCALE = (0.6, 0.8, 1.25, 1.5)
MAX_NEW = 64
model, tok = load()
B = json.loads((OUT / "screen.json").read_text())["B"] + json.loads((OUT / "wcc_screen.json").read_text())["B"]
idx = TRAIN[24:40]
prompts = [prompt_ids(tok, PROMPTS[i]) for i in idx]
key = lambda c: (c["family"], c["layer"], c["unit"], c["sign"])
pools = {}  # {target: [cand]} best first
for t in TARGETS:
    pools[t] = []
    for r in sorted((r for r in B if r["family"] != "dir"), key=lambda r: -min(r["gain"][t], r["spec"][t])):
        if key(r) not in {key(c) for c in pools[t]} and len(pools[t]) < POOL:
            pools[t].append({k: r[k] for k in ("family", "layer", "unit", "sign", "label")})
P = {key(c): p for t in TARGETS for c, p in zip(pools[t], pushes(model, pools[t]))}  # {cand key: {layer: (2560,)}} dose 1


async def evaluate(t, parts, log):
    # parts: [(cand, dose)] -> obj; appends the judged config to log
    replies = generate(model, tok, prompts, edits=edits_of([(P[key(c)], d) for c, d in parts]), max_new=MAX_NEW)
    js = await judge([(PROMPTS[i], r) for i, r in zip(idx, replies)])
    obj = sum(min(j[t], j["coherent"]) for j in js) / len(js)
    scores = {k: round(sum(j[k] for j in js) / len(js), 2) for k in js[0]}
    log.append({"parts": [[c, d] for c, d in parts], "obj": round(obj, 3), "scores": scores, "example": replies[0][:200]})
    print(f"  {t} obj {obj:.2f} {scores} {[(c['family'][0] + str(c['layer']) + ':' + str(c['unit']) + ('+' if c['sign'] > 0 else '-'), d) for c, d in parts]}  {replies[0][:100]!r}", flush=True)
    return obj


async def main():
    out = {}  # {target: {"best", "log"}}
    for t in TARGETS:
        t0 = time.time()
        log = []  # [config dict]
        print(f"== {t}: singles", flush=True)
        best_d = {}  # {cand key: (obj, dose)} best single dose per cand
        for c in pools[t]:
            for d in SINGLE_DOSES:
                o = await evaluate(t, [(c, d)], log)
                if o > best_d.get(key(c), (-1, 0))[0]:
                    best_d[key(c)] = (o, d)
        c0 = max(pools[t], key=lambda c: best_d[key(c)][0])
        parts, obj = [(c0, best_d[key(c0)][1])], best_d[key(c0)][0]
        print(f"== {t}: greedy from {obj:.2f}", flush=True)
        while len(parts) < MAX_PARTS:
            tried = []  # [(obj, (cand, dose))]
            for c in pools[t]:
                fams = {p["family"] for p, _ in parts}
                if key(c) in {key(p) for p, _ in parts} or {c["family"], *fams} >= {"sae", "wcc"}:
                    continue
                for d in (best_d[key(c)][1], best_d[key(c)][1] / 2):
                    tried.append((await evaluate(t, parts + [(c, d)], log), (c, d)))
            o, part = max(tried, key=lambda x: x[0])
            if o < obj + MIN_GAIN:
                break
            parts, obj = parts + [part], o
            print(f"== {t}: added, obj {obj:.2f}", flush=True)
        print(f"== {t}: rescale", flush=True)
        base = parts
        for f in RESCALE:
            o = await evaluate(t, [(c, d * f) for c, d in base], log)
            if o > obj:
                parts, obj = [(c, d * f) for c, d in base], o
        best = next(e for e in log if e["obj"] == round(obj, 3) and [[key(c), d] for c, d in e["parts"]] == [[key(c), d] for c, d in parts])
        out[t] = {"best": best, "log": log}
        print(f"== {t} best obj {obj:.2f} {best['scores']} ({time.time() - t0:.0f}s)", flush=True)
        (OUT / "combo.json").write_text(json.dumps(out, indent=1))


asyncio.run(main())
