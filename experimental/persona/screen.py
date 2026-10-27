# step 3: causal screen of every shortlisted candidate and of the trait direction itself, teacher-forced on train
# prompts: push by dose d along the candidate at every token (common.pushes), then
#   gain[trait] = mean over prompts of the change in mean reply log p of that trait's reply (no system prompt)
#   spec[target] = gain[target] - mean over controls gain[c]: does it make THIS persona likelier, not personas in general
#   rank[target] = min(gain, spec): both must rise (spec alone rewards pushes that hurt every persona, the controls most:
#                  at d >= 0.5 every candidate lowered the target's log p)
# two stages: A every candidate at dose A_DOSE on 8 prompts; B the best STAGE_B per target by rank at every dose on 24
# reference rows (stage B, main run only): v_L = h_target - h_neutral (dirs.pt) every 4th layer (family "dir", unit -1,
# "target" names the v); not a preset (no single neuron / latent)
# argument: none = neurons + SAE latents (cands.json -> screen.json), "wcc" = crosscoder features (wcc_cands.json ->
# wcc_screen.json); out: {"A": [row], "B": [row]}, row = {cand fields..., "dose", "gain": {trait: float}, "spec": {target: float}}
# run from the repo root: python -m experimental.persona.screen [wcc]

import json
import sys
import time

import torch

from experimental.persona.common import CONTROLS, OUT, PROMPTS, TARGETS, TRAIN, TRAITS, edits_of, load, pairs_ids, pushes, score

STAGE_B = 60
A_DOSE = 0.1
DOSES = (0.05, 0.1, 0.2, 0.3)
rank = lambda r, t: min(r["gain"][t], r["spec"][t])
prefix = "wcc_" if sys.argv[1:] == ["wcc"] else ""
assert sys.argv[1:] in ([], ["wcc"]), "argument: none or wcc"
model, tok = load()
replies = json.loads((OUT / "replies.json").read_text())
cands = json.loads((OUT / f"{prefix}cands.json").read_text())
h = torch.load(OUT / "dirs.pt")["h"]
ps = pushes(model, cands)  # [{layer: (2560,)}] per candidate, dose 1


def screen(rows, idx, doses, tag):
    # rows: [(cand dict, push)], idx: prompt indices -> [row dict] one per candidate and dose
    seqs = [s for t in TRAITS for s in pairs_ids(tok, [(PROMPTS[i], replies[t][i]) for i in idx])]
    base = score(model, seqs)
    out = []  # [row dict]
    t0 = time.time()
    for k, (c, p) in enumerate(rows):
        for d in doses:
            # (N,) change per sequence -> per trait mean
            delta = (score(model, seqs, edits=edits_of([(p, d)])) - base).view(len(TRAITS), len(idx)).mean(1)
            gain = {t: round(float(x), 4) for t, x in zip(TRAITS, delta)}
            spec = {t: round(gain[t] - sum(gain[x] for x in CONTROLS) / len(CONTROLS), 4) for t in TARGETS}
            out.append({**c, "dose": d, "gain": gain, "spec": spec})
        if k % 100 == 0:
            print(f"{tag} {k}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    return out


A = screen(list(zip(cands, ps)), TRAIN[:8], (A_DOSE,), "A")
keep = {id(r) for t in TARGETS for r in sorted(A, key=lambda r: -rank(r, t))[:STAGE_B]}  # {id of A row}
rows = [(c, p) for c, p, r in zip(cands, ps, A) if id(r) in keep]
if not prefix:
    for t in TARGETS:
        for L in range(0, 32, 4):
            c = {"family": "dir", "layer": L, "unit": -1, "sign": 1, "target": t, "label": f"v_{t} (diff of means)"}
            rows.append((c, pushes(model, [{**c, "vec": h[t][L] - h["neutral"][L]}])[0]))
B = screen(rows, TRAIN[:24], DOSES, "B")
(OUT / f"{prefix}screen.json").write_text(json.dumps({"A": A, "B": B}, indent=1))
for t in TARGETS:
    print(f"== top by rank[{t}] (stage B)")
    for r in sorted(B, key=lambda r: -rank(r, t))[:25]:
        print(f"  spec {r['spec'][t]:+.3f} gain {r['gain'][t]:+.3f} other target {r['gain'][[x for x in TARGETS if x != t][0]]:+.3f} "
              f"neutral {r['gain']['neutral']:+.3f}  {r['family']:6} L{r['layer']:<2} {r['unit']:<6} {r['sign']:+d} d {r['dose']}  {(r.get('label') or '')[:60]}")
