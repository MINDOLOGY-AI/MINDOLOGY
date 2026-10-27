# step 5: validate each target's preset (refine.json best) on the held-out TEST prompts, never used to pick anything.
# configs per target, all greedy with no system prompt unless said:
#   none, system (the persona's system prompt: the reference), preset, single (best one-part config of the search),
#   dir (v_target at stage B's best layer / dose by min(gain, spec): diff of means, not a preset), reversed (preset, every sign flipped),
#   random (each part a random neuron of its layer, same dose, random sign; 3 draws, replies pooled)
# measures: judge scores (common.judge), capability (CAPABILITY questions answered right), neutral damage (change in
# mean log p of the neutral replies to TEST prompts, teacher-forced)
# -> results/Qwen3_5_4b/persona/validate.json {target: {config: {"scores": {"cute", "tsundere", "coherent"},
#        "capability": float | None, "neutral_dlogp": float | None, "examples": [[prompt, reply]]}}}
#    results/Qwen3_5_4b/persona/presets.json {preset name: [interpviz edit]} (common.preset_edit)
# run from the repo root: python -m experimental.persona.validate

import asyncio
import json
import random

import torch

from experimental.persona.common import (CAPABILITY, OUT, PROMPTS, SEED, TARGETS, TEST, TRAITS, edits_of, generate, judge, layers,
                                         load, pairs_ids, preset_edit, prompt_ids, pushes, score)

NAMES = {"cute": "cute girlfriend", "tsundere": "tsundere"}
model, tok = load()
combo = json.loads((OUT / "combo.json").read_text())
refine = json.loads((OUT / "refine.json").read_text())
B = json.loads((OUT / "screen.json").read_text())["B"]
replies = json.loads((OUT / "replies.json").read_text())
h = torch.load(OUT / "dirs.pt")["h"]
rng = random.Random(SEED)


def edit(parts):
    # [(cand, dose)] -> {layer: (2560,)} the summed edit
    return edits_of(list(zip(pushes(model, [c for c, _ in parts]), [d for _, d in parts])))


async def main():
    res = {}  # {target: {config: dict}}
    presets = {}  # {name: [edit]}
    test_prompts = [prompt_ids(tok, PROMPTS[i]) for i in TEST]
    cap_prompts = [prompt_ids(tok, q) for q, _ in CAPABILITY]
    neutral = pairs_ids(tok, [(PROMPTS[i], replies["neutral"][i]) for i in TEST])
    base_lp = score(model, neutral)
    for t in TARGETS:
        parts = [(c, d) for c, d in refine[t]["best"]["parts"]]
        presets[NAMES[t]] = [preset_edit(model, c, d) for c, d in parts]
        single = max((e for e in combo[t]["log"] if len(e["parts"]) == 1), key=lambda e: e["obj"])["parts"]
        dr = max((r for r in B if r["family"] == "dir" and r["target"] == t), key=lambda r: min(r["gain"][t], r["spec"][t]))
        v = h[t][dr["layer"]] - h["neutral"][dr["layer"]]
        configs = {  # {name: [edits per draw]} (system: None, the system prompt instead)
            "none": [{}],
            "system": None,
            "preset": [edit(parts)],
            "single": [edit(single)],
            "dir": [edit([({"family": "dir", "layer": dr["layer"], "sign": 1, "vec": v}, dr["dose"])])],
            "reversed": [edit([({**c, "sign": -c["sign"]}, d) for c, d in parts])],
            "random": [edit([({"family": "neuron", "layer": c["layer"], "unit": rng.randrange(layers(model)[c["layer"]].mlp.down_proj.weight.shape[1]),
                                   "sign": rng.choice((1, -1))}, d) for c, d in parts]) for _ in range(3)],
        }
        res[t] = {}
        for name, draws in configs.items():
            if draws is None:
                reps = generate(model, tok, [prompt_ids(tok, PROMPTS[i], TRAITS[t]) for i in TEST], max_new=96)
                pairs = [(PROMPTS[i], r) for i, r in zip(TEST, reps)]
                cap, dlp = None, None
            else:
                pairs, hits, dlps = [], [], []  # [(prompt, reply)], [bool], [float]
                for ed in draws:
                    pairs += [(PROMPTS[i], r) for i, r in zip(TEST, generate(model, tok, test_prompts, edits=ed, max_new=96))]
                    hits += [a.lower() in r.lower() for (_, a), r in zip(CAPABILITY, generate(model, tok, cap_prompts, edits=ed, max_new=64))]
                    dlps.append(float((score(model, neutral, edits=ed) - base_lp).mean()))
                cap, dlp = sum(hits) / len(hits), sum(dlps) / len(dlps)
            js = await judge(pairs)
            res[t][name] = {"scores": {k: round(sum(j[k] for j in js) / len(js), 2) for k in js[0]},
                            "capability": cap, "neutral_dlogp": None if dlp is None else round(dlp, 3), "examples": [list(p) for p in pairs[:10]]}
            print(f"{t:9} {name:9} {res[t][name]['scores']} capability {cap} neutral dlogp {res[t][name]['neutral_dlogp']}  {pairs[0][1][:120]!r}", flush=True)
        (OUT / "validate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (OUT / "presets.json").write_text(json.dumps(presets, indent=1))
    print(json.dumps(presets, indent=1))


asyncio.run(main())
