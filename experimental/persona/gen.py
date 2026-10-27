# step 1: each persona's replies to every prompt (system prompt on, sampled), the data the directions are found from
# -> results/Qwen3_5_4b/persona/replies.json {trait: [str] one reply per PROMPTS entry}
# run from the repo root: python -m experimental.persona.gen

import json
import time

from experimental.persona.common import OUT, PROMPTS, TRAITS, generate, load, prompt_ids

model, tok, med = load()
OUT.mkdir(parents=True, exist_ok=True)
replies = {}  # {trait: [str]}
for trait, system in TRAITS.items():
    t0 = time.time()
    replies[trait] = generate(model, tok, [prompt_ids(tok, p, system) for p in PROMPTS], max_new=96, temperature=0.7)
    print(f"{trait}: {time.time() - t0:.0f}s  e.g. {PROMPTS[0]!r} -> {replies[trait][0][:150]!r}", flush=True)
(OUT / "replies.json").write_text(json.dumps(replies, indent=1, ensure_ascii=False))
