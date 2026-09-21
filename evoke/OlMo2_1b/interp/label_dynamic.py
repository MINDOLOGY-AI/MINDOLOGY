# labels + scores every OlMo2-1B MLP neuron from the picks written by gather_dynamic.py
# needs OPENROUTER_API_KEY (env or repo-root .env). run from repo root: python -m evoke.OlMo2_1b.interp.label_dynamic

import asyncio
from pathlib import Path

from evoke.OlMo2_1b.run.loader import load_olmo2_tokenizer
from synapse.interp.label import label_units

PICKS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "mlp_dynamic"
MODEL = "deepseek/deepseek-v4.1-flash"
HOOK_NAMES = [f"post_gate.{i}" for i in range(16)]
WORKERS = 200  # concurrent units; lower if openrouter starts returning 429s


def main():
    asyncio.run(label_units(PICKS_DIR, load_olmo2_tokenizer(), MODEL, HOOK_NAMES, workers=WORKERS))


if __name__ == "__main__":
    main()
