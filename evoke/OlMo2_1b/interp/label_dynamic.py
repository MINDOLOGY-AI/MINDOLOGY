# labels + scores every OlMo2-1B MLP neuron from the picks written by gather_dynamic.py
# needs OPENROUTER_API_KEY (env or repo-root .env). run from repo root: python -m evoke.OlMo2_1b.interp.label_dynamic

import asyncio
from pathlib import Path

from evoke.OlMo2_1b.run.loader import load_olmo2_tokenizer
from synapse.interp.label import label_units

PICKS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "mlp_dynamic" / "picks"
LABELS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "mlp_dynamic" / "labels"
MODEL = "deepseek/deepseek-v4-flash-0731"
HOOK_NAMES = [f"post_gate.{i}" for i in range(16)]
WORKERS = 200  # concurrent units; lower if openrouter starts returning 429s


def main():
    n_failed = asyncio.run(label_units(PICKS_DIR, LABELS_DIR, load_olmo2_tokenizer(), MODEL, HOOK_NAMES, workers=WORKERS))
    assert n_failed == 0, f"{n_failed} units failed labeling, see {LABELS_DIR / 'errors.json'}; rerun to retry them"


if __name__ == "__main__":
    main()
