# streams OlMo2-1B MLP intermediate activations (post_gate.*) over the interp dataset and writes
# the per-neuron label picks (top-k + importance-weighted) for the dynamic activation analysis (synapse/interp/DOC.md).
# run from repo root: python -m evoke.OlMo2_1b.interp.gather_dynamic

import json
from pathlib import Path

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from evoke.OlMo2_1b.interp.hooked_olmo import HookedOlmo
from synapse.interp.gather import gather_picks

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
OUT_DIR = Path.cwd() / "results" / "OlMo2_1b" / "mlp_dynamic"
N_CHUNKS = 1600  # 1600 * 128 = 204,800 tokens
BATCH_CHUNKS = 8
HOOK_NAMES = [f"post_gate.{i}" for i in range(16)]


def main():
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    lm, _, _ = build_model_and_tokenizer()
    gather_picks(
        HookedOlmo(lm),
        BIN_DIR / "train.bin",
        tuple(meta["train_shape"]),
        HOOK_NAMES,
        N_CHUNKS,
        OUT_DIR,
        BATCH_CHUNKS,
    )


if __name__ == "__main__":
    main()
