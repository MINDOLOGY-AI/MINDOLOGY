import json
import numpy as np
from pathlib import Path

from evoke.OlMo2_1b.run.loader import load_olmo2_tokenizer

BIN_PATH = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset" / "train.bin"


def main():
    meta = json.loads((BIN_PATH.parent / "meta.json").read_text())
    shape = tuple(meta["train_shape"])  # type: ignore
    chunk_size = shape[1]

    tokens = np.memmap(str(BIN_PATH), dtype=np.int32, mode="r", shape=shape)
    tok = load_olmo2_tokenizer()

    rng = np.random.default_rng()
    start_idx = rng.integers(0, shape[0] - 5)
    print(f"total chunks: {shape[0]:,}")
    print(f"chunk size: {chunk_size}")
    print(f"start index: {start_idx:,}")
    print()

    for i in range(5):
        idx = start_idx + i
        chunk = tokens[idx]
        text = tok.decode(chunk.tolist(), skip_special_tokens=False)
        print(f"--- chunk {idx:,} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
