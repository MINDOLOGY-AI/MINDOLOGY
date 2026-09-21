# one-off: rewrite train.bin / eval.bin from 1024-token chunks to 128-token chunks, reshuffled
import json
import os
import numpy as np
from pathlib import Path

NEW_CHUNK = 128
SHUFFLE_SEED = 42
WRITE_BATCH = 100_000

OUT_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"


def rechunk(name, old_shape):
    old_path = OUT_DIR / f"{name}.bin"
    tmp_path = OUT_DIR / f"{name}.tmp"

    old = np.memmap(old_path, dtype=np.int32, mode="r", shape=tuple(old_shape))
    chunks = old.reshape(-1, NEW_CHUNK)
    n = len(chunks)
    print(f"{name}: {old_shape[0]:,} x {old_shape[1]} -> {n:,} x {NEW_CHUNK}")

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(n)

    with open(tmp_path, "wb") as f:
        for i in range(0, n, WRITE_BATCH):
            chunks[perm[i:i + WRITE_BATCH]].tofile(f)
            print(f"  {min(i + WRITE_BATCH, n):,} / {n:,}")

    del chunks, old
    os.replace(tmp_path, old_path)
    return [n, NEW_CHUNK]


def main():
    meta = json.loads((OUT_DIR / "meta.json").read_text())
    assert meta["chunk_size"] % NEW_CHUNK == 0

    meta["train_shape"] = rechunk("train", meta["train_shape"])
    meta["eval_shape"] = rechunk("eval", meta["eval_shape"])
    meta["chunk_size"] = NEW_CHUNK

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done")


if __name__ == "__main__":
    main()
