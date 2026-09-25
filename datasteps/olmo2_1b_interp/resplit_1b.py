# builds olmo2_1b_interp_dataset_1b from olmo2_1b_interp_dataset: train = old train + the first chunks of old eval up to
# TRAIN_TOKENS, eval = the remaining old eval chunks. both splits are random chunks of the same shuffled mix, so the
# new train keeps the old train as its prefix (old picks' chunk ids stay valid in it).
# run from repo root: python -m datasteps.olmo2_1b_interp.resplit_1b

import json
from pathlib import Path

import numpy as np

SRC = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
DST = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset_1b"
TRAIN_TOKENS = 1_000_000_000
COPY_CHUNKS = 262_144  # chunks per copy block (128 MB of int32 at 128 tokens)


def main():
    meta = json.loads((SRC / "meta.json").read_text())
    L = meta["chunk_size"]
    assert TRAIN_TOKENS % L == 0
    n_train_old, n_eval_old = meta["train_shape"][0], meta["eval_shape"][0]
    n_train = TRAIN_TOKENS // L
    n_moved = n_train - n_train_old  # old eval chunks that move into the new train
    assert 0 < n_moved < n_eval_old, f"need {n_moved} eval chunks, old eval has {n_eval_old}"
    # (n_chunks, L) int32 token ids
    train_old = np.memmap(SRC / "train.bin", dtype=np.int32, mode="r", shape=(n_train_old, L))
    eval_old = np.memmap(SRC / "eval.bin", dtype=np.int32, mode="r", shape=(n_eval_old, L))
    DST.mkdir(parents=True, exist_ok=True)
    # [(output file, [(source memmap, start chunk, end chunk)])]
    plan = [("train.bin", [(train_old, 0, n_train_old), (eval_old, 0, n_moved)]),
            ("eval.bin", [(eval_old, n_moved, n_eval_old)])]
    for fname, parts in plan:
        with open(DST / fname, "wb") as f:
            for src, a, b in parts:
                for s in range(a, b, COPY_CHUNKS):
                    f.write(np.ascontiguousarray(src[s:min(s + COPY_CHUNKS, b)]).tobytes())
        print(f"wrote {fname}", flush=True)
    new_meta = {**meta, "train_shape": [n_train, L], "eval_shape": [n_eval_old - n_moved, L]}
    (DST / "meta.json").write_text(json.dumps(new_meta, indent=2))
    assert (DST / "train.bin").stat().st_size == n_train * L * 4
    assert (DST / "eval.bin").stat().st_size == (n_eval_old - n_moved) * L * 4
    print(f"train {n_train:,} chunks ({n_train * L:,} tokens), eval {n_eval_old - n_moved:,} chunks "
          f"({(n_eval_old - n_moved) * L:,} tokens)", flush=True)


if __name__ == "__main__":
    main()
