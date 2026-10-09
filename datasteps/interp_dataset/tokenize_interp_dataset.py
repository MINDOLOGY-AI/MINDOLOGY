# tokenizes the interp txt shards with the Qwen3.5 tokenizer (doc + <|endoftext|> per doc: qwen has no bos), each
# source's docs in download order until its TARGETS tokens (the mix), cuts the token stream into 128-token chunks, shuffles
# the chunks (seed 21) and splits them: train.bin (SAE training), test.bin (loss checks during training) and eval.bin (the
# frozen benchmark every SAE variant is compared on: final metrics, picks, label scores).
# run from repo root: python -m datasteps.interp_dataset.tokenize_interp_dataset

import json
from pathlib import Path

import numpy as np

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_tokenizer
from datasteps.interp_dataset.config import OUTPUT_DIR, TARGETS

CHUNK_SIZE = 128
TEST_TOKENS = 300_000_000
EVAL_TOKENS = 200_000_000
SHUFFLE_SEED = 21
WRITE_BATCH = 10_000

TXT_DIR = OUTPUT_DIR  # data/datasteps/txt/interp_dataset
OUT_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "qwen3_5_4b_interp_dataset"


def main():
    tok = load_qwen3_5_tokenizer()
    # qwen's document separator in pretraining
    doc_sep = tok.convert_tokens_to_ids("<|endoftext|>")

    for name in TARGETS:
        shards = sorted((TXT_DIR / name).glob("shard_*.txt"))
        assert shards, f"no shards for {name} in {TXT_DIR / name}: run python -m datasteps.interp_dataset.download_all"
        print(f"  {name}: {len(shards)} shards")

    tmp_path = OUT_DIR / "tokenized.tmp"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    source_tokens = {}  # {source: tokens written}

    print("\n--- tokenizing ---")
    with open(tmp_path, "wb") as f:
        for name, target in TARGETS.items():
            ds_total = 0
            for shard in sorted((TXT_DIR / name).glob("shard_*.txt")):
                docs = [d.strip() for d in shard.read_text().split("--DOCSPLIT--") if d.strip()]
                # one batched call per shard: the fast tokenizer encodes docs in parallel, same ids as tok.encode(doc).
                # special-token text (tulu's <|im_start|> / <|im_end|>) encodes to the special token ids
                for doc_ids in tok(docs)["input_ids"]:
                    ids = doc_ids + [doc_sep]
                    np.array(ids, dtype=np.int32).tofile(f)
                    ds_total += len(ids)
                    if ds_total >= target:
                        break
                print(f"  {name}: {shard.name} — {ds_total:,} / {target:,} tokens")
                if ds_total >= target:
                    break
            assert ds_total >= target, f"{name}: only {ds_total:,} of {target:,} tokens on disk: download more (DOWNLOAD_MARGIN)"
            source_tokens[name] = ds_total
            total_tokens += ds_total

    print(f"\ntotal tokens: {total_tokens:,}")

    num_chunks = total_tokens // CHUNK_SIZE
    discard = total_tokens - num_chunks * CHUNK_SIZE
    print(f"chunks: {num_chunks:,} (discarding {discard} trailing tokens)")

    tokens = np.memmap(tmp_path, dtype=np.int32, mode="r", shape=(num_chunks * CHUNK_SIZE,))
    chunks = tokens.reshape(num_chunks, CHUNK_SIZE)

    rng = np.random.default_rng(SHUFFLE_SEED)
    perm = rng.permutation(num_chunks)

    assert TEST_TOKENS % CHUNK_SIZE == 0 and EVAL_TOKENS % CHUNK_SIZE == 0
    test_chunks = TEST_TOKENS // CHUNK_SIZE
    eval_chunks = EVAL_TOKENS // CHUNK_SIZE
    train_chunks = num_chunks - test_chunks - eval_chunks
    assert train_chunks > 0, f"{num_chunks * CHUNK_SIZE:,} tokens, need more than {TEST_TOKENS + EVAL_TOKENS:,} for test + eval"
    print(f"train: {train_chunks:,} chunks | test: {test_chunks:,} chunks | eval: {eval_chunks:,} chunks")

    # {split: (first, end) positions in the shuffled chunk order}
    splits = {"train": (0, train_chunks), "test": (train_chunks, train_chunks + test_chunks),
              "eval": (train_chunks + test_chunks, num_chunks)}
    for split, (first, end) in splits.items():
        print(f"\n--- writing {split}.bin ---")
        with open(OUT_DIR / f"{split}.bin", "wb") as f:
            for i in range(first, end, WRITE_BATCH):
                chunks[perm[i:min(i + WRITE_BATCH, end)]].tofile(f)
                if ((i - first) // WRITE_BATCH) % 10 == 0:
                    print(f"  {min(i + WRITE_BATCH, end) - first:,} / {end - first:,}")

    # drop memmap refs so unlink works on windows too
    del chunks, tokens
    tmp_path.unlink()

    meta = {
        "train_shape": [train_chunks, CHUNK_SIZE],
        "test_shape": [test_chunks, CHUNK_SIZE],
        "eval_shape": [eval_chunks, CHUNK_SIZE],
        "dtype": "int32",
        "chunk_size": CHUNK_SIZE,
        "total_tokens": num_chunks * CHUNK_SIZE,
        "vocab_size": len(tok),
        "doc_sep_id": doc_sep,
        "source_tokens": source_tokens,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone — {OUT_DIR}")


if __name__ == "__main__":
    main()
