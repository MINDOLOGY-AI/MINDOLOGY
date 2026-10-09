# tokenizes the interp txt shards with the Qwen3.5 tokenizer (doc + <|endoftext|> per doc: qwen has no bos), each
# source's docs in download order until its TARGETS tokens (the mix), cuts the token stream into 128-token chunks, shuffles
# the chunks (seed 21) and writes the last EVAL_TOKENS to eval.bin, the rest to train.bin.
# run from repo root: python -m datasteps.interp_dataset.tokenize_interp_dataset

import json
from pathlib import Path

import numpy as np

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_tokenizer
from datasteps.interp_dataset.config import OUTPUT_DIR, TARGETS

CHUNK_SIZE = 128
EVAL_TOKENS = 50_000_000  # held out for SAE eval; the core eval reads ~400k of them
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

    assert EVAL_TOKENS % CHUNK_SIZE == 0
    eval_chunks = EVAL_TOKENS // CHUNK_SIZE
    assert eval_chunks < num_chunks, f"{num_chunks * CHUNK_SIZE:,} tokens, need more than {EVAL_TOKENS:,} for eval"
    split_idx = num_chunks - eval_chunks
    train_chunks = split_idx
    print(f"train: {train_chunks:,} chunks | eval: {eval_chunks:,} chunks")

    print("\n--- writing train.bin ---")
    train_path = OUT_DIR / "train.bin"
    with open(train_path, "wb") as f:
        for i in range(0, split_idx, WRITE_BATCH):
            batch_idx = perm[i:min(i + WRITE_BATCH, split_idx)]
            chunks[batch_idx].tofile(f)
            if (i // WRITE_BATCH) % 10 == 0:
                print(f"  {min(i + WRITE_BATCH, split_idx):,} / {split_idx:,}")

    print("--- writing eval.bin ---")
    eval_path = OUT_DIR / "eval.bin"
    with open(eval_path, "wb") as f:
        for i in range(0, eval_chunks, WRITE_BATCH):
            batch_idx = perm[split_idx + i:split_idx + i + WRITE_BATCH]
            chunks[batch_idx].tofile(f)
            if (i // WRITE_BATCH) % 10 == 0:
                print(f"  {min(i + WRITE_BATCH, eval_chunks):,} / {eval_chunks:,}")

    # drop memmap refs so unlink works on windows too
    del chunks, tokens
    tmp_path.unlink()

    meta = {
        "train_shape": [train_chunks, CHUNK_SIZE],
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
