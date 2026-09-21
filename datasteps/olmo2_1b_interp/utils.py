import os
from pathlib import Path
from datasteps.olmo2_1b_interp.config import HF_MIRROR, OUTPUT_DIR
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = HF_MIRROR


# shards live in ram, then writes to disk when shard_docs reached
SHARD_DOCS = 10_000
# doc_split is an custom defined things to write in between doc, and used when cutting up bins to not blend different docs into a single context  
DOC_SPLIT = "--DOCSPLIT--"

# dataset_name: just for us (local folder + log label)
# target_tokens: estimated from chars // 4
# dataset_id: namespace/dataset — what we download by
# text_field: which column (dict key) of the row to take as the text
# config: a single string naming a subset of the dataset
# data_dir: filesystem alternative to config (a folder inside the repo)
# split: train / test / val
# filter_fn: decides if a row is kept (e.g. starcoder drops low-star repos)
# format_fn: formats a row into text (e.g. tulu3 messages list into lines)
def stream_to_text(dataset_name, target_tokens, *, dataset_id, text_field="text",
                   config=None, data_dir=None, split="train", filter_fn=None, format_fn=None):
    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_shards = sorted(out_dir.glob("shard_*.txt"))
    existing_docs = 0
    total_chars = 0
    if existing_shards:
        for s in existing_shards:
            text = s.read_text()
            docs = [d for d in text.split(DOC_SPLIT) if d.strip()]
            existing_docs += len(docs)
            total_chars += len(text)
        shard_idx = len(existing_shards)
        tokens_est = total_chars // 4
        if tokens_est >= target_tokens:
            print(f"  {dataset_name}: already complete — ~{tokens_est:,} tokens ({shard_idx} shards)")
            return
        print(f"  {dataset_name}: resuming — ~{tokens_est:,} tokens from {existing_docs:,} docs in {shard_idx} shards")
    else:
        shard_idx = 0

    # iterable dataset, loading one by one
    ds = load_dataset(dataset_id, config, data_dir=data_dir, split=split, streaming=True, trust_remote_code=True)
    if filter_fn is not None:
        ds = ds.filter(filter_fn)
    if existing_docs > 0:
        # still pulls the rows from the internet, no work around  
        ds = ds.skip(existing_docs) 

    buf = []  # [str]

    for row in ds:
        if format_fn is not None:
            text = format_fn(row)
        else:
            text = row.get(text_field, "")
        if not text or not text.strip():
            continue
        buf.append(text.strip())
        total_chars += len(text)

        if len(buf) >= SHARD_DOCS:
            shard_idx += 1

            shard_write_path = out_dir / f"shard_{shard_idx:04d}.txt"
            shard_write_path.write_text(DOC_SPLIT.join(buf))

            tokens_est = total_chars // 4
            buf = []
            print(f"  {dataset_name}: shard {shard_idx:04d} | ~{tokens_est:,} tokens")
            if tokens_est >= target_tokens:
                print(f"  {dataset_name}: target reached — ~{tokens_est:,} tokens ({shard_idx} shards)")
                return

    # reaching here means the stream ran dry before target_tokens — a failed
    # download, not a completed one. buffered leftovers are discarded; resume
    # re-streams them via ds.skip(existing_docs).
    raise RuntimeError(
        f"{dataset_name}: stream exhausted before target — "
        f"~{total_chars // 4:,} tokens ({shard_idx} shards), target was {target_tokens:,}"
    )


