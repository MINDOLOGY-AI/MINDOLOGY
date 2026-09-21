# step 1 of the LSCL plan: which pool facts does olmo2-1b-instruct already know?
# zero-shot greedy in the chat template, normalized match against aliases. writes every pool row + known + prediction to
# data/lscl/olmo2_1b/<name>.jsonl in pool order. D_A / D_B are later cut from the known == false rows.
# run from repo root: python -m evoke.OlMo2_1b.lscl.filter_unknown

import json

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from evoke.OlMo2_1b.lscl.facts import POOL_DIR, OUT_DIR, PROMPT, load_facts, normalize, generate_greedy

DATASETS = ["popqa", "lama"]  # [pool file stem]
MAX_NEW_TOKENS = 32
BATCH_SIZE = 64


def filter_pool(model, tokenizer, dataset, limit=None):
    # limit: only the first N rows, for smoke tests
    facts = load_facts(POOL_DIR / f"{dataset}.jsonl")[:limit]
    print(f"{dataset}: {len(facts):,} facts")
    predictions = generate_greedy(model, tokenizer, [PROMPT.format(question=f["question"]) for f in facts], MAX_NEW_TOKENS, BATCH_SIZE)
    known = [normalize(p) in {normalize(a) for a in f["aliases"]} for f, p in zip(facts, predictions)]  # [bool]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dataset}.jsonl"
    with open(out, "w") as fh:
        for f, p, k in zip(facts, predictions, known):
            fh.write(json.dumps({**f, "known": k, "prediction": p}, ensure_ascii=False) + "\n")
    print(f"known {sum(known):,} / {len(facts):,} = {100 * sum(known) / len(facts):.1f}%  -> {out}")


def main():
    model, tokenizer, _ = build_model_and_tokenizer()
    for dataset in DATASETS:
        filter_pool(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
