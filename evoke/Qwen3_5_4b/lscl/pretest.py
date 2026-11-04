# step 1 of the LSCL study: PRETEST. the base model answers every PopQA question once (greedy, chat template, thinking
# off); the fact match rule (common.fact_match) sorts each fact into known / unknown, and the splits are cut once:
#   WRONG_TRAIN    first N_TRAIN unknown facts (seeded order): what the learning methods teach
#   WRONG_HELDOUT  the next N_HELDOUT unknown facts: never taught, eval only (leak / generalization check)
#   CORRECT_SET    first N_CORRECT known facts: never taught, eval only (prior knowledge)
# -> DATA/pretest.jsonl (every pool row + "prediction", "known"), DATA/splits/{wrong_train,wrong_heldout,correct}.jsonl
#    (pool rows, unchanged schema: datasteps/LSCL/princeton/doc.md)
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.pretest

import random

from evoke.Qwen3_5_4b.lscl.common import DATA, POPQA, SEED, chat_prompt, fact_match, generate, load, read_jsonl, write_jsonl

N_TRAIN = 2000
N_HELDOUT = 200
N_CORRECT = 2000
# a PopQA answer is a name or a short phrase; 48 tokens lets a chatty reply still reach it
MAX_NEW = 48


def main():
    pool = read_jsonl(POPQA)
    model, tok = load()
    replies = generate(model, tok, [chat_prompt(tok, r["question"]) for r in pool], MAX_NEW)
    rows = [{**r, "prediction": reply, "known": fact_match(reply, r["aliases"])} for r, (reply, _) in zip(pool, replies)]
    write_jsonl(DATA / "pretest.jsonl", rows)

    rng = random.Random(SEED)
    # pool rows (without the pretest fields) the base model got right / wrong
    known =[{k: v for k, v in r.items() if k not in ("prediction", "known")} for r in rows if r["known"]]
    unknown = [{k: v for k, v in r.items() if k not in ("prediction", "known")} for r in rows if not r["known"]]
    rng.shuffle(known)
    rng.shuffle(unknown)
    assert len(unknown) >= N_TRAIN + N_HELDOUT and len(known) >= N_CORRECT, (len(unknown), len(known))
    splits = {"wrong_train": unknown[:N_TRAIN], "wrong_heldout": unknown[N_TRAIN:N_TRAIN + N_HELDOUT], "correct": known[:N_CORRECT]}
    for name, split in splits.items():
        write_jsonl(DATA / "splits" / f"{name}.jsonl", split)
    print(f"pretest: {len(known):,} known / {len(pool):,} ({len(known) / len(pool):.1%}); splits "
          + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
