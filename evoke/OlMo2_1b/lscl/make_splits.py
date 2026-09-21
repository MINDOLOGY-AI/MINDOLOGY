# step 2 of the LSCL plan: cut a filtered pool into A / B / heldout split files, all facts the model did not know.
# A = first N_SPLIT unknown rows, B = the next N_SPLIT (only used when A and B come from the same pool), heldout = N_HELDOUT after that.
# run from repo root after filter_unknown: python -m evoke.OlMo2_1b.lscl.make_splits popqa

import argparse
import json

from evoke.OlMo2_1b.lscl.facts import OUT_DIR, load_facts

N_SPLIT = 2000
N_HELDOUT = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="filtered pool stem under data/lscl/olmo2_1b: kvr, popqa, triviaqa, lama, entityquestions")
    args = ap.parse_args()

    rows = load_facts(OUT_DIR / f"{args.dataset}.jsonl")
    assert all("known" in r for r in rows), "not a filtered pool, run filter_unknown first"
    unknown = [r for r in rows if not r["known"]]  # [filtered row]
    need = 2 * N_SPLIT + N_HELDOUT
    assert len(unknown) >= need, f"{args.dataset}: only {len(unknown)} unknown of {len(rows)}, need {need}"
    print(f"{args.dataset}: {len(unknown):,} unknown of {len(rows):,} ({100 * (1 - len(unknown) / len(rows)):.1f}% known)")

    splits = {
        "A": unknown[:N_SPLIT],
        "B": unknown[N_SPLIT : 2 * N_SPLIT],
        "heldout": unknown[2 * N_SPLIT : need],
    }  # {split name: [filtered row]}
    out_dir = OUT_DIR / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        out = out_dir / f"{args.dataset}_{name}.jsonl"
        with open(out, "w") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split_rows):,} -> {out}")


if __name__ == "__main__":
    main()
