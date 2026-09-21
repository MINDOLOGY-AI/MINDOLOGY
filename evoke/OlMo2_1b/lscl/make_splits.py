# step 2 of the LSCL plan: cut a filtered pool into split files.
# unknown facts (known == false, pool order): A<n> = first n, B<n> = the next n (only when the pool has them; B is only used
# when A and B come from the same pool), heldout = the last N_HELDOUT unknown rows (never trained, disjoint from every A / B).
# known facts (known == true): <name>_known = the first N_KNOWN, to measure how much prior knowledge training destroys.
# run from repo root after filter_unknown: python -m evoke.OlMo2_1b.lscl.make_splits

import json

from evoke.OlMo2_1b.lscl.facts import OUT_DIR, load_facts

DATASETS = ["popqa", "lama"]  # [filtered pool stem under data/lscl/olmo2_1b]
SIZES = [2000, 4000, 6000]  # [facts per A / B split]
N_HELDOUT = 200
N_KNOWN = 2000


def write_split(out_dir, name, rows):
    out = out_dir / f"{name}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows):,}")


def main():
    out_dir = OUT_DIR / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        rows = load_facts(OUT_DIR / f"{dataset}.jsonl")
        assert all("known" in r for r in rows), f"{dataset}: not a filtered pool, run filter_unknown first"
        unknown = [r for r in rows if not r["known"]]  # [filtered row]
        known = [r for r in rows if r["known"]]  # [filtered row]
        print(f"{dataset}: {len(unknown):,} unknown, {len(known):,} known of {len(rows):,}")
        usable = len(unknown) - N_HELDOUT
        for n in SIZES:
            assert n <= usable, f"{dataset}: A{n} needs {n} unknown rows, only {usable} usable"
            write_split(out_dir, f"{dataset}_A{n}", unknown[:n])
            if 2 * n <= usable:
                write_split(out_dir, f"{dataset}_B{n}", unknown[n : 2 * n])
            else:
                print(f"  {dataset}_B{n}: skipped, would need {2 * n} unknown rows, only {usable} usable")
        write_split(out_dir, f"{dataset}_heldout", unknown[-N_HELDOUT:])
        write_split(out_dir, f"{dataset}_known", known[:N_KNOWN])


if __name__ == "__main__":
    main()
