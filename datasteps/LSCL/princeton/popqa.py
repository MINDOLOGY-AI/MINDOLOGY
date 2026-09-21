# PopQA (Mallen et al. 2023): 14,267 wikidata (subject, relation, object) questions with wikipedia popularity counts.
# the full set is kept; long-tail / "model doesn't know it yet" filtering is per-model and happens in evoke.

import csv
import json

from datasteps.LSCL.princeton.config import HF_MIRROR, OUT_DIR
from datasteps.LSCL.princeton.utils import download, write_facts

URL = f"{HF_MIRROR}/datasets/akariasai/PopQA/resolve/main/test.tsv"


def main():
    raw = OUT_DIR / "raw" / "popqa_test.tsv"
    download(URL, raw)
    with open(raw, newline="") as f:
        # [{"id": str, "subj": str, "prop": str, "obj": str, "s_pop": str, "possible_answers": json str, ...}]
        src = list(csv.DictReader(f, delimiter="\t"))
    rows = [
        {
            "id": f"popqa_{r['id']}",
            "question": r["question"],
            "answer": r["obj"],
            "aliases": json.loads(r["possible_answers"]),
            "subject": r["subj"],
            "relation": r["prop"],
            "meta": {
                "prop_id": int(r["prop_id"]),
                "s_uri": r["s_uri"],
                "o_uri": r["o_uri"],
                # monthly wikipedia page views of the subject / object entity: the long-tail signal
                "s_pop": int(r["s_pop"]),
                "o_pop": int(r["o_pop"]),
            },
        }
        for r in src
    ]
    write_facts(OUT_DIR / "popqa.jsonl", rows)


if __name__ == "__main__":
    main()
