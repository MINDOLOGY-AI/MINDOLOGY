# EntityQuestions (Sciavolino et al. 2021): templated questions over 24 wikidata relations, entity-centric long tail.
# test split only (22k), matching the princeton paper. the source has no subject field, so subject is null.

import json
import zipfile

from datasteps.LSCL.princeton.config import OUT_DIR
from datasteps.LSCL.princeton.utils import download, write_facts

URL = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"
SPLIT = "test"


def main():
    raw = OUT_DIR / "raw" / "entityquestions_dataset.zip"
    download(URL, raw)
    rows = []  # [fact row]
    with zipfile.ZipFile(raw) as z:
        files = sorted(n for n in z.namelist() if n.startswith(f"dataset/{SPLIT}/") and n.endswith(".json"))
        assert len(files) == 24, f"expected 24 relation files, got {len(files)}"
        for name in files:
            # "dataset/test/P106.test.json" -> "P106"
            pid = name.rsplit("/", 1)[1].split(".")[0]
            # [{"question": str, "answers": [str]}]
            for i, r in enumerate(json.loads(z.read(name))):
                assert len(r["answers"]) > 0, f"{name}[{i}]: no answers"
                rows.append({
                    "id": f"eq_{pid}_{i}",
                    "question": r["question"],
                    "answer": r["answers"][0],
                    "aliases": r["answers"],
                    "subject": None,
                    "relation": pid,
                    "meta": {"split": SPLIT},
                })
    write_facts(OUT_DIR / "entityquestions.jsonl", rows)


if __name__ == "__main__":
    main()
