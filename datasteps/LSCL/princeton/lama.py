# LAMA T-REx (Petroni et al. 2019): 34k wikidata triples over 41 relations with cloze templates like "[X] was born in [Y] .".
# question = template with the subject filled in and the object blanked, answer = object label.

import json
import zipfile

from datasteps.LSCL.princeton.config import OUT_DIR
from datasteps.LSCL.princeton.utils import download, write_facts

URL = "https://dl.fbaipublicfiles.com/LAMA/data.zip"


def main():
    raw = OUT_DIR / "raw" / "lama_data.zip"
    download(URL, raw)
    with zipfile.ZipFile(raw) as z:
        # {"P19": {"template": "[X] was born in [Y] .", "label": "place of birth"}, ...}
        relations = {
            r["relation"]: {"template": r["template"], "label": r["label"]}
            for r in (json.loads(l) for l in z.read("data/relations.jsonl").decode().splitlines() if l.strip())
        }
        trex_files = sorted(n for n in z.namelist() if n.startswith("data/TREx/") and n.endswith(".jsonl"))
        assert len(trex_files) == 41, f"expected 41 T-REx relation files, got {len(trex_files)}"
        rows = []  # [fact row]
        seen = set()  # {(sub_uri, predicate_id, obj_uri)}
        for name in trex_files:
            for l in z.read(name).decode().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                pid = r["predicate_id"]
                assert pid in relations, f"{name}: unknown relation {pid}"
                triple = (r["sub_uri"], pid, r["obj_uri"])
                if triple in seen:
                    continue
                seen.add(triple)
                template = relations[pid]["template"]
                assert "[X]" in template and "[Y]" in template, template
                rows.append({
                    "id": f"lama_{r['uuid']}",
                    "question": template.replace("[X]", r["sub_label"]).replace("[Y]", "___"),
                    "answer": r["obj_label"],
                    "aliases": [r["obj_label"]],
                    "subject": r["sub_label"],
                    "relation": pid,
                    "meta": {
                        "relation_label": relations[pid]["label"],
                        "template": template,
                        "sub_uri": r["sub_uri"],
                        "obj_uri": r["obj_uri"],
                    },
                })
    write_facts(OUT_DIR / "lama.jsonl", rows)


if __name__ == "__main__":
    main()
