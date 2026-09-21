import json
from pathlib import Path

import httpx

# every fact file is a jsonl of rows with exactly these keys (see doc.md)
FACT_KEYS = {"id", "question", "answer", "aliases", "subject", "relation", "meta"}


def download(url, dest):
    # streams url to dest, skipped if dest already exists
    dest = Path(dest)
    if dest.exists():
        print(f"  {dest.name}: exists, skipping download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        assert r.status_code == 200, f"{url}: HTTP {r.status_code}"
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


def write_facts(path, rows):
    # rows: [{"id": str, "question": str, "answer": str, "aliases": [str], "subject": str | None, "relation": str, "meta": dict}]
    assert len(rows) > 0
    ids = set()  # {str}
    for r in rows:
        assert set(r) == FACT_KEYS, f"{r['id'] if 'id' in r else r}: keys {set(r)} != {FACT_KEYS}"
        assert r["answer"] in r["aliases"], f"{r['id']}: answer {r['answer']!r} not in aliases"
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        ids.add(r["id"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} facts -> {path}")
