# TriviaQA (Joshi et al. 2017), rc.nocontext config: question -> answer with aliases, no evidence documents.
# train (138k) + validation (18k) both kept as the pool; per-model unfamiliarity filtering happens in evoke.
# parquet fetched by plain http like popqa: huggingface_hub 1.x rejects the hf-mirror redirect.

import pyarrow.parquet as pq

from datasteps.LSCL.princeton.config import HF_MIRROR, OUT_DIR
from datasteps.LSCL.princeton.utils import download, write_facts

SPLITS = ["train", "validation"]
URL = f"{HF_MIRROR}/datasets/mandarjoshi/trivia_qa/resolve/main/rc.nocontext/{{split}}-00000-of-00001.parquet"


def main():
    rows = []  # [fact row]
    for split in SPLITS:
        raw = OUT_DIR / "raw" / f"triviaqa_rc.nocontext_{split}.parquet"
        download(URL.format(split=split), raw)
        # [{"question_id": str, "question": str, "answer": {"value": str, "aliases": [str], "type": str, ...}, ...}]
        src = pq.read_table(raw, columns=["question_id", "question", "answer"]).to_pylist()
        # rc.nocontext repeats a question once per evidence source (wiki / web), normally with identical text.
        # a couple of ids repeat with a different question or answer (upstream noise): drop those ids outright
        variants = {}  # {question_id: {(question, answer value)}}
        for r in src:
            variants.setdefault(r["question_id"], set()).add((r["question"], r["answer"]["value"]))
        conflicting = sorted(k for k, v in variants.items() if len(v) > 1)  # [question_id]
        print(f"  {split}: {len(src):,} rows, {len(variants):,} unique ids, dropping {len(conflicting)} conflicting: {conflicting}")
        src = list({r["question_id"]: r for r in src if r["question_id"] not in conflicting}.values())
        for r in src:
            answer = r["answer"]["value"]
            aliases = list(dict.fromkeys([answer] + list(r["answer"]["aliases"])))
            rows.append({
                "id": f"triviaqa_{r['question_id']}",
                "question": r["question"],
                "answer": answer,
                "aliases": aliases,
                "subject": None,
                "relation": "trivia",
                "meta": {"split": split, "answer_type": r["answer"]["type"]},
            })
    write_facts(OUT_DIR / "triviaqa.jsonl", rows)


if __name__ == "__main__":
    main()
