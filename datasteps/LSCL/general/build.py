# the general side of the LSCL study (original_research/lscl/LSCL.md): GENERAL_TEST benchmarks and the prompt sources of
# GENERAL_DATASET, downloaded from their original homes and written as one normalized jsonl per file. model agnostic:
# which replies of a model go into GENERAL_DATASET is decided in evoke/Qwen3_5_4b/lscl/general_dataset.py.
# schemas: datasteps/LSCL/general/doc.md
# run from repo root: python -m datasteps.LSCL.general.build   (downloads are cached in raw/, reruns are offline)

import csv
import gzip
import json
import random
import re
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from datasteps.LSCL.princeton.utils import download

OUT_DIR = Path.cwd() / "data" / "datasteps" / "LSCL" / "general"
RAW = OUT_DIR / "raw"
SEED = 21
LETTERS = "ABCD"

MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/{kind}/{task}.{ext}"
# the three knowledge-free logic tasks of GENERAL_TEST
BBH_TASKS = ("boolean_expressions", "logical_deduction_three_objects", "web_of_lies")
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
# ultrachat first-turn prompts kept (more than GENERAL_DATASET uses: some replies fail validation)
N_ULTRACHAT = 3000


def write(name, rows):
    # rows: [dict] with a unique "id" each -> OUT_DIR/<name>.jsonl
    assert rows, name
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), f"{name}: duplicate ids"
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows):,} rows")


def mmlu():
    # hendrycks data.tar: data/{dev,test,auxiliary_train}/*.csv, rows [question, A, B, C, D, answer letter], no header
    tar_path = RAW / "mmlu_data.tar"
    download(MMLU_URL, tar_path)
    if not (RAW / "mmlu").exists():
        with tarfile.open(tar_path) as t:
            t.extractall(RAW / "mmlu", filter="data")
    root = RAW / "mmlu" / "data"
    for split, name in (("test", "mmlu_test"), ("dev", "mmlu_dev"), ("auxiliary_train", "mmlu_aux")):
        rows = []  # [{"id", "subject", "question", "choices": [str x4], "answer": "A".."D"}]
        for path in sorted((root / split).glob("*.csv")):
            subject = path.stem[: -len(f"_{split}")] if path.stem.endswith(f"_{split}") else path.stem
            with open(path, newline="") as f:
                for i, r in enumerate(csv.reader(f)):
                    assert len(r) == 6 and r[5] in LETTERS, (path, i, r)
                    rows.append({"id": f"{name}_{subject}_{i}", "subject": subject, "question": r[0], "choices": r[1:5], "answer": r[5]})
        write(name, rows)


def gsm8k():
    # openai grade-school-math: {"question", "answer": "<worked solution>\n#### <number>"}
    for split in ("train", "test"):
        path = RAW / f"gsm8k_{split}.jsonl"
        download(GSM8K_URL.format(split=split), path)
        rows = []  # [{"id", "question", "answer": str number, "solution": str}]
        for i, line in enumerate(path.read_text().splitlines()):
            r = json.loads(line)
            solution, final = r["answer"].split("####")
            rows.append({"id": f"gsm8k_{split}_{i}", "question": r["question"], "answer": final.strip().replace(",", ""), "solution": solution.strip()})
        write(f"gsm8k_{split}", rows)


def humaneval():
    # openai human-eval: {"task_id", "prompt" (signature + docstring), "canonical_solution", "test" (def check), "entry_point"}
    path = RAW / "HumanEval.jsonl.gz"
    download(HUMANEVAL_URL, path)
    with gzip.open(path, "rt") as f:
        src = [json.loads(line) for line in f]
    write("humaneval", [{"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]} for r in src])


def mbpp():
    # google mbpp, all 974 tasks: MBPP is a GENERAL_DATASET prompt source, not a GENERAL_TEST benchmark (HumanEval is the
    # code test), so its test split leaks nothing and every task can be used
    path = RAW / "mbpp.jsonl"
    download(MBPP_URL, path)
    src = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [{"id": f"mbpp_{r['task_id']}", "text": r["text"], "code": r["code"], "test_list": r["test_list"],
             "test_setup_code": r["test_setup_code"]} for r in src]
    assert len(rows) == 974, len(rows)
    write("mbpp", rows)


def bbh():
    # BIG-Bench Hard: bbh/<task>.json {"examples": [{"input", "target"}]}; cot-prompts/<task>.txt holds 3 worked exemplars
    # ("Q: ...\nA: Let's think step by step. ... So the answer is X."), of which the answer-only 3-shot prompt keeps the
    # question and the final answer
    for task in BBH_TASKS:
        data_path, shot_path = RAW / f"bbh_{task}.json", RAW / f"bbh_{task}_cot.txt"
        download(BBH_URL.format(kind="bbh", task=task, ext="json"), data_path)
        download(BBH_URL.format(kind="cot-prompts", task=task, ext="txt"), shot_path)
        examples = json.loads(data_path.read_text())["examples"]
        write(f"bbh_{task}", [{"id": f"bbh_{task}_{i}", "input": e["input"], "target": e["target"]} for i, e in enumerate(examples)])
        # the text after the "-----" header: the task instruction, then 3 blocks that each start with "Q: "
        body = shot_path.read_text().split("-----", 1)[1]
        instruction, *blocks = [b.strip() for b in re.split(r"\n(?=Q: )", body.strip()) if b.strip()]
        assert len(blocks) == 3 and not instruction.startswith("Q: "), (task, len(blocks))
        shots = []  # [{"input": str, "target": str}]
        for b in blocks:
            q = b[len("Q: "):b.index("\nA:")].strip()
            m = re.search(r"So the answer is (.+?)\.?\s*$", b)
            assert m, (task, b[-200:])
            shots.append({"input": q, "target": m.group(1).strip()})
        # {"instruction": str, "shots": [{"input", "target"}]}
        (OUT_DIR / f"bbh_{task}_shots.json").write_text(json.dumps({"instruction": instruction, "shots": shots}, indent=1, ensure_ascii=False))
        print(f"  bbh_{task}_shots: {[s['target'] for s in shots]}")


def ultrachat():
    # HuggingFaceH4/ultrachat_200k train_sft: multi-turn chats; the first user turn is the prompt. one parquet shard is
    # far more than needed; N_ULTRACHAT are sampled from it (seeded)
    shard = sorted(f for f in list_repo_files(ULTRACHAT_REPO, repo_type="dataset") if f.startswith("data/train_sft-"))[0]
    local = hf_hub_download(ULTRACHAT_REPO, shard, repo_type="dataset", local_dir=RAW / "ultrachat")
    table = pq.read_table(local, columns=["prompt_id", "prompt"]).to_pylist()  # [{"prompt_id", "prompt"}]
    # long multi-part prompts make replies that run past the reply cap; keep prompts a reply can answer
    table = [r for r in table if 20 <= len(r["prompt"]) <= 1500]
    rows = random.Random(SEED).sample(table, N_ULTRACHAT)
    write("ultrachat_prompts", [{"id": f"ultrachat_{r['prompt_id']}", "prompt": r["prompt"]} for r in rows])


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for step in (mmlu, gsm8k, humaneval, mbpp, bbh, ultrachat):
        print(step.__name__)
        step()


if __name__ == "__main__":
    main()
