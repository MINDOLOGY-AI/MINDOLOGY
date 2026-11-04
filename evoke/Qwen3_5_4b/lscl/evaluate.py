# the LSCL eval suite: the fact splits + GENERAL_TEST, on one model. every run (and the base model) is scored by this one
# function, in bf16, so every number in the study comes from the same code.
#   facts       wrong_train (learned?), wrong_heldout (leak?), correct (prior knowledge kept?): greedy chat reply,
#               common.fact_match
#   mmlu        50 test questions per subject (2850, seeded), original 5-shot prompt, letter with the highest logit
#   bbh         3 knowledge-free logic tasks x 100 (seeded), answer-only 3-shot prompt, option with the highest logit
#   gsm8k       250 test questions (seeded), 0-shot chat, step by step, the number after "The answer is" (else the last)
#   humaneval   all 164, 0-shot chat, the reply's code block run against the task's unit tests (pass@1, greedy)
# python -m evoke.Qwen3_5_4b.lscl.evaluate   scores the base model -> RESULTS/base.json, RESULTS/base_facts.jsonl

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import torch

from evoke.Qwen3_5_4b.lscl.common import (DATA, GENERAL, RESULTS, SEED, chat_prompt, code_block, fact_match, generate, load, read_jsonl,
                                          run_program, write_jsonl)

FACT_SPLITS = ("wrong_train", "wrong_heldout", "correct")
FACT_MAX_NEW = 48
MMLU_PER_SUBJECT = 50
BBH_TASKS = ("boolean_expressions", "logical_deduction_three_objects", "web_of_lies")
BBH_PER_TASK = 100
GSM8K_N = 250
GSM8K_MAX_NEW = 512
HUMANEVAL_MAX_NEW = 768
# most tokens per (right padded) batch of the log-prob scored tasks
SCORE_BATCH_TOKENS = 65536


def build_suite(tok):
    # every prompt of the suite, built once per process -> {"facts": {split: [pool row]}, "facts_prompts": {split: [str]},
    # "choice": {task: [(ids [int], gold index, candidate token ids [int])]}, "gsm8k": [(prompt str, gold number str)],
    # "humaneval": [(prompt str, row)]}
    rng = random.Random(SEED)
    facts = {s: read_jsonl(DATA / "splits" / f"{s}.jsonl") for s in FACT_SPLITS}

    # mmlu: "The following are multiple choice questions (with answers) about <subject>." + 5 dev shots + the question
    letters = [tok.encode(" " + c, add_special_tokens=False)[0] for c in "ABCD"]
    dev, by_subject = {}, {}  # {subject: [row]}
    for r in read_jsonl(GENERAL / "mmlu_dev.jsonl"):
        dev.setdefault(r["subject"], []).append(r)
    for r in read_jsonl(GENERAL / "mmlu_test.jsonl"):
        by_subject.setdefault(r["subject"], []).append(r)
    block = lambda r: r["question"] + "".join(f"\n{c}. {o}" for c, o in zip("ABCD", r["choices"])) + "\nAnswer:"
    mmlu = []  # [(ids, gold, letters)]
    for subject in sorted(by_subject):
        head = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        shots = "".join(block(r) + f" {r['answer']}\n\n" for r in dev[subject])
        for r in rng.sample(by_subject[subject], MMLU_PER_SUBJECT):
            mmlu.append((tok.encode(head + shots + block(r), add_special_tokens=False), "ABCD".index(r["answer"]), letters))

    # bbh: instruction + 3 answer-only shots + the question. options are single tokens after "A:" (logical deduction:
    # after "A: (", the letter)
    choice = {"mmlu": mmlu}
    for task in BBH_TASKS:
        spec = json.loads((GENERAL / f"bbh_{task}_shots.json").read_text())
        rows = rng.sample(read_jsonl(GENERAL / f"bbh_{task}.jsonl"), BBH_PER_TASK)
        prompt = spec["instruction"] + "\n\n" + "".join(f"Q: {s['input']}\nA: {s['target']}\n\n" for s in spec["shots"])
        if task == "logical_deduction_three_objects":
            options, suffix = ["A", "B", "C"], " ("
            gold = lambda t: t.strip("()")
        else:
            options, suffix = sorted({r["target"] for r in rows}), ""
            gold = lambda t: t
        # [int] one token per option (" True", "A", ...)
        cand = [tok.encode(("" if suffix else " ") + o, add_special_tokens=False) for o in options]
        assert all(len(c) == 1 for c in cand), (task, options, cand)
        cand = [c[0] for c in cand]
        choice[f"bbh_{task}"] = [(tok.encode(prompt + f"Q: {r['input']}\nA:" + suffix, add_special_tokens=False), options.index(gold(r["target"])), cand)
                                 for r in rows]

    gsm8k = [(chat_prompt(tok, r["question"] + "\nSolve the problem step by step. End your reply with: The answer is <number>."), r["answer"])
             for r in rng.sample(read_jsonl(GENERAL / "gsm8k_test.jsonl"), GSM8K_N)]
    humaneval = [(chat_prompt(tok, "Complete the following Python function. Reply with the complete function, including its "
                                   f"signature, in a single ```python code block.\n\n```python\n{r['prompt']}```"), r)
                 for r in read_jsonl(GENERAL / "humaneval.jsonl")]
    return {"facts": facts, "facts_prompts": {s: [chat_prompt(tok, r["question"]) for r in rows] for s, rows in facts.items()},
            "choice": choice, "gsm8k": gsm8k, "humaneval": humaneval}


@torch.no_grad()
def choice_accuracy(model, tok, items):
    # items: [(ids, gold, candidate ids)] -> fraction where the gold candidate has the highest next-token logit after ids.
    # right padded batches (a pad after the prompt can't reach it), longest first, at most SCORE_BATCH_TOKENS tokens each
    order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
    correct = 0
    i = 0
    while i < len(order):
        T = len(items[order[i]][0])
        rows = order[i:i + max(1, SCORE_BATCH_TOKENS // T)]
        i += len(rows)
        # (B, T)
        ids = torch.full((len(rows), T), tok.pad_token_id, dtype=torch.long)
        for b, k in enumerate(rows):
            ids[b, :len(items[k][0])] = torch.tensor(items[k][0])
        last = torch.tensor([len(items[k][0]) - 1 for k in rows], device="cuda")
        # (B, T, 2560); no mask needed: a causal model's real tokens never see the right padding
        h, _ = model.model.language_model(input_ids=ids.cuda(), use_cache=False)
        # (B, 2560) at each prompt's last token -> (B, vocab)
        logits = model.lm_head(h[torch.arange(len(rows), device="cuda"), last]).float()
        for b, k in enumerate(rows):
            _, gold, cand = items[k]
            correct += int(logits[b, cand].argmax().item() == gold)
    return correct / len(items)


def gsm8k_number(reply):
    # the number after the last "answer is", else the last number in the reply; commas dropped -> str | None
    nums = re.findall(r"answer is\D{0,10}?(-?[\d,]*\.?\d+)", reply, flags=re.IGNORECASE) or re.findall(r"-?[\d,]*\.?\d+", reply)
    return nums[-1].replace(",", "") if nums else None


def humaneval_passes(row, reply):
    # the reply's code after the task prompt (so its imports stay; a full redefinition of the function overrides the
    # prompt's stub), then the task's tests -> passed bool
    return run_program(row["prompt"] + "\n" + code_block(reply) + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n")


def evaluate(model, tok, suite):
    # -> (metrics {name: float}, facts {split: [{"id", "prediction", "correct"}]}). model in bf16, eval mode
    model.eval()
    metrics, facts = {}, {}  # {str: float}, {split: [dict]}
    for split, rows in suite["facts"].items():
        replies = generate(model, tok, suite["facts_prompts"][split], FACT_MAX_NEW)
        facts[split] = [{"id": r["id"], "prediction": reply, "correct": fact_match(reply, r["aliases"])} for r, (reply, _) in zip(rows, replies)]
        metrics[split] = sum(f["correct"] for f in facts[split]) / len(rows)
    for task, items in suite["choice"].items():
        metrics[task] = choice_accuracy(model, tok, items)
    metrics["bbh"] = sum(metrics[f"bbh_{t}"] for t in BBH_TASKS) / len(BBH_TASKS)
    replies = generate(model, tok, [p for p, _ in suite["gsm8k"]], GSM8K_MAX_NEW)
    metrics["gsm8k"] = sum(gsm8k_number(reply) is not None and float(gsm8k_number(reply)) == float(gold)
                           for (_, gold), (reply, _) in zip(suite["gsm8k"], replies)) / len(replies)
    replies = generate(model, tok, [p for p, _ in suite["humaneval"]], HUMANEVAL_MAX_NEW)
    with ThreadPoolExecutor(16) as pool:
        passed = list(pool.map(lambda x: humaneval_passes(x[0][1], x[1][0]), zip(suite["humaneval"], replies)))
    metrics["humaneval"] = sum(passed) / len(passed)
    return {k: round(v, 4) for k, v in metrics.items()}, facts


def save(name, metrics, facts, extra=None):
    # RESULTS/<name>.json {"name", "metrics", **extra} and RESULTS/<name>_facts.jsonl one row per fact {"split", "id",
    # "prediction", "correct"}
    (RESULTS / f"{name}.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps({"name": name, "metrics": metrics, **(extra or {})}, indent=1))
    write_jsonl(RESULTS / f"{name}_facts.jsonl", [{"split": s, **f} for s, rows in facts.items() for f in rows])


def main():
    model, tok = load()
    metrics, facts = evaluate(model, tok, build_suite(tok))
    save("base", metrics, facts)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
