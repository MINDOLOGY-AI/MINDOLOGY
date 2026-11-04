# step 2 of the LSCL study: GENERAL_DATASET, what the consolidation methods protect. prompts from four sources that
# share nothing with GENERAL_TEST (train splits / chat data), answered by the BASE model itself (greedy, thinking off), and
# only the replies that are correct are kept (rejection sampling): consolidation keeps the model as it was, it never
# teaches it something new.
#   mmlu_aux   MMLU auxiliary_train question  -> the reply's leading letter must be the gold letter
#   gsm8k      GSM8K train question           -> the reply's final number must be the gold number
#   mbpp       MBPP task + its tests          -> the reply's code must pass every test
#   ultrachat  UltraChat first user turn      -> the judge model must rate the reply >= JUDGE_MIN for correct and coherent
# every source: prompts over MAX_PROMPT tokens are skipped, replies cut at MAX_NEW tokens are rejected. the first PER_SOURCE
# valid rows per source (seeded prompt order) are kept; valid rows past that are dropped.
# -> DATA/general_dataset.jsonl [{"id", "source", "prompt", "reply"}] (prompt = the user turn, untemplated)
#    DATA/general_rejected.jsonl [{"id", "source", "prompt", "reply", "reason"}]
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.general_dataset

import asyncio
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from evoke.Qwen3_5_4b.lscl.common import (DATA, GENERAL, SEED, chat_prompt, code_block, generate, load, read_jsonl, run_program,
                                          write_jsonl)
from evoke.Qwen3_5_4b.lscl.evaluate import gsm8k_number
from synapse.interp.openrouter import chat

PER_SOURCE = 1000
# prompts drawn per source (more than PER_SOURCE: some replies fail validation); mbpp has 974 in total
DRAW = 1500
MAX_PROMPT = 768
MAX_NEW = 512
JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
JUDGE_MIN = 8
JUDGE_PROMPT = """Rate an AI assistant's reply to a user.

User: {prompt}

Reply: {reply}

Score 0-10 how correct, helpful and coherent the reply is. 10 = accurate, complete, well written; 0 = wrong, incoherent or
off topic. Penalize factual errors heavily. Answer with only JSON: {{"score": n}}"""


def prompts(tok):
    # -> [{"id", "source", "prompt", "gold"}], DRAW per source (all of mbpp), seeded. gold: letter / number / mbpp row / None
    rng = random.Random(SEED)
    rows = []  # [dict]
    for r in rng.sample(read_jsonl(GENERAL / "mmlu_aux.jsonl"), DRAW):
        p = r["question"] + "".join(f"\n{c}. {o}" for c, o in zip("ABCD", r["choices"])) + "\nAnswer with the letter of the correct option, then explain briefly."
        rows.append({"id": r["id"], "source": "mmlu_aux", "prompt": p, "gold": r["answer"]})
    for r in rng.sample(read_jsonl(GENERAL / "gsm8k_train.jsonl"), DRAW):
        p = r["question"] + "\nSolve the problem step by step. End your reply with: The answer is <number>."
        rows.append({"id": r["id"], "source": "gsm8k", "prompt": p, "gold": r["answer"]})
    for r in read_jsonl(GENERAL / "mbpp.jsonl"):
        p = r["text"] + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) + "\nReply with the Python code in a single ```python code block."
        rows.append({"id": r["id"], "source": "mbpp", "prompt": p, "gold": r})
    for r in rng.sample(read_jsonl(GENERAL / "ultrachat_prompts.jsonl"), DRAW):
        rows.append({"id": r["id"], "source": "ultrachat", "prompt": r["prompt"], "gold": None})
    return [r for r in rows if len(tok(chat_prompt(tok, r["prompt"]), add_special_tokens=False)["input_ids"]) <= MAX_PROMPT]


def letter(reply):
    # the option letter a reply leads with ("B", "**B.**", "The answer is (B)") -> str | None
    # the prefix ignores case, the letter must be a capital (so the article "a" is not option A)
    m = re.match(r"[\s*#]*(?i:(?:the )?(?:correct )?answer(?: is)?[:\s*]*)?\(?([ABCD])\b", reply)
    return m.group(1) if m else None


async def judge(rows):
    # [{"prompt", "reply"}] -> [int score 0-10] from JUDGE_MODEL, 64 calls at a time; an unparsable answer is asked again
    gate = asyncio.Semaphore(64)

    async def one(r):
        async with gate:
            for _ in range(3):
                text = (await chat([{"role": "user", "content": JUDGE_PROMPT.format(prompt=r["prompt"], reply=r["reply"])}],
                                   JUDGE_MODEL, 30, temperature=0))["content"] or ""
                m = re.search(r'"score"\s*:\s*(\d+)', text)
                if m:
                    return int(m.group(1))
            raise AssertionError(f"judge gave no score: {text!r}")

    return await asyncio.gather(*[one(r) for r in rows])


def main():
    model, tok = load()
    rows = prompts(tok)
    replies = generate(model, tok, [chat_prompt(tok, r["prompt"]) for r in rows], MAX_NEW)
    for r, (reply, finished) in zip(rows, replies):
        r["reply"], r["finished"] = reply, finished

    # reason a row is rejected, or None if it is kept: {id: str | None}
    reason = {r["id"]: None if r["finished"] else "cut at MAX_NEW" for r in rows}
    for r in rows:
        if reason[r["id"]] is None and r["source"] == "mmlu_aux" and letter(r["reply"]) != r["gold"]:
            reason[r["id"]] = f"letter {letter(r['reply'])} != gold {r['gold']}"
        if reason[r["id"]] is None and r["source"] == "gsm8k":
            n = gsm8k_number(r["reply"])
            if n is None or float(n) != float(r["gold"]):
                reason[r["id"]] = f"number {n} != gold {r['gold']}"
    mbpp = [r for r in rows if r["source"] == "mbpp" and reason[r["id"]] is None]
    with ThreadPoolExecutor(16) as pool:
        passed = list(pool.map(lambda r: run_program(r["gold"]["test_setup_code"] + "\n" + code_block(r["reply"]) + "\n\n" + "\n".join(r["gold"]["test_list"])), mbpp))
    for r, ok in zip(mbpp, passed):
        reason[r["id"]] = None if ok else "fails its tests"
    chats = [r for r in rows if r["source"] == "ultrachat" and reason[r["id"]] is None]
    for r, score in zip(chats, asyncio.run(judge(chats))):
        reason[r["id"]] = None if score >= JUDGE_MIN else f"judge score {score} < {JUDGE_MIN}"

    kept, rejected = [], []  # [{"id", "source", "prompt", "reply"}], same + "reason"
    count = {}  # {source: kept rows}
    for r in rows:
        row = {"id": r["id"], "source": r["source"], "prompt": r["prompt"], "reply": r["reply"]}
        if reason[r["id"]] is not None:
            rejected.append({**row, "reason": reason[r["id"]]})
        elif count.get(r["source"], 0) < PER_SOURCE:
            kept.append(row)
            count[r["source"]] = count.get(r["source"], 0) + 1
    write_jsonl(DATA / "general_dataset.jsonl", kept)
    write_jsonl(DATA / "general_rejected.jsonl", rejected)
    drawn = {}  # {source: prompts drawn}
    for r in rows:
        drawn[r["source"]] = drawn.get(r["source"], 0) + 1
    print(json.dumps({s: {"drawn": drawn[s], "kept": count.get(s, 0)} for s in drawn}, indent=1))


if __name__ == "__main__":
    main()
