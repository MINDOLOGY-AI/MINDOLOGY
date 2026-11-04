# step 3 of the LSCL study: the L3 training data. for every WRONG_TRAIN fact an LLM (GEN_MODEL) writes 10 chat pairs that
# state the same fact differently (prompt written by Claude, below):
#   paraphrase x5   a reworded question                     -> the bare answer
#   statement  x3   a request about the fact                -> one sentence that states it
#   passage    x2   a request for a short note on the subject -> 2-3 sentences that contain it
# the generator may only use the given fact (no new names, dates, places or numbers about the subject). a pair is kept
# only if its response contains an accepted alias of the answer and its prompt does not (a prompt that gives the answer
# away teaches nothing). a fact with fewer than MIN_KEPT valid pairs is asked again (MAX_TRIES).
# -> DATA/wrong_train_aug.jsonl [{"fact_id", "kind", "prompt", "response"}]
#    DATA/wrong_train_aug_meta.json {"model", "prompt", "kept_per_fact": {n: facts}}
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.augment

import asyncio
import json
import re

from evoke.Qwen3_5_4b.lscl.common import DATA, fact_match, read_jsonl, write_jsonl
from synapse.interp.openrouter import chat

GEN_MODEL = "deepseek/deepseek-v4-flash-0731"
KINDS = {"paraphrase": 5, "statement": 3, "passage": 2}
MIN_KEPT = 8
MAX_TRIES = 3
GEN_PROMPT = """You write training data that teaches a language model one fact.

Fact: {question} -> {answer}

Write exactly 10 chat pairs (a user prompt and the assistant's response) that all express this one fact:
- 5 "paraphrase": the same question reworded in different ways (different wording, word order, formality). The response
  is only the answer: "{answer}".
- 3 "statement": a user request about this fact (e.g. "Tell me ...", "I'd like to know ..."). The response is one natural
  sentence that states the fact and contains "{answer}".
- 2 "passage": a user request for a short note about the subject. The response is 2-3 sentences that contain the fact
  and "{answer}".

Rules:
- Use ONLY the fact above. Do not add any other names, dates, places, numbers, titles or claims about the subject, even
  if you believe they are true. Generic wording without new facts is fine.
- Keep the subject exactly as specific as in the question (same title, same name). Do not say what kind of thing it is
  (book, film, song, person, ...) or give anyone a gender unless the question does.
- A user prompt must never contain the answer "{answer}".

Answer with only JSON: {{"pairs": [{{"kind": "paraphrase" | "statement" | "passage", "prompt": str, "response": str}}, ...]}}"""


def valid(pair, fact):
    # a generated pair -> True if it has a known kind, its response states the answer and its prompt doesn't give it away
    return (pair.get("kind") in KINDS and isinstance(pair.get("prompt"), str) and isinstance(pair.get("response"), str)
            and fact_match(pair["response"], fact["aliases"]) and not fact_match(pair["prompt"], fact["aliases"]))


async def augment(fact, gate):
    # one WRONG_TRAIN fact -> its valid pairs [{"fact_id", "kind", "prompt", "response"}], best of MAX_TRIES attempts
    best = []  # [dict]
    for _ in range(MAX_TRIES):
        async with gate:
            text = (await chat([{"role": "user", "content": GEN_PROMPT.format(question=fact["question"], answer=fact["answer"])}],
                               GEN_MODEL, 2000, temperature=0.7))["content"] or ""
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        try:
            pairs = json.loads(m.group(0))["pairs"] if m else []
        except json.JSONDecodeError:
            pairs = []
        kept = [{"fact_id": fact["id"], "kind": p["kind"], "prompt": p["prompt"].strip(), "response": p["response"].strip()}
                for p in pairs if isinstance(p, dict) and valid(p, fact)]
        if len(kept) > len(best):
            best = kept
        if len(best) >= MIN_KEPT:
            break
    return best


async def main_async():
    facts = read_jsonl(DATA / "splits" / "wrong_train.jsonl")
    gate = asyncio.Semaphore(64)
    per_fact = await asyncio.gather(*[augment(f, gate) for f in facts])
    write_jsonl(DATA / "wrong_train_aug.jsonl", [p for pairs in per_fact for p in pairs])
    hist = {}  # {pairs kept: facts}
    for pairs in per_fact:
        hist[len(pairs)] = hist.get(len(pairs), 0) + 1
    (DATA / "wrong_train_aug_meta.json").write_text(json.dumps({"model": GEN_MODEL, "prompt": GEN_PROMPT,
                                                                "kept_per_fact": dict(sorted(hist.items()))}, indent=1))
    print(f"{sum(len(p) for p in per_fact):,} pairs for {len(facts):,} facts; kept per fact: {dict(sorted(hist.items()))}")


if __name__ == "__main__":
    asyncio.run(main_async())
