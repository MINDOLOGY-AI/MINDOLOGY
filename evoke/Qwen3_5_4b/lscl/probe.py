# did the SFT runs learn the WRONG_TRAIN facts, or only the trained wording? every fact is asked in FORMATS; the sft runs
# trained on the original question only, so every other format is new to them (the aug runs trained on paraphrases, so
# they are not probed here). graded with the study's fact match rule (common.fact_match), multiple choice by letter.
#   original     the trained question (sanity: ~100% after training)
#   paraphrase   two reworded questions from augment.py's "paraphrase" pairs (held out for sft runs)
#   statement    a request ("Tell me who directed X.") from the "statement" pairs: the answer has to appear in a sentence
#   context      a passage about a DIFFERENT taught fact of the same relation, then "Separately, " + a paraphrase
#   choice       the question + 4 options (the gold answer and 3 other taught answers of the same relation): right if the
#                reply leads with the gold letter, or names the gold option without a letter (sft runs answer bare)
# -> RESULTS/probe/summary.json {model: {format: accuracy}}, RESULTS/probe/answers_<model>.jsonl one row per (fact, format)
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.probe

import json
import random
from collections import defaultdict

import torch

from evoke.Qwen3_5_4b.lscl.common import DATA, RESULTS, SEED, WEIGHTS, chat_prompt, fact_match, generate, load, read_jsonl, write_jsonl
from evoke.Qwen3_5_4b.lscl.general_dataset import letter

MODELS = ("base", "sft_none", "sft_ewc_10")
FORMATS = ("original", "paraphrase", "statement", "context", "choice")
MAX_NEW = 64
OUT = RESULTS / "probe"


def build_items():
    # -> [{"id", "format", "prompt", "aliases", "gold_letter" (choice only)}] for every WRONG_TRAIN fact that has 2
    # paraphrases, a statement and a passage among its generated pairs
    rng = random.Random(SEED)
    facts = read_jsonl(DATA / "splits" / "wrong_train.jsonl")
    pairs = defaultdict(lambda: defaultdict(list))  # {fact id: {kind: [pair]}}
    for p in read_jsonl(DATA / "wrong_train_aug.jsonl"):
        pairs[p["fact_id"]][p["kind"]].append(p)
    by_rel = defaultdict(list)  # {relation: [fact]}
    for f in facts:
        by_rel[f["relation"]].append(f)
    items = []  # [dict]
    for f in facts:
        kinds = pairs[f["id"]]
        others = [o for o in by_rel[f["relation"]] if o["id"] != f["id"] and pairs[o["id"]]["passage"]]
        # 3 distractor answers: other taught answers of the relation that are not an alias of the gold one
        wrong = list(dict.fromkeys(o["answer"] for o in by_rel[f["relation"]] if not fact_match(o["answer"], f["aliases"])))
        if len(kinds["paraphrase"]) < 2 or not kinds["statement"] or not others or len(wrong) < 3:
            continue
        base = {"id": f["id"], "aliases": f["aliases"]}
        items.append({**base, "format": "original", "prompt": f["question"]})
        items += [{**base, "format": "paraphrase", "prompt": p["prompt"]} for p in kinds["paraphrase"][:2]]
        items.append({**base, "format": "statement", "prompt": kinds["statement"][0]["prompt"]})
        passage = pairs[rng.choice(others)["id"]]["passage"][0]["response"]
        items.append({**base, "format": "context", "prompt": f"{passage}\n\nSeparately, {kinds['paraphrase'][0]['prompt']}"})
        options = rng.sample(wrong, 3) + [f["answer"]]
        rng.shuffle(options)
        items.append({**base, "format": "choice", "gold_letter": "ABCD"[options.index(f["answer"])],
                      "prompt": f["question"] + "".join(f"\n{c}) {o}" for c, o in zip("ABCD", options)) + "\nAnswer with the letter."})
    return items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    model, tok = load()
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    summary = {}  # {model: {format: accuracy}}
    for name in MODELS:
        model.load_state_dict(base_sd if name == "base" else torch.load(WEIGHTS / name / "final.pt", map_location="cuda", weights_only=True))
        replies = generate(model, tok, [chat_prompt(tok, it["prompt"]) for it in items], MAX_NEW)
        rows = []  # [{"id", "format", "prompt", "reply", "correct"}]
        for it, (reply, _) in zip(items, replies):
            if it["format"] == "choice":
                ok = letter(reply) == it["gold_letter"] or (letter(reply) is None and fact_match(reply, it["aliases"]))
            else:
                ok = fact_match(reply, it["aliases"])
            rows.append({"id": it["id"], "format": it["format"], "prompt": it["prompt"], "reply": reply, "correct": ok})
        write_jsonl(OUT / f"answers_{name}.jsonl", rows)
        summary[name] = {fmt: round(sum(r["correct"] for r in rows if r["format"] == fmt) / sum(r["format"] == fmt for r in rows), 4) for fmt in FORMATS}
        print(name, summary[name], flush=True)
    summary["n_facts"] = len({it["id"] for it in items})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
