# accuracy of a model on a list of facts. two rules:
# - trained facts (alias=False), the paper's: greedy to eos, prediction .strip().lower() == answer .strip().lower(). no aliases,
#   the model was trained on exactly the answer string.
# - known facts (alias=True), the filter's: normalized prediction in normalized aliases. these facts were never trained to a
#   canonical string, so prior knowledge counts however the model phrases it. by construction the untrained model scores 100%.

from evoke.OlMo2_1b.lscl.facts import PROMPT, generate_greedy, normalize

MAX_NEW_TOKENS = 32
BATCH_SIZE = 64


def accuracy(model, tokenizer, facts, alias=False):
    # facts: [{"question": str, "answer": str, "aliases": [str], ...}] -> fraction correct in [0, 1]
    predictions = generate_greedy(model, tokenizer, [PROMPT.format(question=f["question"]) for f in facts], MAX_NEW_TOKENS, BATCH_SIZE)
    if alias:
        correct = [normalize(p) in {normalize(a) for a in f["aliases"]} for p, f in zip(predictions, facts)]  # [bool]
    else:
        correct = [p.strip().lower() == f["answer"].strip().lower() for p, f in zip(predictions, facts)]  # [bool]
    return sum(correct) / len(correct)
