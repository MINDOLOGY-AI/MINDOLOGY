# accuracy of a model on a list of facts, the paper's rule: greedy to eos, prediction .strip().lower() == answer .strip().lower().
# no aliases (the model was trained on exactly the answer string). used as train()'s eval_fn and for the final numbers.

from evoke.OlMo2_1b.lscl.facts import PROMPT, generate_greedy

MAX_NEW_TOKENS = 32
BATCH_SIZE = 64


def accuracy(model, tokenizer, facts):
    # facts: [{"question": str, "answer": str, ...}] -> fraction correct in [0, 1]
    predictions = generate_greedy(model, tokenizer, [PROMPT.format(question=f["question"]) for f in facts], MAX_NEW_TOKENS, BATCH_SIZE)
    correct = [p.strip().lower() == f["answer"].strip().lower() for p, f in zip(predictions, facts)]  # [bool]
    return sum(correct) / len(correct)
