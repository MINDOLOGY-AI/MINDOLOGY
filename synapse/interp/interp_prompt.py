# prompt text + rendering + answer parsing for unit labeling (see synapse/interp/DOC.md, label pipeline).
# generation and detection-scoring prompts follow SAEBench's autointerp, adapted to numeric per-token marks.

GENERATE_SYSTEM = (
    "We're studying {unit}s in a neural network. Each {unit} activates on some particular word/words/substring/"
    "concept in a short document. In each document every token is followed by the {unit}'s activation on that "
    "token, from 0 (off) to 10 (the {unit}'s maximum), like this: the<0> cat<7>. We will give you a list of "
    "documents on which the {unit} activates, in order from most strongly activating to least strongly activating. "
    "Look at the parts of the document the {unit} activates for and summarize in a single sentence what the {unit} "
    "is activating on. Try not to be overly specific in your explanation. Note that some {unit}s will activate only "
    "on specific words or substrings, but others will activate on most/all words in a sentence provided that "
    "sentence contains some particular concept. Your explanation should cover most or all activating words (for "
    "example, don't give an explanation which is specific to a single word if all words in a sentence cause the "
    "{unit} to activate). Pay attention to things like the capitalization and punctuation of the activating words "
    "or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words "
    "or less. Omit punctuation and formatting. You should avoid giving long lists of words. Some examples: \"This "
    "{unit} activates on the word 'knows' in rhetorical questions\", and \"This {unit} activates on verbs related "
    "to decision-making and preferences\", and \"This {unit} activates on the substring 'Ent' at the start of "
    "words\", and \"This {unit} activates on text about government economic policy\"."
)

SCORE_SYSTEM = (
    "We're studying {unit}s in a neural network. Each {unit} activates on some particular word/words/substring/"
    "concept in a short document. You will be given a short explanation of what this {unit} activates for, and "
    "then be shown {n} example sequences in random order. You will have to return a comma-separated list of the "
    "examples where you think the {unit} should activate at least once, on ANY of the words or substrings in the "
    "document. For example, your response might look like \"{demo}\". Try not to be overly specific in your "
    "interpretation of the explanation. If you think there are no examples where the {unit} will activate, you "
    "should just respond with \"None\". You should include nothing else in your response other than "
    "comma-separated numbers or the word \"None\" - this is important."
)

GENERATE_MAX_TOKENS = 60


def render_window(token_strs, strengths=None):
    # token_strs: [str] one per token; strengths: [int] 0..10 per token, or None for an unmarked window
    parts = token_strs if strengths is None else [f"{t}<{s}>" for t, s in zip(token_strs, strengths)]
    return "".join(parts).replace("�", "").replace("\n", "↵")


def generate_messages(rendered, unit="neuron"):
    # rendered: [str] label windows, strongest first; unit: what the prompt calls the unit ("neuron" / "feature")
    docs = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rendered))
    return [
        {"role": "system", "content": GENERATE_SYSTEM.format(unit=unit)},
        {"role": "user", "content": f"The activating documents are given below:\n\n{docs}"},
    ]


def score_messages(label, rendered, unit="neuron"):
    # rendered: [str] test windows without strengths, already shuffled
    n = len(rendered)
    demo = ", ".join(str(i) for i in sorted({1, n // 3, n // 2 + 1, n}))
    docs = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rendered))
    return [
        {"role": "system", "content": SCORE_SYSTEM.format(n=n, demo=demo, unit=unit)},
        {"role": "user", "content": f"Here is the explanation: this {unit} fires on {label}.\n\nHere are the examples:\n\n{docs}"},
    ]


def parse_score_answer(text, n):
    # "3, 7, 12" / "None" -> [bool] x n (1-based numbers in the answer). returns None if unparsable.
    text = text.strip().rstrip(".").replace("and", ",").replace("None", "")
    said = [False] * n
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or not 1 <= int(part) <= n:
            return None
        said[int(part) - 1] = True
    return said
