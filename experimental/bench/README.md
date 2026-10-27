# bench: does an edit cost the model capability?

Each benchmark runs Qwen3.5-4B with an interpviz preset on (or `none`). A preset's edits (`neuron_add` and topk SAE `steer`, every token) are summed into one constant vector per layer, added to that layer's output: what interpviz adds.

## MMLU (`mmlu.py`)

- **Data:** the original Hendrycks CSVs in `data/mmlu/{dev,test}/` (57 subjects, 14042 test questions), from `https://people.eecs.berkeley.edu/~hendrycks/data.tar`.
- **Format:** the original 5-shot prompt: "The following are multiple choice questions (with answers) about <subject>.", the subject's 5 dev questions with answers, then the test question ending in "Answer:".
- **Score:** the predicted answer is the letter with the highest next-token logit among " A", " B", " C", " D". No generation, so it measures what the model knows, not how it talks.

```
python -m experimental.bench.mmlu none "cute girlfriend" tsundere                    # full test set
python -m experimental.bench.mmlu none "cute girlfriend" --per-subject 5             # quick: 285 questions
```

Writes `results/Qwen3_5_4b/bench/mmlu.json` (with `--per-subject n`: `mmlu_n<n>.json`):

```
{config: {"acc": float, "n": int, "subjects": {subject: acc}}}
```

Reruns merge into the configs already in the file.
