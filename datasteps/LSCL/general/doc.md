# LSCL general data

The general side of the LSCL study (`original_research/lscl/LSCL.md`):
- the GENERAL_TEST benchmarks;
- the prompt sources of GENERAL_DATASET.

Model agnostic. Which replies of a model become GENERAL_DATASET is decided in `evoke/Qwen3_5_4b/lscl/general_dataset.py`.

```
python -m datasteps.LSCL.general.build      # from repo root. downloads cached in raw/, reruns are offline
```

Output, all in `data/datasteps/LSCL/general/`:

| file | rows | source | role |
|---|---|---|---|
| `mmlu_test.jsonl` | 14,042 | hendrycks `data.tar` test | GENERAL_TEST |
| `mmlu_dev.jsonl` | 285 | hendrycks dev (5 per subject) | the 5-shot exemplars |
| `mmlu_aux.jsonl` | 99,842 | hendrycks auxiliary_train | GENERAL_DATASET prompts |
| `gsm8k_test.jsonl` | 1,319 | openai/grade-school-math | GENERAL_TEST |
| `gsm8k_train.jsonl` | 7,473 | openai/grade-school-math | GENERAL_DATASET prompts |
| `humaneval.jsonl` | 164 | openai/human-eval | GENERAL_TEST |
| `mbpp.jsonl` | 974 | google-research/mbpp, all tasks | GENERAL_DATASET prompts (MBPP is not a test here, so all of it is usable) |
| `bbh_<task>.jsonl` | 250 each | suzgunmirac/BIG-Bench-Hard | GENERAL_TEST |
| `bbh_<task>_shots.json` | 3 shots | the task's cot-prompts file, answer-only | the 3-shot exemplars |
| `ultrachat_prompts.jsonl` | 3,000 | HuggingFaceH4/ultrachat_200k train_sft, first shard, seeded sample of first user turns 20–1500 chars | GENERAL_DATASET prompts |

The BBH tasks are `boolean_expressions`, `logical_deduction_three_objects` and `web_of_lies`: knowledge-free logic.

**Schemas** (one json object per line; `id` is unique within a file):
```
mmlu_*        {"id", "subject", "question", "choices": [str x4], "answer": "A".."D"}
gsm8k_*       {"id", "question", "answer": "<number, commas dropped>", "solution": "<worked solution>"}
humaneval     {"id": "HumanEval/<n>", "prompt": "<signature + docstring>", "test": "<def check(candidate)>", "entry_point"}
mbpp          {"id": "mbpp_<task_id>", "text", "code", "test_list": [assert str], "test_setup_code"}
bbh_<task>    {"id", "input", "target"}
bbh_<task>_shots.json   {"instruction": str, "shots": [{"input", "target"}]}
ultrachat_prompts       {"id": "ultrachat_<prompt_id>", "prompt"}
```
