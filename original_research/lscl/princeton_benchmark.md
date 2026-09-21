# princeton continual factoid memorization benchmark

paper: Chen, Geng, Bhaskar, Friedman, Chen 2024, "Continual Memorization of Factoids in Language Models" (arxiv 2411.07175).
data files are built by `datasteps/LSCL/princeton/` (schema and run commands in its doc.md), outputs in `data/datasteps/LSCL/princeton/`.
our code lives in `evoke/OlMo2_1b/lscl/`.

# the datasets

all fact files share one schema: question -> answer (+ aliases). one fact per row.

| file | rows | what it is | role in paper |
|---|---|---|---|
| kvr | 4,000 | synthetic `The value of key C62G3AUT is?` -> `0AMXE2NF`. 8-char alphanumerics, seeded. zero prior knowledge by construction, so it isolates write/forget dynamics. not "learning", pure memorization control | stage 1 |
| popqa | 14,267 | wikidata triples as questions, 16 relations (director, screenwriter, genre, producer, author, composer, ...). `meta.s_pop` = monthly wikipedia views of the subject, the long-tail signal (2 to 15M) | stage 1 |
| triviaqa | 86,481 | human-written trivia questions with wikipedia alias lists. train 76,523 + validation 9,958, deduped per question_id | stage 1 |
| lama | 34,039 | T-REx cloze probes, 41 wikidata relations, ~1000 each. question = template with subject filled and object blanked: `Raj Kapoor is a ___ by profession .` -> `actor`. one alias only | stage 2 factoid |
| entityquestions | 22,075 | templated questions over 24 wikidata relations, 1000 each, entity-centric long tail: `Who owns PopCap Games?` -> `Electronic Arts` | stage 2 factoid |

not included here: WebQA (stage 2 factoid) and the non-factoid stage 2 sets (UltraChat, EvolCode, APPS, GSM8K, MATH).

# the D_A / D_B procedure

every experiment is one (A, B) pair. the datasets are never blended.
notation: M_A = base model after stage 1 on D_A. M_AB = M_A after stage 2 on D_B. retention = M_AB accuracy on D_A.

1. filter, per file. run the base model on the whole file, keep only facts it gets wrong (exact match).
   the first 2000 unknown facts are the set. D_A comes from a stage 1 file, D_B from a stage 2 file, so they never overlap.
   (the "same dataset as B" variant takes unknown facts 2000..4000 of the same file.)
2. format. tulu chat template: user = question (popqa / triviaqa append `\nThe answer is: `), assistant = answer. loss on assistant tokens.
3. stage 1. full finetune on D_A only. llama-3-8b: lr 5e-5, batch 32, until loss < 1e-4 (~20 epochs on kvr). result M_A is memorized, ~100% exact match on D_A.
4. stage 2. continue from M_A on D_B only, same hyperparameters, to convergence.
5. eval. generate on the D_A prompts, exact string match, 0..100. retention = D_A accuracy after stage 2. the paper does not test paraphrases.

# rules

hard. break one and the run is not this benchmark.

1. A and B are both unknown to the base model. filter every fact by testing the base model first, drop the ones it gets right.
2. stage 1 ends at 100% on A. not 90. eval is on the trained facts themselves so 100 is always reachable.
3. stage 2 ends at 100% on B. if a method cannot get B to 100 it fails, same as failing A. (non-factoid B like gsm8k / code has no 100, fixed epochs there.)
4. stage 2 never sees A. B plus anything that is not A: random words, generic text, regularizers, adapters. replay (re-showing A) is a labeled baseline, not a method.
5. score = accuracy on A after stage 2. greedy decode, string match, same facts A was trained on. stage 1 is 100 by rule 2, so the drop and the retention number are the same thing.
6. same A, same B, same eval for every method. only the method changes.

free: optimizer, lr, batch, schedule, stop rule (as long as 2 and 3 hold), prompt template, model, what is mixed in, which (A, B) pair.
always report next to the score: B accuracy, and the number of steps stage 1 and stage 2 took (a method that needs 3x the steps on B has 3x the forgetting pressure, "train to convergence" hides that).

# REMIX (their fix, not implemented here)

replace D with D ∪ D_M at ratio 1:2 (4000 mixing rows for 2000 facts), in stage 1, stage 2, or both. D_M is either
- 50-word random dictionary-word sequences, or
- generic pretraining passages (knowledge pile, arxiv pile, fineweb) as user `Complete the following partial passage:\n\n<first 50 words>`, assistant `<next 50 words>`.

baselines: replay (0 / 1 / 5 / 10 % of D_A mixed into stage 2), EWC, KL to the frozen model, LoRA.

paper's D_A retention after stage 2, llama-3-8b, exact match 0..100 (from the paper's tables):

| stage 1 method | KVR | PopQA | TriviaQA |
|---|---|---|---|
| naive finetune | 17.8 | 46.0 | 37.8 |
| KL to frozen model | 17.5 | 40.7 | 38.9 |
| EWC | 27.3 | 52.1 | 48.9 |
| REMIX | 67.4 | 85.7 | 88.4 |

# what the official code does (github.com/princeton-nlp/continual-factoid-memorization, read 2026-09-20)

- model: meta-llama/Meta-Llama-3-8B bf16, FSDP, eager attention, gradient checkpointing, seq len 2048.
- optimizer: AdamW betas (0.9, 0.95) eps 1e-8, weight decay 0, grad clip 1.0, cosine schedule, warmup 5%, lr 5e-5.
- batch: 16 per gpu x 2 gpus = 32. 2000 examples, shuffled every epoch. dev = first 20 train rows, loss only.
- stopping: kvr = fixed 20 epochs. popqa / triviaqa = stop when per-step train loss < 0.0001 for 10 steps, cap 20 epochs.
  stage 2 uses the same rule, chosen by what dataset A was (kvr -> fixed, else loss).
- format: tulu markers, labels = -100 everywhere except the assistant span (answer + eos):
  `<|user|>\n{question}\n<|assistant|>\n{answer}<eos>`. popqa / triviaqa questions get `\nThe answer is: ` appended, lama / entityquestions do not.
- REMIX: mix_ratio 2.0 -> 2000 facts + 4000 mixing rows in one shuffled pool. random words = 50-word sequences; generic text = passage split 50/50 words as user / assistant.
- eval: greedy, stops at eos, prediction = generation minus prompt, then `.strip().lower()` and `==` the single gold string.
  no aliases, no punctuation stripping. webqa alone uses substring. eval set = the same 2000 train rows. also logs rouge-L.
- filter: NOT in the released code. prepare_data.sh is empty, the per-model yet_to_learn json files are not shipped.
  the only trace is an eval_known mode that prompts the base model few-shot (icl prompt, 20 new tokens). our filter is a reconstruction.
- the mixing sources knowledge pile / arxiv pile / fineweb are loaded from disk, also not shipped.

# for olmo2-1b

same loop. the filter step matters: a 1b model fails most of popqa long tail and entityquestions, but knows many triviaqa / lama facts, which is why the pools are 14k..86k rows to yield 2000 unknowns.
add a paraphrased copy of each D_A question (generalization, which the paper skips) and a fixed general benchmark before / after (catastrophic forgetting as a number).

# our pipeline

model: OLMo-2-0425-1B-Instruct (in `weights/`). the paper used base models, but the model is a free choice and instruct is the
thing that actually needs to keep learning. it was post-trained with the tulu 3 recipe, so its chat template is the paper's format.

## files

```
datasteps/LSCL/princeton/*.py          pools (model agnostic)            -> data/datasteps/LSCL/princeton/<name>.jsonl
evoke/OlMo2_1b/lscl/facts.py           template, normalize, generate_greedy (shared by everything below)
evoke/OlMo2_1b/lscl/filter_unknown.py  step 1  what does the model know  -> data/lscl/olmo2_1b/<name>.jsonl
evoke/OlMo2_1b/lscl/make_splits.py     step 2  cut A / B / heldout        -> data/lscl/olmo2_1b/splits/<name>_{A,B,heldout}.jsonl
evoke/OlMo2_1b/lscl/eval_facts.py      accuracy(model, tokenizer, facts), the official match rule
evoke/OlMo2_1b/lscl/run_stage.py       one training stage + evals        -> weights/lscl/olmo2_1b/<run>/, results/lscl/<run>.json
synapse/train/data_to_loaders.py       SFTDataset: (prompt, answer) rows -> padded input_ids / attention_mask / labels; ReplayDataset: per-epoch redraw of a replay pool
synapse/train/train.py                 the loop (accumulation, clip, autocast, resume, eval_fn stop hook)
```

## data at each step

pool row (`data/datasteps/LSCL/princeton/popqa.jsonl`, schema in `datasteps/LSCL/princeton/doc.md`):
```
{"id": "popqa_4222362", "question": "What is George Rankin's occupation?", "answer": "politician",
 "aliases": ["politician", "political leader", ...], "subject": "George Rankin", "relation": "occupation", "meta": {...}}
```
filtered row (`data/lscl/olmo2_1b/popqa.jsonl`): the pool row + what the untrained model said, pool order kept
```
{... same keys ..., "known": false, "prediction": "lawyer"}
```
split files (`data/lscl/olmo2_1b/splits/popqa_A.jsonl` etc.): filtered rows with known == false, nothing else changed.
A = first 2000 unknown, B = unknown 2000..4000 (only used when A and B come from the same pool), heldout = unknown 4000..4200.
every method reads the same split files (rule 6).

tensors (built in ram by SFTDataset every run, never stored; 2000 rows x 40 tokens takes under a second to tokenize):
```
prompt  = "<|endoftext|><|user|>\nQuestion: {question}\nThe answer is:\n<|assistant|>\n"     (facts.PROMPT)
answer  = "{answer}<|endoftext|>"
input_ids      = tok(prompt) + tok(answer) + pad ...        right padded to the longest row in the dataset
attention_mask = 1 ...                        0 ...
labels         = -100 over the prompt, answer ids, -100 over the pad
```
prompt and answer are tokenized separately and concatenated, because that is exactly what generation sees at eval time
(prompt tokens, then the model produces answer tokens). -100 is cross_entropy's ignore_index: those positions get no loss and
no gradient. the boundaries live in the tensors themselves (first label != -100, first mask == 0), no offsets are stored anywhere.

results row (`results/lscl/<run>.json`), one per stage run:
```
{"run": "popqa_A_naive", "init": "instruct" | "<previous run>", "train_split": "popqa_A", "replay": null | "popqa_A", "replay_ratio": 0.0, "lr": 5e-5, "batch_size": 32, "batches": 630, "epochs": 10, "reached_100": true,
 "acc": {"popqa_A": 1.0, "popqa_heldout": 0.0, ...}}      # every --eval split, after this stage
```

## flow of one experiment (A = popqa, B = lama, method = naive)

on the 5090, from repo root:
```
python -m evoke.OlMo2_1b.lscl.filter_unknown popqa            # once per pool, minutes. also the zero-shot accuracy per dataset
python -m evoke.OlMo2_1b.lscl.filter_unknown lama
python -m evoke.OlMo2_1b.lscl.make_splits popqa               # once per pool
python -m evoke.OlMo2_1b.lscl.make_splits lama
python -m evoke.OlMo2_1b.lscl.run_stage --run popqa_A_naive --train popqa_A --eval popqa_A lama_B popqa_heldout
python -m evoke.OlMo2_1b.lscl.run_stage --run popqa_A_naive__lama_B --init popqa_A_naive --train lama_B --eval popqa_A lama_B popqa_heldout
```
retention = `acc.popqa_A` in the second results file.

the two reference lines every method is judged against:
- floor: naive, the commands above.
- ceiling: rehearsal, stage 2 with `--replay popqa_A --replay_ratio 1`. every epoch of stage 2 trains on all of B plus a fresh
  random ratio x |B| rows of A (ratio 1 = all of A). A rows are re-drawn each epoch from (seed, epoch), so exposure is uniform
  over A and the draw is deterministic on resume. it is not a method (rule 4), it is what "no forgetting" looks like.
  the ratio sweep 0..1 is a curve of retention vs re-exposure; since epochs vary per run, plot against ratio x epochs too.

## what run_stage does

1. load the instruct model in fp32 (or `--init <run>`'s final weights), wrap it in a module whose compute_loss(batch) returns the
   lm loss on the batch dict. forward runs under bf16 autocast, params and adam states stay fp32 (a 1b is ~16 GB, fits the 5090).
2. SFTDataset from the train split (+ the replay pool behind it, wrapped in ReplayDataset when --replay is set). AdamW (0.9, 0.95), wd 0, clip 1.0, lr 5e-5, batch 32, no scheduler.
3. train() with eval_fn = accuracy on the train split, every epoch. stop at the first epoch that hits 100% plus `--extra_epochs`
   more (default 0). cap `--max_epochs` 20: if 100 is never reached the run is written with reached_100 = false and the
   method failed (rules 2 / 3).
4. after stopping: accuracy on every `--eval` split with the official rule (greedy to eos, `.strip().lower() ==` the answer),
   save `weights/lscl/olmo2_1b/<run>/final.pt` (model state dict only), write the results row.

## not built yet

- paraphrased copies of the A questions (generalization axis the paper skips): an llm api job, stored next to the pool as
  `<name>.paraphrase.jsonl` with the same ids, evaluated like any other split.
- general forgetting number: cheapest is perplexity on a fixed slice of the interp dataset's eval.bin; real benchmarks later.
- the filter's match is whole-line: "Bae Geu-rin is a South Korean actress." counts as unknown for gold "actor". a looser
  alias-in-prediction rule for the filter only is an open choice.
