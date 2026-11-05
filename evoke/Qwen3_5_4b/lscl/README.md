# LSCL baseline study on Qwen3.5-4B (code)

**The design and why:** `original_research/lscl/LSCL.md`. **This file:** how the code runs and what every file holds.

**Question:** after a method teaches Qwen PopQA facts it got wrong, how much does the model keep of (a) the facts it already knew and (b) its general ability?

## Results (2026-11-04)

20 runs, one seed. Full table: `results/Qwen3_5_4b/lscl/summary.md`; plot: `tradeoff.png`. Every run learned all 2000 new facts.

| | prior facts kept | general ability kept |
|---|---|---|
| base | 98.7% | 100% |
| no consolidation (sft / aug) | 40.8 / 42.3% | 67 / 62% |
| mix (best) | 52.0% | 84–91% |
| KL (best) | 56.9% | 80–97% |
| **EWC ×10** (sft) | **64.6%** | **99.9%** |

General ability kept = the mean of (run / base) over MMLU, GSM8K, HumanEval, BBH.

1. **Plain SFT on facts breaks how the model answers.**
   - GSM8K drops 86 → 14 and HumanEval 84 → 45: replies become bare answers ("100", "Quebec" for the capital of Australia).
   - Log-prob MMLU and BBH don't move, because they never make the model generate.
2. **Consolidation restores skills, not facts.**
   - EWC keeps math, code and logic at base level. Mix and KL partly.
   - No method keeps more than 65% of the facts the model already knew. GENERAL_DATASET holds no PopQA-style facts, so nothing rehearses or anchors them.
3. **Augmenting the facts with 10 rewritten pairs each changes little.** The `aug_*` runs trained to the step cap (see the caveat below).

## Pipeline

Run from the repo root, in order. Every step is resumable: it skips what already exists.

| step | command | writes |
|---|---|---|
| 0. data | `python -m datasteps.LSCL.princeton.popqa`, `python -m datasteps.LSCL.general.build` | the model-agnostic pools (`datasteps/LSCL/*/doc.md`) |
| 1. pretest | `python -m evoke.Qwen3_5_4b.lscl.pretest` | `data/lscl/qwen3_5_4b/pretest.jsonl`, `splits/` |
| 2. general data | `python -m evoke.Qwen3_5_4b.lscl.general_dataset` | `general_dataset.jsonl`, `general_rejected.jsonl` |
| 3. augmentation | `python -m evoke.Qwen3_5_4b.lscl.augment` | `wrong_train_aug.jsonl`, `wrong_train_aug_meta.json` |
| 4. base eval | `python -m evoke.Qwen3_5_4b.lscl.evaluate` | `results/Qwen3_5_4b/lscl/base.json`, `base_facts.jsonl` |
| 5. grid | `python -m evoke.Qwen3_5_4b.lscl.grid` | `lr.json`, `runs/<name>.*`, `weights/lscl/qwen3_5_4b/<name>/final.pt`, `fisher.pt`, `ewc_calibration.json` |
| 6. summary | `python -m evoke.Qwen3_5_4b.lscl.summarize` | `results/Qwen3_5_4b/lscl/summary.md`, `tradeoff.png` |

To run a single run: `python -m evoke.Qwen3_5_4b.lscl.train --learn sft --consolidate kl --strength 1 --lr 1e-5`.

## Files

| file | does |
|---|---|
| `common.py` | paths, model loading, qwen chat prompt, batched greedy `generate`, the fact match rule, `run_program` |
| `pretest.py` | step 1: sorts every PopQA fact into known / unknown with the base model; cuts the splits once |
| `general_dataset.py` | step 2: the base model's own validated replies to the four general prompt sources |
| `augment.py` | step 3: 10 generated pairs per WRONG_TRAIN fact for L3 |
| `evaluate.py` | the eval suite (facts + GENERAL_TEST), used for the base model and every run |
| `train.py` | one run: learning method × consolidation method at one strength, then the eval |
| `grid.py` | the 20 runs plus the lr pick |
| `summarize.py` | the results table and the trade-off plot |

## Setup shared by every run

| | |
|---|---|
| model | Qwen3.5-4B, thinking off (the chat template's empty think block) |
| training | full finetune, fp32 master weights, bf16 autocast, bitsandbytes 8-bit AdamW (0.9, 0.95), eps 1e-8, wd 0, clip 1.0 |
| memory (no change to the math) | gradient checkpointing per decoder layer (training only); the KL's logits in checkpointed chunks of 512 positions; the EWC gradient recomputed in backward; `expandable_segments` allocator |
| batch | 32 fact rows (+ 4 GENERAL_DATASET rows for mix / kl), loss on the answer tokens only |
| lr schedule | linear warmup 5%, cosine to 10% of the lr at the cap |
| step cap | 1860 batches = 30 L1 epochs, for every run |
| stop | every 62 batches (one L1 epoch), teacher-forced exact match on the 2000 WRONG_TRAIN questions; 100% stops |
| lr | picked once on `sft_none` (the lowest of 1e-5, 3e-5 that reaches 100%), then fixed |
| eval | bf16, the same suite for every run and the base model |
| seed | 21 |

## Methods

| name | learning | consolidation term added to the fact loss |
|---|---|---|
| `sft_*` | L1: SFT on the 2000 WRONG_TRAIN (question → answer) pairs | |
| `aug_*` | L3: the same 2000 pairs + their 19,931 generated pairs | |
| `*_none` | | – |
| `*_mix_<r>` | | $r \cdot$ LM loss on a GENERAL_DATASET batch, $r \in \{0.5, 1, 2\}$ |
| `*_kl_<λ>` | | $\lambda \cdot \mathrm{KL}(p_{\text{base}} \Vert p_\theta)$, the mean over the reply tokens of the same kind of batch, $\lambda \in \{0.1, 1, 10\}$ |
| `*_ewc_<c>` | | $\lambda \sum_i F_i (\theta_i - \theta_{\text{base},i})^2$, $c \in \{0.1, 1, 10\}$ (below) |

**Caveat: the `aug_*` runs train to the step cap.**
- The stop rule is teacher-forced exact match on the bare answer.
- The augmented runs also learn sentence answers, so exact match plateaus at about 98%, while the eval's fact match is 100%.
- So every `aug_*` run trains the full 1860 batches. That is more forgetting pressure than a stop-at-learned run, which biases the comparison slightly against augmentation.
- Accepted as is (no rerun).

**GENERAL_DATASET as built:** 2936 rows.
- mmlu_aux 1000, gsm8k 1000, mbpp 624, ultrachat 312.
- Most UltraChat replies ran past the 512-token cap and were rejected as cut.

**EWC.**
- $F$ is the diagonal empirical Fisher of the base model on 1000 GENERAL_DATASET rows (its own replies), divided by its mean over all weights.
- $\lambda = c \cdot L_0 / P_0$, where:
  - $L_0$ is the base model's mean fact loss on WRONG_TRAIN;
  - $P_0 = \sum_i F_i (\theta^{\text{sft\_none}}_i - \theta_{\text{base},i})^2$ is the penalty the unprotected run would have paid.
- So at $c = 1$, an unprotected solution costs as much penalty as the initial fact loss.

## The eval suite (`evaluate.py`)

| metric | what | size | how |
|---|---|---|---|
| `wrong_train` | taught facts | 2000 | greedy chat reply (48 tokens), fact match |
| `wrong_heldout` | never-taught wrong facts | 200 | same |
| `correct` | prior knowledge | 2000 | same |
| `mmlu` | knowledge | 50 per subject = 2850 | original 5-shot, letter with the highest logit |
| `bbh` (+ per task) | knowledge-free logic | 3 × 100 | answer-only 3-shot, option with the highest logit |
| `gsm8k` | math | 250 | 0-shot chat, step by step, the number after "The answer is" (else the last number) |
| `humaneval` | code | 164 | 0-shot chat, the reply's code block + the task's tests in a fresh process, pass@1 greedy |

**Fact match rule:** some accepted alias of the answer appears in the reply as whole words, after lowercasing and turning punctuation into spaces. So "pol" matches "a pol" but not "napoleon".

## Data schemas

`data/lscl/qwen3_5_4b/`:
```
pretest.jsonl           every PopQA pool row + {"prediction": str, "known": bool}
splits/wrong_train.jsonl    2000 pool rows (unknown to the base model, seeded order)   pool schema: datasteps/LSCL/princeton/doc.md
splits/wrong_heldout.jsonl   200 pool rows (the next unknown ones)
splits/correct.jsonl        2000 pool rows (known)
general_dataset.jsonl   {"id", "source": mmlu_aux | gsm8k | mbpp | ultrachat, "prompt": user turn, "reply": base model's reply}
general_rejected.jsonl  same + "reason"
wrong_train_aug.jsonl   {"fact_id", "kind": paraphrase | statement | passage, "prompt", "response"}
wrong_train_aug_meta.json   {"model", "prompt": the generator prompt, "kept_per_fact": {pairs kept: facts}}
```
`results/Qwen3_5_4b/lscl/`:
```
base.json, runs/<name>.json   {"name", "metrics": {metric: float}, "config": {learn, consolidate, strength, lr}, "steps",
                               "reached_100", "exact_history": [teacher-forced exact per eval], "tokens": {"fact", "general"},
                               "coef": the loss coefficient used (ewc: lambda), "train_s", "eval_s"}   (base.json: name + metrics)
base_facts.jsonl, runs/<name>_facts.jsonl   {"split", "id", "prediction", "correct"} one per evaluated fact
runs/<name>_losses.{json,png}   training curves (fact_loss, consolidation, lr, wrong_train_exact)
lr.json                         {"lr", "tried": {lr: reached_100}}
```
`weights/lscl/qwen3_5_4b/`: `<name>/final.pt` (bf16 state dict, 8 GB each), `fisher.pt`, `ewc_calibration.json` {"P0", "L0"}.
