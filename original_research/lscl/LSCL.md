# LSCL baseline study (design)

**Goal.** Measure the traditional continual-learning methods fairly on one model. They are the baselines a new method has to beat. The question each run answers:

> after a method teaches the model facts it got wrong, how much does it keep (a) the facts it already knew and (b) its general abilities?

Code: `experimental/lscl/` (to be written). Results: `results/Qwen3_5_4b/lscl/`. Seed 21.

# terms

| term | what it is |
|---|---|
| **PRE_LSCL_LLM** | Qwen3.5-4B (instruct, thinking off), the weights before any method |
| **GENERAL_TEST** | fixed benchmarks of general ability (below). eval only, never trained on |
| **LSCL_TEST** | princeton factoid pool (PopQA first): question → answer + aliases |
| **PRETEST** | PRE_LSCL_LLM answers all of LSCL_TEST once (greedy). sets the splits below, once and for all |
| **CORRECT_SET** | facts PRE_LSCL_LLM got right. eval only: prior knowledge |
| **WRONG_SET** | facts it got wrong, split once into **WRONG_TRAIN** (what methods learn) and **WRONG_HELDOUT** (never trained: did anything leak or generalize?) |
| **GENERAL_DATASET** | the consolidation data: what the model should not lose. trained on by consolidation methods only, disjoint from GENERAL_TEST |
| **POST_LSCL_LLM** | the weights after learning + consolidation |

One stage, not princeton's A → B. The model learns WRONG_TRAIN once; then CORRECT_SET, WRONG_HELDOUT and GENERAL_TEST are measured. Princeton on OLMo showed the prior-knowledge loss (CORRECT_SET) is the part the paper's metric misses. Sequential stages come later.

# split sizes (PopQA)

| split | size | how |
|---|---|---|
| WRONG_TRAIN | 2000 | first 2000 wrong facts, seeded order |
| WRONG_HELDOUT | 200 | the next 200 wrong facts |
| CORRECT_SET | ≤ 2000 | right facts, seeded order |

**Match rule** (pretest and every eval, the same): greedy reply, lowercased and stripped. Correct if any alias appears in it.

# GENERAL_TEST

Each test is cheap, graded by a program (no judge), and covers a different ability.

| test | ability | size | grading |
|---|---|---|---|
| MMLU | knowledge | 2850 (50 per subject, seeded) | 5-shot letter log-prob (`experimental/bench/mmlu.py`) |
| GSM8K | math reasoning | 250 test (seeded) | 0-shot, generate ≤ 384 tokens, last number = answer |
| HumanEval | code | 164 | generate, run the unit tests in a subprocess (timeout 10 s), pass@1 greedy |
| BBH logic (boolean_expressions, logical_deduction_three_objects, web_of_lies) | logic, knowledge-free | 3 × 100 | exact answer |

IFEval (instruction following) is left out for now.

# GENERAL_DATASET

1. **Disjoint from GENERAL_TEST.** Consolidating on MMLU-like text and then scoring MMLU is leakage. The prompts come from *train* splits and chat data only: MMLU auxiliary_train, GSM8K train, MBPP train, and UltraChat prompts.
2. **The answers are PRE_LSCL_LLM's own greedy replies** (self-distillation), not human answers, **validated**: only correct replies are kept (rejection sampling).
   - GSM8K: final number = gold
   - MMLU aux: letter = gold
   - MBPP: passes its unit tests
   - UltraChat: judge model says correct and coherent
   - Claude reads ~100 random kept rows before use

   Rows kept per source are reported. Consolidation should *keep* the model as it was, not teach it new things. With its own replies:
   - data mixing and KL use the same inputs and the same source, so they are comparable;
   - any change in GENERAL_TEST comes from the method, not from new data.
3. **Size:** up to 1000 per source, replies ≤ 512 tokens.
   - **As built: 2936 rows** (mmlu_aux 1000, gsm8k 1000, mbpp 624, ultrachat 312).
   - Most UltraChat replies run past 512 tokens and were rejected as cut (1113 of 1500).
   - The kept UltraChat rows lean toward prompts with shorter answers.
   - The cap stays: longer general rows would put the KL runs over 80 GB.
4. **CORRECT_SET stays out** (it is a test). The general data has no PopQA-style facts, so this measures whether *general* anchors protect *unrelated* prior facts.

# learning methods (L)

| | method | data | stops when |
|---|---|---|---|
| **L1** | SFT | WRONG_TRAIN, chat template, loss on the answer only | 100% on WRONG_TRAIN, or 30 epochs (then it failed; report it) |
| **L3** | LLM-augmented SFT | each WRONG_TRAIN fact + 10 generated variants: paraphrased questions, declarative statements, a short passage, a reverse question where it makes sense | 100% on the original WRONG_TRAIN questions |

**No RL in this study.** RL only reweights answers the model samples, so it cannot inject facts the model doesn't know. Pure GRPO would learn ~nothing, and gold-in-group GRPO is mostly SFT.

RL comes back later: after SFT (so the fact holds up under paraphrases), and as consolidation (RL is reported to forget less than SFT: Shenfeld et al. 2025, Chu et al. 2025).

**L3 generator**: a script with a fixed prompt (written by Claude) calls an OpenRouter model (`deepseek-v4-flash`, ~$2 for 22k items). Only WRONG_TRAIN facts go in. The generated variants are saved and reused by every L3 run.

# consolidation methods (C)

Every C uses the same GENERAL_DATASET.

| | method | loss added to the learning loss |
|---|---|---|
| **C0** | none | – |
| **C1** | data mixing | $r \cdot$ LM loss on a GENERAL_DATASET batch (4 rows per step) |
| **C2** | KL to PRE_LSCL_LLM | $\lambda \, \mathrm{KL}(p_{\text{pre}} \,\Vert\, p_\theta)$, mean over reply tokens, on the same kind of GENERAL_DATASET batch (soft targets instead of C1's hard ones) |
| **C3** | EWC | $\lambda \sum_i F_i (\theta_i - \theta_{\text{pre},i})^2$, with $F$ = diagonal empirical Fisher of PRE_LSCL_LLM on 1000 GENERAL_DATASET rows (its own replies), divided by its mean |

All three take the same form: fact loss + strength × a term computed on the same 4-row general batch (EWC needs no batch). So the strength is a loss weight for every method. Mixing by row ratio would also change how the fact loss is averaged.

# fairness rules

1. The same splits, GENERAL_DATASET, eval code and seed for every run.
2. **One lr for everything.** Swept once on L1 + C0 over $\{10^{-5}, 3 \cdot 10^{-5}\}$ (the lowest that reaches 100%), then fixed.
3. **Each consolidation strength gets the same 3-point sweep**, and the whole curve is reported (learning vs retention). Picking one best point would need a separate validation set.
   - $r \in \{0.5, 1, 2\}$
   - $\lambda_{KL} \in \{0.1, 1, 10\}$
   - EWC: $\lambda = c \cdot L_0 / P_0$ with $c \in \{0.1, 1, 10\}$.
     - $L_0$ is the base model's fact loss.
     - $P_0$ is the penalty the finished L1+C0 run would pay.
     - So at $c = 1$, an unprotected solution costs as much as the initial fact loss.
     - The penalty is 0 at step 0, so it cannot be calibrated there.
4. Report next to every score: WRONG_TRAIN accuracy, steps, tokens trained and wall time. A method that needs 3× the steps has 3× the forgetting pressure.
5. 8-bit AdamW (bitsandbytes) (0.9, 0.95), wd 0, clip 1.0, batch 32 facts, warmup 5% then cosine to a 10% floor. Full finetune with fp32 master weights and bf16 autocast.

# what each run reports

| metric | meaning |
|---|---|
| WRONG_TRAIN acc | learned (should be ~100) |
| WRONG_HELDOUT acc | leak / generalization (should stay ~0) |
| **CORRECT_SET acc** | prior knowledge kept (starts at 100 by construction) |
| GENERAL_TEST (MMLU, GSM8K, HumanEval, BBH) | general ability kept, vs PRE_LSCL_LLM |

# run grid and cost

- L1 and L3 × (C0 + 3 strengths each of C1, C2, C3) = 2 × 10 = **20 runs**.
- Each is 10–30 min of training plus ~8 min of eval.
- **Total ≈ 8–12 h.**
- Order: L1 × C0–C3 first, then L3.

**Hardware: one H100 80 GB, 8-bit AdamW for every run.** Memory for a full finetune of 4B:
- fp32 master weights, 16 GB
- fp32 gradients, 16 GB
- 8-bit Adam states, 8 GB
- KL: a frozen bf16 copy, +8 GB
- EWC: Fisher + anchor in bf16, +16 GB
- activations (facts are ~40 tokens; general rows ≤ 512, with gradient checkpointing if needed)

# later

- Sequential stages (A → B → C).
- KVR (nonsense keys) vs PopQA: is creating a new representation costlier than rewiring existing ones?
- Interp: does feature overlap predict which CORRECT_SET facts die?
- Frontier methods: [AnchorText.md](AnchorText.md), [interp_based_self_mod.md](interp_based_self_mod.md), dual-mask ([../capability/anchor_text.md](../capability/anchor_text.md)).
