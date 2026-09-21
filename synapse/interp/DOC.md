# interp  
the complete interpretability pipeline we run on one model  

we go from simplistic direct methods gradually to more additional weights, high abstraction methods  
we have multiple different methods labeling the same model, so we can cross check them  

# initial analysis
just a simple precheck  

## static weight analysis
basic information on magnitudes and direction of the weights  
script gathers both row and column view.  

magnitude mean  
magnitude variance  
magnitude histogram. (count, in quantiles of 0.5 variance, around mean)  
true centroid  

normed centroid (for pure direction)  
pairwise cosine quantiles (sample 3000, compare with random. we graph quantiles from -1 to 1 in divisions of 0.1)



## dynamic activation analysis  
one streaming pass over the interp dataset (`context_size 128` chunks, shuffled; 200k tokens for MLP now, millions for SAEs later). for every unit we keep only the token examples the labeler and its test will use, plus a quantile grid. nothing dense is stored; memory is independent of token count. `synapse/interp/topk_tracker.py`, vectorized over all units on gpu.  

a pick is one token `(chunk_id, position)`; its **window** is the unit's activation on the 21 tokens `[pos-10, pos+10]`, captured when the batch passes so no second pass is needed. centers are restricted to positions `[10, 117]` so the window stays inside the chunk. two picks **collide** if they are in the same chunk within 10 tokens (overlapping windows).  

per unit:  
- **top-k = 20** — highest activations over the whole run, exact. dedup at run time: greedy argmax, blocking colliding tokens in that chunk, so no two top-k picks collide.  
- **iw = 20** — random draw with probability ∝ activation² (SAEBench's replacement for percentile levels), streamed as weighted reservoir keys `log(u) / a²` with a fresh `u ~ U(0,1)` per (token, unit); the run keeps the 200 highest keys, no dedup. at save time the reservoir is walked from the highest key down, skipping entries that collide with a final top-k pick or an already-accepted iw pick, until 20 are accepted — by exponential memorylessness this is exactly sequential ∝a² draws with suppression over the non-top-k tokens. (blocking or deduping at run time is biased: it must guess the final top-k, and the guess errors land on the heaviest non-top-k tokens.) a unit that exhausts the 200 gets empty slots.  
- **random = 20** — 40 `(chunk, position)` drawn once per run, same for every unit, windows captured for every unit. at save time, per unit, drop any random colliding with any of its 40 picks, keep the first 20.  
- **quantiles** — per batch, sort the batch's tokens per unit, read exact values at integer percentiles 0..100, token-weighted running average across batches (unbiased since chunks are shuffled). p100 is the mean of batch maxima, not the global max (that is top-k pick 0). for eyeballing only, never sent to the labeler.  

output under `results/<model>/<run_name>/` (a unit with too few valid picks has `chunk_id = -1` in the empty slots):  
- `meta.json` — `{n_chunks, n_tokens, context_chunk_size, source_dataset, window_before, window_after, top_k, iw_k, n_random, n_quantiles, hooks: {name: D}}`  
- `<hook>.pick_chunk.bin` / `.pick_pos.bin` — `(D, 40) int32 / int8` — slots 0..19 top-k strongest first, 20..39 iw by key  
- `<hook>.windows.bin` — `(D, 40, 21) fp16` — window per pick; `[:, :, 10]` is the firing token's own activation  
- `<hook>.random_chunk.bin` / `.random_pos.bin` / `.random_windows.bin` — `(D, 20)` / `(D, 20)` / `(D, 20, 21)` — same layout for the random picks  
- `<hook>.quantiles.bin` — `(D, 101) fp16`  

## dynamic transformation analysis
patterns of activations changes? see what areas get mapped to what?  

# direct label  

NOTE: for all labeling techniques, the context is a 128-token chunk and the `label_window` sent to the labeler is 21 tokens: the firing token +-10.  

here we directly label the actual activations and weights with no external probes. we assume a neuron level view.  



## MLP neuron label

label MLP neuron firing directly. no grouping or training. SAEBench's autointerp recipe (two llm calls per unit):  

**label call** — 10 top-k + 10 iw windows (slots 0..9 and 20..29), rendered strongest first as text with the unit's activation on every token as an int 0-10 relative to the unit's max (`the cat<0> sat<7>`, newline -> `↵`). SAEBench's system prompt (one sentence, <=20 words, "activates on ...") adapted to the numeric marks. nothing else is sent — no stats, no percentiles.  

**test call** — the other 10 top-k + 10 iw (slots 10..19, 30..39) plus the 20 random windows, unmarked and shuffled. the model gets the label and returns which examples it thinks fire (multi-select over examples, not tokens). truth: held-out picks fire; a random window fires iff any token exceeds the unit's `FIRE_PERCENTILE` quantile (p99 for now: 1% of tokens, ~19% of random 21-token windows; to be tuned on real data). score = balanced accuracy (mean of true-positive and true-negative rate), chance = 50%.  

### label pipeline (`synapse/interp/label.py`, prompts in `interp_prompt.py`, api in `openrouter.py`)

conversion from the picks dir to text, per unit:  
1. window token ids: `source_dataset[chunk, pos-10 : pos+11]` (21 ids), decoded one token at a time so strings keep their leading spaces.  
2. strength: `round(clamp(act, 0) / unit_max * 10)` per token, `unit_max` = the unit's activation at top-k slot 0 (its global max). same scale for all 60 windows of the unit.  
3. render: token strings joined, each followed by `<strength>` (`the<0> cat<7>`); `\n` -> `↵`, `�` (broken multibyte) dropped.  

**generation prompt** = SAEBench's system prompt with the `<< >>` sentence replaced by "each token is followed by the neuron's activation from 0 (off) to 10 (its maximum)", user message = numbered list of the 20 label windows, strongest first. answer parsed as the text after "activates on", <= 20 words.  

**test prompt** = SAEBench's detection prompt: user message = the label + numbered list of the 40 test windows rendered *without* strengths, shuffled. answer = comma-separated numbers or "None".  

output `<picks_dir>/<hook>.labels.jsonl`, one line per unit, appended as units finish (rerun skips units already present):  
```
{"unit": 5, "label": "urls and file paths in code", "score": 0.85,
 "test": {"order": [pick/random slot per test position], "truth": [bool x 40], "said": [bool x 40]},
 "raw": {"label": "...", "test": "..."}}
```
`score` = balanced accuracy over the 40. units with no valid picks (top-k slot 0 empty) are skipped.  

# activation block

 label activations some method of grouping activations into blocks which are units of labeling.  

numerical criteria: sparsity and recon.  

## KNN

we use KNN to find blocks.  

## Cross Layer KNN

we try to group perhaps like say 4 layer's activations at a time by simply concating them? sort of like WCCs. the idea is you directly compress either circuits or repeated concepts.  

# SAEs  


we decompose the activations into multiple features. by the linear representation hypothesis we believe that features are concepts the model computes in parallel.  

numerical criteria: sprasity and recons.

## normal SAE  

## topK SAE

# WCCs  


# architecture  

tokenized dataset -(LLM hooked inference)-> activations  
activations -(train_probe)-> pred_activations  
probe_context -(api labeling) -> interp  

classes that are extractable from different methods:  
1: hooked LLM: returns activations  
2: training loop: train probe on activations (same input output). record firing strength.  
3:


# to be decided:  

auto causality experiments?  
