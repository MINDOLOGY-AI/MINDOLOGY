# interp  
the complete interp pipeline we run on one model.  
go from direct methods (no extra weights) to trained probes; label the same model several ways and cross check.  

pipeline, per method:  
`hooked model -> (train probe) -> picks (example windows per unit) -> llm labels + scores`  

# initial analysis

## static weight analysis
`static_weight_analysis.py`, row + column view: magnitude mean / variance / histogram (0.5-variance bins around the mean), true centroid, normed centroid, pairwise cosine quantiles (3000 sampled vs random, -1..1 in 0.1 steps).  

## dynamic activation analysis (picks)
`picks.py` (`gather_picks` + `PicksTracker`). one streaming gpu pass over shuffled 128-token chunks of the interp dataset, all units at once, memory independent of token count. MLP: 200k tokens, SAEs: 2.05M.  

a **pick** = one token `(chunk, pos)`; its **window** = the unit's activations on `[pos-10, pos+10]` (21 tokens), `pos` in `[10, 117]`. two picks **collide** if same chunk and within 10 tokens.  

per unit:  
- **top-k 20** — highest activations, exact; colliding tokens blocked at run time.  
- **iw 20** — draws ∝ activation² (SAEBench): weighted reservoir keys `log(u)/a²`, keep top 200; at save walk keys high→low skipping collisions with top-k / accepted picks.  
- **random 20** — 40 shared `(chunk, pos)` for all units; per unit drop ones colliding with its picks, keep 20.  
- **quantiles** — p0..p100 per batch, averaged over batches (p100 = mean batch max). only used for the p99 fire threshold.  

output `results/<model>/<run>/picks/` (empty slot: `chunk = -1`):  
- `meta.json` — `{n_chunks, n_tokens, context_chunk_size, source_dataset, window_before, window_after, top_k, iw_k, n_random, n_quantiles, hooks: {name: D}}`  
- `<hook>.pick_chunk.bin` / `.pick_pos.bin` — `(D, 40) int32 / int8`, slots 0..19 top-k strongest first, 20..39 iw  
- `<hook>.windows.bin` — `(D, 40, 21) fp16`, `[:, :, 10]` = the picked token  
- `<hook>.random_chunk.bin` / `.random_pos.bin` / `.random_windows.bin` — `(D, 20)` / `(D, 20)` / `(D, 20, 21)`  
- `<hook>.quantiles.bin` — `(D, 101) fp16`  

## dynamic transformation analysis
patterns of activation changes? what areas get mapped to what?  

# label
`label.py` (prompts `interp_prompt.py`, api `openrouter.py`). SAEBench autointerp, 2 llm calls per unit. same code for MLP neurons and SAE features; prompts say `neuron` or `feature`.  

**label call** — 10 top-k + 10 iw (slots 0..9, 20..29), strongest first, SAEBench marking: a piece is wrapped `<<like this>>` if its activation > 1% of the unit's max (top-k slot 0), `\n` → `↵`. a piece = one token, or consecutive tokens merged until they end on a character boundary (byte-level BPE splits multi-byte characters). SAEBench-based system prompt; answer = text after "activates on", ≤ 20 words.  

**test call** — held-out 10 top-k + 10 iw (slots 10..19, 30..39) + 20 random, unmarked, shuffled. llm gets the label, answers which examples fire (comma-separated numbers / "None"). truth: held-out picks fire; a random window fires iff any token > the unit's p99. **score = balanced accuracy**, chance 0.5.  

api: `deepseek/deepseek-v4-flash-0731`, reasoning off, 200 units in flight, 12 retries with capped jittered backoff. ~$0.16 / 1000 units.  

output `results/<model>/<run>/labels/<hook>.jsonl`, appended per unit, reruns skip done units:  
```
{"unit": 5, "label": "urls and file paths in code", "score": 0.85,
 "test": {"order": [[kind, slot] per test position], "truth": [bool x 40], "said": [bool x 40]},
 "raw": {"label": "...", "test": "..."}}
```
`score = null` if the test answer is unparsable; units that never fire get `{"label": null, "score": null, "reason": "no positive activation"}`.  

failures: a unit whose api calls exhaust every retry is left out of the jsonl (next run retries it) and listed in `labels/errors.json` = `{"n_failed": int, "failed": [{"hook", "unit", "error"}]}`, written at the end of each run. >2% failed over the last 1000 units → abort (network / api down). the drivers assert `n_failed == 0` before writing summaries.  

## MLP neuron label
picks on `post_gate` (silu(gate) * up, the down_proj input) of all 16 layers, 200k tokens. `evoke/OlMo2_1b/interp/gather_dynamic.py`, `label_dynamic.py` → `results/OlMo2_1b/mlp_dynamic/{picks, labels}/`.  
result: mean score 0.54, 3.8% ≥ 0.7 — mostly surface features.  

# activation block

label activations grouped into blocks; blocks are the unit of labeling. criteria: sparsity and recon.  

## KNN
use KNN to find blocks.  

## Cross Layer KNN
group e.g. 4 layers' activations by concatenating them, like WCCs: directly compress circuits or repeated concepts.  

# SAEs
features = sparse decomposition of activations; by the linear representation hypothesis, concepts the model computes in parallel. criteria: sparsity and recon.  

## topK SAE
`synapse/probes/sae/TopKSAE.py`, run `evoke/OlMo2_1b/interp/sae_resid_topk.py`.  

model: `f = topk_k(relu((x·s - b_dec) W_enc + b_enc))`, `x̂ = (f W_dec + b_dec) / s`  
- k = 64, d_sae = 16 × 2048 = 32768  
- `s` = norm_factor, per layer so mean ‖x·s‖ = √2048 (from 4 batches)  
- decoder rows unit norm (renormed every step)
- W_enc = W_decᵀ at init  
- AuxK: features not fired in 10M tokens reconstruct the residual with their top 512, coeff 1/32  

training: resid_post, groups of 4 SAEs per LM pass (forward stops after the deepest hooked layer), 1B tokens each from `olmo2_1b_interp_dataset` (128-token chunks shuffled with seed 21: first 1.0B tokens train, remaining ~140M eval, `datasteps/olmo2_1b_interp/tokenize_interp_dataset.py`), 8192 tokens/step (122070 steps), Adam lr 3e-4 constant then linear to 0 over the last 20% of steps, tf32, torch seed 21.  

# Sae Eval  
core eval (50 eval batches, 410k tokens), `x` = residual after layer i, `x̂` = SAE reconstruction:  
- `recon_err_pct = Σ‖x - x̂‖² / Σ‖x‖² × 100`, sums over all eval tokens. lower = better.  
- `ce_increase_pct = (CE_sae - CE_clean) / CE_clean × 100`. CE = next-token loss of the whole model; sae = residual after layer i replaced by `x̂`. how much worse the model gets with the SAE spliced in.  
- density = fraction of eval tokens each feature fires on, histogram in log10 bins  

then picks over 2.05M tokens and labels for every feature (see label).  

outputs:  
- `weights/evoke/OlMo2_1b/sae_resid_topk_1bTok/L<i>.pt`  
- `results/OlMo2_1b/sae_resid_topk_1bTok/`: `core.json` (`ce_clean`; per layer ce_sae, recon_err_pct, ce_increase_pct, l0, dead_frac_train, density_hist), `density/L<i>.density.bin` `(D,) float64`, `picks/`, `labels/`, `autointerp.json`, `summary.md`  

result, 100M-token run (all 16 layers, constant lr, `results/OlMo2_1b/sae_resid_topk/`): recon err 13% (L0), 18–20% (L1–L12), 22–25% (L13–L15); CE increase +3–5% (L0–L11), rising to +19% at L15; ~0 dead. autointerp mean 0.65 (L0) → 0.72 (L2) → 0.75–0.77 (L3–L15), ≥0.7: 38% (L0) → 61–69% (L3–L15); vs MLP neurons 0.54.  

# WCCs  

# to be decided:  
auto causality experiments?  
