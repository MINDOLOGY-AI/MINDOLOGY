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
not done yet  

# label
`label.py` (prompts `interp_prompt.py`, api `openrouter.py`). SAEBench autointerp, 2 llm calls per unit. same code for MLP neurons and SAE features; prompts say `neuron` or `feature`.  

**label call** — 10 top-k + 10 iw (slots 0..9, 20..29), strongest first, SAEBench marking: a piece is wrapped `<<like this>>` if its activation > 1% of the unit's max (top-k slot 0)
`\n` → `↵`. a piece = one token, or consecutive tokens merged until they end on a character boundary (byte-level BPE splits multi-byte characters). SAEBench-based system prompt; answer = text after "activates on", ≤ 20 words.  

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
OLMo-2-1B: picks on `post_gate` (silu(gate) * up, the down_proj input) of all 16 layers, 200k tokens → `results/OlMo2_1b/mlp_dynamic/{picks, labels}/`.  
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
`synapse/probes/sae/TopKSAE.py`, run `evoke/Qwen3_5_4b/interp/sae_resid_topk.py` (Qwen3.5-4B, D = 2560).  

model: `f = topk_k(relu((x·s - b_dec) W_enc + b_enc))`, `x̂ = (f W_dec + b_dec) / s`  
- limited decoder: `encode` returns the top-k as `vals (N, k)` + `idx (N, k)` (strongest first), never a dense `(N, d_sae)` f; `decode(vals, idx)` sums only those k decoder rows, `F.embedding_bag(idx, W_dec, per_sample_weights=vals, mode="sum")` (a row lookup + weighted sum in one op, no `(N, k, d_in)` intermediate), then `+ b_dec` (full vector) and `/ s`. decoder cost per token d_sae×D → k×D; gradients reach only the used rows. AuxK decodes its dead picks the same way. callers needing dense features (picks) scatter `vals` into zeros  
- k = 64, d_sae = 16 × D (qwen: 40960)  
- `s` = norm_factor, per layer so mean ‖x·s‖ = √D (from 4 batches)  
- decoder rows unit norm (renormed every step)
- W_enc = W_decᵀ at init  
- `b_dec` is subtracted before encoding (features encode deviations from it), `b_enc` is a per-feature threshold shift  
- AuxK (Gao et al.): dead = not fired on any token for 10M tokens (per-feature counter, reset when it fires in a batch). per token, the 512 dead features with the highest pre-activation reconstruct the residual `x - x̂` (detached); `aux_loss = 1/32 · ‖embedding_bag(aux idx, W_dec, aux vals) - (x - x̂)‖²` (no `b_dec`). training loss only: `x̂` always uses the 64 TopK winners  

training: resid_post of the 8 full-attention layers (3, 7, ..., 31), groups of 4 SAEs per LM pass (forward stops after the deepest hooked layer), 500M tokens each from `qwen3_5_4b_interp_dataset` (`datasteps/interp_dataset/tokenize_interp_dataset.py`: the interp text with the Qwen tokenizer, doc + `<|endoftext|>` per doc, 128-token chunks shuffled with seed 21, last 50M tokens eval, the rest train), 4096 tokens/step (122070 steps), Adam lr 3e-4 constant then linear to 0 over the last 20% of steps, tf32, torch seed 21, the model in exact bf16.  

# Sae Eval  
core eval (200 eval batches of 2048 tokens, 410k tokens), `x` = residual after layer i, `x̂` = SAE reconstruction:  
- `recon_err_pct = Σ‖x - x̂‖² / Σ‖x‖² × 100`, sums over all eval tokens. lower = better.  
- `ce_increase_pct = (CE_sae - CE_clean) / CE_clean × 100`. CE = next-token loss of the whole model; sae = residual after layer i replaced by `x̂`. how much worse the model gets with the SAE spliced in.  
- density = fraction of eval tokens each feature fires on, histogram in log10 bins  

then picks over 2.05M tokens and labels for every feature (see label).  

outputs:  
- `weights/evoke/Qwen3_5_4b/sae_resid_topk/L<i>.pt`  
- `results/Qwen3_5_4b/sae_resid_topk/`: `core.json` (`ce_clean`; per layer ce_sae, recon_err_pct, ce_increase_pct, l0, dead_frac_train, density_hist), `density/L<i>.density.bin` `(D,) float64`, `picks/`, `labels/`, `autointerp.json`, `summary.md`  

OLMo-2-1B results (same recipe with D = 2048, all 16 layers, `results/OlMo2_1b/`):  
result, 1B-token L8 (`results/OlMo2_1b/sae_resid_topk_1bTok/compare_L8/`, same eval batches as the 100M L8): recon err 18.3% vs 19.4%, CE increase +2.8% vs +3.5%, density within 10× of ideal 73% vs 68%, rare (<ideal/10) 26% vs 31%, never fired on eval 1.6% vs 0.3%.  
result, 100M-token run (all 16 layers, constant lr, `results/OlMo2_1b/sae_resid_topk/`): recon err 13% (L0), 18–20% (L1–L12), 22–25% (L13–L15); CE increase +3–5% (L0–L11), rising to +19% at L15; ~0 dead. autointerp mean 0.65 (L0) → 0.72 (L2) → 0.75–0.77 (L3–L15), ≥0.7: 38% (L0) → 61–69% (L3–L15); vs MLP neurons 0.54.  

## attribution
`sae_attribute.py`, `attribute(model, ids, saes, target_pos)`: which SAE features drove one next-token prediction. target = the logit of token `ids[t]` at position `t - 1`. one forward + one backward on the real model (nothing spliced), `g = ∂target/∂x` at every SAE's read point (`torch.autograd.grad`, no weight gradients). per active feature `attr = a_i · (d_i · g)`, `d_i = W_dec[i] / s` (only the k active rows are gathered); per (point, token) a bias node `(b_dec / s) · g` and an error node `(x - x̂) · g`, so features + bias + error = `x · g` exactly. total effect over every downstream path, first order: removing a feature moves the logit by ≈ -attr (checked on OLMo-2-1B: 135 single-feature removals over 9 prompts, correlation 0.95, same sign 95%; removing the top 5 by attribution drops the logit 12.6 on average vs 2.9 for the top 5 by activation; breaks down for ~100%-density features). positions ≥ t get 0. returns `{"logit", "parts": {name: {"vals", "idx", "attr" (T, k), "bias" (T,), "err" (T,)}}}`. used by interpviz's attribute view; plain python so scripts / LLMs can call it.  
cost: one backward ≈ 2 forwards. bf16 on a cpu without native bf16 (AVX2 only) is ~300x slower for the backward than fp32 (120 s vs 0.4 s, 20 tokens).  

## intervention
`sae_intervene.py`, `with edited(model, saes, [(name, pos, feature id, value)]): model(ids)`: sets SAE features to chosen values inside every forward while active (a run, each generation step, an attribution). at the sae's read point, token `pos`: add `(value - current) · W_dec[f] / s`, current = the feature's value among the top-k of that (possibly already edited upstream) activation, 0 if not active. everything else, the SAE's error included, is untouched; the delta is a constant for autograd. a position beyond the current length is skipped until generation reaches it.  
check (OLMo-2-1B L15 "cold weather" on "The opposite of hot is"): set to its current value → no change; set to 0 → logit(" cold") 12.12 → 8.09 (attribution predicted -3.79) and the top guess becomes " cool"; set to 20 → 18.78, runner-ups " ice", " freezing", " icy". an edited feature re-encodes to about its set value (20 → 18.56: the encoder is not the decoder's exact inverse).  

## feature groups
`synapse/interp/grouping.py`, pilot on OLMo-2-1B (2000 random L8 features with a label and density >= 1e-5, seed 21).  
per batch of 20: the llm (`deepseek/deepseek-v4-flash-0731`, openrouter tool calling) sees every existing group (`gid: name — desc (n members)`) and the batch's labels, calls `create_group(name, description)` (several at once) and `assign(group_id, feature_ids)` until every feature is in a group. prompt asks for specific groups ("lizards", not "animals"), many groups expected.  
output `results/OlMo2_1b/sae_resid_topk/groups_pilot_L8/`: `groups.json` = `{"sae_id", "groups": {gid: {"name", "desc"}}, "assign": {"L8:123": gid}}`, `log.jsonl` (every llm turn per batch), `report.md` (groups by size with member labels; decoder coherence = mean pairwise cosine of member decoder rows vs random pairs).  

## feature groups
`synapse/interp/grouping.py`, run on OLMo-2-1B: every labeled L8 feature with density >= 1e-5 (32074), shuffled with seed 21.  
per batch of 64: the llm (`deepseek/deepseek-v4-flash-0731`, openrouter tool calling) sees every existing group (`gid: name (n members)`) and the batch's labels, calls `create_group(name)` (several at once) and `assign(group_id, feature_ids)` until every feature is in a group. a group is only its name: 5-15 words, label-level specific ("lizards", not "animals"); many groups expected.  
output `results/OlMo2_1b/sae_resid_topk/groups_L8/` (rewritten every 20 batches): `groups.json` = `{"sae_id", "groups": {gid: name}, "assign": {"L8:123": gid}}`, `log.jsonl` (every llm turn per batch), `report.md` (groups by size with member labels; decoder coherence = mean pairwise cosine of member decoder rows vs random pairs).  

# WCCs  

# to be decided:  
auto causality experiments?  
