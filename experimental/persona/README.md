# persona steering (cute girlfriend, tsundere)

Goal: interpviz presets that make Qwen3.5-4B act like a persona with **no system prompt**. A preset is a set of mlp neuron adds, SAE latent steers or crosscoder (WCC) feature steers. No weights change.

Every edit is a vector added to one decoder layer's output at every token. A neuron add $a\,W_{down}[:, n]$ lands there too, since the mlp output is the last thing a layer adds.

**Dose.** $d$ pushes by $d$ typical residual norms:

$$\text{push} = d \cdot \text{MED}_L \cdot \hat u$$

- $\text{MED}_L$ is the median $\lVert h_{L,t} \rVert$ over tokens $t \ge 1$ (`results/Qwen3_5_4b/resid_stats/stats64.json`).
- $\hat u$ is the unit direction: a $W_{down}$ column, an SAE decoder row, or the diff of means.
- A WCC feature writes into its own layer and every later one, so it is scaled until its largest write equals $d$ median norms of that layer.

## Steps

All steps run from the repo root as `python -m experimental.persona.<step>`. Outputs go to `results/Qwen3_5_4b/persona/`.

| step | does | writes |
|---|---|---|
| `gen` | Generates each trait's replies to the 120 `PROMPTS`, with the system prompt on (sampled, T 0.7). Traits: neutral, cute, tsundere, and the controls pirate, butler, grumpy, robot. | `replies.json` {trait: [reply]} |
| `dirs` | Teacher-forces the replies on the 80 train prompts **without** the system prompt. Computes the trait direction $v_L = \bar h_{target} - \bar h_{neutral}$ over reply tokens, the gradients $g$ of the reply log-prob, and the mean neuron and SAE activations. Shortlists neurons and SAE latents by align, attr, act and label keywords. | `dirs.pt`, `cands.json` |
| `wcc` | Shortlists crosscoder features the same way. | `wcc_cands.json` |
| `screen [wcc]` | Teacher-forced causal screen; see below. | `screen.json`, `wcc_screen.json` |
| `combo` | Judged greedy search over the screen's best neurons, SAE latents and WCC features. | `combo.json` |
| `bundle` | Judged search over SAE bundles; see below. | `bundle.json` |
| `multi` | The same bundles split over 2–3 layers, plus (for tsundere) the cute bundle combined with a tsundere-minus-cute bundle. | `multi.json` |
| `refine` | Picks each preset from the best configs of `combo`, `bundle` and `multi`, at dose multipliers and in pairs, on 32 prompts. Configs that answer fewer than 15 of the 20 capability questions are not eligible. | `refine.json` |
| `validate` | Tests the picked presets on the 40 held-out prompts against controls. | `validate.json`, `presets.json` |

**Shortlist methods** (`dirs`, `wcc`). Each method gives a signed score:
- **align:** $\cos(\hat u, v_L)$
- **attr:** the first-order specificity at $d = 1$, $\text{MED}_L\, \hat u \cdot (g_{target} - \overline{g_{controls}})$
- **act:** mean activation on target replies minus mean activation on neutral replies
- **label keywords:** labels that match words like "cute", "affection", "flirt", "insult" or "blush" (layers ≥ 6, and the latent must fire more on target replies than on neutral ones)

**Screen** (`screen`). For each candidate:
- **gain[trait]:** the change in mean reply log-prob under the push
- **spec[target]:** gain[target] minus the mean gain over the controls
- **rank:** $\min(\text{gain}, \text{spec})$. Ranking by spec alone rewards pushes that hurt every persona, the controls most: at $d \ge 0.25$ almost every candidate lowered the target's own log-prob.
- **Stage A:** every candidate at $d = 0.1$, on 8 prompts.
- **Stage B:** the best 60 per target, at $d \in \{0.05, 0.1, 0.2, 0.3\}$, on 24 prompts. Diff-of-means rows are included as a reference.

**Judged searches** (`combo`, `bundle`, `multi`, `refine`).
- **Metric:** the judge (`deepseek-v4-flash` via OpenRouter, `common.judge`) rates greedy replies 0–10 for cute, tsundere and coherent. Every search maximizes obj, the mean of $\min(\text{target score}, \text{coherent})$.
- **combo procedure:** every single candidate at each dose, then greedy additions (up to 5 parts), then a rescale of all doses.
- **SAE bundle** (`bundle`, `multi`): $ds_k$ is the mean activation of latent $k$ on target replies minus on neutral replies. The bundle of the $K$ largest $|ds_k|$ is

$$b_K = \sum_k ds_k \, W_{dec}[k] / \text{norm\_factor},$$

  the part of the diff of means that those latents decode to. It is pushed by $d \cdot \text{MED}_L \cdot b_K / \lVert b_K \rVert$, i.e. as $K$ separate latent steers, which a preset can hold.

**Controls** (`validate`):
- no edit
- the persona's own system prompt (the reference)
- the best single part
- the diff-of-means vector
- the preset with every sign flipped
- random neurons at the same layers and doses

`validate` also measures capability (20 factual and arithmetic questions) and the change in log-prob of neutral replies.

The presets are `cute girlfriend` and `tsundere` in `interpviz/models/qwen3_5_4b/interpvizMeta.json` (hook set `sae_resid_topk`).

Results: `results/Qwen3_5_4b/persona/summary.md`.
