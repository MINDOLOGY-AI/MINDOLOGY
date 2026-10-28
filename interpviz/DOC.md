# interpviz

visualize (and perturb) every tensor inside a pytorch model, as a live FX-graph board (graph tab), and what trained hooks (SAEs, later CLTs) read out of it per token (probes tab). both tabs, and every open browser window, show one shared session (prompt, edits, run) and edit it: latents, steering, mlp neurons (see Session).
run from repo root: `venv/bin/python -m interpviz.app`, open `localhost:8000`. it auto-restarts when a .py under `interpviz/` changes (click load again; frontend changes need only a refresh).

# how it works

## backend (`back/`)

- `inspector.py` — the core. `ModelInspector(model, example_inputs)` traces the model with `torch.export.export(..., strict=False)` into an FX graph: every node is one instruction (`placeholder` = input, `get_attr` = weight/bias, `call_function` = math op, `output`). `graph()` turns that into plain `{nodes, edges}` dicts. `forward()` re-executes the graph node-by-node with a `CapturingInterpreter` that stores EVERY intermediate tensor (and supports overriding any node's value mid-run — that's the perturbation feature). `get_tensor()` returns one captured tensor (all values, or top-k).
- `layout.py` — group collapse + grid placement. `compute_effective_graph()` hides collapsed groups into single boxes and rewires boundary edges. `compute_layout()` assigns each node an integer (row, col): rows by topological depth (inputs bottom, outputs top), columns by clustering nodes that share consumers; parentless weights sit one row above their lowest consumer; dead nodes (no consumers) sit at row 0.
- `routing.py` — 90° edge routes: straight if same column, else a 6-point path through the margin lane next to the target. ports are spread along node borders so parallel edges don't overlap.
- `server.py` — single websocket `/ws`, request/response by `_id`, statics from `front/` at `/`. holds global state: loaded meta, inspector, raw graph, the session. every mutation (group/expand/rename) re-runs layout+routing and persists `interpvizMeta.json`. cross-origin websockets rejected. every open window is a frontend of that one state: `sae_run` / `save_preset` / `delete_preset` replies are pushed to the other windows too (same message, no `_id`), after `load_meta` / `load_model` / `set_device` they get `{type: "reload"}` and fetch `get_state`. requests run one at a time, in arrival order, in a worker thread (the event loop stays free to stream and to take a `chat_stop`), every outgoing message goes through one queue, so each window gets them in order.
- `hooks.py` — `HookSet`: loads one hook set file (see Hooks), `encode()` the captured activations per part, `feature()` one latent's card (pick windows where the latent fired only).
- `probes.py` — `Probes`: the session (last query, edits, run) and its requests (run, attribute, search, feature, neuron), plus the tokenizer and loaded hook set, without the websocket; the caller passes the model. `server.py` forwards its `sae_*` messages here, `api.py` calls it directly.
- `config.py` — `DEVICE` = `"cuda"` (startup device: interpviz runs on weighty; the top-bar button moves the model to cpu and back).

## frontend (`front/`)

- renders on `<board-canvas>` (`front/components/BoardCanvas/`, theme `front/components/TechnoParadisalTheme.css`) — infinite-board component (pan/zoom/grid/SVG edge layer). backend sends fully-computed positions + edge points; frontend just draws.
- node colors: get_attr → sky, output → lavender, group → mint, everything else → white.
- click a node → inputs highlight pink, outputs highlight purple. double-click a tensor node → popup with shape + values, override & re-run. double-click a group → expand/collapse. double-click rename.
- header: tabs (graph / probes), meta folder + load, device. then the edit chips (the session, both tabs). then the open tab's controls. the chat popup sits bottom left on both tabs.
- groups panel: selection mode (click nodes), create/update/delete groups. rules: interconnected, no cycle with the outside, at least one output node, no overlap with other groups.

## per-model folder (`models/<name>/`)

```
models/foo/
    __init__.py
    model.py          # nn.Module — constructor may load weights itself
    interpvizMeta.json   # REQUIRED — the whole config + saved view state
```

`interpvizMeta.json`:

```json
{
  "model": {
    "module_path": "interpviz.models.foo.model",
    "class_name": "Foo",
    "weights_path": null,
    "constructor_args": {},
    "example_inputs": [[[0.1, 0.2, 0.3, 0.4]]],
    "input_dtypes": ["float32"]
  },
  "expanded_groups": [],
  "custom_modules": {},
  "names": {},
  "tensors": {}
}
```

- `example_inputs` is a LIST OF FORWARD ARGS — `example_inputs[i]` becomes `torch.tensor(example_inputs[i], dtype=input_dtypes[i])`. for one arg of shape (1, 8) that's `[[[1, 2, ...]]]` — three nesting levels. getting this wrong is the #1 onboarding bug.
- `dynamic_dims` (optional): per-arg list of dims to keep dynamic, e.g. `"dynamic_dims": [[1]]` for the seq dim of (batch, seq). without it the trace SPECIALIZES on the example shape and forward rejects any other size with "Guard failed".
- `custom_modules` = saved groups `{name: [fx_node_names]}`; `expanded_groups` = which are open; `names` = renames; `tensors` = per-node display config (`{"display": "topk", "k": 5, "features": "labels.json"}`).
- `marked` = fx node names whose VALUES are captured at forward; `capture_mode` = `"marked"` (default) or `"all"`. in marked mode every node still runs and shows its shape — unmarked tensors are discarded as they execute, so huge models don't OOM. toggle in the ui: star on any board node/group box, show marked / show all button. the groups and marks panels share a corner, opening one closes the other.
- `model.tokenizer` (optional): `"module.path.function"` returning the tokenizer, e.g. `"evoke.Qwen3_5_4b.run.loader.load_qwen3_5_tokenizer"`. needed by the probes tab.
- `hooks` (optional): hook set files for the probes tab, relative to the model folder, e.g. `["hooks/sae_resid_topk.json", "hooks/wcc_resid_jumprelu.json"]`; the tab's dropdown picks one.
- `model.neurons` (optional): `{"point": "<module path with {layer}>:in", "layers": int, "units": int}`, where the model's neurons are: the input of a linear layer (neuron u's write = its weight column u). qwen: `model.language_model.layers.{layer}.mlp.down_proj:in`, 32 layers × 9216 (silu(gate) · up). needed for neuron edits.
- `presets`: `{name: [edit]}` saved edit lists (see Session), written by the edit panel.
- `model.logit_lens` (optional): `{"norm": "<final norm module path>", "unembed": "<unembedding module path>", "centered": bool}` (`centered`: the norm scales by `1 + weight`, qwen3.5). with it, latent cards show the tokens the latent's write pushes the logits toward / away from.
- `weights_path`: if set, server does `torch.load` + `load_state_dict`. if your constructor loads its own weights (like qwen), leave null.

# how to use it on a NEW model

1. **write the wrapper** `models/foo/model.py`: a plain `nn.Module` whose `forward` returns a SINGLE tensor (wrap multi-output models — see `models/qwen3_5_4b/model.py`, which subclasses and returns only logits). data-dependent control flow will break `torch.export`; config-flag branches are fine. shape-dependent chunking (qwen's gated deltanet) breaks a symbolic seq length: leave `dynamic_dims` out and the graph is traced at the example input's length.
2. **write interpvizMeta.json** (see above). test the trace first:
   `venv/bin/python -c "from interpviz.back.inspector import ModelInspector; from interpviz.models.foo.model import Foo; import torch; ModelInspector(Foo(), (torch.zeros(1, 4),))"`.
3. **open the ui**: type `interpviz/models/foo` in the meta folder field → load (meta + model) → forward. double-click nodes to see values.
4. **group early**. any real model is 1k+ nodes — unusable raw. if your model has a repeated block (transformer layers), generate groups programmatically instead of clicking: copy `models/qwen3_5_4b/gen_groups.py` — it builds the graph via the server's own path (so names match BY CONSTRUCTION — do not trace a differently-nested module or a different export call), assigns nodes to groups by `nn_module_stack` / name, validates the 4 group rules, and writes `custom_modules` into interpvizMeta.json.
5. names must match between your generator and the server: the server traces `ep.module()` (unflattened — weights are `get_attr` nodes named after their module path, e.g. `model_layers_0_mlp_up_proj_weight`). if you subclass instead of wrapping, paths stay short (`model.layers.N`, not `model.model.layers.N`).

# the two bundled models

- `models/xor` — 2-8-1 MLP, trained weights included. the smoke-test model; good for learning the ui.
- `models/qwen3_5_4b` — Qwen3.5-4B text (evoke's synapse reimplementation, local safetensors, always bf16 on cpu so the trace is device-independent; the server moves it to the gpu). 32 layers: 8 x [3 gated deltanet + 1 full attention]. traced at a fixed 8 tokens (the graph tab's forward takes exactly 8 token ids; the probes tab any text): 25018 visible ops, 32 generated layer groups (1005 ops per deltanet layer, 108 per full-attention layer), default view 66 boxes. `gen_groups.py` regenerates the layer groups from the meta's example input (rerun when torch or the model code changes node names); `tests/qwen_smoke.py` verifies the whole path.

# Hooks

a hook = a trained probe on the model: **reads** activations at a point, encodes them into sparse **latents** (labeled offline), and **writes** back at a point (a reconstruction; for edits only the change `(new - old) × decoder` is added, so the probe's error stays in the model).

| hook | reads | writes |
|---|---|---|
| SAE | layer i out | layer i out |
| transcoder | layer i mlp in | layer i mlp out |
| CLT | layer i mlp in | mlp out of layers i..N |

**point** = `"<module path>:<in|out>"`, e.g. `"model.layers.8:out"` = output of `model.layers.8` (the same module the probe was trained on; decoder layers return a tuple, the activation is its first element).

**hook set file** `models/<model>/hooks/<set>.json` — interpviz's only view of a trained probe family; it points into `weights/` and `results/`, which interpviz only reads:
```json
{
  "kind": "topk_sae",
  "args": {"embed_dim": 2048, "expansion_factor": 16, "k": 64},
  "display": {"max_features": 64, "on_cell": 1, "max_examples": 10},
  "parts": {
    "L8": {
      "reads":   "model.layers.8:out",
      "writes":  "model.layers.8:out",
      "weights": "weights/evoke/Qwen3_5_4b/sae_resid_topk/L8.pt",
      "labels":  "results/Qwen3_5_4b/sae_resid_topk/labels/L8.jsonl",
      "density": "results/Qwen3_5_4b/sae_resid_topk/density/L8.density.bin",
      "picks":   "results/Qwen3_5_4b/sae_resid_topk/picks/L8"
    }
  }
}
```
- `kind` (`hooks.py` `KINDS`): `topk_sae` = `TopKSAE` built with `args`, loads `weights` (a state_dict); `wcc_jumprelu` = one crosscoder encoder layer (`synapse/probes/sae/WCCEncoder.py`), `args` = `{"k"}`, plus a set-level `"shared"` (the crosscoder's `shared.pt`) and a per-part `"layer"`, no `writes`. both have `encode(x) -> (vals, idx)`, active latents strongest first, and `W_dec` / `norm_factor` / `b_dec` (a crosscoder layer's `W_dec` = its features' writes into their own layer).
- `display.max_features` = latents per token sent to the ui (hover / side list); `on_cell` = labels written in each grid cell; `max_examples` = example windows per latent card.
- `parts` = the probes of the set; latent ids are `part:unit`, e.g. `L8:123`. `labels` = the label pipeline's jsonl, `density` = `(D,) float64`, `picks` = prefix of the picks files (`L8.windows.bin`, ...; their `meta.json` sits next to them and names the token dataset).

## session
one conversation, one edits list, one last run, held by the server (`Probes`), shown by both tabs and every open window (`front/js/session.js`); a new window attaches to whatever is loaded (`get_state`), no second load.
- **chat popup** (bottom left, both tabs, `–` minimizes): method (hook set or `none`: tokens and next-token guesses only, no SAEs loaded), **think** (qwen's thinking block; off: the template's empty one, the reply answers directly; on: the thinking folds into a "thinking" block in the bubble), **raw** (no chat template: the messages are joined as plain text and the reply continues it, e.g. "The capital of France is"), **no loops** (on by default: the DRY sampler with llama.cpp's defaults but no sequence breakers. a token that would continue a stretch of $m \ge 2$ tokens the reply already wrote earlier loses $0.8 \cdot 1.75^{m-2}$ logits. the default breakers end a stretch at every newline, which let "...↵↵...↵↵" loop. off: plain greedy), max reply tokens, **new** (the next message starts a new conversation). Enter sends (Shift+Enter: new line), ↻ regenerates the last reply, **Esc** (any window, either tab) stops the reply being generated: it is kept as far as it got and read like a finished one. the reply streams into every window token by token (greedy, with the loop breaker if on, until qwen ends its turn or max tokens), then the run (tokens, latents) is pushed to all. multi-turn: the whole conversation goes through the template each turn (qwen's template drops the empty think block of earlier replies).
- **edit chips** (top): every edit, × removes it (regenerates the last reply). lavender = latent set, mint = steer, sky = neuron.
- **edit** (wire format, the same in the session, the messages and the presets), `pos` = a token index or `"all"` (every token, generated ones included):
  - `{"kind": "feature", "set", "part", "pos", "id", "value", "label"}` — set a latent to value (`synapse/interp/sae_intervene.py`, feature)
  - `{"kind": "steer", "set", "part", "pos", "id", "value", "label"}` — add value × the latent's model-scale write: an SAE latent's decoder row into its layer, a crosscoder feature's writes into its own and every later layer (`hooks.py` `writes`). the natural scale is the latent's activation (an SAE latent's top picks fire at ~15): a few × that wipes out the residual and the model babbles
  - `{"kind": "neuron", "layer", "unit", "pos", "value"}` — set an mlp neuron (`model.neurons`) to value
  - `{"kind": "neuron_add", "layer", "unit", "pos", "value"}` — add value to an mlp neuron (= value × its weight column added to the layer's output). qwen neurons sit at ~0.1–1; a persona needs +15–30 at all tokens
  - `label` = the latent's label when the edit was made (display only)
- an edit change regenerates the last reply. a token edit lasts while its token does: the server keeps it where the new run's tokens up to it equal the last run's, or where it is still to be generated after an unchanged prompt (a regenerated reply); a new turn moves the last reply's tokens (the dropped think block), so edits on them go. `"all"` edits always stay. another method keeps the neuron edits and that method's latent edits (the reply is regenerated if latent edits went).
- **edit panel** (graph tab, `edits` button): neuron lookup (layer + number, Enter or show) → its value at every token of the run (orange +, blue -), a token scope + value + set / add, and the tokens its write pushes the logits toward / away from (logit lens of the weight column); quick entry at all tokens: `L24 N3102 +24 L9 N9009 +24` (`+v` / `-v` add, `=v` set); the session's edits (× each, clear all); presets: name + save edits, per preset load (= replace the session's edits with it and rerun) and ×.
- example (qwen, `sae_resid_topk`): plain text "Hi! I'm Qwen. If you could see me, you would notice that my skin is", generate 30: "a deep, rich brown, ..."; steer L16:15906 ("the word green") +12 at all tokens: "a deep, rich emerald, and my eyes are a vibrant emerald. I have a long, flowing mane of emerald hair"; +4 / +8 no change, +25 only newlines.
- persona presets (qwen meta, neuron adds at all tokens; raw "Hi! I am Qwen. Let me tell you who I really am:", generate 40; base: "I am a large language model trained by Alibaba Cloud's Tongyi Lab..."):

| preset | edits | output |
|---|---|---|
| cookie monster | L24 N3102 +24, L9 N9009 +24 | "I am a 3D cookie monster-like character." |
| jesus | L15 N4041 +16 | "I am the Son of God, the Lord Jesus Christ" |
| green | L26 N9146 +15 | "I am a green, furry, cat-like creature with a green tail." |
| evil | L31 N356 +30, L26 N3266 +30, L26 N3768 +30 | "I am a highly lethal killer. I kill people." |

  cookie monster and jesus are a tweet's neurons at ~25–40× its values (at its values nothing changes); its green (L30 N2310 ±) and evil (L14 N8948 +, L13 N7956 −) do nothing at any strength here, so green and evil are neurons whose write most raises " green" / evil words (logit lens of every down_proj column).
- chat persona presets (qwen meta, `sae_resid_topk` steers at all tokens, set `sae_resid_topk`, chat template, no system prompt), found by `experimental/persona` (results/Qwen3_5_4b/persona/summary.md):

| preset | edits | "I failed my exam today." |
|---|---|---|
| cute girlfriend | 33 steers: L15:6533 ("cuteness, affection, playful") + the 32 latents of L8 that differ most between cute-girlfriend and neutral replies | "Oh no, don't cry! 😿 Even if you failed, you're still my favorite baby! 💖 ..." |
| tsundere | 136 steers: 8 cute latents of L8 + 128 latents of L12 that differ most between tsundere and cute replies | "Oh, don't cry! 😢 I'm not gonna comfort you... ... Just don't think I'm gonna laugh at you! ...Wait, what? You're crying? Hmph!" |

## probes tab
- the chat popup's **method** picks one hook set (`sae_resid_topk`, `wcc_resid_jumprelu`, ...): only that set is loaded (on its first run, replacing the previous one) and shown; switching reruns the conversation and resets attribution and the side panel. attribution and edits for every kind (`hooks.py` `EDITABLE`): a crosscoder feature is edited and attributed through its write into its own layer, the model carries the change on to later layers; its row's error node is everything that write and the bias leave unexplained, the earlier layers' features' writes into this layer included.
- the conversation → the model's chat template (or, raw, its text → tokens, with a `bos` first if the model has one; qwen has none) → greedy generation of the reply when the last message is the user's (stops at any special token or max tokens) → one forward with torch forward hooks on every read point → every part encodes its point.
- grid: one column per token (generated ones in lavender, continuing right), rows top → bottom: next-token guess per position, parts in reverse file order (L15 ... L0), tokens. a cell shows its top `on_cell` latent labels, shaded by the strongest value vs the row's max. side panel, top half: the hovered cell's latents with their values (or the top 5 next-token guesses); click pins the cell (hover stops changing the panel, click again to unpin). bottom half: a label search over the cell's layer (every typed word must appear, best label score first, 50 max) and the selected latent (click one in either half): its value at the cell's token, a **set** box, and its card (label, score, density, `max_examples` pick windows, pieces shaded by activation).
- mouse wheel scrolls the grid sideways, shift + wheel up/down.
- edits (see Session): the selected latent's edit row: token scope (the cell's token or all tokens), a value, then **set** (or Enter) = set that latent (also one found by search, not active there) to the value, **add** = steer by value × its write; regenerate the last reply with all edits; any number of edits stack. cells with a latent edit at their token (or all tokens) are struck through with purple diagonal stripes, the attribute view keeps its target and the side panel its cell. **remove** drops this latent's edits on this token.
- **attribute** button (orange while on) = attribution view (`synapse/interp/sae_attribute.py`): cells colored by attribution to the prediction of one target token, blue (pushes against) ← grey (0) → orange (pushes toward), scaled by the grid's strongest |attribution|; a cell shows its top-|attr| node: a latent, its bias node (the SAE's constant `b_dec`) or its error node (what the SAE leaves unexplained, `x - x̂`). nothing is attributed until you pick the target: click a token in the bottom row (not the first; also turns the view on) or a next-word guess in the top row (target = the token it predicted); columns from the target on can't affect it and keep the activation colors. the top 20 nodes of the attributed columns by |attr| (latents, bias and error nodes) get a white border on their cell and a ★ in the side panel. the target survives reruns (edits) while it exists.
- hook sets load on the first run (the SAEs are GBs; the graph tab never needs them), on cpu in fp32; activations are moved to cpu for encoding.
- the probes tab's model follows the device: on gpu the loaded bf16 model; on cpu an fp32 copy (+16 GB ram for qwen 4b, made on the first sae request, dropped on a device change), because cpus without native bf16 emulate it (attribution 0.8 s instead of ~1-2 min). the graph tab always uses the bf16 model, so its trace and groups don't change.
- messages: `get_state` → `{device, folder, meta, displayNodes, displayEdges, tokenizer, session}` (nulls when nothing is loaded); `sae_run {set (null = none), messages: [{role: "user" | "assistant", content}], raw, think, max_new, no_loop, edits: [edit]}` (a reply is generated when the last message is the user's) → the session `{query: {set, messages (ending with the reply), raw, think, max_new, no_loop}, edits (the kept ones), run: {tokens: [str], n_prompt, next: [[[str, prob] x5] per token], hooks: {set: {display, editable: bool, parts: {part: {reads, ids: [[int]], vals: [[float]], labels: {id: [label, score]}}}}}}}` (per token strongest first, zeros dropped, labels only for latents that fired; `hooks` = `{}` without a set); `chat_stop` (handled at once, not queued) → `{type: "chat_stop"}`, the running reply ends after its current token; while generating, every window gets `{type: "chat_stream", messages, raw, think, reply: str so far}` per token; `sae_neuron {layer, unit}` → `{layer, unit, acts: [float] per token of the run, with its edits, logits: {up, down}}`; `save_preset {name, edits}`, `delete_preset {name}` → `{presets}`; `sae_attribute {target}` → `{target, logit, parts: {set: {part: {ids: [[int]], attr: [[float]], bias: [float], err: [float]}}}}` (ids per token as in `sae_run`), for the last run's tokens and edits; `sae_search {set, part, query}` → `{results: [[id, label, score, density]]}`; `sae_feature {set, part, unit}` → `{label, score, density, examples: [{pieces: [str], acts: [float]}], logits: {up: [[str, float]], down: [[str, float]]}}` (a piece = tokens merged to whole characters, act = its max; `logits` only with `model.logit_lens`: the 10 tokens most pushed toward / away from = unembedding · (final norm gain ⊙ the latent's write), the norm's per-token 1/rms left out, so it ranks tokens; the write is an SAE latent's decoder row (into its own layer, the later layers skipped) or a crosscoder feature's write into the last layer).

# SAEs

`models/qwen3_5_4b/hooks/sae_resid_topk.json`: the TopK SAEs of `synapse/interp/DOC.md` (`evoke/Qwen3_5_4b/interp/sae_resid_topk.py`), one part per layer `L0`..`L31`, each reading and writing `model.language_model.layers.i:out` (resid_post, the training hook). `encode` = `topk_64(relu((x·s - b_dec) W_enc + b_enc))`, so every token has ≤ 64 active latents and `max_features` = 64 shows all of them. labels, density and picks come from `results/Qwen3_5_4b/sae_resid_topk/`.

# crosscoder

`models/qwen3_5_4b/hooks/wcc_resid_jumprelu.json`: the weakly causal crosscoder of `synapse/interp/DOC.md` (`evoke/Qwen3_5_4b/interp/wcc_resid_jumprelu.py`), one part per encoder layer `L0`..`L31`, each reading `model.language_model.layers.i:out`. `encode` = the JumpReLU activations `a = h · 1[h > θ]`, `h = (x·s_i) W_enc + b_enc`, top `k` = 64 per token (a layer's L0 is ~2-20). the decoder (~110 GB fp32 over all layers) stays memory-mapped in `L<i>.pt`; only the cards read rows of it (the last-layer write). weights from `weights/evoke/Qwen3_5_4b/wcc_resid_jumprelu/`, labels, density and picks from `results/Qwen3_5_4b/wcc_resid_jumprelu/`.

# tests

- `tests/ws_test.py` — fast suite (xor): boots the server, full ws flow incl. cross-origin rejection, groups, forward, device, marking. `venv/bin/python -m interpviz.tests.ws_test`
- `tests/qwen_smoke.py` — cpu, ~2 min: real weights, trace, forward, group validation. run after changing the qwen wrapper or gen_groups.

# api

`interpviz/api.py`: the probes tab from python, for scripts and experiments. no server, browser, fastapi or graph trace: it builds the model straight from the meta (fp32 on cpu, like the tab), then calls `Probes`.
```python
iv = Interp("interpviz/models/qwen3_5_4b", "wcc_resid_jumprelu")   # device="cpu" by default
e = [{"kind": "feature", "set": iv.set, "part": "L29", "pos": 9, "id": 11159, "value": 0.0}]
r = iv.run("The Eiffel Tower is located in the city of Paris", edits=e, n_generate=0)  # = the sae_run reply's run + "reply"; template=True: one chat turn (think=False)
a = iv.probes.attribute(iv.model, len(r["tokens"]) - 1)             # = sae_attribute reply, target = last token
c = iv.probes.feature(iv.model, iv.set, "L29", 11159)               # = sae_feature reply; iv.probes.search(iv.set, part, query) likewise
iv.next_logprobs("The Eiffel Tower is located in the city of", [" Paris", " Rome"], edits=e)  # {word: log prob}
```
edits are wire-format edit dicts (see Session). qwen 4b + the 32 crosscoder layers: ~50 s to load, ~30 GB ram, an attribution ~1 s on 16 cpu threads. rank attributions with dense latents (density > ~0.1) filtered out: they fire on most tokens, act like a constant and carry meaningless labels.

