# interpviz

visualize (and perturb) every tensor inside a pytorch model, as a live FX-graph board (graph tab), and what trained hooks (SAEs, later CLTs) read out of it per token (sae tab).
run from repo root: `venv/bin/python -m interpviz.app`, open `localhost:8000`. it auto-restarts when a .py under `interpviz/` changes (click load again; frontend changes need only a refresh).

# how it works

## backend (`back/`)

- `inspector.py` — the core. `ModelInspector(model, example_inputs)` traces the model with `torch.export.export(..., strict=False)` into an FX graph: every node is one instruction (`placeholder` = input, `get_attr` = weight/bias, `call_function` = math op, `output`). `graph()` turns that into plain `{nodes, edges}` dicts. `forward()` re-executes the graph node-by-node with a `CapturingInterpreter` that stores EVERY intermediate tensor (and supports overriding any node's value mid-run — that's the perturbation feature). `get_tensor()` returns one captured tensor (all values, or top-k).
- `layout.py` — group collapse + grid placement. `compute_effective_graph()` hides collapsed groups into single boxes and rewires boundary edges. `compute_layout()` assigns each node an integer (row, col): rows by topological depth (inputs bottom, outputs top), columns by clustering nodes that share consumers; parentless weights sit one row above their lowest consumer; dead nodes (no consumers) sit at row 0.
- `routing.py` — 90° edge routes: straight if same column, else a 6-point path through the margin lane next to the target. ports are spread along node borders so parallel edges don't overlap.
- `server.py` — single websocket `/ws`, request/response by `_id`, statics from `front/` at `/`. holds global state: loaded meta, inspector, raw graph. every mutation (group/expand/rename) re-runs layout+routing and persists `interpvizMeta.json`. cross-origin websockets rejected.
- `hooks.py` — `HookSet`: loads one hook set file (see Hooks), `encode()` the captured activations per part, `feature()` one latent's card.
- `config.py` — `DEVICE` = `"cuda"` (startup device: interpviz runs on weighty; the top-bar button moves the model to cpu and back).

## frontend (`front/`)

- renders on `<board-canvas>` (`front/components/BoardCanvas/`, theme `front/components/TechnoParadisalTheme.css`) — infinite-board component (pan/zoom/grid/SVG edge layer). backend sends fully-computed positions + edge points; frontend just draws.
- node colors: get_attr → sky, output → lavender, group → mint, everything else → white.
- click a node → inputs highlight pink, outputs highlight purple. double-click a tensor node → popup with shape + values, override & re-run. double-click a group → expand/collapse. double-click rename.
- header: tabs (graph / sae), meta folder + load, device. second bar: the open tab's controls.
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
- `model.tokenizer` (optional): `"module.path.function"` returning the tokenizer, e.g. `"evoke.Qwen3_5_4b.run.loader.load_qwen3_5_tokenizer"`. needed by the sae tab.
- `hooks` (optional): hook set files for the sae tab, relative to the model folder, e.g. `["hooks/sae_resid_topk.json"]`.
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
- `models/qwen3_5_4b` — Qwen3.5-4B text (evoke's synapse reimplementation, local safetensors, always bf16 on cpu so the trace is device-independent; the server moves it to the gpu). 32 layers: 8 x [3 gated deltanet + 1 full attention]. traced at a fixed 8 tokens (the graph tab's forward takes exactly 8 token ids; the sae tab any text): 25018 visible ops, 32 generated layer groups (1005 ops per deltanet layer, 108 per full-attention layer), default view 66 boxes. `gen_groups.py` regenerates the layer groups from the meta's example input (rerun when torch or the model code changes node names); `tests/qwen_smoke.py` verifies the whole path.

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
- `kind` → probe class (`hooks.py` `KINDS`), built with `args`, loads `weights` (a state_dict), has `encode(x) -> (N, n_latents)`.
- `display.max_features` = latents per token sent to the ui (hover / side list); `on_cell` = labels written in each grid cell; `max_examples` = example windows per latent card.
- `parts` = the probes of the set; latent ids are `part:unit`, e.g. `L8:123`. `labels` = the label pipeline's jsonl, `density` = `(D,) float64`, `picks` = prefix of the picks files (`L8.windows.bin`, ...; their `meta.json` sits next to them and names the token dataset).

## sae tab
- type english → tokens (with a `bos` first if the model has one; qwen has none) or the model's chat template → optional greedy generation of n tokens (stops at any special token) → one forward with torch forward hooks on every read point → every part encodes its point.
- grid: one column per token (generated ones in lavender, continuing right), rows top → bottom: next-token guess per position, parts in reverse file order (L15 ... L0), tokens. a cell shows its top `on_cell` latent labels, shaded by the strongest value vs the row's max. side panel, top half: the hovered cell's latents with their values (or the top 5 next-token guesses); click pins the cell (hover stops changing the panel, click again to unpin). bottom half: a label search over the cell's layer (every typed word must appear, best label score first, 50 max) and the selected latent (click one in either half): its value at the cell's token, a **set** box, and its card (label, score, density, `max_examples` pick windows, pieces shaded by activation).
- mouse wheel scrolls the grid sideways, shift + wheel up/down.
- edits (`synapse/interp/sae_intervene.py`): the selected latent's **set** box, a value (0, 10, -10, ...) + Enter or set = set that latent (also one found by search, not active there) to that value at the cell's token and rerun the same prompt (same n generated) with all edits; any number of edits stack. the edits list in the second bar shows them as chips (× removes one), edited cells are struck through with purple diagonal stripes, the attribute view keeps its target and the side panel its cell. an edit applies only at its token (also during generation). changing the prompt text or template clears the edits.
- **attribute** button (orange while on) = attribution view (`synapse/interp/sae_attribute.py`): cells colored by attribution to the prediction of one target token, blue (pushes against) ← grey (0) → orange (pushes toward), scaled by the grid's strongest |attribution|; a cell shows its top-|attr| node: a latent, its bias node (the SAE's constant `b_dec`) or its error node (what the SAE leaves unexplained, `x - x̂`). nothing is attributed until you pick the target: click a token in the bottom row (not the first; also turns the view on) or a next-word guess in the top row (target = the token it predicted); columns from the target on can't affect it and keep the activation colors. the top 20 nodes of the attributed columns by |attr| (latents, bias and error nodes) get a white border on their cell and a ★ in the side panel. the target survives reruns (edits) while it exists.
- second bar: the generated text in a one-line horizontally scrolling box next to generate.
- hook sets load on the first run (the SAEs are GBs; the graph tab never needs them), on cpu in fp32; activations are moved to cpu for encoding.
- the sae tab's model follows the device: on gpu the loaded bf16 model; on cpu an fp32 copy (+16 GB ram for qwen 4b, made on the first sae request, dropped on a device change), because cpus without native bf16 emulate it (attribution 0.8 s instead of ~1-2 min). the graph tab always uses the bf16 model, so its trace and groups don't change.
- messages: `sae_run {text, template, n_generate, edits: [[set, part, token, latent id, value]]}` → `{tokens: [str], n_prompt, output: str (generated text), next: [[[str, prob] x5] per token], hooks: {set: {display, parts: {part: {reads, ids: [[int]], vals: [[float]], labels: {id: [label, score]}}}}}}` (per token strongest first, zeros dropped, labels only for latents that fired); `sae_attribute {target}` → `{target, logit, parts: {set: {part: {ids: [[int]], attr: [[float]], bias: [float], err: [float]}}}}` (ids per token as in `sae_run`), for the last run's tokens and edits; `sae_search {set, part, query}` → `{results: [[id, label, score, density]]}`; `sae_feature {set, part, unit}` → `{label, score, density, examples: [{pieces: [str], acts: [float]}]}` (a piece = tokens merged to whole characters, act = its max).

# SAEs

`models/qwen3_5_4b/hooks/sae_resid_topk.json`: the TopK SAEs of `synapse/interp/DOC.md` (`evoke/Qwen3_5_4b/interp/sae_resid_topk.py`), one part per trained layer `L3`, `L7`, ..., `L31` (the full-attention layers), each reading and writing `model.language_model.layers.i:out` (resid_post, the training hook). `encode` = `topk_64(relu((x·s - b_dec) W_enc + b_enc))`, so every token has ≤ 64 active latents and `max_features` = 64 shows all of them. labels, density and picks come from `results/Qwen3_5_4b/sae_resid_topk/`.

# tests

- `tests/ws_test.py` — fast suite (xor): boots the server, full ws flow incl. cross-origin rejection, groups, forward, device, marking. `venv/bin/python -m interpviz.tests.ws_test`
- `tests/qwen_smoke.py` — cpu, ~2 min: real weights, trace, forward, group validation. run after changing the qwen wrapper or gen_groups.
