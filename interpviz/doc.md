# interpviz

visualize (and perturb) every tensor inside a pytorch model, as a live FX-graph board.
mounted under the hub at `/interpviz/` — start the hub (`fab.hub.back.app:app`), log in, click intvrpviz.

# how it works

## backend (`back/`)

- `inspector.py` — the core. `ModelInspector(model, example_inputs)` traces the model with `torch.export.export(..., strict=False)` into an FX graph: every node is one instruction (`placeholder` = input, `get_attr` = weight/bias, `call_function` = math op, `output`). `graph()` turns that into plain `{nodes, edges}` dicts. `forward()` re-executes the graph node-by-node with a `CapturingInterpreter` that stores EVERY intermediate tensor (and supports overriding any node's value mid-run — that's the perturbation feature). `get_tensor()` returns one captured tensor (all values, or top-k).
- `layout.py` — group collapse + grid placement. `compute_effective_graph()` hides collapsed groups into single boxes and rewires boundary edges. `compute_layout()` assigns each node an integer (row, col): rows by topological depth (inputs bottom, outputs top), columns by clustering nodes that share consumers; parentless weights sit one row above their lowest consumer; dead nodes (no consumers) sit at row 0.
- `routing.py` — 90° edge routes: straight if same column, else a 6-point path through the margin lane next to the target. ports are spread along node borders so parallel edges don't overlap.
- `server.py` — single websocket `/interpviz/ws`, request/response by `_id`. holds global state: loaded meta, inspector, raw graph. every mutation (group/expand/rename) re-runs layout+routing and persists `visualMeta.json`. auth: hub login cookie required, cross-origin rejected.
- `config.py` — `DEVICE` (cuda if available).

## frontend (`front/`)

- renders on `<board-canvas>` (`fab/hub/components/BoardCanvas/`) — the shared infinite-board component (pan/zoom/grid/SVG edge layer). backend sends fully-computed positions + edge points; frontend just draws.
- node colors: get_attr → sky, output → lavender, group → mint, everything else → white.
- click a node → inputs highlight pink, outputs highlight purple. double-click a tensor node → popup with shape + values, override & re-run. double-click a group → expand/collapse. double-click rename.
- groups panel: selection mode (click nodes), create/update/delete groups. rules: interconnected, no cycle with the outside, exactly one output node, no overlap with other groups.

## per-model folder (`models/<name>/`)

```
models/foo/
    __init__.py
    model.py          # nn.Module — constructor may load weights itself
    visualMeta.json   # REQUIRED — the whole config + saved view state
```

`visualMeta.json`:

```json
{
  "model": {
    "module_path": "mind.interpviz.models.foo.model",
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
- `marked` = fx node names whose VALUES are captured at forward; `capture_mode` = `"all"` (default, old behavior) or `"marked"`. in marked mode every node still runs and shows its shape — unmarked tensors are discarded as they execute, so huge models don't OOM. toggle in the ui: star on any node/group, show all / show marked buttons.
- `weights_path`: if set, server does `torch.load` + `load_state_dict`. if your constructor loads its own weights (like olmo), leave null.

# how to use it on a NEW model

1. **write the wrapper** `models/foo/model.py`: a plain `nn.Module` whose `forward` returns a SINGLE tensor (wrap multi-output models — see `models/olmo2_1b/model.py`, which subclasses and returns only logits). data-dependent control flow will break `torch.export`; config-flag branches are fine.
2. **write visualMeta.json** (see above). test the trace first:
   `venv/bin/python -c "from interpviz.back.inspector import ModelInspector; from interpviz.models.foo.model import Foo; import torch; ModelInspector(Foo(), (torch.zeros(1, 4),))"`.
3. **open the ui**: type `interpviz/models/foo` in the meta folder field → load meta → load model → forward. click nodes to see values.
4. **group early**. any real model is 1k+ nodes — unusable raw. if your model has a repeated block (transformer layers), generate groups programmatically instead of clicking: copy `models/olmo2_1b/gen_groups.py` — it builds the graph via the server's own path (so names match BY CONSTRUCTION — do not trace a differently-nested module or a different export call), assigns nodes to groups by `nn_module_stack` / name, validates the 4 group rules, and writes `custom_modules` into visualMeta.json.
5. names must match between your generator and the server: the server traces `ep.module()` (unflattened — weights are `get_attr` nodes named after their module path, e.g. `model_layers_0_mlp_up_proj_weight`). if you subclass instead of wrapping, paths stay short (`model.layers.N`, not `model.model.layers.N`).

# the two bundled models

- `models/xor` — 2-8-1 MLP, trained weights included. the smoke-test model; good for learning the ui.
- `models/olmo2_1b` — OlMo2 1B Instruct (evoke's synapse reimplementation, local safetensors, bf16). 1505 visible ops; ships with 16 pre-generated layer groups (default view: 65 boxes). expand any layer to see its 91 ops; click a softmax to see attention weights per head. `gen_groups.py` regenerates the groups; `tests/olmo_smoke.py` verifies the whole path on GPU.

# tests

- `tests/ws_test.py` — fast CPU suite (xor): boots the hub, full ws flow incl. auth, groups, forward. `venv/bin/python interpviz/tests/ws_test.py`
- `tests/olmo_smoke.py` — GPU, ~1 min: real weights, trace, forward, group validation. run after changing the olmo wrapper or gen_groups.
