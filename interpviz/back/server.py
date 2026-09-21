# server.py -- FastAPI WebSocket server, handles all frontend messages
# single websocket at /ws, message protocol: {type, _id, ...} -> {type, _id, ...}
# static files served from front/, mounted under the hub at /interpviz

import importlib
import json
import traceback
from pathlib import Path
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from interpviz.back.config import DEVICE
from interpviz.back.inspector import ModelInspector
from interpviz.back.layout import compute_layout, compute_effective_graph
from interpviz.back.routing import compute_routes

# repo convention: everything runs as modules from the repo root
REPO_ROOT = Path.cwd()


def load_meta(folder):
    path = REPO_ROOT / folder / "visualMeta.json"
    assert path.exists(), f"no visualMeta.json in {folder}"
    meta = json.loads(path.read_text())
    # normalize — ensure all optional fields exist with empty defaults
    meta.setdefault("custom_modules", {})
    meta.setdefault("expanded_groups", [])
    meta.setdefault("names", {})
    meta.setdefault("display", {})
    meta.setdefault("tensors", {})
    # selective capture: marked nodes keep values at forward (others are
    # discarded as soon as they run — shapes still shown). "all" = keep everything.
    meta.setdefault("marked", [])
    meta.setdefault("capture_mode", "all")
    return meta


def save_meta(folder, meta):
    (REPO_ROOT / folder / "visualMeta.json").write_text(json.dumps(meta, indent=4) + "\n")

app = FastAPI()

# global state
inspector: ModelInspector | None = None
meta: dict | None = None
meta_folder: str | None = None
# immutable graph from inspector — never mutated
raw_graph: dict | None = None
# current compute device — mutable via the set_device message (cpu <-> cuda)
device = DEVICE

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "long": torch.long,
}


def _validate_group(node_ids, edges, existing_groups, exclude_group=None):
    # checks 4 rules (1-4) for custom group validity, returns list of violation messages
    # existing_groups: {group_name: [fx_id]} — for overlap check
    # exclude_group: name of the group being updated (its old members don't count as overlap)
    group = set(node_ids)
    violations = []

    # rule 1: no overlap — members can't already belong to another group
    overlapping = {} # {fx_id: other_group_name}
    for gname, members in existing_groups.items():
        if gname == exclude_group:
            continue
        for m in members:
            if m in group:
                overlapping[m] = gname
    if overlapping:
        violations.append(
            "overlapping members — " + ", ".join(
                f"{nid} already in group '{gname}'" for nid, gname in sorted(overlapping.items())
            )
        )

    # rule 2: interconnected — all nodes reachable via edges within the group
    neighbors = {nid: set() for nid in group}
    for e in edges:
        if e["from"] in group and e["to"] in group:
            neighbors[e["from"]].add(e["to"])
            neighbors[e["to"]].add(e["from"])
    start = next(iter(group))
    visited = {start}
    queue = [start]
    while queue:
        nid = queue.pop()
        for nb in neighbors[nid]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    unreachable = group - visited
    if unreachable:
        violations.append(f"disconnected — {', '.join(sorted(unreachable))} not reachable from the rest")

    # rule 3: no cycle — no outside node is simultaneously fed by the group AND feeds into it
    feeds_group = set()    # outside nodes with edges INTO the group
    fed_by_group = set()   # outside nodes with edges FROM the group
    for e in edges:
        if e["from"] not in group and e["to"] in group:
            feeds_group.add(e["from"])
        if e["from"] in group and e["to"] not in group:
            fed_by_group.add(e["to"])
    cycle_nodes = feeds_group & fed_by_group
    if cycle_nodes:
        violations.append(f"would create cycle — {', '.join(sorted(cycle_nodes))} are both input and output of the group")

    # rule 4: at least one output — some node must output to outside the group.
    # (multiple outputs allowed: multi-result functions like rope's cos/sin pair
    # are legitimate groups; the collapse machinery dedups parallel edges anyway)
    output_nodes = {e["from"] for e in edges if e["from"] in group and e["to"] not in group}
    if len(output_nodes) == 0:
        violations.append("no output — group must have at least one node that outputs to outside the group")

    return violations


def _build_display():
    # generates displayNodes + displayEdges from raw graph + groups + expanded state
    assert raw_graph and meta is not None
    # {group_name: [node_names]}
    groups = meta["custom_modules"]
    expanded = set(meta["expanded_groups"])

    # eff_nodes: [{"name": str, "op": str}, ...], eff_edges: [{"from": str, "to": str}, ...] -- post group-collapse
    eff_nodes, eff_edges = compute_effective_graph(raw_graph["nodes"], raw_graph["edges"], groups, expanded)
    # {node_name: {"x": float, "y": float, "w": float, "h": float}} -- pixel positions
    positions = compute_layout(eff_nodes, eff_edges)
    # {"from->to": [[x, y], ...]} -- polyline points per edge
    routes = compute_routes(positions, eff_edges)

    # custom display names from meta
    names = meta["names"]

    # build displayNodes list
    display_nodes = []
    for n in eff_nodes:
        fx_name = n["name"]
        display_nodes.append({
            "name": names.get(fx_name, fx_name),
            "id": fx_name,
            "op": n["op"],
            "shape": None,
            "pos": positions[fx_name],
        })

    # build displayEdges list with route points
    display_edges = []
    for e in eff_edges:
        key = f"{e['from']}->{e['to']}"
        display_edges.append({
            "from": e["from"],
            "to": e["to"],
            "points": routes.get(key, []),
        })

    return display_nodes, display_edges


def handle(msg):
    global inspector, meta, meta_folder, raw_graph, device
    t = msg["type"]

    if t == "load_meta":
        inspector = None
        raw_graph = None
        meta_folder = msg["folder"]
        meta = load_meta(meta_folder)
        return {"type": t, "device": device, **meta}

    elif t == "load_model":
        assert meta is not None, "load_meta first"
        m = meta["model"]
        mod = importlib.import_module(m["module_path"])
        cls = getattr(mod, m["class_name"])
        model = cls(**m.get("constructor_args", {})).to(device)
        if m.get("weights_path"):
            state = torch.load(REPO_ROOT / m["weights_path"], map_location=device, weights_only=True)
            model.load_state_dict(state)
        input_dtypes = m.get("input_dtypes", ["float32"])
        example_inputs = tuple(
            torch.tensor(inp, dtype=DTYPE_MAP[dt], device=device)
            for inp, dt in zip(m["example_inputs"], input_dtypes)
        )
        inspector = ModelInspector(model, example_inputs, dynamic_dims=m.get("dynamic_dims"))
        raw_graph = inspector.graph()

        display_nodes, display_edges = _build_display()
        return {
            "type": t,
            "device": device,
            "displayNodes": display_nodes,
            "displayEdges": display_edges,
        }

    elif t == "set_device":
        # move the loaded model + graph module between cpu and gpu.
        # dtype is NOT cast: the exported graph (and saved groups, which
        # reference node names) must stay identical across devices — an
        # fp32 trace elides the dtype-conversion nodes a bf16 trace has.
        target = msg["device"]
        assert target in ("cpu", "cuda"), f"device must be cpu or cuda, got {target!r}"
        if target == "cuda":
            assert torch.cuda.is_available(), "no cuda on this machine"
        if target != device and inspector is not None:
            inspector.model.to(device=target)
            inspector.gm.to(device=target)
            # captured tensors are stale (old device) — force a fresh forward
            inspector.captured_tensors = {}
        device = target
        return {"type": t, "device": device}

    elif t == "forward":
        assert inspector, "no model loaded"
        input_dtypes = msg.get("input_dtypes", ["float32"])
        inputs = tuple(
            torch.tensor(inp, dtype=DTYPE_MAP[dt], device=device)
            for inp, dt in zip(msg["inputs"], input_dtypes)
        )
        # set_device casts the model (bf16 cuda / fp32 cpu) — float inputs must
        # match its dtype or matmuls reject (guards only check shapes)
        model_dtype = next(inspector.model.parameters()).dtype
        inputs = tuple(t.to(model_dtype) if t.is_floating_point() else t for t in inputs)
        # show-marked mode: only marked nodes' values are kept (shapes for all)
        marked = set(meta["marked"]) if meta and meta.get("capture_mode") == "marked" else None
        result = inspector.forward(*inputs, overrides=msg.get("overrides"), marked=marked)
        return {"type": t, **result}

    elif t == "get_tensor":
        assert inspector, "no model loaded"
        assert hasattr(inspector, "captured_tensors"), "run forward first"
        fx_name = msg["name"]
        if meta and meta.get("capture_mode") == "marked" and fx_name not in meta["marked"]:
            raise AssertionError(f"'{fx_name}' is not marked — values are only captured for marked nodes in show-marked mode")
        display = "all"
        k = 5
        features = None
        if meta and "tensors" in meta:
            tensor_config = meta["tensors"].get(fx_name, {})
            display = tensor_config.get("display", "all")
            k = tensor_config.get("k", 5)
            if "features" in tensor_config and meta_folder:
                features_path = REPO_ROOT / meta_folder / tensor_config["features"]
                assert features_path.exists(), f"features file not found: {tensor_config['features']}"
                features = json.loads(features_path.read_text())
        result = inspector.get_tensor(fx_name, display=display, k=k)
        if features:
            result["features"] = features
        return {"type": t, **result}

    elif t == "expand_group":
        assert meta and meta_folder and raw_graph
        group_name = msg["group"]
        if group_name not in meta["expanded_groups"]:
            meta["expanded_groups"].append(group_name)
        save_meta(meta_folder, meta)
        display_nodes, display_edges = _build_display()
        return {"type": t, "displayNodes": display_nodes, "displayEdges": display_edges}

    elif t == "collapse_group":
        assert meta and meta_folder and raw_graph
        group_name = msg["group"]
        meta["expanded_groups"] = [g for g in meta["expanded_groups"] if g != group_name]
        save_meta(meta_folder, meta)
        display_nodes, display_edges = _build_display()
        return {"type": t, "displayNodes": display_nodes, "displayEdges": display_edges}

    elif t == "rename_node":
        assert meta and meta_folder
        node_id = msg["node_id"]
        name = msg["name"].strip()
        if name:
            meta["names"][node_id] = name
        else:
            meta["names"].pop(node_id, None)
        save_meta(meta_folder, meta)
        return {"type": t, "status": "ok"}

    elif t == "create_group":
        assert meta and meta_folder and raw_graph
        name = msg["name"]
        node_ids = msg["nodes"]
        existing = meta["custom_modules"]
        assert name not in existing, f"group '{name}' already exists"
        # group name must not collide with any raw node name
        raw_names = {n["name"] for n in raw_graph["nodes"]}
        assert name not in raw_names, f"group name '{name}' collides with a node name"
        violations = _validate_group(node_ids, raw_graph["edges"], existing)
        assert not violations, "invalid group:\n" + "\n".join([f"{i+1}) {v}" for i, v in enumerate(violations)])
        meta["custom_modules"][name] = node_ids
        display_nodes, display_edges = _build_display()
        save_meta(meta_folder, meta)
        return {"type": t, "displayNodes": display_nodes, "displayEdges": display_edges}

    elif t == "update_group":
        assert meta and meta_folder and raw_graph
        name = msg["group"]
        node_ids = msg["nodes"]
        assert name in meta["custom_modules"], f"group '{name}' not found"
        violations = _validate_group(node_ids, raw_graph["edges"], meta["custom_modules"], exclude_group=name)
        assert not violations, "invalid group:\n" + "\n".join([f"{i+1}) {v}" for i, v in enumerate(violations)])
        meta["custom_modules"][name] = node_ids
        display_nodes, display_edges = _build_display()
        save_meta(meta_folder, meta)
        return {"type": t, "displayNodes": display_nodes, "displayEdges": display_edges}

    elif t == "delete_group":
        assert meta and meta_folder and raw_graph
        name = msg["group"]
        assert name in meta["custom_modules"], f"group '{name}' not found"
        del meta["custom_modules"][name]
        meta["expanded_groups"] = [g for g in meta["expanded_groups"] if g != name]
        save_meta(meta_folder, meta)
        display_nodes, display_edges = _build_display()
        return {"type": t, "displayNodes": display_nodes, "displayEdges": display_edges}

    elif t == "set_marked":
        assert meta and meta_folder
        # mark/unmark a node — or a whole group at once (name resolves to members)
        name = msg["name"]
        marked = set(meta["marked"])
        members = meta["custom_modules"].get(name, [name])
        if msg["marked"]:
            marked.update(members)
        else:
            marked.difference_update(members)
        meta["marked"] = sorted(marked)
        save_meta(meta_folder, meta)
        return {"type": t, "marked": meta["marked"]}

    elif t == "set_capture_mode":
        assert meta and meta_folder
        assert msg["mode"] in ("all", "marked"), "mode must be 'all' or 'marked'"
        meta["capture_mode"] = msg["mode"]
        save_meta(meta_folder, meta)
        return {"type": t, "capture_mode": meta["capture_mode"]}

    elif t == "update_display":
        assert meta and meta_folder
        meta["display"][msg["key"]] = msg["value"]
        save_meta(meta_folder, meta)
        return {"type": t, "status": "ok"}

    else:
        return {"type": "error", "message": f"unknown type: {t}"}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    # cybersec: block cross-site WebSocket hijacking — browsers send Origin, and if it doesn't match our Host another site is driving the victim's session. closing before accept makes the handshake fail with 403
    origin = websocket.headers.get("origin")
    if origin is not None and urlparse(origin).netloc != websocket.headers.get("host"):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    while True:
        msg = await websocket.receive_json()
        _id = msg.get("_id")
        try:
            response = handle(msg)
        except Exception as e:
            traceback.print_exc()
            response = {"type": "error", "message": str(e)}
        if _id is not None:
            response["_id"] = _id
        await websocket.send_json(response)


# frontend statics — the hub mounts this whole app at /interpviz, so this serves /interpviz/
app.mount("/", StaticFiles(directory=str(REPO_ROOT / "interpviz" / "front"), html=True), name="front")
