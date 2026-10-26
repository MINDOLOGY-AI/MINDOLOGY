# server.py -- FastAPI WebSocket server, handles all frontend messages
# single websocket at /ws, message protocol: {type, _id, ...} -> {type, _id, ...}
# every open window is a frontend of the same state (model, meta, session): a run / preset change is pushed to the other
# windows (no _id), a (re)load or device change tells them to fetch the state again (interpviz/DOC.md, session).
# requests run one at a time in a worker thread (the event loop stays free to stream a chat reply to every window);
# every outgoing message goes through one queue, so each window gets them in order
# static files served from front/
# launched by interpviz/app.py

import asyncio
import copy
import importlib
import json
import threading
import traceback
from pathlib import Path
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from interpviz.back.config import DEVICE
from interpviz.back.inspector import ModelInspector
from interpviz.back.probes import Probes
from interpviz.back.layout import compute_layout, compute_effective_graph
from interpviz.back.routing import compute_routes

# repo convention: everything runs as modules from the repo root
REPO_ROOT = Path.cwd()


def load_meta(folder):
    path = REPO_ROOT / folder / "interpvizMeta.json"
    assert path.exists(), f"no interpvizMeta.json in {folder}"
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
    meta.setdefault("capture_mode", "marked")
    # {name: [edit]} saved edit lists (wire format, interpviz/DOC.md, session)
    meta.setdefault("presets", {})
    return meta


def save_meta(folder, meta):
    (REPO_ROOT / folder / "interpvizMeta.json").write_text(json.dumps(meta, indent=4) + "\n")

app = FastAPI()

# global state
inspector: ModelInspector | None = None
meta: dict | None = None
meta_folder: str | None = None
# immutable graph from inspector — never mutated
raw_graph: dict | None = None
# current compute device — mutable via the set_device message (cpu <-> cuda)
device = DEVICE
# the probes tab (its tokenizer, loaded hook set, last run and edits), made by load_model when the meta names a tokenizer
probes: Probes | None = None
# the sae tab's model on cpu: an fp32 copy of the loaded bf16 model. cpus without native bf16 (avx2) emulate it,
# ~300x slower for attribution's backward. made on the first sae request on cpu, dropped on a device change
cpu_fp32_model = None
# every open websocket (one per browser window)
clients = set()  # {WebSocket}
# replies pushed to the other windows too: they change what every frontend shows
SHARED = ("sae_run", "save_preset", "delete_preset")
# replies after which the other windows fetch the whole state again (get_state)
RELOAD = ("load_meta", "load_model", "set_device")
# one request at a time: handle() owns the globals and the model
busy = asyncio.Lock()
# [(targets: {WebSocket} | None = every window, message)] everything sent, in order, by the send_loop task
outbox = asyncio.Queue()
send_task = None
# set by chat_stop (any window; handled at once, not queued behind the running request): the reply being generated
# ends after its current token
stop_reply = threading.Event()
# the running request tasks (kept referenced until done)
tasks = set()  # {asyncio.Task}


def sae_model():
    # the eager model the sae tab runs: on gpu the loaded (bf16) model, on cpu its fp32 copy
    global cpu_fp32_model
    if device == "cuda":
        return inspector.model
    if cpu_fp32_model is None:
        cpu_fp32_model = copy.deepcopy(inspector.model).float()
    return cpu_fp32_model

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


def handle(msg, emit):
    # emit(message): send to every window now (from the worker thread), e.g. a chat reply as it is generated
    global inspector, meta, meta_folder, raw_graph, device, probes, cpu_fp32_model
    t = msg["type"]

    if t == "get_state":
        # everything a newly opened window needs to show the loaded model without loading it again
        display_nodes, display_edges = _build_display() if raw_graph else (None, None)
        return {"type": t, "device": device, "folder": meta_folder, "meta": meta, "displayNodes": display_nodes,
                "displayEdges": display_edges, "tokenizer": probes is not None,
                "session": probes.session() if probes else None}

    elif t == "load_meta":
        inspector = None
        raw_graph = None
        probes = None
        cpu_fp32_model = None
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
        probes = Probes(meta, REPO_ROOT / meta_folder) if m.get("tokenizer") else None

        display_nodes, display_edges = _build_display()
        return {
            "type": t,
            "device": device,
            "displayNodes": display_nodes,
            "displayEdges": display_edges,
            "tokenizer": probes is not None,
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
            cpu_fp32_model = None
            if probes is not None:
                probes.ids = None
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

    elif t == "sae_run":
        assert inspector and probes, "load a model with a tokenizer first"
        # every window shows the reply growing: {"type": "chat_stream", "messages": the conversation so far, "raw", "think", "reply": str}
        stop_reply.clear()
        on_reply = lambda r: emit({"type": "chat_stream", "messages": msg["messages"], "raw": msg["raw"], "think": msg["think"], "reply": r})
        return {"type": t, **probes.run(sae_model(), msg["set"], msg["messages"], msg["raw"], msg["think"], msg["max_new"],
                                        msg["edits"], on_reply, stop_reply)}

    elif t == "sae_neuron":
        assert probes, "load a model with a tokenizer first"
        return {"type": t, **probes.neuron(sae_model(), msg["layer"], msg["unit"])}

    elif t == "save_preset":
        assert meta and meta_folder
        name = msg["name"].strip()
        assert name, "preset needs a name"
        meta["presets"][name] = msg["edits"]
        save_meta(meta_folder, meta)
        return {"type": t, "presets": meta["presets"]}

    elif t == "delete_preset":
        assert meta and meta_folder
        assert msg["name"] in meta["presets"], f"no preset {msg['name']!r}"
        del meta["presets"][msg["name"]]
        save_meta(meta_folder, meta)
        return {"type": t, "presets": meta["presets"]}

    elif t == "sae_attribute":
        assert probes, "load a model with a tokenizer first"
        return {"type": t, **probes.attribute(sae_model(), msg["target"])}

    elif t == "sae_search":
        assert probes, "load a model with a tokenizer first"
        return {"type": t, **probes.search(msg["set"], msg["part"], msg["query"])}

    elif t == "sae_feature":
        assert probes, "load a model with a tokenizer first"
        return {"type": t, **probes.feature(sae_model(), msg["set"], msg["part"], msg["unit"])}

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

    global send_task
    await websocket.accept()
    clients.add(websocket)
    if send_task is None:
        send_task = asyncio.create_task(send_loop())
    loop = asyncio.get_running_loop()
    emit = lambda m: loop.call_soon_threadsafe(outbox.put_nowait, (None, m))
    async def process(msg):
        _id = msg.get("_id")
        # the lock queues requests in arrival order
        async with busy:
            try:
                response = await asyncio.to_thread(handle, msg, emit)
            except Exception as e:
                traceback.print_exc()
                response = {"type": "error", "message": str(e)}
        # the other windows: the same reply without _id, or a nudge to fetch the state again
        push = response if response["type"] in SHARED else {"type": "reload"} if response["type"] in RELOAD else None
        outbox.put_nowait(({websocket}, response if _id is None else {**response, "_id": _id}))
        if push is not None:
            outbox.put_nowait((clients - {websocket}, push))

    try:
        while True:
            msg = await websocket.receive_json()
            if msg["type"] == "chat_stop":
                # at once, even from the window whose request is running: its sae_run reply follows with the reply so far
                stop_reply.set()
                outbox.put_nowait(({websocket}, {"type": "chat_stop", "_id": msg.get("_id")}))
                continue
            task = asyncio.create_task(process(msg))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


async def send_loop():
    # sends the outbox in order; a window that closed meanwhile is skipped
    while True:
        targets, m = await outbox.get()
        for c in list(clients if targets is None else targets & clients):
            try:
                await c.send_json(m)
            except (WebSocketDisconnect, RuntimeError):
                clients.discard(c)


# frontend statics at /
app.mount("/", StaticFiles(directory=str(REPO_ROOT / "interpviz" / "front"), html=True), name="front")
