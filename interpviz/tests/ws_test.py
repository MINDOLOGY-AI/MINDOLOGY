# headless smoke test for the interpviz websocket backend.
# boots the standalone server on a test port, connects to /ws and exercises the fx-graph visualizer protocol against the
# bundled xor demo: load_meta / load_model / forward / get_tensor / create_group / set_device / marking.
# exits non-zero on failure.
# run from repo root: venv/bin/python -m interpviz.tests.ws_test

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import torch
import websockets

PROJECT_ROOT = Path.cwd()
TEST_PORT = 2998
BASE = f"http://127.0.0.1:{TEST_PORT}"
XOR_FOLDER = "interpviz/models/xor"

FAILURES = []  # [str] labels of failed checks


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f" FAIL {label}")
        FAILURES.append(label)


def wait_for_server(proc, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        assert proc.poll() is None, "server exited early"
        try:
            if httpx.get(f"{BASE}/", timeout=1).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    raise RuntimeError("server did not come up")


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def flat_len(values):
    # total scalar count of an arbitrarily nested list
    if isinstance(values, list):
        return sum(flat_len(v) for v in values)
    return 1


async def run_ws_tests():
    # cross-origin -> handshake rejected before accept
    try:
        async with websockets.connect(f"ws://127.0.0.1:{TEST_PORT}/ws",
                                      additional_headers=[("Origin", "http://evil.example")]):
            check("cross-origin ws rejected", False)
    except websockets.exceptions.InvalidStatus:
        check("cross-origin ws rejected", True)

    async with websockets.connect(f"ws://127.0.0.1:{TEST_PORT}/ws", additional_headers=[("Origin", BASE)]) as ws:
        req_id = 0

        async def rpc(msg):
            # protocol: send {type, _id, ...}, wait for the response with the same _id
            nonlocal req_id
            req_id += 1
            msg["_id"] = req_id
            await ws.send(json.dumps(msg))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 30))
                if m.get("_id") == req_id:
                    return m

        # load_meta returns the xor model config
        meta = await rpc({"type": "load_meta", "folder": XOR_FOLDER})
        model_cfg = meta.get("model", {})
        check("load_meta xor returns model config",
              meta.get("type") == "load_meta"
              and model_cfg.get("class_name") == "XORNet"
              and model_cfg.get("module_path") == "interpviz.models.xor.model")

        # load_model returns displayNodes/displayEdges with sane positions
        loaded = await rpc({"type": "load_model"})
        nodes = loaded.get("displayNodes", [])
        edges = loaded.get("displayEdges", [])
        check("load_model returns displayNodes", loaded.get("type") == "load_model" and len(nodes) > 0)
        check("load_model returns displayEdges", len(edges) > 0)
        check("node positions sane",
              all(n["pos"]["x"] >= 0 and n["pos"]["y"] >= 0 and n["pos"]["w"] > 0 and n["pos"]["h"] > 0
                  for n in nodes))
        check("edge routes are point lists",
              all(isinstance(e["points"], list) and len(e["points"]) >= 2 for e in edges))

        # forward returns output + shapes for every node
        fwd = await rpc({"type": "forward", "inputs": [[[0.0, 1.0]]]})
        output = fwd.get("output")
        shapes = fwd.get("shapes", {})
        check("forward returns output", fwd.get("type") == "forward" and isinstance(output, list))
        # xor(0, 1) == 1, the bundled weights are trained
        check("forward output is xor(0,1) ~ 1", isinstance(output[0][0], float) and output[0][0] > 0.9)
        check("forward shapes cover every node", all(n["id"] in shapes for n in nodes))

        # get_tensor for a middle node returns values matching shape
        middle = next(n["id"] for n in nodes if n["op"] == "call_function")
        tensor = await rpc({"type": "get_tensor", "name": middle})
        check("get_tensor middle node", tensor.get("type") == "get_tensor" and tensor.get("name") == middle)
        check("get_tensor values match shape",
              numel(tensor.get("shape", [])) == flat_len(tensor.get("values")))

        # create_group with an invalid group (disconnected nodes) returns error
        bad = await rpc({"type": "create_group", "name": "g_bad", "nodes": ["x", "output"]})
        check("create_group invalid returns error", bad.get("type") == "error")

        # device toggle: cpu roundtrip, forward keeps working after each move
        if torch.cuda.is_available():
            to_cpu = await rpc({"type": "set_device", "device": "cpu"})
            check("set_device cpu", to_cpu.get("device") == "cpu")
            fwd_cpu = await rpc({"type": "forward", "inputs": [[[0.0, 1.0]]]})
            check("forward works on cpu", fwd_cpu.get("output", [[0]])[0][0] > 0.9)
            back = await rpc({"type": "set_device", "device": "cuda"})
            check("set_device back to cuda", back.get("device") == "cuda")
            fwd_gpu = await rpc({"type": "forward", "inputs": [[[0.0, 1.0]]]})
            check("forward works back on cuda", fwd_gpu.get("output", [[0]])[0][0] > 0.9)
        bogus = await rpc({"type": "set_device", "device": "tpu"})
        check("set_device bogus rejected", bogus.get("type") == "error")

        # selective capture: mark one node, switch to show-marked, values only for it
        mark = await rpc({"type": "set_marked", "name": middle, "marked": True})
        check("set_marked stores node", middle in (mark.get("marked") or []))
        mode = await rpc({"type": "set_capture_mode", "mode": "marked"})
        check("set_capture_mode marked", mode.get("capture_mode") == "marked")
        fwd_sel = await rpc({"type": "forward", "inputs": [[[0.0, 1.0]]]})
        check("selective forward shapes cover every node", all(n["id"] in fwd_sel.get("shapes", {}) for n in nodes))
        t_marked = await rpc({"type": "get_tensor", "name": middle})
        check("marked node has values", t_marked.get("type") == "get_tensor" and "values" in t_marked)
        other = next(n["id"] for n in nodes if n["op"] == "call_function" and n["id"] != middle)
        t_other = await rpc({"type": "get_tensor", "name": other})
        check("unmarked node rejected in marked mode", t_other.get("type") == "error" and "not marked" in t_other.get("message", ""))
        # re-mark + fresh forward → values appear (capture happens at forward time)
        await rpc({"type": "set_marked", "name": other, "marked": True})
        await rpc({"type": "forward", "inputs": [[[0.0, 1.0]]]})
        t_other2 = await rpc({"type": "get_tensor", "name": other})
        check("re-marked node readable after fresh forward", t_other2.get("type") == "get_tensor")
        # restore meta to its original state (the file must end byte-identical)
        await rpc({"type": "set_capture_mode", "mode": "all"})
        await rpc({"type": "set_marked", "name": middle, "marked": False})
        unmark2 = await rpc({"type": "set_marked", "name": other, "marked": False})
        check("unmark works", not unmark2.get("marked"))


def main():
    meta_path = PROJECT_ROOT / XOR_FOLDER / "interpvizMeta.json"
    meta_before = meta_path.read_text()

    server = subprocess.Popen(
        [str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "uvicorn",
         "interpviz.back.server:app", "--host", "127.0.0.1", "--port", str(TEST_PORT)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(server)
        asyncio.run(run_ws_tests())
        # the test only sends read-only messages + one rejected create_group, and undoes its marks
        check("interpvizMeta.json untouched", meta_path.read_text() == meta_before)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
