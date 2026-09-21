'''
Standalone smoke test for the olmo2_1b interpviz model (not part of any suite).

Goes through the exact server path (Olmo1B + ModelInspector, same trace_server_graph
call gen_groups.py uses), so node names match the baked-in groups by construction.
Verifies the 4 group rules, runs one forward through the server's
CapturingInterpreter keeping all intermediates, and prints params / node counts /
key tensor shapes / default-view box count.
'''

import json
import sys
from pathlib import Path

# make the repo root importable when run as a script from the project root
sys.path.insert(0, str(Path.cwd()))

import torch

from interpviz.back.inspector import CapturingInterpreter
from interpviz.back.layout import compute_effective_graph
from interpviz.models.olmo2_1b.gen_groups import (
    EXAMPLE_INPUT,
    OUT_PATH,
    build_adjacency,
    group_violations,
    trace_server_graph,
)


def main():
    inspector, raw = trace_server_graph()
    model = inspector.model
    params = sum(p.numel() for p in model.parameters())
    print(f"total params: {params:,} ({params / 1e9:.3f}B)")
    print(f"visible fx nodes: {len(raw['nodes'])}")

    # baked-in groups: names exist in the served graph and pass the 4 rules
    meta = json.loads(OUT_PATH.read_text())
    groups = meta["custom_modules"]
    raw_names = {n["name"] for n in raw["nodes"]}
    users, inputs = build_adjacency(raw["edges"])
    for gname, names in groups.items():
        assert set(names) <= raw_names, f"{gname}: names missing from server graph"
        v = group_violations(set(names), users, inputs)
        assert not v, f"{gname} violates: {v}"
    print(f"all {len(groups)} baked-in groups valid on the served graph: OK")

    # one forward keeping every intermediate value (server's interpreter)
    device = next(model.parameters()).device
    ids = torch.tensor(EXAMPLE_INPUT, dtype=torch.long, device=device)
    interp = CapturingInterpreter(inspector.gm)
    with torch.no_grad():
        interp.run(ids)
    vals = interp.tensor_values
    logits = vals["output"]
    print(f"logits: shape={tuple(logits.shape)} dtype={logits.dtype}")

    # sample intermediates by name substring
    for sub in ("softmax", "silu", "add_8"):
        name = next(k for k in vals if sub in k)
        t = vals[name]
        print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype}")

    # default-view box count: collapsed groups + loose nodes (server's collapse logic)
    eff_nodes, _ = compute_effective_graph(raw["nodes"], raw["edges"], groups, [])
    print(f"default view: {len(eff_nodes) - len(groups)} loose nodes + {len(groups)} groups "
          f"= {len(eff_nodes)} boxes")


if __name__ == "__main__":
    main()
