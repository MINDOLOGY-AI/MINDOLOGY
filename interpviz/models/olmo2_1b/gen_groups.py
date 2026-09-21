'''
Generate visualMeta.json for the olmo2_1b interpviz model.

Goes through the exact server path: instantiate the real Olmo1B (GPU, weights)
and trace it with back.inspector.ModelInspector — the same code the server
runs — so grouped node names match the served graph by construction.

Layer attribution (layer_0..layer_15):
- call nodes via nn_module_stack -> model.layers.N
- get_attr weight nodes via "layers_N" in their mangled FQN name
  (ep.module() turns lifted params into get_attr without nn_module_stack)

Validates the 4 group rules on the server's raw edge set, ejecting strays
(the unused rope_inv_freq buffers are disconnected dead nodes), then writes
visualMeta.json next to this file. Embedding / lm_head / rope / mask one-shot
ops stay ungrouped on purpose.
'''

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# make the repo root importable when run as a script from the project root
sys.path.insert(0, str(Path.cwd()))

import torch

from evoke.OlMo2_1b.run.config import OLMO2_1B_INSTRUCT_CONFIG
from interpviz.back.inspector import ModelInspector
from interpviz.models.olmo2_1b.model import Olmo1B

OUT_PATH = Path(__file__).parent / "visualMeta.json"
NUM_LAYERS = OLMO2_1B_INSTRUCT_CONFIG.num_hidden_layers
EXAMPLE_INPUT = [[1, 2, 3, 4, 5, 6, 7, 8]]
# seq dim stays dynamic so forward accepts any length (matches the server, which reads this from the meta)
DYNAMIC_DIMS = [[1]]


def trace_server_graph():
    # exactly what the server's load_model handler does
    model = Olmo1B()
    device = next(model.parameters()).device
    example = (torch.tensor(EXAMPLE_INPUT, dtype=torch.long, device=device),)
    inspector = ModelInspector(model, example, dynamic_dims=DYNAMIC_DIMS)
    return inspector, inspector.graph()


def layer_of(node):
    stack = node.meta.get("nn_module_stack")
    if stack:
        m = re.search(r"layers\.(\d+)", list(stack.values())[-1][0])
        if m:
            return int(m.group(1))
    if node.op == "get_attr":
        m = re.search(r"layers_(\d+)", node.name)
        if m:
            return int(m.group(1))
    return None


def build_adjacency(edges):
    # name-based adjacency from the raw graph's edge dicts
    users = defaultdict(set)    # name -> names it feeds
    inputs = defaultdict(set)   # name -> names feeding it
    for e in edges:
        users[e["from"]].add(e["to"])
        inputs[e["to"]].add(e["from"])
    return users, inputs


def connected_components(members, users, inputs):
    seen, comps = set(), []
    for start in members:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(y for y in (users[x] | inputs[x]) & members if y not in seen)
        comps.append(comp)
    return comps


def group_violations(members, users, inputs):
    # the server's 4 group rules on raw-graph edges; members: set of names
    v = []
    comps = connected_components(members, users, inputs)
    if len(comps) > 1:
        v.append(f"rule2: {len(comps)} disconnected components")
    outside_in = set()    # outside nodes feeding into the group
    outside_out = set()   # outside nodes fed by the group
    for m in members:
        outside_in |= inputs[m] - members
        outside_out |= users[m] - members
    both = outside_in & outside_out
    if both:
        v.append(f"rule3: {sorted(both)} both input and output of the group")
    out_nodes = {m for m in members if users[m] - members}
    if len(out_nodes) != 1:
        v.append(f"rule4: {len(out_nodes)} output nodes {sorted(out_nodes)}")
    return v


def fix_group(members, users, inputs, order):
    # adjust membership until all rules pass; returns (members, ejected_names)
    ejected = []
    for _ in range(16):
        comps = connected_components(members, users, inputs)
        if len(comps) > 1:
            # keep the largest component, eject the strays (e.g. unused inv_freq buffers)
            comps.sort(key=len, reverse=True)
            for stray in comps[1:]:
                for name in stray:
                    members.discard(name)
                    ejected.append(name)
            continue
        out_nodes = sorted((m for m in members if users[m] - members), key=order.get)
        if len(out_nodes) > 1:
            # keep the topologically-last output (the layer's residual add),
            # eject earlier boundary nodes to loose
            for name in out_nodes[:-1]:
                members.discard(name)
                ejected.append(name)
            continue
        outside_in = set()
        outside_out = set()
        for m in members:
            outside_in |= inputs[m] - members
            outside_out |= users[m] - members
        both = outside_in & outside_out
        if both:
            # eject members consuming an outside node that also consumes the group
            for m in list(members):
                if inputs[m] & both:
                    members.discard(m)
                    ejected.append(m)
            continue
        break
    return members, ejected


def main():
    inspector, raw = trace_server_graph()
    names = [n["name"] for n in raw["nodes"]]
    order = {name: i for i, name in enumerate(names)}
    users, inputs = build_adjacency(raw["edges"])

    # attribute gm nodes to layers (nn_module_stack for ops, name for get_attr)
    raw_set = set(names)
    groups = {}
    for node in inspector.gm.graph.nodes:
        if node.name not in raw_set:
            continue  # underscore _assert guards, already filtered by the server
        l = layer_of(node)
        if l is not None:
            groups.setdefault(l, set()).add(node.name)

    # rule 1 (no overlap) holds by construction: layer_of returns one layer per node
    custom_modules = {}
    total_ejected = 0
    for l in sorted(groups):
        members, ejected = fix_group(set(groups[l]), users, inputs, order)
        total_ejected += len(ejected)
        v = group_violations(members, users, inputs)
        assert not v, f"layer_{l} still violates: {v}"
        out_node = next(m for m in members if users[m] - members)
        print(f"layer_{l}: {len(members)} nodes, output={out_node}, "
              f"ejected={len(ejected)} ({', '.join(ejected)})")
        custom_modules[f"layer_{l}"] = [name for name in names if name in members]

    grouped_names = {name for names_ in custom_modules.values() for name in names_}
    loose = [name for name in names if name not in grouped_names]
    print(f"\nvisible fx nodes: {len(names)}")
    print(f"grouped: {len(grouped_names)} in {len(groups)} groups, ejected to loose: {total_ejected}")
    print(f"loose nodes: {len(loose)} -> default view shows {len(loose) + len(groups)} boxes")

    meta = {
        "model": {
            "module_path": "mind.interpviz.models.olmo2_1b.model",
            "class_name": "Olmo1B",
            "weights_path": None,
            "constructor_args": {},
            # one entry per forward arg; the server does torch.tensor on each
            "example_inputs": [EXAMPLE_INPUT],
            "input_dtypes": ["long"],
            "dynamic_dims": DYNAMIC_DIMS,
        },
        "expanded_groups": [],
        "custom_modules": custom_modules,
        "names": {},
        "tensors": {},
    }
    OUT_PATH.write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
