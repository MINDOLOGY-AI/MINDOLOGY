# layout.py -- group collapse + integer grid layout
# used by server.py: compute_effective_graph() collapses groups, compute_layout() assigns positions
# flow direction: inputs at bottom (high Y), outputs at top (low Y)
# algorithm: topological sort → row assignment → cluster-by-downstream → column assignment

from collections import defaultdict, deque

# all nodes are the same size
NODE_W = 140
NODE_H = 40
# pixel spacing between grid cells
CELL_X = 200
CELL_Y = 100

# nodes: [{"name": str, "op": str}, ...]
# edges: [{"from": str, "to": str}, ...]
# groups: {"group_name": ["node_name", ...]}
# expanded_groups: ["group_name", ...] 
def compute_effective_graph(nodes, edges, groups, expanded_groups):
    # collapses non-expanded groups into single synthetic nodes, remaps edges

    expanded = set(expanded_groups)
    # node names hidden inside collapsed groups
    # converted to set for O(1) lookup
    hidden = set() 
    # node_name -> group_node_name for edge remapping
    node_to_group = {}
    effective_nodes = []

    # fills in collapsed groups as effective nodes
    # node to group mapping for collapsed groups
    for gname, members in groups.items():
        if gname in expanded:
            continue
        # collapsed group — hide members, create synthetic node
        member_set = set(members)
        hidden.update(member_set)
        effective_nodes.append({"name": gname, "op": "group"})
        for m in member_set:
            node_to_group[m] = gname

    # visible raw nodes
    effective_nodes.extend(n for n in nodes if n["name"] not in hidden)

    # rewire edges
    effective_names = {n["name"] for n in effective_nodes}
    effective_edges = []
    seen = set()
    for e in edges:
        # if it's in a collapsed group, get group name. otherwise, get it's own name
        src = node_to_group.get(e["from"], e["from"])
        dst = node_to_group.get(e["to"], e["to"])
        assert src in effective_names and dst in effective_names, f"edge {src}->{dst} has unknown node"
        #  if within same collapsed group
        if src == dst:
            continue
        key = (src, dst)
        # if multiple nodes in group point to the same node outside we just show one group edge
        if key not in seen:
            seen.add(key)
            effective_edges.append({"from": src, "to": dst})

    return effective_nodes, effective_edges


def compute_layout(nodes, edges):
    # integer grid layout: topological rows, clustered columns
    # returns {node_name: {x, y, w, h}} with inputs at bottom, outputs at top
    node_map = {n["name"]: n for n in nodes}
    all_names = set(node_map.keys())

    # build adjacency
    outputs_of = defaultdict(list) #what a node's output is
    inputs_of = defaultdict(list) #what a node's input is
    for e in edges:
        assert e["from"] in all_names and e["to"] in all_names, f"edge {e['from']}->{e['to']} has unknown node"
        outputs_of[e["from"]].append(e["to"])
        inputs_of[e["to"]].append(e["from"])

    # --- row assignment: topological BFS, each node at max(input rows) + 1 ---
    # cycles can occur when a collapsed group has gap nodes (e.g. group_x → relu → group_x)
    # so after BFS we handle any unvisited nodes
    in_degree = {name: len(inputs_of[name]) for name in all_names}
    queue = deque(name for name in all_names if in_degree[name] == 0)
    # {"node name" : row number}
    row = {}
    for name in queue:
        row[name] = 0
    while queue:
        name = queue.popleft()
        for output in outputs_of[name]:
            row[output] = max(row.get(output, 0), row[name] + 1)
            in_degree[output] -= 1
            if in_degree[output] == 0:
                queue.append(output)

    assert set(row.keys()) == all_names, f"cycle detected: {all_names - set(row.keys())} never reached"

    # relocate parentless nodes (get_attr, collapsed groups): place 1 row below lowest consumer
    for name in all_names:
        if not inputs_of[name]:
            output_rows = [row[c] for c in outputs_of[name]]
            # dead nodes (no consumers at all, e.g. unused buffers) stay at row 0
            if output_rows:
                row[name] = min(output_rows) - 1

    max_row = max(row.values()) if row else 0
    # shift all rows so the minimum is 0 (get_attr relocation can create negative rows)
    min_row = min(row.values()) if row else 0
    if min_row < 0:
        for name in row:
            row[name] -= min_row
        max_row -= min_row

    # group nodes by row
    rows = defaultdict(list)
    for name in all_names:
        rows[row[name]].append(name)

    # --- column assignment: top-down, cluster by shared consumers, align under consumers ---

    col = {} # {node_name : int}

    for r in sorted(rows.keys(), reverse=True):
        nodes_in_row = rows[r]

        # collect all consumers of this row's nodes, sorted by (row , column, lexicographic)
        consumers = set()
        for name in nodes_in_row:
            consumers.update(outputs_of[name])
        sorted_consumers = sorted(consumers, key=lambda o: (row[o], col[o], o))

        # cluster by consumer in order — each consumer claims its unclaimed inputs from this row
        # get_attr nodes go after operations within each cluster
        ordered = []
        assigned = set()
        for consumer in sorted_consumers:
            cluster = [n for n in nodes_in_row if n not in assigned and consumer in outputs_of[n]]
            if not cluster:
                continue
            cluster.sort(key=lambda n: (1 if node_map[n]["op"] == "get_attr" else 0, n))
            ordered.extend(cluster)
            assigned.update(cluster)
        # remaining nodes with no consumers (e.g. output node), lexicographic
        for name in sorted(nodes_in_row):
            if name not in assigned:
                ordered.append(name)

        # assign columns sequentially — ordering already handles alignment
        for i, name in enumerate(ordered):
            col[name] = i

    # --- convert grid (col, row) to pixel coordinates of top left corner  ---
    positions = {}
    for name in all_names:
        x = col[name] * CELL_X + (CELL_X - NODE_W) / 2
        # flips earlier nodes like input to high css so it's below
        y = (max_row - row[name]) * CELL_Y + (CELL_Y - NODE_H) / 2
        positions[name] = {"x": x, "y": y, "w": NODE_W, "h": NODE_H}

    return positions
