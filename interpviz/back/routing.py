# routing.py -- 90-degree edge routing
# same column: straight vertical
# different column: 6-point path via vertical lane next to target

from collections import defaultdict

from interpviz.back.layout import CELL_X, CELL_Y, NODE_W, NODE_H

MARGIN_X = (CELL_X - NODE_W) / 2
MARGIN_Y = (CELL_Y - NODE_H) / 2


# positions: {str: {"x": float, "y": float, "w": float, "h": float}}
# edges: [{"from": str, "to": str}]
def compute_routes(positions, edges):
    # {node_id: [edge_keys]} — edges leaving from top of source
    top_out = defaultdict(list)
    # {node_id: [edge_keys]} — edges entering bottom of destination
    bottom_in = defaultdict(list)
    edge_src, edge_dst = {}, {} # {edge_key: node_id}

    for e in edges:
        src, dst = e["from"], e["to"]
        assert src in positions and dst in positions, f"edge {src}->{dst} has unknown node"
        key = f"{src}->{dst}"
        top_out[src].append(key)
        bottom_in[dst].append(key)
        edge_src[key], edge_dst[key] = src, dst

    # spread ports evenly along each node border
    start_ports, end_ports = {}, {} # {edge_key: (x, y)}

    for nid, keys in top_out.items():
        # sort by dest x, tiebreak by dest y descending (closer dest = left port)
        keys.sort(key=lambda k: (positions[edge_dst[k]]["x"], -positions[edge_dst[k]]["y"]))
        pos = positions[nid]
        step = pos["w"] / (len(keys) + 1)
        for i, key in enumerate(keys):
            start_ports[key] = (pos["x"] + step * (i + 1), pos["y"])

    for nid, keys in bottom_in.items():
        # sort by src x, tiebreak by src y ascending (closer src = left port)
        keys.sort(key=lambda k: (positions[edge_src[k]]["x"], positions[edge_src[k]]["y"]))
        pos = positions[nid]
        step = pos["w"] / (len(keys) + 1)
        for i, key in enumerate(keys):
            end_ports[key] = (pos["x"] + step * (i + 1), pos["y"] + pos["h"])

    # --- route ---

    routes = {} # {edge_key: [[x, y], ...]}
    for e in edges:
        src, dst = e["from"], e["to"]
        key = f"{src}->{dst}"
        sp, dp = positions[src], positions[dst] # start position, destination position
        sx, sy = start_ports[key]
        ex, ey = end_ports[key]

        # same column AND adjacent row: straight (or small jog for inner ports).
        # anything farther goes through the lane like cross-column edges —
        # a dozen parallel verticals spanning the whole board is unreadable
        adjacent = abs(sp["y"] - dp["y"]) <= CELL_Y + 0.001
        if abs(sp["x"] - dp["x"]) < 0.001 and adjacent:
            # border edge: leftmost or rightmost at BOTH src and dst → straight vertical
            is_left = (top_out[src][0] == key and bottom_in[dst][0] == key)
            is_right = (top_out[src][-1] == key and bottom_in[dst][-1] == key)
            if is_left:
                straight_x = min(sx, ex)
                routes[key] = [[straight_x, sy], [straight_x, ey]]
            elif is_right:
                straight_x = max(sx, ex)
                routes[key] = [[straight_x, sy], [straight_x, ey]]
            else:
                # inner edge: small jog near target to reach port
                mid_y = ey + MARGIN_Y
                routes[key] = [[sx, sy], [sx, mid_y], [ex, mid_y], [ex, ey]]
        else:
            # different column: 6-point path via vertical lane next to target
            lane_x = dp["x"] - MARGIN_X if sx < ex else dp["x"] + dp["w"] + MARGIN_X
            turn_y_src, turn_y_dst = sy - MARGIN_Y, ey + MARGIN_Y
            routes[key] = [[sx, sy], [sx, turn_y_src], [lane_x, turn_y_src], [lane_x, turn_y_dst], [ex, turn_y_dst], [ex, ey]]

    return routes
