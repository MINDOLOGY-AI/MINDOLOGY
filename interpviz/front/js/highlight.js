// highlight.js -- click a node to highlight it, its edges, and its neighbors
// orange = clicked node, pink = input edges/nodes, purple = output edges/nodes
// click empty board space to clear (except during selection mode)

import { highlightEdge, resetAllEdges } from "./edges.js";
import { isSelectionMode } from "./groups.js";

// theme colors (svg attributes can't read css vars, so resolve them once)
const rootStyle = getComputedStyle(document.documentElement);
const PINK = rootStyle.getPropertyValue("--neon-pink").trim() || "#FF80BF";
const PURPLE = rootStyle.getPropertyValue("--lavender").trim() || "#B8A0FF";

let selectedNode = null;

export function initHighlight() {
    // <board-canvas> fires board-tap on empty-space clicks without dragging
    document.getElementById("board").addEventListener("board-tap", () => {
        // don't clear highlights during selection mode (prevents deselect on drag)
        if (isSelectionMode()) return;
        clearHighlight();
    });
}

export function clearHighlight() {
    if (!selectedNode) return;
    resetAllEdges();
    selectedNode = null;
    document.querySelectorAll(".tensor-node.selected, .tensor-node.hl-input, .tensor-node.hl-output").forEach(el => {
        el.classList.remove("selected", "hl-input", "hl-output");
    });
}

export function highlightEdges(nodeId, displayEdges) {
    clearHighlight();
    selectedNode = nodeId;

    // highlight clicked node orange
    const nodeEl = document.querySelector(`[data-id="${CSS.escape(nodeId)}"]`);
    if (nodeEl) nodeEl.classList.add("selected");

    for (const edge of displayEdges) {
        const key = `${edge.from}->${edge.to}`;
        if (edge.to === nodeId) {
            // input edge + input neighbor: pink
            highlightEdge(key, PINK);
            const el = document.querySelector(`[data-id="${CSS.escape(edge.from)}"]`);
            if (el) el.classList.add("hl-input");
        } else if (edge.from === nodeId) {
            // output edge + output neighbor: purple
            highlightEdge(key, PURPLE);
            const el = document.querySelector(`[data-id="${CSS.escape(edge.to)}"]`);
            if (el) el.classList.add("hl-output");
        }
    }
}
