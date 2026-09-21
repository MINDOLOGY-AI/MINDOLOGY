// nodes.js -- renders nodes on the board (into the <board-canvas> node layer)
// single click: highlight edges (pink inputs, purple outputs). in selection mode: toggle selection.
// double click: open popup (tensor or group). on group nodes: expand if collapsed.
// in selection mode (groups.js), clicking non-group nodes toggles them for group creation

import { showTensorPopup } from "./popup.js";
import { highlightEdges } from "./highlight.js";
import { isSelectionMode, toggleNodeSelection, refreshGroupsPanel } from "./groups.js";
// static circular import — safe for function declarations (live bindings),
// and CRITICAL: never dynamic-import main.js by a different specifier — a
// mismatched url spawns a SECOND module instance with its own empty state + websocket
import { fullRender, showError } from "./main.js";

const board = document.getElementById("board");
// world-space layer inside <board-canvas> — node divs go here
const nodeLayer = board.nodeLayer;

let sendFn = null;
let stateRef = null;

export function initNodes(send, state) {
    sendFn = send;
    stateRef = state;
}

// --- marking (selective capture) ---
// a node is marked if it's in meta.marked; a group is marked if ANY member is

// mark mode: every node click toggles its mark instead of highlighting
let markMode = false;
export function setMarkMode(on) { markMode = on; }

function isMarked(node) {
    const marked = stateRef.meta?.marked || [];
    if (node.op === "group") {
        const members = stateRef.meta?.custom_modules?.[node.id] || [];
        return members.some(m => marked.includes(m));
    }
    return marked.includes(node.id);
}

async function toggleMark(node) {
    try {
        const res = await sendFn("set_marked", { name: node.id, marked: !isMarked(node) });
        stateRef.meta.marked = res.marked;
        refreshStars();
    } catch (err) {
        showError(err);
    }
}

export function refreshStars() {
    if (!stateRef.displayNodes) return;
    nodeLayer.querySelectorAll(".tensor-node").forEach(el => {
        const star = el.querySelector(".node-star");
        const node = stateRef.displayNodes.find(n => n.id === el.dataset.id);
        if (!star || !node) return;
        const on = isMarked(node);
        star.classList.toggle("marked", on);
        star.textContent = on ? "★" : "☆";
    });
    if (marksPanel.style.display !== "none") renderMarksPanel();
}

// --- marks panel: list of everything marked, × to unmark ---

const marksPanel = document.createElement("div");
marksPanel.className = "group-panel";
marksPanel.style.display = "none";
document.body.appendChild(marksPanel);

export function toggleMarksPanel() {
    if (marksPanel.style.display === "none") {
        renderMarksPanel();
        marksPanel.style.display = "block";
    } else {
        marksPanel.style.display = "none";
    }
}

function renderMarksPanel() {
    marksPanel.innerHTML = "";
    const h = document.createElement("h3");
    h.textContent = "marked";
    marksPanel.appendChild(h);
    const marked = stateRef.meta?.marked || [];
    if (!marked.length) {
        const msg = document.createElement("div");
        msg.textContent = "nothing marked — mark mode + click nodes";
        msg.style.color = "var(--dim)";
        msg.style.fontSize = "11px";
        marksPanel.appendChild(msg);
        return;
    }
    for (const name of marked) {
        const row = document.createElement("div");
        row.className = "group-row";
        const label = document.createElement("span");
        label.className = "group-name";
        label.textContent = name;
        row.appendChild(label);
        const x = document.createElement("button");
        x.textContent = "×";
        x.title = "unmark";
        x.style.color = "var(--red)";
        x.style.marginLeft = "auto";
        x.addEventListener("click", async () => {
            try {
                const res = await sendFn("set_marked", { name, marked: false });
                stateRef.meta.marked = res.marked;
                refreshStars();
            } catch (err) {
                showError(err);
            }
        });
        row.appendChild(x);
        marksPanel.appendChild(row);
    }
}

function makeStar(node) {
    const star = document.createElement("div");
    const on = isMarked(node);
    star.className = "node-star" + (on ? " marked" : "");
    star.textContent = on ? "★" : "☆";
    star.title = "mark for selective capture (show marked mode)";
    star.addEventListener("click", (e) => { e.stopPropagation(); toggleMark(node); });
    star.addEventListener("dblclick", (e) => e.stopPropagation());
    return star;
}

export function clearNodes() {
    nodeLayer.querySelectorAll(".tensor-node").forEach(el => el.remove());
}

export function renderNodes(displayNodes) {
    for (const node of displayNodes) {
        const pos = node.pos;
        const el = createNode(node);
        el.style.left = pos.x + "px";
        el.style.top = pos.y + "px";
        el.style.width = pos.w + "px";
        el.style.height = pos.h + "px";
        el.dataset.id = node.id;
        nodeLayer.appendChild(el);
    }
}

function createNode(node) {
    const el = document.createElement("div");
    const nameEl = document.createElement("div");
    nameEl.className = "node-name";
    nameEl.textContent = node.name;

    if (node.op === "group") {
        el.className = "tensor-node group";
        el.appendChild(nameEl);
        el.appendChild(makeStar(node));
        el.addEventListener("click", (e) => {
            e.stopPropagation();
            if (markMode) { toggleMark(node); return; }
            highlightEdges(node.id, stateRef.displayEdges);
            if (isSelectionMode()) {
                toggleNodeSelection(node.id);
                refreshGroupsPanel();
            }
        });
        el.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            if (isSelectionMode() || markMode) return;
            expandGroup(node.id);
        });
    } else {
        el.className = "tensor-node " + node.op;
        el.appendChild(nameEl);
        const shapeEl = document.createElement("div");
        shapeEl.className = "node-shape";
        shapeEl.textContent = node.shape ? node.shape.join("×") : "";
        el.appendChild(shapeEl);
        el.appendChild(makeStar(node));
        el.addEventListener("click", (e) => {
            e.stopPropagation();
            if (markMode) { toggleMark(node); return; }
            highlightEdges(node.id, stateRef.displayEdges);
            if (isSelectionMode()) {
                toggleNodeSelection(node.id);
                refreshGroupsPanel();
            }
        });
        el.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            if (isSelectionMode() || markMode) return;
            showTensorPopup(node.id);
        });
    }

    return el;
}

async function expandGroup(groupName) {
    let result;
    try {
        result = await sendFn("expand_group", { group: groupName });
    } catch (err) {
        showError(err);
        return;
    }
    stateRef.displayNodes = result.displayNodes;
    stateRef.displayEdges = result.displayEdges;
    stateRef.tensorCache = {};
    // sync local meta so groups panel shows expanded state
    if (!stateRef.meta.expanded_groups) stateRef.meta.expanded_groups = [];
    if (!stateRef.meta.expanded_groups.includes(groupName)) {
        stateRef.meta.expanded_groups.push(groupName);
    }

    fullRender();
    // note: no reframe — the viewport stays where the user put it
}

export function updateNodeShapes(shapes) {
    for (const [id, shape] of Object.entries(shapes)) {
        const el = nodeLayer.querySelector(`[data-id="${CSS.escape(id)}"]`);
        if (!el) continue;
        const shapeEl = el.querySelector(".node-shape");
        if (shapeEl) shapeEl.textContent = shape.join("×");
    }
}
