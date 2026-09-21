// popup.js -- inspection popups that appear next to clicked nodes
// tensor popup: shows shape, tensor values (editable for override)
// group popup: shows group name, "double-click to expand" hint
// header is contentEditable for rename (blur/Enter sends rename_node to backend)

// static circular import with nodes.js — safe for function declarations;
// NEVER dynamic-import sibling modules: an unversioned url spawns a second
// module instance with its own state (this bug wiped the board on expand)
import { updateNodeShapes } from "./nodes.js";

const board = document.getElementById("board");
// popups live in the node layer so they pan/zoom with the board
const nodeLayer = board.nodeLayer;

let sendFn = null;
let stateRef = null;

export function initPopup(send, state) {
    sendFn = send;
    stateRef = state;
}

export function closeAllPopups() {
    nodeLayer.querySelectorAll(".tensor-popup").forEach(el => el.remove());
}

export async function showTensorPopup(nodeId) {
    closeAllPopups();

    const node = stateRef.displayNodes.find(n => n.id === nodeId);
    if (!node) return;
    const pos = node.pos;

    const popup = document.createElement("div");
    popup.className = "tensor-popup";
    popup.style.left = (pos.x + pos.w + 20) + "px";
    popup.style.top = pos.y + "px";
    popup.addEventListener("mousedown", e => e.stopPropagation());
    popup.addEventListener("click", e => e.stopPropagation());

    // header (editable to rename)
    const boardNode = nodeLayer.querySelector(`[data-id="${CSS.escape(nodeId)}"] .node-name`);
    const displayLabel = boardNode ? boardNode.textContent : nodeId;
    _addHeader(popup, displayLabel, nodeId);

    // shape
    const shape = stateRef.tensorShapes ? stateRef.tensorShapes[nodeId] : null;
    if (shape) {
        _addLabel(popup, `shape [${shape.join(" × ")}]`);
    }

    // tensor values on demand
    if (stateRef.tensorShapes && stateRef.tensorShapes[nodeId]) {
        if (!stateRef.tensorCache[nodeId]) {
            try {
                stateRef.tensorCache[nodeId] = await sendFn("get_tensor", { name: nodeId });
            } catch (err) {
                // e.g. unmarked node in show-marked mode, or forward not run yet
                _addLabel(popup, String(err.message || err));
            }
        }
        const tv = stateRef.tensorCache[nodeId];
        if (tv) {
            _addLabel(popup, "values");
            _showValuesInPopup(popup, tv.values);
        }
    }

    // override controls
    _addOverrideControls(popup, nodeId);

    nodeLayer.appendChild(popup);
}


function _addHeader(popup, title, nodeId) {
    const header = document.createElement("div");
    header.className = "popup-header";
    const titleEl = document.createElement("span");
    titleEl.textContent = title;
    titleEl.contentEditable = "true";
    titleEl.spellcheck = false;
    titleEl.title = "click to rename";
    const doRename = async () => {
        const newName = titleEl.textContent.trim();
        if (newName && newName !== title && nodeId) {
            await sendFn("rename_node", { node_id: nodeId, name: newName });
            const boardNode = nodeLayer.querySelector(`[data-id="${CSS.escape(nodeId)}"] .node-name`);
            if (boardNode) boardNode.textContent = newName;
        }
    };
    titleEl.addEventListener("blur", doRename);
    titleEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            titleEl.blur();
        }
    });
    header.appendChild(titleEl);
    const closeBtn = document.createElement("span");
    closeBtn.className = "popup-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", () => popup.remove());
    header.appendChild(closeBtn);
    popup.appendChild(header);
}

function _addLabel(popup, text) {
    const label = document.createElement("div");
    label.className = "detail-label";
    label.textContent = text;
    popup.appendChild(label);
}

function _showValuesInPopup(popup, values) {
    popup.querySelectorAll(".values-display, .batch-container").forEach(el => el.remove());

    // 3D+ tensor: split first dim as batch, show separate panels per batch element
    if (Array.isArray(values) && values.length > 0 && Array.isArray(values[0]) && Array.isArray(values[0][0])) {
        const outer = document.createElement("div");
        outer.className = "batch-container";
        for (let i = 0; i < values.length; i++) {
            const col = document.createElement("div");
            col.className = "batch-col";
            const label = document.createElement("div");
            label.className = "detail-label";
            label.textContent = `batch ${i}`;
            col.appendChild(label);
            const pre = document.createElement("pre");
            pre.className = "values-display";
            pre.contentEditable = "true";
            pre.spellcheck = false;
            pre.textContent = _formatValues(values[i]);
            col.appendChild(pre);
            outer.appendChild(col);
        }
        popup.appendChild(outer);
        return;
    }

    // 1D/2D tensor: single panel
    const container = document.createElement("pre");
    container.className = "values-display";
    container.contentEditable = "true";
    container.spellcheck = false;
    container.textContent = _formatValues(values);
    popup.appendChild(container);
}

function _formatValues(data, indent = 0) {
    if (!Array.isArray(data)) {
        return String(data);
    }
    if (data.length > 0 && !Array.isArray(data[0])) {
        return "[" + data.map(v => String(v)).join(", ") + "]";
    }
    const pad = "  ".repeat(indent);
    const inner = data.map(row => pad + "  " + _formatValues(row, indent + 1));
    return "[\n" + inner.join(",\n") + "\n" + pad + "]";
}

function _addOverrideControls(popup, nodeId) {
    const section = document.createElement("div");
    section.className = "override-controls";
    const btn = document.createElement("button");
    btn.textContent = "Override & Re-run";
    btn.addEventListener("click", async () => {
        const allPanels = popup.querySelectorAll(".values-display");
        if (allPanels.length === 0) return;
        // if batch display (multiple panels), reassemble the full tensor from all batch slices
        const batchContainer = popup.querySelector(".batch-container");
        let val;
        if (batchContainer && allPanels.length > 1) {
            val = Array.from(allPanels).map(el => JSON.parse(el.textContent));
        } else {
            val = JSON.parse(allPanels[0].textContent);
        }
        const inputsEl = document.getElementById("input-tensor");
        const inputs = JSON.parse(inputsEl.value);
        const inputDtypes = stateRef.meta?.model?.input_dtypes || ["float32"];
        const result = await sendFn("forward", {
            inputs,
            input_dtypes: inputDtypes,
            overrides: { [nodeId]: val },
        });
        stateRef.tensorShapes = result.shapes;
        stateRef.tensorCache = {};
        updateNodeShapes(result.shapes);
        showTensorPopup(nodeId);
    });
    section.appendChild(btn);
    popup.appendChild(section);
}
