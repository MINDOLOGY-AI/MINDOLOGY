// main.js -- app entry point, WebSocket connection, global state, button wiring
// all other modules import from here: send() for WebSocket calls, state for shared data
// flow: Load Meta -> Load Model -> (optional) Forward -> click nodes to inspect

import { renderNodes, clearNodes, updateNodeShapes, initNodes, setMarkMode, toggleMarksPanel } from "./nodes.js";
import { renderEdges, clearEdges } from "./edges.js";
import { initPopup, closeAllPopups } from "./popup.js";
import { initHighlight } from "./highlight.js";
import { initGroups, refreshGroupsPanel } from "./groups.js";

// the <board-canvas> element — pan/zoom/grid/edges all live in the component
const board = document.getElementById("board");

// --- WebSocket (request/response with message IDs) ---

let ws = null;
// {_id: {resolve, reject}} -- pending promises waiting for server response
const pending = {};
let msgId = 0;

// fire a WS message tagged with a fresh _id, return a promise that resolves once the server replies with the same _id
export function send(type, data = {}) {
    const _id = ++msgId;
    return new Promise((resolve, reject) => {
        pending[_id] = { resolve, reject };
        ws.send(JSON.stringify({ type, _id, ...data }));
    });
}

// every `await send(...)` waits for onmessage here to resolve it (matched by _id) then returns whatever you pass to resolve
function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/interpviz/ws`);
    ws.onmessage = (e) => {
        // msg: {type: str, _id: int, ...payload}
        const msg = JSON.parse(e.data);
        // server demands login — surface the auth overlay
        if (msg.type === "auth_required") {
            document.getElementById("auth-overlay").style.display = "flex";
            return;
        }
        const cb = pending[msg._id];
        if (cb) {
            delete pending[msg._id];
            if (msg.type === "error") {
                cb.reject(new Error(msg.message));
            } else {
                cb.resolve(msg);
            }
            return;
        }
        // server-pushed display refresh (no _id): swap in the new graph
        if (msg.type === "update_display" && msg.displayNodes) {
            state.displayNodes = msg.displayNodes;
            state.displayEdges = msg.displayEdges;
            fullRender();
        }
    };
    ws.onclose = () => {
        document.getElementById("model-status").textContent = "disconnected — reconnecting...";
        // reject in-flight requests so awaits don't hang forever
        for (const id in pending) {
            pending[id].reject(new Error("connection lost"));
            delete pending[id];
        }
        setTimeout(connect, 1000);
    };
}

// --- Global state ---

export const state = {
    // [{name, id, op, shape, pos: {x, y, w, h}}] -- backend-computed display list
    displayNodes: null,
    // [{from, to, points: [[x,y],...]}] -- backend-computed routed edges
    displayEdges: null,
    meta: null,             // visualMeta.json contents
    tensorShapes: null,     // {node_id: [dims]} after forward
    tensorCache: {},        // {node_id: tensor_data} fetched on click
};

// --- Render helper ---

export function fullRender() {
    clearNodes();
    clearEdges();
    closeAllPopups();
    if (state.displayNodes && state.displayEdges) {
        renderNodes(state.displayNodes);
        renderEdges(state.displayEdges);
    }
    refreshGroupsPanel();
}

// frame the whole graph on load (was board.js — bbox math stays here, the
// component does the actual camera work)
export function centerView(displayNodes) {
    if (!displayNodes || displayNodes.length === 0) return;
    // compute bounding box of all nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const node of displayNodes) {
        const pos = node.pos;
        minX = Math.min(minX, pos.x);
        minY = Math.min(minY, pos.y);
        maxX = Math.max(maxX, pos.x + pos.w);
        maxY = Math.max(maxY, pos.y + pos.h);
    }
    board.centerView({ minX, minY, maxX, maxY });
}

function status(text) {
    document.getElementById("model-status").textContent = text;
}

// server errors (bad folder, trace failure, invalid group) show in the status bar
// instead of dying as unhandled promise rejections
export function showError(err) {
    status(String(err.message || err));
}

// --- capture mode toggle (one button: grey "show all" <-> green "show marked") ---

const captureModeBtn = document.getElementById("btn-capture-mode");

function setCaptureModeUI(mode) {
    captureModeBtn.textContent = mode === "marked" ? "show marked" : "show all";
    captureModeBtn.classList.toggle("active", mode === "marked");
}

captureModeBtn.addEventListener("click", async () => {
    const target = captureModeBtn.classList.contains("active") ? "all" : "marked";
    try {
        await send("set_capture_mode", { mode: target });
        if (state.meta) state.meta.capture_mode = target;
        setCaptureModeUI(target);
    } catch (err) { showError(err); }
});

// --- device toggle (cpu <-> gpu) ---

const deviceBtn = document.getElementById("btn-device");

function setDeviceUI(device) {
    deviceBtn.textContent = device === "cuda" ? "gpu" : "cpu";
    deviceBtn.classList.toggle("on", device === "cuda");
}

deviceBtn.addEventListener("click", async () => {
    const target = deviceBtn.classList.contains("on") ? "cpu" : "cuda";
    try {
        const result = await send("set_device", { device: target });
        setDeviceUI(result.device);
    } catch (err) {
        showError(err);
    }
});

// --- Event handlers ---

async function loadMeta() {
    const folder = document.getElementById("meta-folder").value;
    let result;
    try {
        result = await send("load_meta", { folder });
    } catch (err) {
        showError(err);
        return;
    }
    state.meta = result;
    if (result.device) setDeviceUI(result.device);
    setCaptureModeUI(state.meta.capture_mode || "all");
    state.displayNodes = null;
    state.displayEdges = null;
    state.tensorShapes = null;
    state.tensorCache = {};
    clearNodes();
    clearEdges();
    closeAllPopups();
    status("meta loaded");
    if (state.meta.model) {
        document.getElementById("input-tensor").value =
            JSON.stringify(state.meta.model.example_inputs || [[[0, 1]]]);
    }
}

async function loadModel() {
    if (!state.meta?.model) {
        status("load meta first");
        return;
    }
    status("loading...");

    let result;
    try {
        result = await send("load_model");
    } catch (err) {
        showError(err);
        return;
    }

    state.displayNodes = result.displayNodes;
    state.displayEdges = result.displayEdges;
    if (result.device) setDeviceUI(result.device);
    state.tensorShapes = null;
    state.tensorCache = {};

    fullRender();
    centerView(state.displayNodes);

    status("");
}

async function runForward() {
    if (!state.meta?.model) return;
    const inputs = JSON.parse(document.getElementById("input-tensor").value);
    const inputDtypes = state.meta.model.input_dtypes || ["float32"];

    let result;
    try {
        result = await send("forward", { inputs, input_dtypes: inputDtypes });
    } catch (err) {
        showError(err);
        return;
    }
    state.tensorShapes = result.shapes;
    state.tensorCache = {};

    updateNodeShapes(state.tensorShapes);
}

// --- Init ---

connect();
initNodes(send, state);
initPopup(send, state);
initHighlight();
initGroups(send, state, fullRender);

// one load button: meta then model — there is no use for meta without the model
document.getElementById("btn-load").addEventListener("click", async () => {
    await loadMeta();
    if (state.meta?.model) await loadModel();
});
document.getElementById("btn-forward").addEventListener("click", runForward);

// mark mode: purple while active — every node click toggles its mark
const markModeBtn = document.getElementById("btn-mark-mode");
markModeBtn.addEventListener("click", () => {
    const on = !markModeBtn.classList.contains("active");
    markModeBtn.classList.toggle("active", on);
    setMarkMode(on);
});
document.getElementById("btn-marks").addEventListener("click", toggleMarksPanel);
