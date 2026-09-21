// groups.js -- custom groups side panel
// flat list of custom groups with expand/collapse/create/edit/delete
// selection mode: user clicks nodes on the board to build a custom group
// used by: nodes.js (isSelectionMode, toggleNodeSelection), main.js (initGroups, refreshGroupsPanel)

// static circular import — safe for function declarations; never dynamic-import
// main.js by a different specifier (a mismatched url spawns a second module
// instance, see nodes.js)
import { showError } from "./main.js";
import { refreshStars } from "./nodes.js";

let sendFn = null;
let stateRef = null;
let fullRenderFn = null;

// selection mode state
let selectionMode = false;
// when true, clicking a node also selects all its direct inputs and outputs
let relatedSelect = false;
// {node_id: true} -- which nodes are currently selected
let selectedNodes = {};
// null = creating new group, "groupname" = editing existing group
let editingGroup = null;

// called by nodes.js to check if clicks should select instead of inspect
export function isSelectionMode() { return selectionMode; }

// called by nodes.js on node click during selection mode
// returns list of all node ids that were toggled (for CSS class updates)
export function toggleNodeSelection(nodeId) {
    // collect ids to toggle: just the clicked node, or it + all direct neighbors (excluding groups)
    const ids = [nodeId];
    if (relatedSelect && stateRef.displayEdges) {
        // group node ids — don't include them in related select
        const groupIds = new Set(stateRef.displayNodes.filter(n => n.op === "group").map(n => n.id));
        for (const e of stateRef.displayEdges) {
            if (e.from === nodeId && !groupIds.has(e.to)) ids.push(e.to);
            if (e.to === nodeId && !groupIds.has(e.from)) ids.push(e.from);
        }
    }
    // if the clicked node was already selected, deselect all; otherwise select all
    const wasSelected = selectedNodes[nodeId];
    for (const id of ids) {
        if (wasSelected) {
            delete selectedNodes[id];
        } else {
            selectedNodes[id] = true;
        }
    }
    return ids;
}

const panel = document.createElement("div");
panel.className = "group-panel";
panel.style.display = "none";
document.body.appendChild(panel);

export function initGroups(send, state, fullRender) {
    sendFn = send;
    stateRef = state;
    fullRenderFn = fullRender;
    document.getElementById("btn-groups").addEventListener("click", togglePanel);
}

export function refreshGroupsPanel() {
    if (panel.style.display === "none") return;
    // in selection mode, only update the count + sync selecting CSS — don't nuke the input value
    if (selectionMode) {
        const countEl = panel.querySelector("#selection-count");
        if (countEl) {
            const count = Object.keys(selectedNodes).length;
            countEl.textContent = `${count} node${count !== 1 ? "s" : ""} selected`;
        }
        // sync selecting class on all nodes
        document.querySelectorAll(".tensor-node").forEach(el => {
            el.classList.toggle("selecting", !!selectedNodes[el.dataset.id]);
        });
        return;
    }
    renderPanel();
}

function togglePanel() {
    if (panel.style.display === "none") {
        panel.style.display = "block";
        renderPanel();
    } else {
        panel.style.display = "none";
    }
}

function exitSelectionMode() {
    selectionMode = false;
    relatedSelect = false;
    selectedNodes = {};
    editingGroup = null;
    document.querySelectorAll(".tensor-node.selecting").forEach(el => el.classList.remove("selecting"));
}

function renderPanel() {
    panel.innerHTML = "";

    const closeBtn = document.createElement("span");
    closeBtn.className = "panel-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", () => { panel.style.display = "none"; });
    panel.appendChild(closeBtn);

    const h = document.createElement("h3");
    h.textContent = "Groups";
    panel.appendChild(h);

    if (!stateRef.displayNodes) {
        const msg = document.createElement("div");
        msg.textContent = "load a model first";
        msg.style.color = "var(--dim)";
        msg.style.fontSize = "11px";
        panel.appendChild(msg);
        return;
    }

    // flat list of custom groups from meta
    const groups = stateRef.meta?.custom_modules || {};
    const expanded = new Set(stateRef.meta?.expanded_groups || []);

    for (const name of Object.keys(groups).sort()) {
        const row = document.createElement("div");
        row.className = "group-row";

        const label = document.createElement("span");
        label.className = "group-name";
        label.textContent = name;
        row.appendChild(label);

        // mark star: marks/unmarks every member for selective capture
        const markedSet = new Set(stateRef.meta?.marked || []);
        const anyMarked = (groups[name] || []).some(m => markedSet.has(m));
        const starBtn = document.createElement("button");
        starBtn.textContent = anyMarked ? "★" : "☆";
        starBtn.title = "mark group for selective capture";
        if (anyMarked) starBtn.style.color = "var(--mint)";
        starBtn.addEventListener("click", async () => {
            try {
                const res = await sendFn("set_marked", { name, marked: !anyMarked });
                stateRef.meta.marked = res.marked;
                renderPanel();
                refreshStars();
            } catch (err) {
                showError(err);
            }
        });
        row.appendChild(starBtn);

        const btns = document.createElement("span");
        btns.style.marginLeft = "auto";
        btns.style.display = "flex";
        btns.style.gap = "4px";

        // expand/collapse toggle
        const isExpanded = expanded.has(name);
        const toggleBtn = document.createElement("button");
        toggleBtn.textContent = isExpanded ? "▼" : "▶";
        toggleBtn.title = isExpanded ? "Collapse" : "Expand";
        if (isExpanded) {
            toggleBtn.style.color = "var(--sky)";
            toggleBtn.style.borderColor = "color-mix(in srgb, var(--sky) 40%, transparent)";
        }
        toggleBtn.addEventListener("click", () => toggleGroup(name, isExpanded));
        btns.appendChild(toggleBtn);

        // edit button
        const editBtn = document.createElement("button");
        editBtn.textContent = "✎";
        editBtn.title = "Edit group";
        editBtn.addEventListener("click", () => startEditGroup(name));
        btns.appendChild(editBtn);

        // delete button
        const delBtn = document.createElement("button");
        delBtn.textContent = "×";
        delBtn.title = "Delete group";
        delBtn.style.color = "var(--red)";
        delBtn.style.borderColor = "color-mix(in srgb, var(--red) 30%, transparent)";
        delBtn.addEventListener("click", () => deleteGroup(name));
        btns.appendChild(delBtn);

        row.appendChild(btns);
        panel.appendChild(row);
    }

    const sep = document.createElement("hr");
    sep.style.borderColor = "var(--border)";
    sep.style.margin = "8px 0";
    panel.appendChild(sep);

    if (selectionMode) {
        renderSelectionControls();
    } else {
        const createBtn = document.createElement("button");
        createBtn.textContent = "Create Group";
        createBtn.style.width = "100%";
        createBtn.addEventListener("click", () => {
            selectionMode = true;
            selectedNodes = {};
            editingGroup = null;
            renderPanel();
        });
        panel.appendChild(createBtn);
    }
}

function renderSelectionControls() {
    const isEditing = editingGroup !== null;

    const info = document.createElement("div");
    info.className = "detail-label";
    info.textContent = isEditing
        ? `editing ${editingGroup} — click nodes to add/remove`
        : "click nodes on the board to select them";
    panel.appendChild(info);

    const count = Object.keys(selectedNodes).length;
    const countEl = document.createElement("div");
    countEl.id = "selection-count";
    countEl.style.fontSize = "11px";
    countEl.style.color = "var(--red)";
    countEl.style.margin = "4px 0";
    countEl.textContent = `${count} node${count !== 1 ? "s" : ""} selected`;
    panel.appendChild(countEl);

    // related select toggle — click a node to also select its direct inputs/outputs
    const relBtn = document.createElement("button");
    relBtn.textContent = relatedSelect ? "Related Select: ON" : "Related Select: OFF";
    relBtn.style.width = "100%";
    relBtn.style.marginBottom = "6px";
    if (relatedSelect) {
        relBtn.style.borderColor = "var(--red)";
        relBtn.style.color = "var(--red)";
    }
    relBtn.addEventListener("click", () => {
        relatedSelect = !relatedSelect;
        relBtn.textContent = relatedSelect ? "Related Select: ON" : "Related Select: OFF";
        relBtn.style.borderColor = relatedSelect ? "var(--red)" : "";
        relBtn.style.color = relatedSelect ? "var(--red)" : "";
    });
    panel.appendChild(relBtn);

    // name input (only for create, not edit)
    let nameInput = null;
    if (!isEditing) {
        nameInput = document.createElement("input");
        nameInput.type = "text";
        nameInput.placeholder = "group name";
        nameInput.style.width = "100%";
        nameInput.style.marginBottom = "6px";
        nameInput.style.background = "var(--background)";
        nameInput.style.border = "1px solid var(--border)";
        nameInput.style.color = "var(--white)";
        nameInput.style.padding = "4px 6px";
        nameInput.style.borderRadius = "3px";
        nameInput.style.fontFamily = "var(--mono, monospace)";
        nameInput.style.fontSize = "11px";
        panel.appendChild(nameInput);
    }

    const btnRow = document.createElement("div");
    btnRow.style.display = "flex";
    btnRow.style.gap = "6px";

    const confirmBtn = document.createElement("button");
    confirmBtn.textContent = isEditing ? "Save" : "Create";
    confirmBtn.style.flex = "1";
    confirmBtn.addEventListener("click", async () => {
        const ids = Object.keys(selectedNodes);
        if (ids.length === 0) return;

        try {
            if (isEditing) {
                const result = await sendFn("update_group", {
                    group: editingGroup,
                    nodes: ids,
                });
                stateRef.displayNodes = result.displayNodes;
                stateRef.displayEdges = result.displayEdges;
                stateRef.tensorCache = {};
                exitSelectionMode();
                fullRenderFn();
            } else {
                const name = nameInput.value.trim();
                if (!name) return;
                const result = await sendFn("create_group", {
                    name,
                    nodes: ids,
                });
                stateRef.displayNodes = result.displayNodes;
                stateRef.displayEdges = result.displayEdges;
                // add to local meta so panel shows it
                if (!stateRef.meta.custom_modules) stateRef.meta.custom_modules = {};
                stateRef.meta.custom_modules[name] = ids;
                stateRef.tensorCache = {};
                exitSelectionMode();
                fullRenderFn();
            }
        } catch (err) {
            console.error("group operation failed:", err);
            const countEl = panel.querySelector("#selection-count");
            if (countEl) {
                countEl.textContent = err.message;
                countEl.style.color = "var(--red)";
                countEl.style.whiteSpace = "pre-wrap";
            }
        }
    });
    btnRow.appendChild(confirmBtn);

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.flex = "1";
    cancelBtn.addEventListener("click", () => {
        exitSelectionMode();
        renderPanel();
    });
    btnRow.appendChild(cancelBtn);

    panel.appendChild(btnRow);
}

async function toggleGroup(name, isExpanded) {
    const type = isExpanded ? "collapse_group" : "expand_group";
    let result;
    try {
        result = await sendFn(type, { group: name });
    } catch (err) {
        showError(err);
        return;
    }
    stateRef.displayNodes = result.displayNodes;
    stateRef.displayEdges = result.displayEdges;
    // update local meta expanded_groups
    if (!stateRef.meta.expanded_groups) stateRef.meta.expanded_groups = [];
    if (isExpanded) {
        stateRef.meta.expanded_groups = stateRef.meta.expanded_groups.filter(g => g !== name);
    } else {
        stateRef.meta.expanded_groups.push(name);
    }
    stateRef.tensorCache = {};
    fullRenderFn();
    // note: no reframe — the viewport stays where the user put it
    renderPanel();
}

async function deleteGroup(name) {
    let result;
    try {
        result = await sendFn("delete_group", { group: name });
    } catch (err) {
        showError(err);
        return;
    }
    stateRef.displayNodes = result.displayNodes;
    stateRef.displayEdges = result.displayEdges;
    // remove from local meta
    if (stateRef.meta.custom_modules) delete stateRef.meta.custom_modules[name];
    if (stateRef.meta.expanded_groups) {
        stateRef.meta.expanded_groups = stateRef.meta.expanded_groups.filter(g => g !== name);
    }
    stateRef.tensorCache = {};
    fullRenderFn();
    renderPanel();
}

function startEditGroup(name) {
    // enter selection mode with existing nodes pre-selected
    selectionMode = true;
    editingGroup = name;
    selectedNodes = {};
    // get nodes from meta
    const groupNodes = stateRef.meta?.custom_modules?.[name];
    if (groupNodes) {
        for (const nid of groupNodes) {
            selectedNodes[nid] = true;
        }
    }
    fullRenderFn();
    for (const nid of Object.keys(selectedNodes)) {
        const el = document.querySelector(`[data-id="${CSS.escape(nid)}"]`);
        if (el) el.classList.add("selecting");
    }
    renderPanel();
}
