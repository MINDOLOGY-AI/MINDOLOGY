// sae.js -- the sae tab: type english, run / generate, see every hook part's latents per token
// grid: one column per token (generated tokens continue to the right), rows top -> bottom:
//   next-token guesses, hook parts in reverse file order (L15 ... L0), the tokens themselves.
// side panel, top half: the hovered cell's latents (or next-token guesses); click a cell = pin it (hover stops changing
// the panel, click again to unpin). bottom half: search the cell's layer by label, and the selected latent (click one
// in either half): its value at the cell's token, a box to set it (an edit, synapse/interp/sae_intervene.py), its card.
// attribute view: nothing is attributed until a click picks the target: a token in the bottom row (also turns the view
// on) or a next-token guess in the top row (target = the token it predicted). cells before the target are colored by
// attribution to the target's prediction (blue - / grey 0 / orange +), the target and later columns keep the activation
// colors; the top TOP_N nodes get a border.
// edits: every change reruns the same prompt with all edits; the edits list sits in the second bar, edited cells are
// striped purple.

// static circular import — safe for function declarations (see nodes.js)
import { send, showError, status } from "./main.js";

const grid = document.getElementById("sae-grid");
const sideTop = document.getElementById("sae-side-top");
const sideBottom = document.getElementById("sae-side-bottom");

// last sae_run reply: {tokens: [str], n_prompt: int, output: str, next: [[[str, prob] x5] per token],
//   hooks: {set: {display: {max_features, on_cell, max_examples}, parts: {part: {reads, ids: [[int]], vals: [[float]], labels: {id: [label, score]}}}}}}
let run = null;
// the pinned cell element, or null: while set, hovering other cells leaves the side panel alone
let pinned = null;
// "act" (activation colors) or "attr" (attribution view)
let view = "act";
// last sae_attribute reply: {target: int, logit: float,
//   parts: {set: {part: {ids: [[int]], attr: [[float]], bias: [float], err: [float]}}}} (ids per token as in run, strongest first)
let attr = null;
// {"set|part|t|id"}: the top N nodes by |attribution|; a cell's bias / error node has id "bias" / "err"
let top = new Set();
// [[set, part, token, latent id, value]] feature edits applied to every run until removed (cleared when the prompt changes)
let edits = [];
// {text, template, n}: the last run's request, rerun when the edits change
let lastQuery = null;
// {set, part, t}: the cell shown in the side panel's top half (the bottom half searches / edits in it), or null
let ctx = null;
// latent id selected in the bottom half, or null
let sel = null;
// search box text (kept across cells) and the last search's results for ctx: [[id, label, score, density]] or null
let searchText = "";
let results = null;

// nodes of the whole grid with a border in the attribute view (by |attribution|)
const TOP_N = 20;

// names of a cell's non-latent attribution nodes
const NODE_NAMES = { err: "error (unexplained by the SAE)", bias: "bias (the SAE's constant b_dec)" };

// token text with visible whitespace
function show(t) {
    return t.replace(/ /g, "␣").replace(/\n/g, "↵");
}

function latentName(part, labels, id) {
    return labels[id] ? labels[id][0] : `${part}:${id} (no label)`;
}

async function runSae(nGenerate) {
    const text = document.getElementById("sae-text").value;
    const template = document.getElementById("sae-template").checked;
    // edits point at token positions of one prompt
    if (lastQuery && (lastQuery.text !== text || lastQuery.template !== template)) edits = [];
    lastQuery = { text, template, n: nGenerate };
    status(nGenerate ? `generating ${nGenerate}...` : "running...");
    const prevTarget = attr ? attr.target : null;
    try {
        run = await send("sae_run", { text, template, n_generate: nGenerate, edits });
    } catch (err) {
        showError(err);
        return;
    }
    status("");
    pinned = null;
    attr = null;
    document.getElementById("sae-output").value = run.output.replace(/\n/g, "↵");
    renderEdits();
    // attribute view: keep the clicked target if it still exists, else wait for a click
    if (view === "attr" && prevTarget !== null && prevTarget < run.tokens.length) await attribute(prevTarget);
    else setAttrInfo();
    renderGrid();
    // keep the side panel on the same cell (e.g. after an edit) if it still exists
    if (ctx && ctx.t < run.tokens.length) showCell(ctx.set, ctx.part, ctx.t);
    else clearSide();
}

// set latent `id` of part (set, part) at token t to value (null = remove the edit), then rerun
function setEdit(setName, partName, t, id, value) {
    edits = edits.filter(e => !(e[0] === setName && e[1] === partName && e[2] === t && e[3] === id));
    if (value !== null) edits.push([setName, partName, t, id, value]);
    runSae(lastQuery.n);
}

// the edits list in the second bar: one chip per edit, × removes it
function renderEdits() {
    const box = document.getElementById("sae-edits");
    box.innerHTML = "";
    for (const [sn, pn, t, id, value] of edits) {
        const chip = document.createElement("span");
        chip.className = "sae-edit-chip";
        const label = run.hooks[sn].parts[pn].labels[id];
        chip.title = label ? label[0] : "(no label)";
        chip.textContent = `${pn}:${id} @${t} "${show(run.tokens[t] ?? "?")}" = ${value} `;
        const x = document.createElement("span");
        x.className = "sae-edit-x";
        x.textContent = "×";
        x.addEventListener("click", () => setEdit(sn, pn, t, id, null));
        chip.appendChild(x);
        box.appendChild(chip);
    }
}

// attribution of every cell's latents (+ bias and error node) to the prediction of token column `target`
async function attribute(target) {
    status(`attributing to "${show(run.tokens[target])}"...`);
    try {
        attr = await send("sae_attribute", { target });
    } catch (err) {
        showError(err);
        return;
    }
    status("");
    rankTop();
    setAttrInfo();
}

// the text next to the attribute button: the target, or how to pick one
function setAttrInfo() {
    const info = document.getElementById("sae-attr-info");
    if (view !== "attr") info.textContent = "";
    else if (!attr) info.textContent = "click a token (bottom row) or a next-word guess (top row)";
    else info.textContent = `target "${show(run.tokens[attr.target])}" · logit ${attr.logit}`;
}

// make token column `target` the attribution target (turns the attribute view on) and redraw
async function pickTarget(target) {
    if (view !== "attr") {
        view = "attr";
        document.getElementById("btn-sae-attr").classList.add("active");
    }
    await attribute(target);
    pinned = null;
    renderGrid();
    // the side panel showed the old numbers
    if (ctx) showCell(ctx.set, ctx.part, ctx.t);
}

// top TOP_N nodes of the columns before the target by |attribution|
function rankTop() {
    const nodes = [];  // [[key, |attr|]]
    for (const [sn, parts] of Object.entries(attr.parts)) {
        for (const [pn, p] of Object.entries(parts)) {
            for (let t = 0; t < attr.target; t++) {
                nodes.push([`${sn}|${pn}|${t}|err`, Math.abs(p.err[t])]);
                nodes.push([`${sn}|${pn}|${t}|bias`, Math.abs(p.bias[t])]);
                p.ids[t].forEach((id, j) => nodes.push([`${sn}|${pn}|${t}|${id}`, Math.abs(p.attr[t][j])]));
            }
        }
    }
    nodes.sort((a, b) => b[1] - a[1]);
    top = new Set(nodes.slice(0, TOP_N).map(([key]) => key));
}

// diverging color: blue (negative) - grey (0) - orange (positive), strength |v| / max
function attrColor(v, max) {
    const p = Math.round(100 * Math.min(1, Math.abs(v) / max));
    return `color-mix(in srgb, ${v > 0 ? "var(--attr-pos)" : "var(--attr-neg)"} ${p}%, var(--attr-zero))`;
}

// a cell's attribution nodes, strongest |attr| first: [{id: int | "bias" | "err", attr: float}]
function cellNodes(sn, pn, t) {
    const p = attr.parts[sn][pn];
    const nodes = p.ids[t].map((id, j) => ({ id, attr: p.attr[t][j] }));
    nodes.push({ id: "err", attr: p.err[t] }, { id: "bias", attr: p.bias[t] });
    return nodes.sort((a, b) => Math.abs(b.attr) - Math.abs(a.attr));
}

// true if cell (t) shows attribution: attribute view and a column the target can see
function attributed(t) {
    return attr !== null && t < attr.target;
}

function renderGrid() {
    grid.innerHTML = "";
    const T = run.tokens.length;
    grid.style.gridTemplateColumns = `70px repeat(${T}, 130px)`;

    // row: next-token guesses (column t = the model's guess for the token after t); in the attribute view a click
    // makes the token it predicted (t + 1) the target
    addHead("next", "sae-next-row");
    run.next.forEach((guesses, t) => {
        const cell = addCell("sae-next-row");
        cell.textContent = show(guesses[0][0]);
        if (attr && t + 1 === attr.target) cell.classList.add("attr-target");
        sideOnHover(cell, () => showGuesses(t));
        if (view === "attr" && t + 1 < run.tokens.length) cell.addEventListener("click", () => pickTarget(t + 1));
    });

    // strongest |attribution| over the attributed columns, for the attribute view's color scale
    const attrMax = attr ? Math.max(1e-6, ...Object.entries(attr.parts).flatMap(([sn, parts]) =>
        Object.keys(parts).flatMap(pn => [...Array(attr.target).keys()].map(t => Math.abs(cellNodes(sn, pn, t)[0].attr))))) : 0;

    // rows: every hook part, last part on top
    for (const [setName, hs] of Object.entries(run.hooks)) {
        for (const [partName, part] of Object.entries(hs.parts).reverse()) {
            addHead(partName, "");
            // strongest latent value in this row, for the activation shading
            const rowMax = Math.max(1e-6, ...part.vals.map(v => v[0] || 0));
            for (let t = 0; t < T; t++) {
                const cell = addCell("sae-feat");
                // [id | "bias" | "err"] shown in the cell
                let shown;
                if (attributed(t)) {
                    const nodes = cellNodes(setName, partName, t);
                    cell.style.backgroundColor = attrColor(nodes[0].attr, attrMax);
                    if (nodes.some(nd => top.has(`${setName}|${partName}|${t}|${nd.id}`))) cell.classList.add("attr-top");
                    shown = nodes.slice(0, hs.display.on_cell).map(nd => nd.id);
                } else {
                    const share = Math.round(40 * (part.vals[t][0] || 0) / rowMax);
                    cell.style.backgroundColor = `color-mix(in srgb, var(--mint) ${share}%, transparent)`;
                    shown = part.ids[t].slice(0, hs.display.on_cell);
                }
                for (const id of shown) {
                    const line = document.createElement("div");
                    line.className = "sae-feat-line";
                    line.textContent = NODE_NAMES[id] ?? latentName(partName, part.labels, id);
                    cell.appendChild(line);
                }
                if (edits.some(e => e[0] === setName && e[1] === partName && e[2] === t)) cell.classList.add("edited");
                sideOnHover(cell, () => showCell(setName, partName, t));
            }
        }
    }

    // row: the tokens, generated ones highlighted; a click makes a token the attribution target
    addHead("tokens", "sae-token-row");
    run.tokens.forEach((tok, t) => {
        const cell = addCell("sae-token-row" + (t >= run.n_prompt ? " generated" : ""));
        cell.textContent = show(tok);
        cell.title = `token ${t}${t >= run.n_prompt ? " (generated)" : ""}`;
        if (attr && t === attr.target) cell.classList.add("attr-target");
        // the first token has no prediction to explain
        if (t === 0) return;
        cell.classList.add("attr-pickable");
        cell.addEventListener("click", () => pickTarget(t));
    });
}

function addHead(text, cls) {
    const el = document.createElement("div");
    el.className = `sae-head ${cls}`;
    el.textContent = text;
    grid.appendChild(el);
}

function addCell(cls) {
    const el = document.createElement("div");
    el.className = `sae-cell ${cls}`;
    grid.appendChild(el);
    return el;
}

// hover a cell = fill the side panel with fill() unless another cell is pinned; click = pin / unpin this cell
function sideOnHover(cell, fill) {
    cell.addEventListener("mouseenter", () => { if (!pinned) fill(); });
    cell.addEventListener("click", () => {
        if (pinned) pinned.classList.remove("pinned");
        pinned = pinned === cell ? null : cell;
        if (pinned) pinned.classList.add("pinned");
        fill();
    });
}

function clearSide() {
    sideTop.innerHTML = "";
    ctx = null;
    sel = null;
    results = null;
    renderBottom();
}

// one row of a side list: a value span (colored when given) + text; onClick selects it (null = not clickable)
function sideRow(parent, value, color, text, onClick) {
    const r = document.createElement("div");
    r.className = "sae-latent" + (onClick ? " pickable" : "");
    const vEl = document.createElement("span");
    vEl.className = "sae-val";
    if (color) vEl.style.color = color;
    vEl.textContent = value;
    r.appendChild(vEl);
    r.appendChild(document.createTextNode(text));
    if (onClick) r.addEventListener("click", onClick);
    parent.appendChild(r);
    return r;
}

// side panel top: top next-token guesses after token t
function showGuesses(t) {
    clearSide();
    const h = document.createElement("h3");
    h.textContent = `next after token ${t} "${show(run.tokens[t])}"`;
    sideTop.appendChild(h);
    for (const [tok, p] of run.next[t]) sideRow(sideTop, p.toFixed(3), null, show(tok), null);
}

// side panel top: a layer cell's latents (activations, or attributions strongest |attr| first with top-N starred);
// click a latent = select it in the bottom half
function showCell(setName, partName, t) {
    if (!ctx || ctx.set !== setName || ctx.part !== partName || ctx.t !== t) {
        ctx = { set: setName, part: partName, t };
        sel = null;
        results = null;
    }
    const part = run.hooks[setName].parts[partName];
    sideTop.innerHTML = "";
    const h = document.createElement("h3");
    h.textContent = `${partName} · token ${t} "${show(run.tokens[t])}"` + (attributed(t) ? ` → "${show(run.tokens[attr.target])}"` : "");
    sideTop.appendChild(h);
    if (attributed(t)) {
        for (const nd of cellNodes(setName, partName, t)) {
            const star = top.has(`${setName}|${partName}|${t}|${nd.id}`) ? "★ " : "";
            const name = nd.id in NODE_NAMES ? NODE_NAMES[nd.id] : `${partName}:${nd.id} ${latentName(partName, part.labels, nd.id)}`;
            sideRow(sideTop, (nd.attr > 0 ? "+" : "") + nd.attr.toFixed(3), nd.attr > 0 ? "var(--attr-pos)" : "var(--attr-neg)",
                star + name, nd.id in NODE_NAMES ? null : () => select(nd.id));
        }
    } else {
        part.ids[t].forEach((id, j) => sideRow(sideTop, part.vals[t][j], null, `${partName}:${id} ${latentName(partName, part.labels, id)}`, () => select(id)));
    }
    renderBottom();
}

function select(id) {
    sel = id;
    renderBottom();
}

// side panel bottom: search the cell's layer by label + the selected latent (value here, set box, card)
function renderBottom() {
    sideBottom.innerHTML = "";
    if (!ctx) {
        const msg = document.createElement("div");
        msg.className = "sae-card-stats";
        msg.textContent = "hover or pin a layer cell to search / edit its latents";
        sideBottom.appendChild(msg);
        return;
    }
    const { set: sn, part: pn, t } = ctx;
    const part = run.hooks[sn].parts[pn];

    const search = document.createElement("input");
    search.type = "text";
    search.className = "sae-search";
    search.placeholder = `search ${pn} labels, Enter`;
    search.value = searchText;
    search.addEventListener("input", () => { searchText = search.value; });
    search.addEventListener("keydown", async (ev) => {
        if (ev.key !== "Enter" || !searchText.trim()) return;
        try {
            results = (await send("sae_search", { set: sn, part: pn, query: searchText })).results;
        } catch (err) {
            showError(err);
            return;
        }
        renderBottom();
    });
    sideBottom.appendChild(search);
    if (results) {
        const box = document.createElement("div");
        box.className = "sae-results";
        if (!results.length) box.textContent = "no label contains all of those words";
        for (const [id, label, score] of results) {
            const r = sideRow(box, score === null ? "-" : score.toFixed(2), "var(--dim)", `${pn}:${id} ${label}`, () => select(id));
            r.title = "label score";
        }
        sideBottom.appendChild(box);
    }

    if (sel === null) return;
    const id = sel;
    const j = part.ids[t].indexOf(id);
    const edit = edits.find(e => e[0] === sn && e[1] === pn && e[2] === t && e[3] === id);
    const head = document.createElement("div");
    head.className = "sae-selected";
    head.textContent = `${pn}:${id} at token ${t} "${show(run.tokens[t])}": ${j >= 0 ? part.vals[t][j] : 0}` + (edit ? ` (edited to ${edit[4]})` : "");
    sideBottom.appendChild(head);

    // set box: Enter or "set" adds / replaces the edit and reruns; "remove" drops it
    const row = document.createElement("div");
    row.className = "sae-set-row";
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.className = "sae-set";
    input.placeholder = "value";
    if (edit) input.value = edit[4];
    const apply = () => {
        const v = parseFloat(input.value);
        if (Number.isNaN(v)) { status("set: type a number"); return; }
        setEdit(sn, pn, t, id, v);
    };
    input.addEventListener("keydown", ev => { if (ev.key === "Enter") apply(); });
    const setBtn = document.createElement("button");
    setBtn.textContent = "set";
    setBtn.addEventListener("click", apply);
    row.append("set to ", input, setBtn);
    if (edit) {
        const rm = document.createElement("button");
        rm.textContent = "remove";
        rm.addEventListener("click", () => setEdit(sn, pn, t, id, null));
        row.appendChild(rm);
    }
    sideBottom.appendChild(row);

    const card = document.createElement("div");
    card.className = "sae-card";
    sideBottom.appendChild(card);
    showCard(card, sn, pn, id);
}

// latent card: label, score, density, example windows with pieces shaded by activation
async function showCard(card, setName, partName, id) {
    let f;
    try {
        f = await send("sae_feature", { set: setName, part: partName, unit: id });
    } catch (err) {
        showError(err);
        return;
    }
    card.innerHTML = "";
    const title = document.createElement("div");
    title.className = "sae-card-title";
    title.textContent = `${partName}:${id} ${f.label ?? "(no label)"}`;
    card.appendChild(title);
    const stats = document.createElement("div");
    stats.className = "sae-card-stats";
    stats.textContent = `score ${f.score === null ? "-" : f.score.toFixed(2)} · fires on ${(100 * f.density).toFixed(3)}% of tokens`;
    card.appendChild(stats);
    // strongest activation over all shown examples, for the shading
    const maxAct = Math.max(1e-6, ...f.examples.flatMap(ex => ex.acts));
    for (const ex of f.examples) {
        const line = document.createElement("div");
        line.className = "sae-example";
        ex.pieces.forEach((piece, j) => {
            const span = document.createElement("span");
            span.textContent = piece.replace(/\n/g, "↵");
            const a = Math.max(0, ex.acts[j]);
            if (a > 0) span.style.background = `color-mix(in srgb, var(--mint) ${Math.round(70 * a / maxAct)}%, transparent)`;
            span.title = String(ex.acts[j]);
            line.appendChild(span);
        });
        card.appendChild(line);
    }
}

export function initSae() {
    // the grid is wide (one column per token) and short (one row per layer): the wheel scrolls sideways,
    // shift + wheel scrolls up/down (chrome reports shift + wheel as deltaX, so take whichever is set)
    grid.addEventListener("wheel", (e) => {
        e.preventDefault();
        const d = e.deltaY || e.deltaX;
        if (e.shiftKey) grid.scrollTop += d;
        else grid.scrollLeft += e.deltaX + e.deltaY;
    }, { passive: false });
    document.getElementById("btn-sae-run").addEventListener("click", () => runSae(0));
    // attribute view toggle: on = wait for a clicked target, off = activations only
    const attrBtn = document.getElementById("btn-sae-attr");
    attrBtn.addEventListener("click", () => {
        view = view === "act" ? "attr" : "act";
        attrBtn.classList.toggle("active", view === "attr");
        attr = null;
        setAttrInfo();
        pinned = null;
        if (!run) return;
        renderGrid();
        if (ctx) showCell(ctx.set, ctx.part, ctx.t);
    });
    document.getElementById("btn-sae-generate").addEventListener("click", () => {
        const n = parseInt(document.getElementById("sae-n").value, 10);
        if (!(n >= 0)) { status("generate: type a number of tokens"); return; }
        runSae(n);
    });
    renderBottom();
}
