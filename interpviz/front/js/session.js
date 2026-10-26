// session.js -- the session every tab and every open window shares (interpviz/DOC.md, session): one prompt, one edits
// list, one last run, held by the server. the chat bar (method, prompt, run / generate, output) and the edit chips sit
// above both tabs; a run from any window is pushed to the others, which redraw from it.
// edit panel (graph tab, "edits" button): look a neuron up by layer + number (its value per token of the run, the tokens
// its write pushes the logits toward / away from) and set it at one token or all; every edit of the session; presets =
// named edit lists saved in the meta, loaded onto any prompt.
// edits (wire format, the same in the server's session and the meta's presets):
//   {kind: "feature", set, part, pos, id, value, label}  set an SAE / crosscoder latent to value
//   {kind: "steer",   set, part, pos, id, value, label}  add value × the latent's write (a crosscoder: into every layer it writes to)
//   {kind: "neuron",  layer, unit, pos, value}           set an mlp neuron to value
//   {kind: "neuron_add", layer, unit, pos, value}        add value to an mlp neuron (= value × its write)
// pos = a token index or "all" (every token, generated ones included). a new prompt keeps only the "all" edits; another
// method keeps only the neuron edits and that method's latents.

// static circular import — safe for function declarations (see nodes.js)
import { send, showError, status } from "./main.js";

// {query: {set, text, template, n_generate}, edits: [edit], run: {tokens, n_prompt, output, next, hooks}} or null before
// the first run
export let session = null;
// {name: [edit]} the meta's presets
let presets = {};
// [fn(prev session, session)] called on every new session
const listeners = [];
// {layer, unit} the neuron shown in the edit panel, or null; its last sae_neuron reply {layer, unit, acts, logits}
let neuron = null;
let neuronCard = null;

const setSelect = document.getElementById("sae-set");
const panel = document.getElementById("edit-panel");

// token text with visible whitespace
export function show(t) {
    return t.replace(/ /g, "␣").replace(/\n/g, "↵");
}

export function onSession(fn) {
    listeners.push(fn);
}

// fill the method dropdown from the meta's hook set files ("hooks/sae_resid_topk.json" -> "sae_resid_topk"), plus none
export function setHookSets(files) {
    setSelect.innerHTML = "";
    for (const name of [...files.map(f => f.split("/").pop().replace(/\.json$/, "")), "none"]) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        setSelect.appendChild(opt);
    }
}

export function setPresets(p) {
    presets = p;
    renderPanel();
}

// show a session from the server (this window's run, another window's push, or get_state): the chat bar follows it
export function applySession(s) {
    const prev = session;
    session = s;
    if (s) {
        setSelect.value = s.query.set ?? "none";
        document.getElementById("sae-text").value = s.query.text;
        document.getElementById("sae-template").checked = s.query.template;
        document.getElementById("sae-output").value = s.run.output.replace(/\n/g, "↵");
    } else {
        document.getElementById("sae-output").value = "";
    }
    renderChips();
    listeners.forEach(fn => fn(prev, s));
    // the shown neuron's values belong to the old run
    if (s && neuron) showNeuron(neuron.layer, neuron.unit);
    else renderPanel();
}

// one run: the query (missing fields from the chat bar) with edits (default: the session's, minus the ones the new
// prompt / method can't keep)
async function run(query, edits) {
    const q = {
        set: setSelect.value === "none" ? null : setSelect.value,
        text: document.getElementById("sae-text").value,
        template: document.getElementById("sae-template").checked,
        n_generate: 0,
        ...query,
    };
    if (edits === undefined) {
        edits = session ? session.edits : [];
        // token positions point into one prompt
        if (session && (session.query.text !== q.text || session.query.template !== q.template)) edits = edits.filter(e => e.pos === "all");
        // latents belong to one method
        edits = edits.filter(e => isNeuron(e) || e.set === q.set);
    }
    status(q.n_generate ? `generating ${q.n_generate}...` : "running...");
    let s;
    try {
        s = await send("sae_run", { ...q, edits });
    } catch (err) {
        showError(err);
        return;
    }
    status("");
    applySession({ query: s.query, edits: s.edits, run: s.run });
}

// rerun the session's prompt with new edits
export function setEdits(edits) {
    if (!session) { status("run a prompt first"); return; }
    run(session.query, edits);
}

export const isNeuron = e => e.kind === "neuron" || e.kind === "neuron_add";

// two edits on the same thing (same kind, latent / neuron and token): one replaces the other
function sameTarget(a, b) {
    if (a.kind !== b.kind || a.pos !== b.pos) return false;
    if (isNeuron(a)) return a.layer === b.layer && a.unit === b.unit;
    return a.set === b.set && a.part === b.part && a.id === b.id;
}

// add an edit (replacing one on the same target) and rerun
export function putEdit(e) {
    setEdits([...session.edits.filter(x => !sameTarget(x, e)), e]);
}

export function removeEdit(e) {
    setEdits(session.edits.filter(x => x !== e));
}

// "@5 "tok"" or "@all"
function where(pos) {
    if (pos === "all") return "@all";
    return `@${pos} "${show(session.run.tokens[pos] ?? "?")}"`;
}

function editText(e) {
    if (isNeuron(e)) return `n${e.layer}:${e.unit} ${where(e.pos)} ${e.kind === "neuron_add" ? "+=" : "="} ${e.value}`;
    return `${e.part}:${e.id} ${where(e.pos)} ${e.kind === "steer" ? "+=" : "="} ${e.value}`;
}

// one chip per edit, × removes it
function editChip(e) {
    const chip = document.createElement("span");
    chip.className = "sae-edit-chip" + (e.kind === "steer" ? " steer" : isNeuron(e) ? " neuron" : "");
    chip.title = isNeuron(e) ? `mlp neuron ${e.unit} of layer ${e.layer}` : `${e.kind} ${e.set}: ${e.label ?? "(no label)"}`;
    chip.textContent = editText(e) + " ";
    const x = document.createElement("span");
    x.className = "sae-edit-x";
    x.textContent = "×";
    x.addEventListener("click", () => removeEdit(e));
    chip.appendChild(x);
    return chip;
}

function renderChips() {
    const box = document.getElementById("sae-edits");
    box.innerHTML = "";
    if (!session) return;
    for (const e of session.edits) box.appendChild(editChip(e));
}

async function showNeuron(layer, unit) {
    neuron = { layer, unit };
    try {
        neuronCard = await send("sae_neuron", { layer, unit });
    } catch (err) {
        neuronCard = null;
        showError(err);
    }
    renderPanel();
}

// a number input with a placeholder
function numInput(placeholder, value, width) {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.className = "sae-set";
    input.style.width = width;
    input.placeholder = placeholder;
    if (value !== null) input.value = value;
    return input;
}

// a token scope dropdown: "all" or one token of the run
export function tokenSelect(selected) {
    const sel = document.createElement("select");
    sel.className = "sae-scope";
    const opts = [["all", "all tokens"], ...session.run.tokens.map((t, i) => [String(i), `${i} "${show(t)}"`])];
    for (const [v, text] of opts) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = text;
        sel.appendChild(o);
    }
    sel.value = String(selected);
    return sel;
}

function section(title) {
    const h = document.createElement("h3");
    h.textContent = title;
    panel.appendChild(h);
}

function renderPanel() {
    panel.innerHTML = "";
    const close = document.createElement("span");
    close.className = "panel-close";
    close.textContent = "×";
    close.addEventListener("click", () => { panel.style.display = "none"; });
    panel.appendChild(close);

    // neuron lookup: layer + number, Enter or show
    section("neuron");
    const row = document.createElement("div");
    row.className = "sae-set-row";
    const layerIn = numInput("layer", neuron ? neuron.layer : null, "60px");
    const unitIn = numInput("neuron", neuron ? neuron.unit : null, "80px");
    const go = () => {
        const layer = parseInt(layerIn.value, 10), unit = parseInt(unitIn.value, 10);
        if (Number.isNaN(layer) || Number.isNaN(unit)) { status("neuron: type a layer and a neuron number"); return; }
        if (!session) { status("run a prompt first"); return; }
        showNeuron(layer, unit);
    };
    for (const input of [layerIn, unitIn]) input.addEventListener("keydown", ev => { if (ev.key === "Enter") go(); });
    const goBtn = document.createElement("button");
    goBtn.textContent = "show";
    goBtn.addEventListener("click", go);
    row.append("layer ", layerIn, " neuron ", unitIn, goBtn);
    panel.appendChild(row);

    // quick entry, at every token: "L24 N3102 +24 L9 N9009 -5" adds, "L20 N1234 =30" sets; Enter or apply
    const quickRow = document.createElement("div");
    quickRow.className = "sae-set-row";
    const quick = document.createElement("input");
    quick.type = "text";
    quick.className = "sae-search";
    quick.style.marginBottom = "0";
    quick.placeholder = "L24 N3102 +24 L9 N9009 +24";
    const applyQuick = () => {
        // [[_, layer, unit, op, value]] one per "L<layer> N<unit> <+v | -v | =v>"
        const found = [...quick.value.matchAll(/L\s*(\d+)\s*N\s*(\d+)\s*([+=-])\s*(\d*\.?\d+)/gi)];
        if (!found.length) { status("quick: type e.g. L24 N3102 +24"); return; }
        if (!session) { status("run a prompt first"); return; }
        const added = found.map(([, l, u, op, v]) => ({ kind: op === "=" ? "neuron" : "neuron_add", layer: parseInt(l, 10),
            unit: parseInt(u, 10), pos: "all", value: (op === "-" ? -1 : 1) * parseFloat(v) }));
        setEdits([...session.edits.filter(x => !added.some(e => sameTarget(x, e))), ...added]);
    };
    quick.addEventListener("keydown", ev => { if (ev.key === "Enter") applyQuick(); });
    const quickBtn = document.createElement("button");
    quickBtn.textContent = "apply";
    quickBtn.addEventListener("click", applyQuick);
    quickRow.append(quick, quickBtn);
    panel.appendChild(quickRow);
    if (neuronCard && session) renderNeuronCard();

    // every edit of the session
    section("edits");
    const list = document.createElement("div");
    list.className = "edit-list";
    if (!session || !session.edits.length) list.textContent = "none";
    else for (const e of session.edits) list.appendChild(editChip(e));
    panel.appendChild(list);
    if (session && session.edits.length) {
        const clear = document.createElement("button");
        clear.textContent = "clear all";
        clear.addEventListener("click", () => setEdits([]));
        panel.appendChild(clear);
    }

    // presets: save the session's edits under a name; load = replace the session's edits with a preset's and rerun
    section("presets");
    const saveRow = document.createElement("div");
    saveRow.className = "sae-set-row";
    const name = document.createElement("input");
    name.type = "text";
    name.className = "sae-search";
    name.style.marginBottom = "0";
    name.placeholder = "name";
    const save = async () => {
        if (!name.value.trim()) { status("preset: type a name"); return; }
        if (!session || !session.edits.length) { status("preset: no edits to save"); return; }
        try {
            setPresets((await send("save_preset", { name: name.value, edits: session.edits })).presets);
        } catch (err) { showError(err); }
    };
    name.addEventListener("keydown", ev => { if (ev.key === "Enter") save(); });
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "save edits";
    saveBtn.addEventListener("click", save);
    saveRow.append(name, saveBtn);
    panel.appendChild(saveRow);
    for (const [pname, pedits] of Object.entries(presets)) {
        const r = document.createElement("div");
        r.className = "group-row";
        const label = document.createElement("span");
        label.className = "preset-name";
        label.textContent = `${pname} (${pedits.length})`;
        label.title = pedits.map(editText).join("\n");
        const load = document.createElement("button");
        load.textContent = "load";
        load.addEventListener("click", () => setEdits(pedits));
        const del = document.createElement("button");
        del.textContent = "×";
        del.addEventListener("click", async () => {
            try {
                setPresets((await send("delete_preset", { name: pname })).presets);
            } catch (err) { showError(err); }
        });
        r.append(label, load, del);
        panel.appendChild(r);
    }
}

// the shown neuron: its value per token (shaded), the set row, and the tokens its write pushes the logits toward / away from
function renderNeuronCard() {
    const { layer, unit, acts, logits } = neuronCard;
    const card = document.createElement("div");
    card.className = "sae-card";
    const title = document.createElement("div");
    title.className = "sae-card-title";
    title.textContent = `n${layer}:${unit} value per token`;
    card.appendChild(title);
    // strongest |value| over the run, for the shading
    const maxAct = Math.max(1e-6, ...acts.map(Math.abs));
    const line = document.createElement("div");
    line.className = "sae-example";
    acts.forEach((a, t) => {
        const span = document.createElement("span");
        span.textContent = session.run.tokens[t].replace(/\n/g, "↵");
        const p = Math.round(70 * Math.abs(a) / maxAct);
        span.style.background = `color-mix(in srgb, ${a > 0 ? "var(--attr-pos)" : "var(--attr-neg)"} ${p}%, transparent)`;
        span.title = `token ${t}: ${a}`;
        line.appendChild(span);
    });
    card.appendChild(line);
    const stats = document.createElement("div");
    stats.className = "sae-card-stats";
    stats.textContent = `max ${Math.max(...acts).toFixed(2)} · min ${Math.min(...acts).toFixed(2)} · last token ${acts[acts.length - 1]}`;
    card.appendChild(stats);

    // edit row: token scope + value, set (Enter) or add
    const row = document.createElement("div");
    row.className = "sae-set-row";
    const scope = tokenSelect("all");
    const value = numInput("value", null, "70px");
    const apply = kind => {
        const v = parseFloat(value.value);
        if (Number.isNaN(v)) { status(`${kind === "neuron" ? "set" : "add"}: type a number`); return; }
        putEdit({ kind, layer, unit, pos: scope.value === "all" ? "all" : parseInt(scope.value, 10), value: v });
    };
    value.addEventListener("keydown", ev => { if (ev.key === "Enter") apply("neuron"); });
    const setBtn = document.createElement("button");
    setBtn.textContent = "set";
    setBtn.addEventListener("click", () => apply("neuron"));
    const addBtn = document.createElement("button");
    addBtn.textContent = "add";
    addBtn.title = "add value to the neuron (= value × its write into the residual stream)";
    addBtn.addEventListener("click", () => apply("neuron_add"));
    row.append(scope, " value ", value, setBtn, addBtn);
    card.appendChild(row);

    if (logits) {
        for (const [dir, list, color] of [["pushes toward", logits.up, "var(--attr-pos)"], ["pushes away from", logits.down, "var(--attr-neg)"]]) {
            const r = document.createElement("div");
            r.className = "sae-logits";
            const head = document.createElement("span");
            head.className = "sae-logits-head";
            head.style.color = color;
            head.textContent = dir;
            r.appendChild(head);
            for (const [tok, v] of list) {
                const chip = document.createElement("span");
                chip.className = "sae-logit";
                chip.textContent = show(tok);
                chip.title = String(v);
                r.appendChild(chip);
            }
            card.appendChild(r);
        }
    }
    panel.appendChild(card);
}

export function hideEditPanel() {
    panel.style.display = "none";
}

export function initSession() {
    document.getElementById("btn-sae-run").addEventListener("click", () => run({}));
    document.getElementById("btn-sae-generate").addEventListener("click", () => {
        const n = parseInt(document.getElementById("sae-n").value, 10);
        if (!(n >= 0)) { status("generate: type a number of tokens"); return; }
        run({ n_generate: n });
    });
    // another method: rerun the session's prompt with it (only its latents' edits and the neuron edits survive)
    setSelect.addEventListener("change", () => {
        if (session) run({ ...session.query, set: setSelect.value === "none" ? null : setSelect.value });
    });
    document.getElementById("btn-edits").addEventListener("click", () => {
        panel.style.display = panel.style.display === "none" ? "" : "none";
        renderPanel();
    });
    renderPanel();
}
