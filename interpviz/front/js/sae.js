// sae.js -- the sae tab: type english, run / generate, see every hook part's latents per token
// grid: one column per token (generated tokens continue to the right), rows top -> bottom:
//   next-token guesses, hook parts in reverse file order (L15 ... L0), the tokens themselves.
// hover a cell = its latents (or next-token guesses) in the side panel; click = pin it there (hover stops
// changing the panel, click again to unpin); click a latent = its card (label, score, density, example windows).

// static circular import — safe for function declarations (see nodes.js)
import { send, showError, status } from "./main.js";

const grid = document.getElementById("sae-grid");
const side = document.getElementById("sae-side");

// last sae_run reply: {tokens: [str], n_prompt: int, next: [[[str, prob] x5] per token],
//   hooks: {set: {display: {max_features, on_cell, max_examples}, parts: {part: {reads, ids: [[int]], vals: [[float]], labels: {id: [label, score]}}}}}}
let run = null;
// the pinned cell element, or null: while set, hovering other cells leaves the side panel alone
let pinned = null;

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
    status(nGenerate ? `generating ${nGenerate}...` : "running...");
    try {
        run = await send("sae_run", { text, template, n_generate: nGenerate });
    } catch (err) {
        showError(err);
        return;
    }
    status("");
    side.innerHTML = "";
    pinned = null;
    document.getElementById("sae-output").value = run.output.replace(/\n/g, "↵");
    renderGrid();
}

function renderGrid() {
    grid.innerHTML = "";
    const T = run.tokens.length;
    grid.style.gridTemplateColumns = `70px repeat(${T}, 130px)`;

    // row: next-token guesses (column t = the model's guess for the token after t)
    addHead("next", "sae-next-row");
    run.next.forEach((guesses, t) => {
        const cell = addCell("sae-next-row");
        cell.textContent = show(guesses[0][0]);
        sideOnHover(cell, () => showGuesses(t));
    });

    // rows: every hook part, last part on top
    for (const [setName, hs] of Object.entries(run.hooks)) {
        for (const [partName, part] of Object.entries(hs.parts).reverse()) {
            addHead(partName, "");
            // strongest latent value in this row, for the cell shading
            const rowMax = Math.max(1e-6, ...part.vals.map(v => v[0] || 0));
            for (let t = 0; t < T; t++) {
                const cell = addCell("sae-feat");
                const ids = part.ids[t], vals = part.vals[t];
                const share = Math.round(40 * (vals[0] || 0) / rowMax);
                cell.style.background = `color-mix(in srgb, var(--mint) ${share}%, transparent)`;
                for (let j = 0; j < Math.min(hs.display.on_cell, ids.length); j++) {
                    const line = document.createElement("div");
                    line.className = "sae-feat-line";
                    line.textContent = latentName(partName, part.labels, ids[j]);
                    cell.appendChild(line);
                }
                sideOnHover(cell, () => showCellLatents(setName, partName, t));
            }
        }
    }

    // row: the tokens, generated ones highlighted
    addHead("tokens", "sae-token-row");
    run.tokens.forEach((tok, t) => {
        const cell = addCell("sae-token-row" + (t >= run.n_prompt ? " generated" : ""));
        cell.textContent = show(tok);
        cell.title = `token ${t}${t >= run.n_prompt ? " (generated)" : ""}`;
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

// side panel: top next-token guesses after token t
function showGuesses(t) {
    side.innerHTML = "";
    const h = document.createElement("h3");
    h.textContent = `next after token ${t} "${show(run.tokens[t])}"`;
    side.appendChild(h);
    for (const [tok, p] of run.next[t]) {
        const r = document.createElement("div");
        r.className = "sae-guess";
        const vEl = document.createElement("span");
        vEl.className = "sae-val";
        vEl.textContent = p.toFixed(3);
        r.appendChild(vEl);
        r.appendChild(document.createTextNode(show(tok)));
        side.appendChild(r);
    }
}

// side panel: every latent of one cell, click one for its card
function showCellLatents(setName, partName, t) {
    const part = run.hooks[setName].parts[partName];
    side.innerHTML = "";
    const h = document.createElement("h3");
    h.textContent = `${partName} · token ${t} "${show(run.tokens[t])}"`;
    side.appendChild(h);
    const card = document.createElement("div");
    card.className = "sae-card";
    side.appendChild(card);
    part.ids[t].forEach((id, j) => {
        const r = document.createElement("div");
        r.className = "sae-latent";
        const vEl = document.createElement("span");
        vEl.className = "sae-val";
        vEl.textContent = part.vals[t][j];
        r.appendChild(vEl);
        r.appendChild(document.createTextNode(`${partName}:${id} ${part.labels[id] ? part.labels[id][0] : "(no label)"}`));
        r.addEventListener("click", () => showCard(card, setName, partName, id));
        side.appendChild(r);
    });
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
    document.getElementById("btn-sae-generate").addEventListener("click", () => {
        const n = parseInt(document.getElementById("sae-n").value, 10);
        if (!(n >= 0)) { status("generate: type a number of tokens"); return; }
        runSae(n);
    });
}
