// <board-canvas> — shared infinite 2D board (pan + zoom + grid + svg edges).
// extracted from the shared interaction model of TH0UGHTGR4PH's board and the
// old intvrpviz frontend, so any graph/board app can reuse it.
//
// layout created inside the element (light DOM — app css/theming applies):
//   .bc-viewport    clip area; owns the grid background (moves/scales with the transform)
//     .bc-world     transformed layer: translate(panX, panY) scale(zoom)
//       svg.bc-edges  edge layer (below)
//       div.bc-nodes  node layer (above) — the app positions its own node
//                     elements absolutely inside it (world coordinates)
//
// api:
//   nodeLayer                                  the .bc-nodes div (append node elements here)
//   setEdges(edges)                            replace all edges; edges =
//     [{key, points: [[x,y],...], color?, width?, opacity?, arrow?}]
//     renders svg polylines (+ arrowhead at destination unless arrow: false)
//   updateEdgeStyle(key, {color?, width?, opacity?})   restyle one edge (highlight)
//   resetEdgeStyles()                          restore every edge to its setEdges style
//   centerView(bbox)                           frame {minX, minY, maxX, maxY} (resets zoom to 1)
//   screenToWorld(clientX, clientY)            client px -> world coords {x, y}
//
// events:
//   board-tap — CustomEvent(detail = world coords {x, y}) fired when empty board
//               space is clicked without dragging (apps use it to clear
//               selection/highlight)

const GRID = 100;      // grid cell size in world px
const MIN_ZOOM = 0.05;
const MAX_ZOOM = 4;

class BoardCanvas extends HTMLElement {
    connectedCallback() {
        this.panX = 0;
        this.panY = 0;
        this.zoom = 1;
        // {key: {color, width, opacity}} — base styles from the last setEdges
        this.edgeBase = {};

        this.innerHTML = `
            <div class="bc-viewport">
                <div class="bc-world">
                    <svg class="bc-edges"></svg>
                    <div class="bc-nodes"></div>
                </div>
            </div>`;
        this.viewport = this.querySelector(".bc-viewport");
        this.world = this.querySelector(".bc-world");
        this.edgeLayer = this.querySelector(".bc-edges");
        this.nodeLayer = this.querySelector(".bc-nodes");

        this.updateTransform();

        let panning = false;
        let moved = false;
        let last = { x: 0, y: 0 };

        this.viewport.addEventListener("mousedown", (e) => {
            // only pan if pressing empty board/svg area, not nodes or popups
            if (e.target !== this.viewport && e.target !== this.world &&
                e.target.tagName !== "svg" && e.target.tagName !== "SVG") return;
            panning = true;
            moved = false;
            last = { x: e.clientX, y: e.clientY };
            e.preventDefault();
        });

        window.addEventListener("mousemove", (e) => {
            if (!panning) return;
            if (Math.abs(e.clientX - last.x) + Math.abs(e.clientY - last.y) > 3) moved = true;
            this.panX += e.clientX - last.x;
            this.panY += e.clientY - last.y;
            last = { x: e.clientX, y: e.clientY };
            this.updateTransform();
        });

        window.addEventListener("mouseup", (e) => {
            if (!panning) return;
            panning = false;
            // press + release on empty space without dragging -> tap
            if (!moved) {
                this.dispatchEvent(new CustomEvent("board-tap", {
                    detail: this.screenToWorld(e.clientX, e.clientY),
                }));
            }
        });

        this.viewport.addEventListener("wheel", (e) => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoomAt(factor, e.clientX, e.clientY);
        }, { passive: false });
    }

    // cursor-anchored zoom: the world point under the cursor stays fixed
    zoomAt(factor, clientX, clientY) {
        const rect = this.viewport.getBoundingClientRect();
        const cx = clientX - rect.left;
        const cy = clientY - rect.top;
        const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.zoom * factor));
        const f = newZoom / this.zoom;
        this.panX = cx - (cx - this.panX) * f;
        this.panY = cy - (cy - this.panY) * f;
        this.zoom = newZoom;
        this.updateTransform();
    }

    updateTransform() {
        this.world.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        // grid is a css background on the viewport: scale the cell, shift with pan
        const cell = GRID * this.zoom;
        this.viewport.style.backgroundSize = `${cell}px ${cell}px`;
        this.viewport.style.backgroundPosition = `${this.panX}px ${this.panY}px`;
    }

    screenToWorld(clientX, clientY) {
        const rect = this.viewport.getBoundingClientRect();
        return {
            x: (clientX - rect.left - this.panX) / this.zoom,
            y: (clientY - rect.top - this.panY) / this.zoom,
        };
    }

    centerView(bbox) {
        const cx = (bbox.minX + bbox.maxX) / 2;
        const cy = (bbox.minY + bbox.maxY) / 2;
        const rect = this.viewport.getBoundingClientRect();
        this.zoom = 1;
        this.panX = rect.width / 2 - cx * this.zoom;
        this.panY = rect.height / 2 - cy * this.zoom;
        this.updateTransform();
    }

    // edges = [{key, points: [[x,y],...], color?, width?, opacity?, arrow?}]
    setEdges(edges) {
        this.edgeLayer.innerHTML = "";
        this.edgeBase = {};
        for (const edge of edges) {
            if (!edge.points || edge.points.length < 2) continue;

            const base = {
                color: edge.color || "#ffffff",
                width: edge.width ?? 1.2,
                opacity: edge.opacity ?? 0.35,
            };
            this.edgeBase[edge.key] = base;

            const points = edge.points.map(([x, y]) => `${x},${y}`).join(" ");
            const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
            polyline.setAttribute("points", points);
            polyline.setAttribute("fill", "none");
            polyline.dataset.edgeKey = edge.key;
            this.edgeLayer.appendChild(polyline);

            // arrowhead pointing in the direction of the last segment
            if (edge.arrow !== false) {
                const [px, py] = edge.points[edge.points.length - 2];
                const [ex, ey] = edge.points[edge.points.length - 1];
                this.drawArrow(px, py, ex, ey, edge.key);
            }
            this.applyEdgeStyle(edge.key, base);
        }
    }

    updateEdgeStyle(key, style) {
        if (!this.edgeBase[key]) return;
        this.applyEdgeStyle(key, { ...this.edgeBase[key], ...style });
    }

    resetEdgeStyles() {
        for (const key of Object.keys(this.edgeBase)) {
            this.applyEdgeStyle(key, this.edgeBase[key]);
        }
    }

    applyEdgeStyle(key, style) {
        const el = this.edgeLayer.querySelector(`polyline[data-edge-key="${CSS.escape(key)}"]`);
        if (el) {
            el.setAttribute("stroke", style.color);
            el.setAttribute("stroke-width", style.width);
            el.setAttribute("opacity", style.opacity);
        }
        const arrow = this.edgeLayer.querySelector(`.edge-arrow[data-edge-key="${CSS.escape(key)}"]`);
        if (arrow) {
            arrow.setAttribute("fill", style.color);
            arrow.setAttribute("opacity", style.opacity);
        }
    }

    drawArrow(fromX, fromY, toX, toY, key) {
        // small triangle at (toX, toY) pointing in the direction of travel
        const size = 4;
        const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        arrow.classList.add("edge-arrow");

        if (Math.abs(toX - fromX) < 1) {
            // vertical segment
            const yOff = toY < fromY ? size * 1.5 : -size * 1.5;
            arrow.setAttribute("points",
                `${toX},${toY} ${toX - size},${toY + yOff} ${toX + size},${toY + yOff}`);
        } else {
            // horizontal segment
            const xOff = toX > fromX ? -size * 1.5 : size * 1.5;
            arrow.setAttribute("points",
                `${toX},${toY} ${toX + xOff},${toY - size} ${toX + xOff},${toY + size}`);
        }
        arrow.dataset.edgeKey = key;
        this.edgeLayer.appendChild(arrow);
    }
}

customElements.define("board-canvas", BoardCanvas);
