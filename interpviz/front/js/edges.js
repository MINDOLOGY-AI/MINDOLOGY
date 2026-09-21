// edges.js -- renders edges through the <board-canvas> api
// each edge is a polyline with 90-degree segments + arrowhead at destination
// (rendered by the component). highlight.js sets edge color on selection,
// resetAllEdges restores the defaults.

const board = document.getElementById("board");

export function clearEdges() {
    board.setEdges([]);
}

export function renderEdges(displayEdges) {
    board.setEdges(displayEdges.map(edge => ({
        key: `${edge.from}->${edge.to}`,
        points: edge.points,
    })));
}

export function highlightEdge(key, color) {
    board.updateEdgeStyle(key, { color, width: 2, opacity: 1 });
}

export function resetAllEdges() {
    board.resetEdgeStyles();
}
