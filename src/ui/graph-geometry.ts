import type { ComponentRegistry, GraphJSON, GraphNode } from '../engine/graph';

/** Short glyph drawn in the palette and in the node body. Falls back to the localized name. */
export const SYMBOLS: Readonly<Record<string, string>> = {
  input: 'x',
  output: 'y',
  const: 'c',
  param: 'θ',
  add: '+',
  mul: '×',
  max: 'max',
  ge: '≥',
  exp: 'eˣ',
  log: 'ln',
  sum: 'Σ',
  reduce_max: 'max',
  reshape: '⇄',
  unsqueeze: '…×1',
  transpose: 'ᵀ',
  gather: '[ ]',
  arange: '0‥n',
  neg: '−x',
  sub: 'a−b',
  div: 'a÷b',
  tanh: 'tanh',
  relu: 'ReLU',
  sqrt: '√x',
  recip: '1/x',
  sigmoid: 'σ',
  matmul: '@',
  softmax: 'soft',
  layernorm: 'LN',
  mse: 'MSE',
  dsigmoid: 'σ′',
  dtanh: 'tanh′',
  cross_entropy: 'CE',
};

/* Node geometry. Ports, wires and the CSS in theme.css all derive from these numbers. */
export const NODE_W = 150;
export const HEAD_H = 26;
export const BODY_PAD = 8;
export const ROW_H = 24;
export const FOOT_H = 24;

export interface Point {
  x: number;
  y: number;
}

export function portRows(node: GraphNode, registry: ComponentRegistry): number {
  const def = registry.get(node.component);
  if (node.component === 'input') return 1;
  if (node.component === 'output') return 1;
  return Math.max(def?.inputs.length ?? 0, def?.outputs.length ?? 0, 1);
}
export function bodyHeight(node: GraphNode, registry: ComponentRegistry): number {
  return portRows(node, registry) * ROW_H + BODY_PAD * 2;
}
export function nodeHeight(node: GraphNode, registry: ComponentRegistry): number {
  return HEAD_H + bodyHeight(node, registry) + FOOT_H;
}
/** Centre of port row `i`, relative to the node's top edge. */
export function portY(i: number): number {
  return HEAD_H + BODY_PAD + ROW_H * i + ROW_H / 2;
}
/** Single-port sides are centred on the body rather than pinned to the first row. */
export function sidePortY(node: GraphNode, registry: ComponentRegistry, i: number, count: number) {
  if (count === 1) return HEAD_H + bodyHeight(node, registry) / 2;
  return portY(i);
}

/**
 * Positions for every node. Nodes saved with `ui` coordinates keep them; the rest are laid out
 * in columns by longest path from the sources, so solutions and component internals that were
 * never placed by hand still read left to right.
 */
export function layoutGraph(graph: GraphJSON, registry: ComponentRegistry): Map<string, Point> {
  const positions = new Map<string, Point>();
  const placed = (n: GraphNode) => typeof n.ui?.x === 'number' && typeof n.ui?.y === 'number';
  for (const n of graph.nodes)
    if (placed(n)) positions.set(n.id, { x: n.ui!.x as number, y: n.ui!.y as number });
  const free = graph.nodes.filter((n) => !placed(n));
  if (!free.length) return positions;

  const depth = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (id: string): number => {
    if (depth.has(id)) return depth.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const parents = graph.edges.filter((e) => e.to.node === id).map((e) => e.from.node);
    const d = parents.length ? Math.max(...parents.map(depthOf)) + 1 : 0;
    visiting.delete(id);
    depth.set(id, d);
    return d;
  };
  graph.nodes.forEach((n) => depthOf(n.id));
  // Constants and parameters sit one column left of their first consumer, not in column 0.
  for (const n of graph.nodes) {
    if (n.component === 'input' || graph.edges.some((e) => e.to.node === n.id)) continue;
    const consumers = graph.edges
      .filter((e) => e.from.node === n.id)
      .map((e) => depth.get(e.to.node) ?? 1);
    if (consumers.length) depth.set(n.id, Math.max(0, Math.min(...consumers) - 1));
  }
  // Outputs go in the last column so the result always sits on the right.
  const last = Math.max(0, ...depth.values());
  for (const n of graph.nodes) if (n.component === 'output') depth.set(n.id, last);

  const origin = { x: 60, y: 60 };
  const placedBottom = Math.max(
    0,
    ...[...positions.entries()].map(([id, p]) => {
      const n = graph.nodes.find((m) => m.id === id)!;
      return p.y + nodeHeight(n, registry);
    }),
  );
  if (positions.size) origin.y = placedBottom + 60;
  const columns = new Map<number, number>();
  for (const n of free) {
    const d = depth.get(n.id) ?? 0;
    const y = columns.get(d) ?? origin.y;
    positions.set(n.id, { x: origin.x + d * (NODE_W + 70), y });
    columns.set(d, y + nodeHeight(n, registry) + 36);
  }
  return positions;
}

/** Bounding box of all nodes, used by "fit to view". */
export function graphBounds(
  graph: GraphJSON,
  positions: Map<string, Point>,
  registry: ComponentRegistry,
) {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const n of graph.nodes) {
    const p = positions.get(n.id);
    if (!p) continue;
    x0 = Math.min(x0, p.x);
    y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x + NODE_W);
    y1 = Math.max(y1, p.y + nodeHeight(n, registry));
  }
  return Number.isFinite(x0) ? { x0, y0, x1, y1 } : null;
}
