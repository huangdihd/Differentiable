import type { ComponentRegistry, GraphJSON, GraphNode } from '../engine/graph';

/** Short glyph drawn in the palette and in the node body. Falls back to the localized name. */
export const SYMBOLS: Readonly<Record<string, string>> = {
  input: 'x',
  output: 'y',
  const: 'c',
  param: 'θ',
  add: 'a+b',
  mul: 'a×b',
  max: 'max',
  ge: '≥',
  exp: 'eˣ',
  log: 'ln',
  sum: 'Σ',
  reduce_max: 'max',
  reshape: '⇄',
  size: '|x|ₖ',
  unsqueeze: '…×1',
  transpose: 'ᵀ',
  gather: '[ ]',
  scatter_add: 'Σ[ ]',
  arange: '0‥n',
  neg: '−x',
  sub: 'a−b',
  div: 'a÷b',
  tanh: 'tanh',
  relu: 'ReLU',
  sqrt: '√x',
  square: 'x²',
  recip: '1/x',
  sigmoid: 'σ',
  matmul: '@',
  linear: 'xW+b',
  softmax: 'soft',
  layernorm: 'LN',
  mse: 'MSE',
  dsigmoid: 'σ′',
  dtanh: 'tanh′',
  cross_entropy: 'CE',
  dot: 'a·b',
  outer: 'a⊗b',
  mean: 'x̄',
  one_hot: '1-hot',
  embedding: 'E[i]',
  sample: 'draw',
  attn_scores: 'qkᵀ',
  causal_mask: '◣',
  attention: 'attn',
  embed_pos: 'E+P',
  multi_head: 'attn×2',
  feed_forward: 'FFN',
  transformer_block: 'block',
};

/**
 * Families of components, each with its own colour strip on palette tiles and node headers, so
 * a circuit shows at a glance where the arithmetic, activations, tensor plumbing, derivatives,
 * learning parts and attention sit. Inputs, outputs and unknown names have no family.
 */
export type Category = 'op' | 'act' | 'tensor' | 'grad' | 'learn' | 'attn';
const FAMILIES: Record<Category, readonly string[]> = {
  op: [
    'const',
    'add',
    'mul',
    'max',
    'ge',
    'exp',
    'log',
    'neg',
    'sub',
    'div',
    'sqrt',
    'square',
    'recip',
  ],
  act: ['sigmoid', 'tanh', 'relu', 'softmax'],
  tensor: [
    'sum',
    'reduce_max',
    'reshape',
    'size',
    'unsqueeze',
    'transpose',
    'gather',
    'scatter_add',
    'arange',
    'matmul',
    'dot',
    'outer',
    'mean',
    'layernorm',
    'one_hot',
    'embedding',
  ],
  grad: ['dsigmoid', 'dtanh'],
  learn: ['param', 'linear', 'mse', 'cross_entropy'],
  attn: [
    'sample',
    'attn_scores',
    'causal_mask',
    'attention',
    'embed_pos',
    'multi_head',
    'feed_forward',
    'transformer_block',
  ],
};
const CATEGORY = new Map(
  Object.entries(FAMILIES).flatMap(([cat, names]) => names.map((n) => [n, cat as Category])),
);
export const categoryOf = (component: string): Category | undefined => CATEGORY.get(component);

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

/** Reshape exposes only dimensions present in its target shape (up to four wired axes). */
export function nodePorts(
  node: GraphNode,
  registry: ComponentRegistry,
  side: 'inputs' | 'outputs',
): readonly string[] {
  if (node.component === 'input') return side === 'outputs' ? ['out'] : [];
  if (node.component === 'output') return side === 'inputs' ? ['x'] : [];
  const ports = registry.get(node.component)?.[side] ?? [];
  if (node.component !== 'reshape' || side !== 'inputs') return ports;
  const shape = node.attrs?.shape ?? [-1];
  const rank = Array.isArray(shape) ? shape.length : 0;
  return ports.filter((port) => !/^d[0-3]$/.test(port) || Number(port.slice(1)) < rank);
}

export function portRows(node: GraphNode, registry: ComponentRegistry): number {
  return Math.max(
    nodePorts(node, registry, 'inputs').length,
    nodePorts(node, registry, 'outputs').length,
    1,
  );
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

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Room left around a placed node for its value notes and for wires to pass between nodes. */
const PLACE_GAP = 24;

/**
 * Where a clicked-in node of size `w`×`h` should go: `want` if nothing is there, otherwise the
 * nearest free spot found by walking outward in rows below and columns beside `want`. Without
 * this, clicking the palette twice stacks the second node exactly over the first.
 */
export function freeSpot(want: Point, w: number, h: number, occupied: readonly Rect[]): Point {
  const clear = (p: Point) =>
    occupied.every(
      (r) =>
        p.x + w + PLACE_GAP <= r.x ||
        r.x + r.w + PLACE_GAP <= p.x ||
        p.y + h + PLACE_GAP <= r.y ||
        r.y + r.h + PLACE_GAP <= p.y,
    );
  const stepX = NODE_W + 40,
    stepY = 60;
  for (let ring = 0; ring <= 12; ring++) {
    const candidates: Point[] = [];
    for (let dx = -ring; dx <= ring; dx++)
      for (let dy = -ring; dy <= ring; dy++)
        if (Math.max(Math.abs(dx), Math.abs(dy)) === ring)
          candidates.push({ x: want.x + dx * stepX, y: want.y + dy * stepY });
    // Nearest first; at equal distance prefer spots below, where a chain usually has room.
    const cost = (p: Point) =>
      Math.hypot((p.x - want.x) / stepX, (p.y - want.y) / stepY) - (p.y > want.y ? 0.1 : 0);
    candidates.sort((a, b) => cost(a) - cost(b));
    const spot = candidates.find(clear);
    if (spot) return spot;
  }
  return want;
}

/**
 * The input port nearest to `at` within `radius` (graph units), for snapping a wire being dragged.
 * Free ports win over occupied ones at similar distance, so a drop near a half-wired node lands
 * on its empty input.
 */
export function nearestInput(
  graph: GraphJSON,
  registry: ComponentRegistry,
  positions: Map<string, Point>,
  at: Point,
  radius: number,
  skip?: string,
): { node: string; port: string; point: Point } | undefined {
  let best: { node: string; port: string; point: Point } | undefined;
  let bestScore = Infinity;
  for (const n of graph.nodes) {
    if (n.id === skip) continue;
    const p = positions.get(n.id);
    if (!p) continue;
    const list = nodePorts(n, registry, 'inputs');
    list.forEach((port, i) => {
      const point = { x: p.x, y: p.y + sidePortY(n, registry, i, list.length) };
      const d = Math.hypot(point.x - at.x, point.y - at.y);
      if (d > radius) return;
      const taken = graph.edges.some((e) => e.to.node === n.id && e.to.port === port);
      const score = d + (taken ? radius / 3 : 0);
      if (score < bestScore) {
        bestScore = score;
        best = { node: n.id, port, point };
      }
    });
  }
  return best;
}
