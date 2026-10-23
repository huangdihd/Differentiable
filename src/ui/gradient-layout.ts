import type { BackwardGraph } from '../engine/backward-graph';
import type { GraphJSON } from '../engine/graph';

/** One forward node and, below it, the card for its gradient. */
export interface GradientColumnNode {
  id: string;
  component: string;
  column: number;
  row: number;
  /** On a path from a differentiated input or parameter to the output: the gradient flows through it. */
  carries: boolean;
}
/** A forward wire u→v; its backward twin runs v'→u' when both ends carry gradient. */
export interface GradientEdge {
  from: string;
  to: string;
  port: string;
  backward: boolean;
  /** i18n key of the local derivative used on the way back (v's rule), when known. */
  rule?: string;
}
export interface GradientLayout {
  nodes: GradientColumnNode[];
  edges: GradientEdge[];
  columns: number;
  rows: number;
  /** Nodes whose gradient is a sum of several backward contributions (fan-out in the forward). */
  sums: Record<string, number>;
}

export interface BackwardTeachingSegments {
  /** The supplied upstream-gradient card: the hand-built backward circuit starts here. */
  start: string;
  /** Cards and wires that reproduce the backward circuit the player just built. */
  playerNodes: Set<string>;
  playerEdges: Set<string>;
  /** Cards and wires upstream of `start`, expanded by the engine from the outer forward graph. */
  engineNodes: Set<string>;
  engineEdges: Set<string>;
}

export const gradientEdgeKey = (from: string, to: string, port: string) =>
  `${from}\u0000${to}\u0000${port}`;

/** Use a multiplication node's actual operand names in a local-derivative label. */
export function namedMultiplicationRule(
  graph: GraphJSON,
  edge: GradientEdge,
  label: (node: string) => string,
): { product: string; variable: string; other: string } | undefined {
  if (edge.rule !== 'backward.rule.mul') return undefined;
  const operands = Object.fromEntries(
    graph.edges
      .filter((candidate) => candidate.to.node === edge.to)
      .map((candidate) => [candidate.to.port, label(candidate.from.node)]),
  );
  if (!operands.a || !operands.b || (edge.port !== 'a' && edge.port !== 'b')) return undefined;
  return {
    product: `${operands.a}·${operands.b}`,
    variable: operands[edge.port],
    other: edge.port === 'a' ? operands.b : operands.a,
  };
}

/** Split a reveal's backward path at the gradient supplied to the hand-built local circuit. */
export function backwardTeachingSegments(
  layout: GradientLayout,
  start: string,
): BackwardTeachingSegments | undefined {
  if (!layout.nodes.some((node) => node.id === start && node.carries)) return undefined;

  // In the forward-edge set, descendants of start are the engine's outer step, while
  // ancestors of start are the local backward circuit the player implemented.
  const engineNodes = new Set([start]);
  const playerNodes = new Set([start]);
  for (let changed = true; changed;) {
    changed = false;
    for (const edge of layout.edges) {
      if (!edge.backward) continue;
      if (engineNodes.has(edge.from) && !engineNodes.has(edge.to)) {
        engineNodes.add(edge.to);
        changed = true;
      }
      if (playerNodes.has(edge.to) && !playerNodes.has(edge.from)) {
        playerNodes.add(edge.from);
        changed = true;
      }
    }
  }

  const playerEdges = new Set<string>();
  const engineEdges = new Set<string>();
  for (const edge of layout.edges) {
    if (!edge.backward) continue;
    const key = gradientEdgeKey(edge.from, edge.to, edge.port);
    if (playerNodes.has(edge.from) && playerNodes.has(edge.to)) playerEdges.add(key);
    if (engineNodes.has(edge.from) && engineNodes.has(edge.to)) engineEdges.add(key);
  }
  return { start, playerNodes, playerEdges, engineNodes, engineEdges };
}

/**
 * Forward reads left to right by depth; the gradient row mirrors it underneath, so the same
 * column holds a value and its gradient and backward arrows naturally point right to left.
 */
export function gradientLayout(
  graph: GraphJSON,
  backward: BackwardGraph,
  wrt: readonly string[],
): GradientLayout {
  const ids = graph.nodes.map((n) => n.id);
  const incoming = (id: string) => graph.edges.filter((e) => e.to.node === id);
  const depth = new Map<string, number>();
  const visit = (id: string, stack = new Set<string>()): number => {
    if (depth.has(id)) return depth.get(id)!;
    if (stack.has(id)) return 0;
    stack.add(id);
    const d = Math.max(-1, ...incoming(id).map((e) => visit(e.from.node, stack))) + 1;
    depth.set(id, d);
    return d;
  };
  ids.forEach((id) => visit(id));
  // Constants hug their consumer instead of sitting in column 0 with the inputs.
  for (const n of graph.nodes)
    if (n.component === 'const' || n.component === 'param') {
      const users = graph.edges
        .filter((e) => e.from.node === n.id)
        .map((e) => depth.get(e.to.node)!);
      if (users.length) depth.set(n.id, Math.max(0, Math.min(...users) - 1));
    }

  // Gradient flows where a differentiated input or parameter reaches the output.
  const output = graph.nodes.find((n) => n.component === 'output');
  const downstream = new Set(
    graph.nodes
      .filter(
        (n) =>
          (n.component === 'input' && wrt.includes(String(n.attrs?.name))) ||
          (n.component === 'param' && wrt.includes(n.id)),
      )
      .map((n) => n.id),
  );
  for (let changed = true; changed;) {
    changed = false;
    for (const e of graph.edges)
      if (downstream.has(e.from.node) && !downstream.has(e.to.node)) {
        downstream.add(e.to.node);
        changed = true;
      }
  }
  const upstream = new Set(output ? [output.id] : []);
  for (let changed = true; changed;) {
    changed = false;
    for (const e of graph.edges)
      if (upstream.has(e.to.node) && !upstream.has(e.from.node)) {
        upstream.add(e.from.node);
        changed = true;
      }
  }
  const carries = (id: string) => downstream.has(id) && upstream.has(id);

  const columns = Math.max(0, ...depth.values()) + 1;
  const byColumn = new Map<number, string[]>();
  const y = (id: string) => Number(graph.nodes.find((n) => n.id === id)?.ui?.y ?? 0);
  for (const id of ids) byColumn.set(depth.get(id)!, [...(byColumn.get(depth.get(id)!) ?? []), id]);
  const row = new Map<string, number>();
  for (const list of byColumn.values())
    list.sort((a, b) => y(a) - y(b)).forEach((id, i) => row.set(id, i));

  const rules = new Map<string, string>();
  for (const role of Object.values(backward.roles))
    if (role.role === 'backward' && role.mirrorOf && role.rule && !rules.has(role.mirrorOf))
      rules.set(role.mirrorOf, role.rule);

  const edges = graph.edges.map((e) => {
    const backwardEdge = carries(e.from.node) && carries(e.to.node);
    const consumer = graph.nodes.find((n) => n.id === e.to.node)!;
    return {
      from: e.from.node,
      to: e.to.node,
      port: e.to.port,
      backward: backwardEdge,
      rule: backwardEdge && consumer.component !== 'output' ? rules.get(e.to.node) : undefined,
    };
  });
  const sums: Record<string, number> = {};
  for (const e of edges) if (e.backward) sums[e.from] = (sums[e.from] ?? 0) + 1;
  for (const id of Object.keys(sums)) if (sums[id] < 2) delete sums[id];

  return {
    nodes: graph.nodes.map((n) => ({
      id: n.id,
      component: n.component,
      column: depth.get(n.id)!,
      row: row.get(n.id)!,
      carries: carries(n.id),
    })),
    edges,
    columns,
    rows: Math.max(1, ...[...byColumn.values()].map((l) => l.length)),
    sums,
  };
}
