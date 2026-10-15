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
