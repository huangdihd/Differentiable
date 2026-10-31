// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { GraphEdge, GraphJSON, GraphNode } from '../engine/graph';
import type { CircuitLevel } from '../engine/levels';

export interface LockMark {
  /** Input ports of the locked node whose wires are fixed. */
  ports: string[];
}
export const lockMark = (node: GraphNode | undefined): LockMark | undefined => {
  const mark = node?.ui?.locked as LockMark | undefined;
  return mark && Array.isArray(mark.ports) ? mark : undefined;
};
export const isLockedNode = (node: GraphNode | undefined) => !!lockMark(node);

/** Locked wires: the target into the loss, and the loss into its output. */
export function isLockedEdge(graph: GraphJSON, edge: GraphEdge): boolean {
  const from = graph.nodes.find((n) => n.id === edge.from.node);
  const to = graph.nodes.find((n) => n.id === edge.to.node);
  if (isLockedNode(from) && to?.component === 'output') return true;
  return !!lockMark(to)?.ports.includes(edge.to.port);
}

const sameAttrs = (a: unknown = {}, b: unknown = {}) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Seed or repair the level's locked loss in a draft. Never removes player nodes: a conflicting
 * player loss is adopted when it already matches, otherwise it stays beside the locked one.
 */
export function ensureLockedLoss(level: CircuitLevel, graph: GraphJSON): GraphJSON {
  if (level.kind !== 'train' || !level.lockedLoss) return graph;
  const lock = level.lockedLoss;
  const outputName = level.lossOutput ?? level.interface.output;
  const output = graph.nodes.find(
    (n) => n.component === 'output' && (n.attrs?.name ?? 'out') === outputName,
  );
  const target = graph.nodes.find(
    (n) => n.component === 'input' && (n.attrs?.name ?? '') === lock.target,
  );
  if (!output || !target) return graph;
  const nodes = graph.nodes.map((n) => ({ ...n }));
  let loss = nodes.find((n) => isLockedNode(n) && n.component === lock.component);
  // Drop stale marks (e.g. a differently configured old lock) so exactly one node is locked.
  for (const n of nodes)
    if (n !== loss && isLockedNode(n)) {
      const { locked: _locked, ...ui } = n.ui ?? {};
      n.ui = ui;
    }
  if (!loss) {
    // Adopt the player's own matching loss that already feeds the output.
    const feeding = graph.edges.find((e) => e.to.node === output.id)?.from.node;
    loss = nodes.find(
      (n) =>
        n.id === feeding && n.component === lock.component && sameAttrs(n.attrs, lock.attrs ?? {}),
    );
  }
  if (!loss) {
    let id = 'locked_loss';
    for (let i = 2; nodes.some((n) => n.id === id); i++) id = `locked_loss_${i}`;
    const ox = typeof output.ui?.x === 'number' ? output.ui.x : 660;
    const oy = typeof output.ui?.y === 'number' ? output.ui.y : 120;
    // Beside the output; move the output right so the loss sits between it and the model.
    loss = {
      id,
      component: lock.component,
      attrs: structuredClone(lock.attrs ?? {}),
      ui: { x: ox, y: oy },
    };
    const index = nodes.findIndex((n) => n.id === output.id);
    nodes[index] = { ...output, ui: { ...output.ui, x: ox + 240, y: oy } };
    nodes.push(loss);
  }
  loss.attrs = structuredClone(lock.attrs ?? {});
  loss.ui = { ...loss.ui, locked: { ports: [lock.targetPort] } satisfies LockMark };
  const lossId = loss.id;
  const edges = graph.edges.filter(
    (e) =>
      !(e.to.node === output.id && e.from.node !== lossId) &&
      !(e.to.node === lossId && e.to.port === lock.targetPort && e.from.node !== target.id),
  );
  const want: GraphEdge[] = [
    { from: { node: target.id, port: 'out' }, to: { node: lossId, port: lock.targetPort } },
    { from: { node: lossId, port: 'out' }, to: { node: output.id, port: 'x' } },
  ];
  for (const w of want)
    if (!edges.some((e) => JSON.stringify(e) === JSON.stringify(w))) edges.push(w);
  return { ...graph, nodes, edges };
}
