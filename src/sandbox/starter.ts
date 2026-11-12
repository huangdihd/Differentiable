// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Starter models for an empty sandbox canvas: a working network for the slot's data, built
 * from ordinary components so the player can take it apart. Linear layers own their weights.
 */
import type { GraphJSON } from '../engine/graph';
import type { SandboxTask } from './datasets';

export type StarterKind = 'linear' | 'mlp';
export const STARTER_HIDDEN = 16;

const COL = 190,
  ROW = 200,
  X0 = 60,
  Y0 = 60;

/**
 * x and t shapes as the level declares them ([N] for a one-column curve, else [N, columns]).
 * The output is the scalar loss, named like the sandbox level's output.
 */
export function starterGraph(
  shapes: { x: readonly number[]; t: readonly number[] },
  task: SandboxTask,
  kind: StarterKind,
): GraphJSON {
  const graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  const vector = shapes.x.length === 1;
  const inFeatures = vector ? 1 : shapes.x[1];
  const outFeatures = shapes.t.length === 1 ? 1 : shapes.t[1];
  const at = (col: number, row: number) => ({ x: X0 + col * COL, y: Y0 + row * ROW });
  const node = (
    id: string,
    component: string,
    col: number,
    row: number,
    from: Record<string, string> = {},
    attrs?: Record<string, unknown>,
  ) => {
    graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}), ui: at(col, row) } as never);
    for (const [port, source] of Object.entries(from))
      graph.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  };
  const linear = (inF: number, outF: number) => ({
    automatic_parameters: true,
    in_features: inF,
    out_features: outF,
    bias: true,
  });
  node('x', 'input', 0, 0, {}, { name: 'x' });
  node('t', 'input', 0, 1, {}, { name: 't' });
  let col = 1;
  let p: string;
  if (kind === 'linear')
    p = node('layer', 'linear', col++, 0, { x: 'x' }, linear(inFeatures, outFeatures));
  else {
    node('hidden', 'linear', col++, 0, { x: 'x' }, linear(inFeatures, STARTER_HIDDEN));
    node('activation', 'tanh', col++, 0, { x: 'hidden' });
    p = node('layer', 'linear', col++, 0, { x: 'activation' }, linear(STARTER_HIDDEN, outFeatures));
  }
  // A one-column curve keeps its targets as [N]; the layer gives [N, 1].
  if (vector && shapes.t.length === 1)
    p = node('flat', 'reshape', col++, 0, { x: p }, { shape: [-1] });
  if (task === 'classification') p = node('probabilities', 'softmax', col++, 0, { x: p });
  const lossCol = col - 1;
  const loss = node('error', task === 'classification' ? 'cross_entropy' : 'mse', lossCol, 1, {
    p,
    t: 't',
  });
  node('output', 'output', lossCol + 1, 1, { x: loss }, { name: 'loss' });
  return graph;
}
