// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { initialProgress, type Progress } from '../engine/levels';
import { gameLevels } from '../levels/catalog';

/** Test-only completed route; production must receive the actual player's snapshot. */
export function recurrentCharacterTestProgress(): Progress {
  const progress = initialProgress([]);
  for (const level of gameLevels) {
    progress.passed.push(level.id);
    progress.unlockedComponents.push(...(level.unlockComponents ?? []));
    if (level.kind !== 'observation' && level.referenceGraph)
      progress.passedSolutions[level.id] = structuredClone(level.referenceGraph);
    if (level.id === 'time-rnn') break;
  }
  const graph = progress.passedSolutions['time-rnn'];
  // Equivalent but observably player-authored: commute additions and add an identity step.
  for (const edge of graph.edges) {
    if (edge.to.node === 'affine' || edge.to.node === 'input_projection')
      edge.to.port = edge.to.port === 'a' ? 'b' : 'a';
    if (edge.to.node === 'output_hiddenState') edge.from.node = 'player_identity';
  }
  graph.nodes.push(
    { id: 'player_one', component: 'const', attrs: { value: 1 } },
    { id: 'player_identity', component: 'mul' },
  );
  graph.edges.push(
    { from: { node: 'next', port: 'out' }, to: { node: 'player_identity', port: 'a' } },
    { from: { node: 'player_one', port: 'out' }, to: { node: 'player_identity', port: 'b' } },
  );
  return progress;
}

/** Legal learned scale with an ID that deliberately overlaps an experiment fixture. */
export function recurrentCharacterParameterTestProgress(): Progress {
  const progress = recurrentCharacterTestProgress();
  const graph = progress.passedSolutions['time-rnn'];
  const scale = graph.nodes.find(({ id }) => id === 'player_one')!;
  scale.id = 'embedding';
  scale.component = 'param';
  scale.attrs = { shape: [], init: 'ones' };
  for (const edge of graph.edges) if (edge.from.node === 'player_one') edge.from.node = scale.id;
  return progress;
}
