// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { evaluateLevel, initialProgress } from '../engine/levels';
import { gameLevels } from '../levels/catalog';
import { buildPlayerRegistry } from '../levels/registry';
import { SANDBOX_EXAMPLES, builtinStarter, builtinTask } from './datasets';
import { buildSandboxLevel } from './level';
import { starterGraph } from './starter';

import { sandboxComponents } from './level';

// The sandbox opens after every level, so the player owns every component by then.
const ids = gameLevels.map((l) => l.id);
const registry = buildPlayerRegistry(gameLevels, {
  ...initialProgress(ids, sandboxComponents()),
  passed: ids,
  passedSolutions: Object.fromEntries(
    gameLevels.flatMap((l) =>
      'referenceGraph' in l && l.referenceGraph ? [[l.id, l.referenceGraph]] : [],
    ),
  ),
});

describe('sandbox starter models', () => {
  for (const id of SANDBOX_EXAMPLES)
    it(`trains on ${id} and the loss falls`, async () => {
      const level = buildSandboxLevel('s', { kind: 'builtin', id });
      const graph = starterGraph(
        { x: level.interface.inputs.x, t: level.interface.inputs.t },
        builtinTask(id),
        builtinStarter(id),
      );
      const losses: number[] = [];
      const result = await evaluateLevel(level, graph, {
        registry,
        steps: 300,
        optimizer: 'adam',
        learningRate: 0.03,
        onReport: (r) => losses.push(r.loss),
      });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(false);
      expect(losses.length).toBeGreaterThan(10);
      expect(losses.at(-1)!).toBeLessThan(losses[0] * 0.5);
    });

  it('builds a table model for imported multi-column data', () => {
    const graph = starterGraph({ x: [10, 3], t: [10, 4] }, 'classification', 'mlp');
    const components = graph.nodes.map((n) => n.component);
    expect(components).toEqual([
      'input',
      'input',
      'linear',
      'tanh',
      'linear',
      'softmax',
      'cross_entropy',
      'output',
    ]);
    expect(graph.nodes.find((n) => n.id === 'hidden')?.attrs?.in_features).toBe(3);
    expect(graph.nodes.find((n) => n.id === 'layer')?.attrs?.out_features).toBe(4);
  });
});
