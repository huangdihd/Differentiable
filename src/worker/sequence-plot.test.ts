// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { compileGraph } from '../engine/graph';
import type { ForwardLevel } from '../engine/levels';
import { gameLevels } from '../levels/catalog';
import { referenceRegistry } from '../levels/level-helpers';
import { Random } from '../engine/random';
import { temporalRnnDelayLevel, temporalTrainLevel } from '../levels/chapter-temporal';
import { sequenceExample, sequenceProbe, sequenceRule } from '../levels/sequence-probe';
import { sequencePlot } from './training-plot';

it.each([temporalRnnDelayLevel, temporalTrainLevel])(
  'plots the reference model over the whole probe sweep for $id',
  async (level) => {
    const plan = compileGraph(level.referenceGraph!, {
      registry: referenceRegistry(
        gameLevels.filter((l): l is ForwardLevel => l.kind === 'forward'),
      ),
      inputShapes: level.interface.inputs,
      random: new Random(level.seed),
    });
    try {
      const plot = await sequencePlot(level, level.referenceGraph!, plan);
      const probe = sequenceProbe(level.sequenceProbe!, level.interface.inputs.x[1]);
      expect(plot?.kind).toBe('regression');
      if (plot?.kind !== 'regression') return;
      expect(plot.fit?.x).toEqual(probe.at);
      expect(plot.fit?.p).toHaveLength(probe.rows.length);
      expect(plot.fit!.p.every(Number.isFinite)).toBe(true);
    } finally {
      plan.dispose();
    }
  },
);

it('gives each try-it example a right answer', () => {
  const delay = sequenceExample('delay', 6, 3);
  expect(delay.row).toHaveLength(6);
  expect(delay.expected).toBe(delay.row[0]);
  expect(sequenceRule('delay', [0.4, 0, 0, 0, 0, 0])).toBe(0.4);
  expect(sequenceRule('forecast', [0, 0, 0, 0, 0, 0])).toBeUndefined();
  const wave = sequenceExample('forecast', 6, 3);
  expect(Math.abs(wave.expected)).toBeLessThanOrEqual(1);
});
