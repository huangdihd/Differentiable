// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { evaluateLevel, type LevelCase } from '../engine/levels';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { referenceRegistry } from './level-helpers';

const registry = referenceRegistry(chapter0Levels);

describe('signed division', () => {
  const level = chapter0Levels.find((entry) => entry.id === '0-4')!;
  const available = [...level.available, 'neg'];
  it('tests both signs of b without zero, deterministically', () => {
    expect(level.fixed.map((test) => test.inputs.b[0])).toEqual([
      -3, -2, -0.5, -0.25, 0.25, 0.5, 1, 3,
    ]);
    const generated = level.generated!;
    const sample = () => {
      const random = new Random(generated.seed);
      return Array.from({ length: generated.count }, (_, i) => generated.generate(random, i));
    };
    expect(sample()).toEqual(sample());
    for (const test of sample()) {
      expect(Math.abs(test.inputs.b[0])).toBeGreaterThanOrEqual(0.1);
      expect(Math.abs(test.inputs.b[0])).toBeLessThanOrEqual(3.1);
    }
  });
  it('reference graph agrees with independent division near zero', async () => {
    const values = [-3, -0.001, -0.00001, 0.00001, 0.001, 3];
    const fixed = values.map((x, i): LevelCase => ({
      id: `near-zero-${i}`,
      inputs: { a: [-2], b: [x] },
    }));
    const result = await evaluateLevel(
      { ...level, available, fixed, generated: undefined },
      level.referenceGraph!,
      { registry },
    );
    expect(result.passed).toBe(true);
    expect(result.tests).toHaveLength(fixed.length);
    for (const [i, test] of result.tests.entries()) {
      const expected = -2 / values[i];
      expect(Math.abs((test.actual[0] - expected) / expected)).toBeLessThan(1e-4);
      expect(Math.abs((test.expected[0] - expected) / expected)).toBeLessThan(1e-4);
    }
  });
  it('offers reciprocal as a basic primitive rather than a lesson', () => {
    expect(level.available).toContain('recip');
    expect(chapter0Levels.some((entry) => entry.id === '0-3')).toBe(false);
    expect(chapter0Levels.find((entry) => entry.id === 'square')!.unlock).toEqual(['0-4']);
  });
});
