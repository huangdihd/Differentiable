import { describe, expect, it } from 'vitest';
import { evaluateLevel, type LevelCase } from '../engine/levels';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { formulaGraph, referenceRegistry } from './level-helpers';

const registry = referenceRegistry(chapter0Levels);

describe('signed reciprocal and division', () => {
  for (const id of ['0-3', '0-4']) {
    const level = chapter0Levels.find((entry) => entry.id === id)!;
    const name = id === '0-3' ? 'x' : 'b';
    const available = [...level.available, 'neg', ...(id === '0-4' ? ['recip'] : [])];
    it(`${id} tests both signs without zero, deterministically`, () => {
      expect(level.fixed.map((test) => test.inputs[name][0])).toEqual([
        -3, -2, -0.5, -0.25, 0.25, 0.5, 1, 3,
      ]);
      const generated = level.generated!;
      const sample = () => {
        const random = new Random(generated.seed);
        return Array.from({ length: generated.count }, (_, i) => generated.generate(random, i));
      };
      expect(sample()).toEqual(sample());
      for (const test of sample()) {
        expect(Math.abs(test.inputs[name][0])).toBeGreaterThanOrEqual(0.1);
        expect(Math.abs(test.inputs[name][0])).toBeLessThanOrEqual(3.1);
      }
    });
    it(`${id} reference graph agrees with independent division near zero`, async () => {
      const values = [-3, -0.001, -0.00001, 0.00001, 0.001, 3];
      const fixed = values.map((x, i): LevelCase => ({
        id: `near-zero-${i}`,
        inputs: id === '0-3' ? { x: [x] } : { a: [-2], b: [x] },
      }));
      const result = await evaluateLevel(
        {
          ...level,
          available,
          fixed,
          generated: undefined,
        },
        level.referenceGraph!,
        { registry },
      );
      expect(result.passed).toBe(true);
      expect(result.tests).toHaveLength(fixed.length);
      for (const [i, test] of result.tests.entries()) {
        const expected = (id === '0-3' ? 1 : -2) / values[i];
        expect(Math.abs((test.actual[0] - expected) / expected)).toBeLessThan(1e-4);
        expect(Math.abs((test.expected[0] - expected) / expected)).toBeLessThan(1e-4);
      }
    });
  }
  it('rejects the positive-only answer and accepts the signed answer', async () => {
    const level = { ...chapter0Levels[2], available: [...chapter0Levels[2].available, 'neg'] };
    expect(
      (await evaluateLevel(level, formulaGraph('exp(neg(log(x)))'), { registry })).passed,
    ).toBe(false);
    expect((await evaluateLevel(level, level.referenceGraph!, { registry })).passed).toBe(true);
  });
});
