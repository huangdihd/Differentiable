import { describe, expect, it } from 'vitest';
import { applyResult, evaluateLevel, graphMetrics, initialProgress } from '../engine/levels';
import * as T from '../engine/tensor';
import { gameLevels, catalog } from './catalog';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels, chapter3Levels, chapter3TrainLevel } from './chapter3';
import { referenceRegistry } from './level-helpers';
import { availableComponents, buildPlayerRegistry } from './registry';

const preceding = [...chapter0Levels, ...chapter1Levels, ...chapter2Levels];
const registry = referenceRegistry([...preceding, ...chapter3ForwardLevels]);
const playable = (level: (typeof chapter3Levels)[number]) => ({
  ...level,
  available: [
    ...level.available,
    ...preceding.flatMap((l) => l.unlockComponents ?? []),
    ...(level.kind === 'train' ? ['mse'] : []),
  ],
});

describe('gradient descent curriculum', () => {
  it('has references, two hints, expanded par and the complete route', () => {
    expect(chapter2Levels.at(-1)!.unlock).toEqual(['3-1']);
    expect(chapter3Levels.map((l) => l.unlock)).toEqual([
      ['3-2'],
      ['3-3'],
      ['3-4'],
      ['3-5'],
      ['3-6'],
      ['3-7'],
      ['3-8'],
      ['3-9'],
      ['xor'],
    ]);
    expect(catalog.at(-1)!.code).toBe('3-10');
    expect(chapter3Levels.slice(0, 4).map((l) => l.par)).toEqual([7, 6, 11, 13]);
    for (const level of chapter3Levels) {
      expect(level.hintsKeys).toHaveLength(2);
      expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
    }
  });

  for (const [i, level] of chapter3ForwardLevels.slice(0, 3).entries()) {
    it(`${level.id}: reference passes all shapes and rejects the typical mistake`, async () => {
      const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(true);
      const wrong = structuredClone(level.referenceGraph!);
      if (i === 0) wrong.nodes.find((n) => n.id === 'negativeRate')!.attrs = { value: 0.1 };
      if (i === 1) wrong.nodes.find((n) => n.id === 'two')!.attrs = { value: 1 };
      if (i === 2) {
        const mean = wrong.nodes.find((n) => n.id === 'mean')!;
        mean.component = 'sum';
        mean.attrs = { axes: [-1], keepdims: true };
      }
      const rejected = await evaluateLevel(playable(level), wrong, { registry });
      expect(rejected.errorKey).toBeUndefined();
      expect(rejected.passed).toBe(false);
      expect(rejected.tests.some((t) => !t.passed && !t.errorKey)).toBe(true);
    });
  }

  for (const level of chapter3ForwardLevels.slice(0, 2)) {
    it(`${level.id}: explicit gradient agrees with independent backward`, async () => {
      const groups = [
        { inputs: level.interface.inputs, fixed: level.fixed },
        ...(level.shapeVariants ?? []),
      ];
      for (const group of groups)
        for (const test of group.fixed) {
          const inputs = Object.fromEntries(
            Object.entries(test.inputs).map(([name, values]) => [
              name,
              T.tensor(values, group.inputs[name]),
            ]),
          );
          const { w, x, t } = inputs;
          const prediction = level.id === '3-1' ? T.mul(w, x) : T.sum(T.mul(w, x), [-1]);
          const error = T.add(prediction, T.mul(t, T.tensor([-1], [])));
          const loss = T.mul(error, error);
          const [gradient] = await T.backward(loss, [w]);
          const output = level.reference(inputs);
          try {
            const actual = await output.read(),
              grad = await gradient.read(),
              weights = await w.read();
            actual.forEach((value, i) =>
              expect(value).toBeCloseTo(
                level.id === '3-1' ? weights[i] - 0.1 * grad[i] : grad[i],
                4,
              ),
            );
            if (level.id === '3-1') expect(Math.abs(actual[0] - weights[0])).toBeGreaterThan(0.1);
          } finally {
            gradient.dispose();
            T.disposeGraph([loss, output, ...Object.values(inputs)]);
          }
        }
    });
  }

  it('earns mse as the player subgraph with p/t ports and no native kernel', async () => {
    let progress = initialProgress(['0-1']);
    for (const level of [...preceding, ...chapter3ForwardLevels.slice(0, 3)]) {
      const result = await evaluateLevel(
        { ...level, available: availableComponents(level, progress) },
        level.referenceGraph!,
        { registry: buildPlayerRegistry(gameLevels, progress) },
      );
      expect(result.passed, level.id).toBe(true);
      progress = applyResult(progress, level, result, level.referenceGraph!);
    }
    const mse = buildPlayerRegistry(gameLevels, progress).get('mse')!;
    expect(mse.inputs).toEqual(['p', 't']);
    expect(mse.native).toBeUndefined();
    expect(mse.subgraph).toBeDefined();
    expect(progress.unlocked).toContain('3-4');
    expect(availableComponents(chapter3TrainLevel, progress)).toContain('mse');
  });

  it('measures slow, convergent and divergent learning rates', async () => {
    const losses: Record<string, number | undefined> = {};
    for (const rate of [undefined, 0.1, 0.3, 0.5, 1.5]) {
      const result = await evaluateLevel(
        playable(chapter3TrainLevel),
        chapter3TrainLevel.referenceGraph!,
        { registry, learningRate: rate },
      );
      losses[String(rate ?? 'default')] = result.training?.loss;
      expect(result.passed).toBe(rate !== undefined && rate < 1);
      if (rate === undefined) expect(result.training!.loss).toBeGreaterThan(0.05);
      else if (rate < 1) expect(result.training!.loss).toBeLessThan(0.05);
      else expect(result.errorKey).toMatch(/diverged/);
    }
    console.info('chapter 3 learning-rate losses', losses);
  });
});
