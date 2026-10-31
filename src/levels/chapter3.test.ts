// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import type { GraphJSON } from '../engine/graph';
import { applyResult, evaluateLevel, graphMetrics, initialProgress } from '../engine/levels';
import * as T from '../engine/tensor';
import { gameLevels, catalog } from './catalog';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels, chapter3Levels, chapter3TrainLevel } from './chapter3';
import { referenceRegistry } from './level-helpers';
import { availableComponents, buildPlayerRegistry } from './registry';

const circuitLevels = gameLevels.filter((level) => level.kind !== 'observation');
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
    expect(chapter2Levels.at(-1)!.unlock).toEqual(['loss-intro']);
    expect(chapter3Levels.map((l) => l.unlock)).toEqual([
      ['3-1'],
      ['neuron'],
      ['3-2'],
      ['layer'],
      ['3-3'],
      ['3-4'],
      ['3-5'],
      ['3-6'],
      ['3-7'],
      ['3-8'],
      ['3-9'],
      ['xor'],
    ]);
    expect(
      catalog.filter((entry) => entry.chapter === 3).map(({ level, code }) => [level.id, code]),
    ).toEqual([
      ['loss-intro', '3-1'],
      ['3-1', '3-2'],
      ['neuron', '3-3'],
      ['3-2', '3-4'],
      ['layer', '3-5'],
      ['3-3', '3-6'],
      ['3-4', '3-7'],
      ['3-5', '3-8'],
      ['3-6', '3-9'],
      ['3-7', '3-10'],
      ['3-8', '3-11'],
      ['3-9', '3-12'],
      ['xor', '3-13'],
    ]);
    expect(
      ['loss-intro', '3-1', '3-2', '3-3', '3-4'].map(
        (id) => chapter3Levels.find((l) => l.id === id)!.par,
      ),
    ).toEqual([2, 3, 1, 6, 8]);
    for (const level of chapter3Levels) {
      expect(level.hintsKeys).toHaveLength(2);
      expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
    }
  });

  for (const level of chapter3ForwardLevels.filter((candidate) =>
    ['loss-intro', '3-1', '3-2', '3-3'].includes(candidate.id),
  )) {
    it(`${level.id}: reference passes all shapes and rejects the typical mistake`, async () => {
      const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(true);
      const wrong = structuredClone(level.referenceGraph!);
      if (level.id === 'loss-intro')
        wrong.edges.find((edge) => edge.to.node === 'output')!.from.node = 'error';
      if (level.id === '3-1')
        wrong.nodes.find((node) => node.id === 'negativeRate')!.attrs = { value: 0.1 };
      if (level.id === '3-2') wrong.nodes.find((node) => node.id === 'gradient')!.component = 'add';
      if (level.id === '3-3') {
        const mean = wrong.nodes.find((node) => node.id === 'mean')!;
        mean.component = 'sum';
        mean.attrs = { axes: [-1], keepdims: true };
      }
      const rejected = await evaluateLevel(playable(level), wrong, { registry });
      expect(rejected.errorKey).toBeUndefined();
      expect(rejected.passed).toBe(false);
      expect(rejected.tests.some((t) => !t.passed && !t.errorKey)).toBe(true);
    });
  }

  for (const level of chapter3ForwardLevels.filter((candidate) =>
    ['3-1', '3-2'].includes(candidate.id),
  )) {
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
          const { x, '∂L/∂p': upstream } = inputs;
          const w =
            inputs.w ??
            T.tensor(
              new Float32Array(x.shape.reduce((size, dimension) => size * dimension, 1)).fill(1),
              x.shape,
              x.backend,
            );
          const prediction = level.id === '3-1' ? T.mul(w, x) : T.sum(T.mul(w, x), [-1]);
          // Seed the model-side VJP with the supplied loss-side derivative, exactly as 1-11 does.
          const seeded = T.mul(prediction, upstream);
          const [gradient] = await T.backward(seeded, [w]);
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
            T.disposeGraph([seeded, output, ...Object.values(inputs), w]);
          }
        }
    });
  }

  it('takes the upstream loss gradient directly in both model-side backward lessons', () => {
    const step = chapter3ForwardLevels.find((candidate) => candidate.id === '3-1')!;
    const vector = chapter3ForwardLevels.find((candidate) => candidate.id === '3-2')!;
    expect(Object.keys(step.interface.inputs)).toEqual(['w', 'x', '∂L/∂p']);
    expect(Object.keys(vector.interface.inputs)).toEqual(['x', '∂L/∂p']);
    for (const level of [step, vector]) {
      expect(level.revalidateOnLoad).toBe(true);
      expect(level.referenceGraph!.nodes.some((node) => node.attrs?.name === 't')).toBe(false);
      expect(level.referenceGraph!.nodes.some((node) => node.attrs?.name === '∂L/∂p')).toBe(true);
    }
    expect(vector.referenceGraph!.nodes.some((node) => node.attrs?.name === 'w')).toBe(false);
  });

  it('computes the vector weight gradient from x and upstream without a weight input', async () => {
    const level = chapter3ForwardLevels.find((candidate) => candidate.id === '3-2')!;
    const x = T.tensor([1, -2, 3], [3]);
    const upstream = T.tensor([0.5], []);
    const output = level.reference({ x, '∂L/∂p': upstream });
    try {
      expect(Array.from(await output.read())).toEqual([0.5, -1, 1.5]);
    } finally {
      T.disposeGraph([output, x, upstream]);
    }
  });

  it('requires momentum steps to output both the updated weight and recurrent momentum', async () => {
    const level = chapter3ForwardLevels.find((candidate) => candidate.id === '3-7')!;
    expect(level.interface.outputs).toEqual({ updatedWeight: [], updatedMomentum: [] });
    expect(level.revalidateOnLoad).toBe(true);
    const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
    expect(result.errorKey).toBeUndefined();
    expect(result.passed).toBe(true);
    expect(result.tests.every((test) => test.actual.length === 2)).toBe(true);

    const inputs = {
      w: T.tensor([2], []),
      g: T.tensor([3], []),
      m: T.tensor([-1], []),
    };
    const outputs = level.referenceOutputs!(inputs);
    try {
      expect(Array.from(await outputs.updatedMomentum.read())).toEqual([2.0999999046325684]);
      expect(Array.from(await outputs.updatedWeight.read())).toEqual([1.7899999618530273]);
    } finally {
      T.disposeGraph([...Object.values(outputs), ...Object.values(inputs)]);
    }
  });

  it('earns mse as the player subgraph with p/t ports and no native kernel', async () => {
    let progress = initialProgress(['0-1']);
    for (const level of [...preceding, ...chapter3Levels.slice(0, 6)]) {
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

  it('earns linear before losses without giving construction lessons future components', async () => {
    let progress = initialProgress(['0-1']);
    for (const level of circuitLevels) {
      if (level.id === '3-3') {
        for (const id of ['3-3', '3-4', '3-5', '3-6']) {
          expect(
            availableComponents(
              circuitLevels.find((l) => l.id === id)!,
              progress,
            ),
          ).toContain('linear');
        }
        break;
      }
      expect(progress.unlocked).toContain(level.id);
      const available = availableComponents(level, progress);
      if (['neuron', 'layer'].includes(level.id)) {
        expect(available).toContain(level.id === 'neuron' ? 'dot' : 'matmul');
        for (const name of ['mse', 'cross_entropy', 'linear'])
          expect(available).not.toContain(name);
        expect(level.par).toBe(level.id === 'neuron' ? 3 : 5);
      }
      const result = await evaluateLevel({ ...level, available }, level.referenceGraph!, {
        registry: buildPlayerRegistry(gameLevels, progress),
      });
      expect(result.passed, level.id).toBe(true);
      progress = applyResult(progress, level, result, level.referenceGraph!);
    }
  });

  for (const id of ['neuron', 'layer']) {
    it(`${id}: player components pass all four shapes and award only linear`, async () => {
      const level = chapter3ForwardLevels.find((level) => level.id === id)!;
      const progress = initialProgress([id]);
      for (const prior of preceding) {
        progress.passed.push(prior.id);
        progress.passedSolutions[prior.id] = prior.referenceGraph!;
        progress.unlockedComponents.push(...(prior.unlockComponents ?? []));
      }
      const playerRegistry = buildPlayerRegistry(gameLevels, progress);
      const result = await evaluateLevel(
        { ...level, available: availableComponents(level, progress) },
        level.referenceGraph!,
        { registry: playerRegistry },
      );
      expect(result.passed).toBe(true);
      expect(result.tests).toHaveLength(36);
      expect(level.par).toBe(id === 'neuron' ? 3 : 5);
      expect(level.available).toEqual(chapter3ForwardLevels.find((l) => l.id === '3-5')!.available);
      const next = applyResult(progress, level, result, level.referenceGraph!);
      if (id === 'layer') {
        expect(level.forbidden).toEqual(['linear']);
        const linear = buildPlayerRegistry(gameLevels, next).get('linear')!;
        expect(linear.inputs).toEqual(['x', 'w', 'b']);
        expect(linear.subgraph).toBeDefined();
        expect(linear.native).toBeUndefined();
        expect(
          availableComponents(
            circuitLevels.find((l) => l.id === '3-6')!,
            next,
          ),
        ).toContain('linear');
      } else {
        expect(level.unlockComponents).toBeUndefined();
        // Axis 0 works for a single point, but mistakenly contracts the batch axis.
        const wrong = structuredClone(level.referenceGraph!);
        Object.assign(
          wrong.nodes.find((node) => node.id === 'weighted')!,
          { component: 'mul' },
        );
        wrong.nodes.push({ id: 'sum', component: 'sum', attrs: { axes: [0] } });
        wrong.edges.find((edge) => edge.from.node === 'weighted')!.from.node = 'sum';
        wrong.edges.push({
          from: { node: 'weighted', port: 'out' },
          to: { node: 'sum', port: 'x' },
        });
        expect(
          (
            await evaluateLevel(
              { ...level, available: playable(level).available, shapeVariants: [] },
              wrong,
              { registry },
            )
          ).passed,
        ).toBe(true);
        expect((await evaluateLevel(playable(level), wrong, { registry })).passed).toBe(false);
      }
    });
  }

  it('lets the regression lesson reuse a one-feature automatic Linear on scalar batches', async () => {
    expect(chapter3TrainLevel.componentDefaults?.linear).toEqual({
      automatic_parameters: true,
      in_features: 1,
      out_features: 1,
      bias: true,
    });
    const graph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 't', component: 'input', attrs: { name: 't' } },
        {
          id: 'prediction',
          component: 'linear',
          attrs: chapter3TrainLevel.componentDefaults!.linear,
        },
        { id: 'error', component: 'mse' },
        { id: 'output', component: 'output', attrs: { name: 'loss' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'prediction', port: 'x' } },
        { from: { node: 'prediction', port: 'out' }, to: { node: 'error', port: 'p' } },
        { from: { node: 't', port: 'out' }, to: { node: 'error', port: 't' } },
        { from: { node: 'error', port: 'out' }, to: { node: 'output', port: 'x' } },
      ],
    };
    const level = playable(chapter3TrainLevel);
    const result = await evaluateLevel(
      { ...level, available: [...level.available, 'linear'] },
      graph,
      { registry, learningRate: 0.3 },
    );
    expect(result.errorKey).toBeUndefined();
    expect(result.training?.loss).toBeLessThan(0.05);
    expect(result.passed).toBe(true);
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
