// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { evaluateLevel, graphMetrics, type TrainLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { Random } from '../engine/random';
import { catalog, gameLevels } from './catalog';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels } from './chapter4';
import { temporalForwardLevels } from './chapter-temporal';
import { chapter5ForwardLevels } from './chapter5';
import { chapter6ForwardLevels } from './chapter6';
import { formulaGraph, formulaOutputsGraph, referenceRegistry } from './level-helpers';
import en from '../locales/en.json';
import zh from '../locales/zh-CN.json';

const registry = referenceRegistry([
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
  ...temporalForwardLevels,
  ...chapter5ForwardLevels,
  ...chapter6ForwardLevels,
]);
const formalLevels = [...chapter0Levels, ...chapter1Levels];
const chapter1LevelById = (id: string) => {
  const level = chapter1Levels.find((candidate) => candidate.id === id);
  if (!level) throw new Error(`missing level ${id}`);
  return level;
};
const semanticOutputs: Readonly<Record<string, string>> = {
  '1-1': 'f′',
  '1-2': 'f′',
  '1-3': 'f′',
  '1-4': 'f′',
  '1-5': 'σ′',
  '1-6': 'tanh′',
  '1-7': 'f′',
  '1-8': '∂f/∂a',
  '1-9': '∂f/∂b',
};

describe('formal chapter catalog', () => {
  it('contains the sequential lessons and the preserved XOR challenge', () => {
    expect(gameLevels.map((level) => level.id)).toEqual([
      '0-1',
      '0-2',
      'square',
      ...Array.from({ length: 5 }, (_, i) => `0-${i + 4}`),
      ...Array.from({ length: 11 }, (_, i) => `1-${i + 1}`),
      'axes',
      ...Array.from({ length: 8 }, (_, i) => `2-${i + 1}`),
      'loss-intro',
      '3-1',
      'neuron',
      '3-2',
      'layer',
      ...Array.from({ length: 7 }, (_, i) => `3-${i + 3}`),
      'xor',
      ...Array.from({ length: 4 }, (_, i) => `4-${i + 1}`),
      'reshape',
      '4-5',
      'time-window-limit',
      'time-rnn',
      'time-rnn-window',
      'time-rnn-delay',
      'time-attention',
      'time-forecast',
      ...Array.from({ length: 5 }, (_, i) => `5-${i + 1}`),
      ...Array.from({ length: 4 }, (_, i) => `6-${i + 1}`),
    ]);
    expect(catalog.map((entry) => entry.level)).toEqual(gameLevels);
    expect(
      catalog.filter((entry) => entry.chapter === 4).map(({ level, code }) => [level.id, code]),
    ).toEqual([
      ['4-1', '4-1'],
      ['4-2', '4-2'],
      ['4-3', '4-3'],
      ['4-4', '4-4'],
      ['reshape', '4-5'],
      ['4-5', '4-6'],
    ]);
    expect(
      catalog.filter((entry) => entry.chapter === 5).map(({ level, code }) => [level.id, code]),
    ).toEqual([
      ['time-window-limit', '5-1'],
      ['time-rnn', '5-2'],
      ['time-rnn-window', '5-3'],
      ['time-rnn-delay', '5-4'],
    ]);
    expect(
      catalog.filter((entry) => entry.chapter === 6).map(({ level, code }) => [level.id, code]),
    ).toEqual([
      ['time-attention', '6-1'],
      ['time-forecast', '6-2'],
      ...Array.from({ length: 5 }, (_, i) => [`5-${i + 1}`, `6-${i + 3}`]),
    ]);
    expect(
      catalog.filter((entry) => entry.chapter === 7).map(({ level, code }) => [level.id, code]),
    ).toEqual(Array.from({ length: 4 }, (_, i) => [`6-${i + 1}`, `7-${i + 1}`]));
    for (const [index, level] of gameLevels.entries()) {
      expect(level.id).not.toBe('');
      expect(level.titleKey).not.toBe('');
      expect(level.descriptionKey).not.toBe('');
      expect(level.unlock).toEqual(
        index === gameLevels.length - 1 ? [] : [gameLevels[index + 1].id],
      );
      for (const key of [
        level.titleKey,
        level.descriptionKey,
        ...(level.hintsKeys ?? []),
        ...(level.unlockFeatures ?? []),
      ]) {
        expect(en).toHaveProperty(key);
        expect(zh).toHaveProperty(key);
      }
      if (level.kind === 'observation') {
        for (const property of ['graph', 'referenceGraph', 'par', 'interface', 'available'])
          expect(level).not.toHaveProperty(property);
        continue;
      }
      expect(level.referenceGraph).toBeDefined();
      expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
      // Palette parts are primitives, plus the built-in tensor plumbing ops (Slice/Stack/
      // Concat/Split), which run as fixed copy subgraphs and count as one operation each.
      for (const name of level.available)
        expect(
          registry.get(name)?.primitive ??
            (registry.get(name)?.builtin ? registry.get(name)?.subgraph : undefined),
        ).toBeDefined();
      for (const key of [
        ...(level.revealGraph?.introKey ? [level.revealGraph.introKey] : []),
        ...(level.revealGraph?.playerBackwardLabelKey
          ? [level.revealGraph.playerBackwardLabelKey]
          : []),
        ...(level.revealGraph?.engineBackwardLabelKey
          ? [level.revealGraph.engineBackwardLabelKey]
          : []),
        ...(level.revealGraph?.graph.nodes.flatMap((node) =>
          typeof node.attrs?.labelKey === 'string' ? [node.attrs.labelKey] : [],
        ) ?? []),
      ]) {
        expect(en).toHaveProperty(key);
        expect(zh).toHaveProperty(key);
      }
    }
    expect(
      gameLevels.find((level): level is TrainLevel => level.kind === 'train' && level.id === 'xor')
        ?.presets,
    ).toEqual(['matmul', 'softmax']);
    expect(chapter1LevelById('1-11').unlockFeatures).toContain('levels.features.autodiff');
  });

  it('matches the required lesson order, interfaces and primitive progression', () => {
    expect(chapter0Levels.map((level) => [level.id, level.unlockComponents])).toEqual([
      ['0-1', ['neg']],
      ['0-2', ['sub']],
      ['square', ['square']],
      ['0-4', ['div']],
      ['0-5', ['sigmoid']],
      ['0-6', ['tanh']],
      ['0-7', ['relu']],
      ['0-8', ['sqrt']],
    ]);
    for (const level of formalLevels) {
      const names =
        level.id === '1-11'
          ? ['w', 'x', '∂f/∂z']
          : ['0-2', '0-4', '1-8', '1-9', '1-10'].includes(level.id)
            ? ['a', 'b']
            : ['x'];
      const gradientOutputs =
        level.id === '1-10'
          ? { '∂a': [], '∂b': [] }
          : level.id === '1-11'
            ? { '∂f/∂w': [], '∂f/∂x': [], '∂f/∂b': [] }
            : undefined;
      expect(level.interface).toEqual({
        inputs: Object.fromEntries(names.map((name) => [name, []])),
        output:
          level.id === '1-10'
            ? '∂a'
            : level.id === '1-11'
              ? '∂f/∂w'
              : (semanticOutputs[level.id] ?? 'y'),
        outputShape: [],
        ...(gradientOutputs ? { outputs: gradientOutputs } : {}),
      });
      expect(level.available).toEqual([
        'const',
        'add',
        'mul',
        'exp',
        'log',
        'recip',
        ...(level.chapter === '1' || ['0-7', '0-8'].includes(level.id) ? ['max'] : []),
      ]);
      const random = new Random(level.generated!.seed);
      const cases = [
        ...level.fixed,
        ...Array.from({ length: level.generated!.count }, (_, i) =>
          level.generated!.generate(random, i),
        ),
      ];
      for (const test of cases) {
        expect(Object.keys(test.inputs)).toEqual(names);
        for (const [name, data] of Object.entries(test.inputs)) {
          expect(data).toHaveLength(1);
          expect(data.every(Number.isFinite)).toBe(true);
          if (['0-8', '1-4'].includes(level.id)) expect(data.every((x) => x > 0)).toBe(true);
          if (level.id === '0-4' && name === 'b') expect(data.every((x) => x !== 0)).toBe(true);
        }
      }
      if (level.id === '0-4') expect(level.fixed.some((test) => test.inputs.a[0] < 0)).toBe(true);
    }
  });

  // Independent scalar mathematics anchors the curriculum, not just agreement
  // between two graphs that could encode the same wrong lesson.
  const functions: Record<string, (x: number, b: number, bias: number) => number> = {
    '0-1': (x) => -x,
    '0-2': (a, b) => a - b,
    '0-4': (a, b) => a / b,
    '0-5': (x) => 1 / (1 + Math.exp(-x)),
    '0-6': Math.tanh,
    '0-7': (x) => Math.max(x, 0),
    '0-8': Math.sqrt,
    square: (x) => x * x,
    '1-1': (x) => x * x,
    '1-2': (x) => x * Math.exp(x),
    '1-3': (x) => Math.exp(3 * x),
    '1-4': Math.log,
    '1-5': (x) => 1 / (1 + Math.exp(-x)),
    '1-6': Math.tanh,
    '1-7': (x) => Math.log(1 + x * x),
    '1-8': (a, b) => a * b + Math.exp(a),
    '1-9': (a, b) => a * Math.exp(a * b),
    '1-10': (a, b) => a * b + Math.exp(a),
    '1-11': (w, x) => w * x,
  };
  const derivatives: Record<string, (x: number, b: number, bias: number) => number[]> = {
    '1-1': (x) => [2 * x],
    '1-2': (x) => [Math.exp(x) * (1 + x)],
    '1-3': (x) => [3 * Math.exp(3 * x)],
    '1-4': (x) => [1 / x],
    '1-5': (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return [s * (1 - s)];
    },
    '1-6': (x) => [1 - Math.tanh(x) ** 2],
    '1-7': (x) => [(2 * x) / (1 + x * x)],
    '1-8': (a, b) => [b + Math.exp(a)],
    '1-9': (a, b) => [a * a * Math.exp(a * b)],
    '1-10': (a, b) => [b + Math.exp(a), a],
    '1-11': (w, x, g) => [g * x, g * w, g],
  };
  for (const level of formalLevels) {
    it(`${level.id}: reference is the specified f, with the specified derivative`, async () => {
      for (const test of level.fixed) {
        const data = test.inputs;
        const inputs = Object.fromEntries(
          Object.entries(data).map(([name, values]) => [name, T.tensor(values, [])]),
        );
        const output = level.reference(inputs);
        let gradients: T.Tensor[] = [],
          explicit: T.Tensor[] = [];
        try {
          expect(output.shape).toEqual([]);
          const x = (data.w ?? data.x ?? data.a)[0];
          const second = (data.w ? data.x : data.b)?.[0] ?? 0;
          const third = data['∂f/∂z']?.[0] ?? data.b?.[0] ?? 0;
          const [actual] = await output.read();
          const expected = functions[level.id](x, second, third);
          expect(Math.abs(actual - expected)).toBeLessThan(1e-5 * Math.max(1, Math.abs(expected)));
          if (level.referenceOutputs) {
            const named = level.referenceOutputs(inputs);
            explicit = Object.values(named);
            const values = await Promise.all(
              Object.keys(level.interface.outputs!).map(
                async (name) => (await named[name].read())[0],
              ),
            );
            const slopes = derivatives[level.id](x, second, third);
            expect(values).toHaveLength(slopes.length);
            values.forEach((value, index) =>
              expect(Math.abs(value - slopes[index])).toBeLessThan(
                1e-5 * Math.max(1, Math.abs(slopes[index])),
              ),
            );
          } else if (level.kind === 'derivative') {
            const derivativeInputs = level.derivativeInputs ?? [level.derivativeInput ?? 'x'];
            gradients = await T.backward(
              output,
              derivativeInputs.map((name) => inputs[name]),
            );
            const values = await Promise.all(
              gradients.map(async (gradient) => (await gradient.read())[0]),
            );
            const slopes = derivatives[level.id](x, second, third);
            expect(values).toHaveLength(slopes.length);
            values.forEach((value, index) =>
              expect(Math.abs(value - slopes[index])).toBeLessThan(
                1e-5 * Math.max(1, Math.abs(slopes[index])),
              ),
            );
          }
        } finally {
          gradients.forEach((gradient) => gradient.dispose());
          if (explicit.length) T.disposeGraph(explicit, Object.values(inputs));
          T.disposeGraph([output, ...Object.values(inputs)]);
        }
      }
    });
  }

  for (const level of formalLevels) {
    it(`${level.id}: reference answer passes fixed and reproducible seeded cases`, async () => {
      const randomA = new Random(level.generated!.seed);
      const randomB = new Random(level.generated!.seed);
      for (let i = 0; i < level.generated!.count; i++)
        expect(level.generated!.generate(randomA, i)).toEqual(
          level.generated!.generate(randomB, i),
        );
      // Simulate the palette supplied by progression; available itself is primitives only.
      const currentIndex = gameLevels.findIndex((candidate) => candidate.id === level.id);
      expect(currentIndex).toBeGreaterThanOrEqual(0);
      const available = [
        ...level.available,
        ...gameLevels
          .filter((_, index) => index < currentIndex)
          .flatMap((item) => item.unlockComponents ?? []),
      ];
      const result = await evaluateLevel({ ...level, available }, level.referenceGraph!, {
        registry,
      });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(true);
      expect(result.tests).toHaveLength(level.fixed.length + level.generated!.count);
      const outputCount = Object.keys(
        level.interface.outputs ?? { [level.interface.output]: level.interface.outputShape },
      ).length;
      expect(
        result.tests.every(
          (test) =>
            test.passed &&
            test.actual.length === outputCount &&
            test.expected.length === outputCount,
        ),
      ).toBe(true);
    });

    it(`${level.id}: rejects a constant impostor`, async () => {
      const inputs = Object.keys(level.interface.inputs);
      const formula = `0 * ${inputs[0]}`;
      const wrong = level.interface.outputs
        ? formulaOutputsGraph(
            Object.fromEntries(Object.keys(level.interface.outputs).map((name) => [name, formula])),
            inputs,
          )
        : formulaGraph(formula, inputs, level.interface.output);
      const result = await evaluateLevel(level, wrong, { registry });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(false);
      expect(result.tests).toHaveLength(level.fixed.length + level.generated!.count);
      const outputCount = Object.keys(
        level.interface.outputs ?? { [level.interface.output]: level.interface.outputShape },
      ).length;
      expect(
        result.tests.every(
          (test) =>
            !test.errorKey &&
            test.actual.length === outputCount &&
            test.expected.length === outputCount,
        ),
      ).toBe(true);
      expect(
        result.tests.some(
          (test) =>
            !test.passed &&
            test.actual.some((value, i) => Math.abs(value - test.expected[i]) > 0.01),
        ),
      ).toBe(true);
    });
  }

  it('preserved XOR reference trains with earned tensor components', async () => {
    const level = gameLevels.find(
      (level): level is TrainLevel => level.kind === 'train' && level.id === 'xor',
    )!;
    const available = [
      ...level.available,
      ...[...chapter0Levels, ...chapter2Levels].flatMap((item) => item.unlockComponents ?? []),
      ...(level.presets ?? []),
    ];
    const result = await evaluateLevel({ ...level, available }, level.referenceGraph!, {
      registry,
    });
    expect(result.errorKey).toBeUndefined();
    expect(result.passed).toBe(true);
    expect(result.training?.loss).toBeLessThan(0.05);
    expect(result.training?.accuracy).toBe(1);
  }, 20000);
});
