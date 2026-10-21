import { describe, expect, it } from 'vitest';
import { evaluateLevel, graphMetrics } from '../engine/levels';
import * as T from '../engine/tensor';
import { Random } from '../engine/random';
import { catalog, gameLevels } from './catalog';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels } from './chapter4';
import { chapter5ForwardLevels } from './chapter5';
import { chapter6ForwardLevels } from './chapter6';
import { formulaGraph, referenceRegistry } from './level-helpers';
import en from '../locales/en.json';
import zh from '../locales/zh-CN.json';

const registry = referenceRegistry([
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
  ...chapter5ForwardLevels,
  ...chapter6ForwardLevels,
]);
const formalLevels = [...chapter0Levels, ...chapter1Levels];

describe('formal chapter catalog', () => {
  it('contains the sequential lessons and the preserved XOR challenge', () => {
    expect(gameLevels.map((level) => level.id)).toEqual([
      '0-1',
      '0-2',
      'square',
      ...Array.from({ length: 5 }, (_, i) => `0-${i + 4}`),
      ...Array.from({ length: 10 }, (_, i) => `1-${i + 1}`),
      'axes',
      ...Array.from({ length: 8 }, (_, i) => `2-${i + 1}`),
      '3-1',
      '3-2',
      'neuron',
      'layer',
      ...Array.from({ length: 7 }, (_, i) => `3-${i + 3}`),
      'xor',
      ...Array.from({ length: 4 }, (_, i) => `4-${i + 1}`),
      'reshape',
      '4-5',
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
    for (const [index, level] of gameLevels.entries()) {
      expect(level.unlock).toEqual(
        index === gameLevels.length - 1 ? [] : [gameLevels[index + 1].id],
      );
      expect(level.referenceGraph).toBeDefined();
      expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
      for (const name of level.available) expect(registry.get(name)?.primitive).toBeDefined();
      for (const key of [
        level.titleKey,
        level.descriptionKey,
        ...(level.hintsKeys ?? []),
        ...(level.unlockFeatures ?? []),
      ]) {
        expect(en).toHaveProperty(key);
        expect(zh).toHaveProperty(key);
      }
    }
    expect(gameLevels.find((level) => level.id === 'xor')?.presets).toEqual(['matmul', 'softmax']);
    expect(chapter1Levels.at(-1)?.unlockFeatures).toContain('levels.features.autodiff');
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
        level.id === '1-10'
          ? ['w', 'x', 'b']
          : ['0-2', '0-4', '1-8', '1-9'].includes(level.id)
            ? ['a', 'b']
            : ['x'];
      expect(level.interface).toEqual({
        inputs: Object.fromEntries(names.map((name) => [name, []])),
        output: 'y',
        outputShape: [],
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
    '1-10': (w, x, b) => 1 / (1 + Math.exp(-(w * x + b))),
  };
  const derivatives: Record<string, (x: number, b: number, bias: number) => number> = {
    '1-1': (x) => 2 * x,
    '1-2': (x) => Math.exp(x) * (1 + x),
    '1-3': (x) => 3 * Math.exp(3 * x),
    '1-4': (x) => 1 / x,
    '1-5': (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    '1-6': (x) => 1 - Math.tanh(x) ** 2,
    '1-7': (x) => (2 * x) / (1 + x * x),
    '1-8': (a, b) => b + Math.exp(a),
    '1-9': (a, b) => a * a * Math.exp(a * b),
    '1-10': (w, x, b) => {
      const s = 1 / (1 + Math.exp(-(w * x + b)));
      return s * (1 - s) * x;
    },
  };
  for (const level of formalLevels) {
    it(`${level.id}: reference is the specified f, with the specified derivative`, async () => {
      for (const test of level.fixed) {
        const data = test.inputs;
        const inputs = Object.fromEntries(
          Object.entries(data).map(([name, values]) => [name, T.tensor(values, [])]),
        );
        const output = level.reference(inputs);
        let gradient: T.Tensor | undefined;
        try {
          expect(output.shape).toEqual([]);
          const x = (data.w ?? data.x ?? data.a)[0];
          const second = (data.w ? data.x : data.b)?.[0] ?? 0;
          const bias = data.b?.[0] ?? 0;
          const [actual] = await output.read();
          const expected = functions[level.id](x, second, bias);
          expect(Math.abs(actual - expected)).toBeLessThan(1e-5 * Math.max(1, Math.abs(expected)));
          if (level.kind === 'derivative') {
            [gradient] = await T.backward(output, [inputs[level.derivativeInput ?? 'x']]);
            const [value] = await gradient.read();
            const slope = derivatives[level.id](x, second, bias);
            expect(Math.abs(value - slope)).toBeLessThan(1e-5 * Math.max(1, Math.abs(slope)));
          }
        } finally {
          gradient?.dispose();
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
      const preceding =
        level.kind === 'derivative'
          ? [
              ...chapter0Levels.slice(0, 8),
              ...chapter1Levels.slice(0, chapter1Levels.indexOf(level)),
            ]
          : chapter0Levels.slice(0, chapter0Levels.indexOf(level));
      const available = [
        ...level.available,
        ...preceding.flatMap((item) => item.unlockComponents ?? []),
      ];
      const result = await evaluateLevel({ ...level, available }, level.referenceGraph!, {
        registry,
      });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(true);
      expect(result.tests).toHaveLength(level.fixed.length + level.generated!.count);
      expect(
        result.tests.every(
          (test) => test.passed && test.actual.length === 1 && test.expected.length === 1,
        ),
      ).toBe(true);
    });

    it(`${level.id}: rejects a constant impostor`, async () => {
      const wrong = formulaGraph(
        `0 * ${Object.keys(level.interface.inputs)[0]}`,
        Object.keys(level.interface.inputs),
      );
      const result = await evaluateLevel(level, wrong, { registry });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(false);
      expect(result.tests).toHaveLength(level.fixed.length + level.generated!.count);
      expect(
        result.tests.every(
          (test) => !test.errorKey && test.actual.length === 1 && test.expected.length === 1,
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
    const level = gameLevels.find((level) => level.id === 'xor')!;
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
