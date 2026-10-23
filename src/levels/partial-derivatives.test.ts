import { expect, it } from 'vitest';
import { buildBackwardGraph, traceBackward } from '../engine/backward-graph';
import { evaluateLevel, graphMetrics, type ForwardLevel } from '../engine/levels';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { formulaGraph, formulaOutputsGraph, referenceRegistry } from './level-helpers';

const registry = referenceRegistry([...chapter0Levels, ...chapter1Levels]);
const levelById = (id: string): ForwardLevel => {
  const level = chapter1Levels.find((candidate) => candidate.id === id);
  if (!level) throw new Error(`missing level ${id}`);
  return level;
};
const earned = [...chapter0Levels, ...chapter1Levels].flatMap(
  (level) => level.unlockComponents ?? [],
);
const playable = (level: ForwardLevel): ForwardLevel => ({
  ...level,
  available: [...level.available, ...earned],
});

it('keeps 1-11 focused on the local backward step for z=w·x+b', () => {
  expect(
    Object.fromEntries(
      chapter1Levels
        .filter((level) => level.unlockComponents?.length)
        .map((level) => [level.id, level.unlockComponents]),
    ),
  ).toEqual({ '1-5': ['dsigmoid'], '1-6': ['dtanh'] });

  const level = levelById('1-11');
  expect(
    level
      .referenceGraph!.nodes.filter((node) => !['input', 'output'].includes(node.component))
      .map((node) => node.component),
  ).toEqual(['mul', 'mul']);
  expect(
    level
      .referenceGraph!.nodes.filter((node) => node.component === 'input')
      .map((node) => node.attrs?.name),
  ).toEqual(['w', 'x', '∂f/∂z']);
  expect(level.referenceGraph!.nodes.some((node) => node.attrs?.name === 'b')).toBe(false);
  expect(
    level
      .referenceGraph!.edges.filter((edge) => edge.from.node === 'g')
      .map((edge) => edge.to.node)
      .sort(),
  ).toEqual(['dw', 'dx', 'out_b']);
});

it('reveals Linear followed by Sigmoid as a first-order engine backward pass', async () => {
  const level = playable(levelById('1-11'));
  const reveal = level.revealGraph!;
  expect(reveal).toBeDefined();
  expect(reveal.inputShapes).toEqual({ w: [], x: [], b: [] });
  expect(reveal.output).toBe('f');
  expect(reveal.wrt).toEqual(['w', 'x', 'b']);
  expect(reveal.playerBackwardStart).toBe('z');
  expect(reveal.playerBackwardLabelKey).toBe('gradView.backprop.playerRegion');
  expect(reveal.engineBackwardLabelKey).toBe('gradView.backprop.engineRegion');
  expect(reveal.graph.nodes.filter((node) => node.component === 'output')).toEqual([
    expect.objectContaining({ attrs: { name: 'f' } }),
  ]);
  expect(
    reveal.graph.nodes
      .filter((node) => !['input', 'output'].includes(node.component))
      .map((node) => node.component),
  ).toEqual(['mul', 'add', 'sigmoid']);
  expect(
    reveal.graph.nodes.some((node) => node.component === 'input' && node.attrs?.name === 'b'),
  ).toBe(true);
  expect(reveal.graph.nodes.some((node) => node.attrs?.name === '∂f/∂z')).toBe(false);
  expect(reveal.graph.nodes.some((node) => node.attrs?.name === '∂f/∂w')).toBe(false);

  const options = {
    registry,
    availableComponents: level.available,
    inputShapes: reveal.inputShapes,
    output: reveal.output,
    wrt: reveal.wrt,
  };
  expect(() => buildBackwardGraph(reveal.graph, options)).not.toThrow();
  const trace = await traceBackward(reveal.graph, { ...options, inputs: reveal.inputs });
  const [w] = reveal.inputs.w;
  const [x] = reveal.inputs.x;
  const [b] = reveal.inputs.b;
  const z = w * x + b;
  const s = 1 / (1 + Math.exp(-z));
  const g = s * (1 - s);
  expect(trace.gradient.z.values[0]).toBeCloseTo(g, 6);
  expect(trace.gradient.w.values[0]).toBeCloseTo(g * x, 6);
  expect(trace.gradient.x.values[0]).toBeCloseTo(g * w, 6);
  expect(trace.gradient.b.values[0]).toBeCloseTo(g, 6);
});

it('orders the gradient lessons before backpropagation and then axes', () => {
  for (const [id, successor] of [
    ['1-9', '1-10'],
    ['1-10', '1-11'],
    ['1-11', 'axes'],
  ]) {
    expect(levelById(id).unlock).toEqual([successor]);
  }

  expect(
    chapter1Levels
      .filter((level) => level.id !== '1-11')
      .every((level) => !level.unlockFeatures?.length),
  ).toBe(true);
  expect(levelById('1-11').unlockFeatures).toEqual([
    'levels.features.autodiff',
    'levels.features.expression',
  ]);
  expect(levelById('1-8').derivativeInput).toBe('a');
  expect(levelById('1-9').derivativeInput).toBe('b');
  expect(levelById('1-10').derivativeInputs).toEqual(['a', 'b']);
  expect(levelById('1-11').derivativeInput).toBeUndefined();
  expect(levelById('1-11').derivativeInputs).toBeUndefined();
  expect(levelById('1-11').referenceOutputs).toBeDefined();
});

for (const { id, wrong } of [
  { id: '1-8', wrong: 'b' },
  { id: '1-9', wrong: 'a * exp(a * b)' },
]) {
  it(`${id} accepts the reference at par and rejects the missing derivative term`, async () => {
    const level = levelById(id);
    const correct = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
    expect(correct.passed).toBe(true);
    expect(correct.tests).toHaveLength(12);
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
    const result = await evaluateLevel(
      playable(level),
      formulaGraph(wrong, Object.keys(level.interface.inputs), level.interface.output),
      { registry },
    );
    expect(result.errorKey).toBeUndefined();
    expect(result.passed).toBe(false);
    expect(result.tests.some((test) => !test.passed && !test.errorKey)).toBe(true);
  });
}

const gradientLessons = [
  {
    id: '1-10',
    formulas: { '∂a': 'b + exp(a)', '∂b': 'a' },
    expected: (a: number, b: number) => [b + Math.exp(a), a],
  },
] as const;

for (const { id, formulas, expected } of gradientLessons) {
  it(`${id} checks both gradient components against the independent mathematics`, async () => {
    const level = levelById(id);
    const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
    expect(result.errorKey).toBeUndefined();
    expect(result.passed).toBe(true);
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);

    const random = new Random(level.generated!.seed);
    const cases = [
      ...level.fixed,
      ...Array.from({ length: level.generated!.count }, (_, index) =>
        level.generated!.generate(random, index),
      ),
    ];
    expect(result.tests).toHaveLength(cases.length);
    result.tests.forEach((test, index) => {
      const { a, b } = cases[index].inputs;
      const independent = expected(a[0], b[0]);
      expect(test.actual).toEqual(test.expected);
      expect(test.expected[0]).toBeCloseTo(independent[0], 6);
      expect(test.expected[1]).toBeCloseTo(independent[1], 6);
    });
  });

  it(`${id} rejects either missing gradient output`, async () => {
    const level = playable(levelById(id));
    for (const output of ['∂a', '∂b'] as const) {
      const result = await evaluateLevel(
        level,
        formulaOutputsGraph({ [output]: formulas[output] }, ['a', 'b']),
        { registry },
      );
      expect(result.passed).toBe(false);
      expect(result.errorKey).toBe('levels.output_shape');
    }
  });

  it(`${id} rejects swapped and individually wrong gradient components`, async () => {
    const level = playable(levelById(id));
    const candidates = [
      { '∂a': formulas['∂b'], '∂b': formulas['∂a'] },
      { '∂a': '0 * a', '∂b': formulas['∂b'] },
      { '∂a': formulas['∂a'], '∂b': '0 * b' },
    ];
    for (const candidate of candidates) {
      const result = await evaluateLevel(level, formulaOutputsGraph(candidate, ['a', 'b']), {
        registry,
      });
      expect(result.errorKey).toBeUndefined();
      expect(result.passed).toBe(false);
      expect(result.tests.some((test) => !test.passed)).toBe(true);
    }
  });
}

it('1-11 sends an upstream gradient back along all three local input edges', async () => {
  const level = playable(levelById('1-11'));
  const result = await evaluateLevel(level, level.referenceGraph!, { registry });
  expect(result.errorKey).toBeUndefined();
  expect(result.passed).toBe(true);
  expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);

  const random = new Random(level.generated!.seed);
  const cases = [
    ...level.fixed,
    ...Array.from({ length: level.generated!.count }, (_, index) =>
      level.generated!.generate(random, index),
    ),
  ];
  expect(level.fixed.map((test) => test.inputs['∂f/∂z'][0])).toEqual([
    1, -1, 0.5, 2, -0.25, 0, 1.5, -2,
  ]);
  result.tests.forEach((test, index) => {
    const { w, x } = cases[index].inputs;
    const [g] = cases[index].inputs['∂f/∂z'];
    test.actual.forEach((actual, output) => expect(actual).toBeCloseTo(test.expected[output], 6));
    [g * x[0], g * w[0], g].forEach((expected, output) =>
      expect(Math.abs(test.expected[output] - expected)).toBeLessThan(
        1e-5 * Math.max(1, Math.abs(expected)),
      ),
    );
  });

  const formulas = {
    '∂f/∂w': 'g * x',
    '∂f/∂x': 'g * w',
    '∂f/∂b': 'g',
  };
  for (const missing of Object.keys(formulas)) {
    const candidate = Object.fromEntries(
      Object.entries(formulas).filter(([output]) => output !== missing),
    );
    const graph = formulaOutputsGraph(candidate, ['w', 'x', 'g']);
    graph.nodes.find((node) => node.component === 'input' && node.attrs?.name === 'g')!.attrs = {
      name: '∂f/∂z',
    };
    const missingResult = await evaluateLevel(level, graph, { registry });
    expect(missingResult.passed).toBe(false);
    expect(missingResult.errorKey).toBe('levels.output_shape');
  }
});
