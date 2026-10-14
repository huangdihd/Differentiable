import { expect, it } from 'vitest';
import { evaluateLevel, graphMetrics } from '../engine/levels';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { formulaGraph, referenceRegistry } from './level-helpers';

const registry = referenceRegistry([...chapter0Levels, ...chapter1Levels]);
const lessons = chapter1Levels.slice(7);
const earned = [...chapter0Levels, ...chapter1Levels.slice(0, 7)].flatMap(
  (level) => level.unlockComponents ?? [],
);

it('turns the σ′ and tanh′ derivations into parts that 1-10 reuses', async () => {
  expect(chapter1Levels.map((level) => level.unlockComponents ?? [])).toEqual([
    [],
    [],
    [],
    [],
    ['dsigmoid'],
    ['dtanh'],
    [],
    [],
    [],
    [],
  ]);
  expect(lessons[2].referenceGraph!.nodes.some((n) => n.component === 'dsigmoid')).toBe(true);
  // Deriving σ′ again by hand is still a correct answer, just a longer one.
  const longhand = formulaGraph('sigmoid(w * x + b) * (1 + neg(sigmoid(w * x + b))) * x', [
    'w',
    'x',
    'b',
  ]);
  const playable = { ...lessons[2], available: [...lessons[2].available, ...earned] };
  expect((await evaluateLevel(playable, longhand, { registry })).passed).toBe(true);
});

it('moves the chapter-end tools to 1-10 and preserves the successor chain', () => {
  expect(chapter1Levels.slice(6).map((level) => level.unlock)).toEqual([
    ['1-8'],
    ['1-9'],
    ['1-10'],
    ['2-1'],
  ]);
  expect(chapter1Levels.slice(0, 9).every((level) => !level.unlockFeatures?.length)).toBe(true);
  expect(lessons[2].unlockFeatures).toEqual([
    'levels.features.autodiff',
    'levels.features.expression',
  ]);
  expect(lessons.map((level) => level.derivativeInput)).toEqual(['a', 'b', 'w']);
});

for (const [index, wrong] of [
  'b',
  'a * exp(a * b)',
  'sigmoid(w * x + b) * (1 + neg(sigmoid(w * x + b)))',
].entries()) {
  const level = lessons[index];
  it(`${level.id} accepts the reference at par and rejects the missing derivative term`, async () => {
    const playable = { ...level, available: [...level.available, ...earned] };
    const correct = await evaluateLevel(playable, level.referenceGraph!, { registry });
    expect(correct.passed).toBe(true);
    expect(correct.tests).toHaveLength(12);
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
    const result = await evaluateLevel(
      playable,
      formulaGraph(wrong, Object.keys(level.interface.inputs)),
      { registry },
    );
    expect(result.errorKey).toBeUndefined();
    expect(result.passed).toBe(false);
    expect(result.tests.some((test) => !test.passed && !test.errorKey)).toBe(true);
  });
}

it('keeps all fixed and seeded sigmoid logits away from saturation with an independent bias column', () => {
  const level = lessons[2];
  const random = new Random(level.generated!.seed);
  const cases = [
    ...level.fixed,
    ...Array.from({ length: level.generated!.count }, (_, i) =>
      level.generated!.generate(random, i),
    ),
  ];
  expect(level.fixed.map((test) => test.inputs.b[0])).toEqual([3, -1, 1, -0.5, 0.25, -1.5, 2, -2]);
  const logits = cases.map(({ inputs: { w, x, b } }) => w[0] * x[0] + b[0]);
  expect(logits).toMatchSnapshot();
  expect(Math.min(...logits)).toBeGreaterThanOrEqual(-6);
  expect(Math.max(...logits)).toBeLessThanOrEqual(6);
});
