import { expect, it } from 'vitest';
import { evaluateLevel, graphMetrics } from '../engine/levels';
import { chapter6ForwardLevels, chapter6Preceding } from './chapter6-forward';
import { referenceRegistry } from './level-helpers';

const levels = [...chapter6Preceding, ...chapter6ForwardLevels];
const registry = referenceRegistry(levels);
const earned = levels.flatMap((level) => level.unlockComponents ?? []);
for (const level of chapter6ForwardLevels) {
  it(`${level.id} passes primary, generated and shape-variant fixtures`, async () => {
    const playable = { ...level, available: [...level.available, ...earned] };
    const result = await evaluateLevel(playable, level.referenceGraph!, { registry });
    expect(result.passed, JSON.stringify(result)).toBe(true);
    expect(result.tests).toHaveLength(10 * (1 + level.shapeVariants!.length));
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
    const wrong = structuredClone(level.referenceGraph!);
    wrong.edges.find((edge) => edge.to.node === 'output')!.from.node =
      level.id === '6-1' ? 'q' : 'x';
    expect((await evaluateLevel(playable, wrong, { registry })).passed).toBe(false);
  });
}
for (const [id, value] of [
  ['width', 8],
  ['length', 4],
] as const) {
  it(`6-1 rejects hardcoded ${id} despite passing the primary fixtures`, async () => {
    const level = chapter6ForwardLevels[0];
    const graph = structuredClone(level.referenceGraph!);
    const dimension = graph.nodes.find((node) => node.id === id)!;
    dimension.component = 'const';
    dimension.attrs = { value };
    graph.edges = graph.edges.filter((edge) => edge.to.node !== id);
    const result = await evaluateLevel(
      { ...level, available: [...level.available, ...earned] },
      graph,
      { registry },
    );
    expect(result.tests.slice(0, 10).every((test) => test.passed)).toBe(true);
    expect(result.passed).toBe(false);
    expect(result.tests.some((test) => test.id.startsWith('shape-') && !test.passed)).toBe(true);
  });
}
it('6-3 exposes all eight parameter inputs and forward progression', () => {
  expect(Object.keys(chapter6ForwardLevels[2].interface.inputs)).toEqual([
    'x',
    'wq',
    'wk',
    'wv',
    'wo',
    'w1',
    'b1',
    'w2',
    'b2',
  ]);
  expect(chapter6ForwardLevels.map((level) => level.unlock)).toEqual([['6-2'], ['6-3'], ['6-4']]);
});

it('preserves the multi_head and feed_forward order, ports and rank variants', () => {
  const [attention, feedforward] = chapter6ForwardLevels;
  expect(attention.unlockComponents).toEqual(['multi_head']);
  expect(Object.keys(attention.interface.inputs)).toEqual(['q', 'k', 'v']);
  expect(attention.interface.outputShape).toEqual([2, 4, 8]);
  expect(attention.shapeVariants![0].inputs.q).toEqual([3, 5, 12]);
  expect(feedforward.unlockComponents).toEqual(['feed_forward']);
  expect(Object.keys(feedforward.interface.inputs)).toEqual(['x', 'w1', 'b1', 'w2', 'b2']);
});
