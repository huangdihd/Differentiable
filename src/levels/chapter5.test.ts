import { expect, it } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { evaluateLevel, graphMetrics } from '../engine/levels';
import { Random } from '../engine/random';
import { crossEntropy, TrainingSession } from '../engine/training';
import { createInitialProgress, loadProgressWithMigration, saveProgress } from '../storage';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels, chapter4TrainLevels } from './chapter4';
import { temporalForwardLevels } from './chapter-temporal';
import {
  attentionConfig,
  chapter5ForwardLevels,
  chapter5Levels,
  chapter5TrainLevels,
} from './chapter5';
import { referenceRegistry } from './level-helpers';

const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
  ...temporalForwardLevels,
];
const registry = referenceRegistry([...preceding, ...chapter5ForwardLevels]);
const earned = [...preceding, ...chapter5ForwardLevels].flatMap(
  (level) => level.unlockComponents ?? [],
);
for (const level of chapter5ForwardLevels) {
  it(`${level.id} reference passes all fixed, generated and batched shape variants`, async () => {
    const playable = { ...level, available: [...level.available, ...earned] };
    const result = await evaluateLevel(playable, level.referenceGraph!, { registry });
    expect(result.passed, JSON.stringify(result)).toBe(true);
    expect(result.tests).toHaveLength(30);
    const wrong = structuredClone(level.referenceGraph!);
    wrong.edges.find((edge) => edge.to.node === 'output')!.from.node = Object.keys(
      level.interface.inputs,
    )[0];
    expect((await evaluateLevel(playable, wrong, { registry })).passed).toBe(false);
  });
}
for (const id of ['5-2', '5-4']) {
  it(`${id} rejects hardcoded T even when the primary fixtures pass`, async () => {
    const level = chapter5ForwardLevels.find((entry) => entry.id === id)!;
    const hardcoded = structuredClone(level.referenceGraph!);
    const length = hardcoded.nodes.find((node) => node.id === 'length')!;
    length.component = 'const';
    length.attrs = { value: 4 };
    hardcoded.edges = hardcoded.edges.filter((edge) => edge.to.node !== 'length');
    const result = await evaluateLevel(
      { ...level, available: [...level.available, ...earned] },
      hardcoded,
      { registry },
    );
    expect(result.tests.slice(0, 10).every((test) => test.passed)).toBe(true);
    expect(result.passed).toBe(false);
    expect(result.tests.some((test) => test.id.startsWith('shape-') && !test.passed)).toBe(true);
  });
}

it('Attention contracts preserve progression, interfaces and expanded par', () => {
  console.info(
    'chapter5 par',
    chapter5Levels.map((level) => [level.id, level.par]),
  );
  expect(chapter5Levels.map((level) => level.id)).toEqual(['5-1', '5-2', '5-3', '5-4', '5-5']);
  expect(chapter5Levels.map((level) => level.unlock)).toEqual([
    ['5-2'],
    ['5-3'],
    ['5-4'],
    ['5-5'],
    ['6-1'],
  ]);
  expect(chapter5ForwardLevels.flatMap((level) => level.unlockComponents!)).toEqual([
    'attn_scores',
    'causal_mask',
    'attention',
    'embed_pos',
  ]);
  for (const level of chapter5Levels) {
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
    expect(level.hintsKeys).toHaveLength(2);
  }
});

it('old sequence graduates unlock temporal prediction without losing saved circuits', async () => {
  const data = new Map<string, string>();
  const source = () => ({
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
  });
  const graph = chapter4TrainLevels[1].referenceGraph!;
  const progress = {
    ...createInitialProgress(),
    unlocked: ['4-5', '5-1'],
    passed: ['4-5'],
    savedGraphs: { '4-5': graph },
    passedSolutions: { '4-5': graph },
  };
  expect(saveProgress(progress, source)).toBe(true);
  const levels = [...chapter4TrainLevels, ...chapter5Levels];
  const restored = await loadProgressWithMigration(levels, source);
  expect(restored.unlocked).toEqual(['4-5', '5-1', 'time-window-limit']);
  expect(restored.savedGraphs).toEqual(progress.savedGraphs);
  expect(restored.passedSolutions).toEqual(progress.passedSolutions);
  expect(saveProgress(restored, source)).toBe(true);
  expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
});

it('5-5 reference beats the threshold and no-attention embed_pos → linear control on CPU', async () => {
  const level = chapter5TrainLevels[0];
  const train = async (graph: GraphJSON) => {
    const start = performance.now();
    const plan = compileGraph(graph, {
      registry,
      random: new Random(level.seed),
      inputShapes: level.interface.inputs,
    });
    const batch = level.dataset(new Random(level.seed));
    const evaluation = level.evaluationDataset!(new Random(level.seed + 1));
    const session = new TrainingSession(
      plan,
      batch,
      crossEntropy,
      level.optimizer,
      level.interface.output,
      { evaluationBatch: evaluation },
    );
    try {
      const initial = await session.evaluate();
      const final = await session.run(level.steps, { yieldEvery: 1000 });
      return {
        initial: initial.loss,
        loss: final.loss,
        runtimeMs: performance.now() - start,
        ...graphMetrics(graph, registry),
      };
    } finally {
      session.dispose();
      plan.dispose();
      for (const data of [batch, evaluation]) {
        Object.values(data.inputs).forEach((tensor) => tensor.dispose());
        data.targets?.dispose();
      }
    }
  };
  const control = structuredClone(level.referenceGraph!);
  const removed = new Set(['q', 'k', 'v', 'qWeight', 'kWeight', 'vWeight', 'attended']);
  control.nodes = control.nodes.filter((node) => !removed.has(node.id));
  control.edges = control.edges.filter((edge) => !removed.has(edge.to.node));
  control.edges.forEach((edge) => {
    if (edge.from.node === 'attended') edge.from.node = 'embedded';
  });
  const reference = await train(level.referenceGraph!);
  const baseline = await train(control);
  console.info(
    JSON.stringify({
      id: level.id,
      ...attentionConfig,
      seed: level.seed,
      steps: level.steps,
      threshold: level.threshold.loss,
      chapter4Threshold: chapter4TrainLevels[1].threshold.loss,
      par: level.par,
      reference,
      baseline,
    }),
  );
  expect(reference.loss).toBeLessThan(reference.initial);
  expect(reference.loss).toBeLessThan(level.threshold.loss);
  // Not compared with 4-5: its line was set on 512 memorizable samples, a different task.
  expect(baseline.loss).toBeGreaterThan(level.threshold.loss);
  expect(reference.loss).toBeLessThan(baseline.loss);
}, 60000);
