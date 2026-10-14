import { expect, it } from 'vitest';
import { compileGraph } from '../engine/graph';
import { evaluateLevel, graphMetrics } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { crossEntropy, TrainingSession } from '../engine/training';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import {
  chapter4ForwardLevels,
  chapter4TrainLevels,
  chapter4Stats,
  chapter4Levels,
} from './chapter4';
import {
  bigramEntropy,
  contextBigramEntropy,
  corpusStats,
  decode,
  encode,
  sampleToken,
  vocabularySize,
} from './corpus';
import { referenceRegistry } from './level-helpers';
const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
];
const registry = referenceRegistry([...preceding, ...chapter4ForwardLevels]);
const earned = [...preceding, ...chapter4ForwardLevels].flatMap(
  (level) => level.unlockComponents ?? [],
);
for (const level of chapter4ForwardLevels) {
  it(`${level.id} checks actual circuits and shape variants`, async () => {
    const playable = { ...level, available: [...level.available, ...earned] };
    const result = await evaluateLevel(playable, level.referenceGraph!, { registry });
    expect(result.passed, JSON.stringify(result)).toBe(true);
    expect(result.tests.length).toBeGreaterThan(9);
    const wrong = structuredClone(level.referenceGraph!);
    wrong.edges.find((edge) => edge.to.node === 'output')!.from.node = Object.keys(
      level.interface.inputs,
    )[0];
    expect((await evaluateLevel(playable, wrong, { registry })).passed).toBe(false);
  });
}
for (const level of chapter4TrainLevels) {
  it(`${level.id} full reference training, calibrated distribution loss and seeded generation`, async () => {
    const plan = compileGraph(level.referenceGraph!, {
      registry,
      random: new Random(level.seed),
      inputShapes: level.interface.inputs,
    });
    const batch = level.dataset(new Random(level.seed));
    const evaluation = level.evaluationDataset!(new Random(level.seed + 1));
    const session = new TrainingSession(plan, batch, crossEntropy, level.optimizer, 'y', {
      evaluationBatch: evaluation,
    });
    try {
      const initial = await session.evaluate();
      const final = await session.run(level.steps, { yieldEvery: 1000 });
      expect(final.loss).toBeLessThan(initial.loss);
      expect(final.loss).toBeLessThan(level.threshold.loss);
      if (level.id === '4-3') expect(level.threshold.loss).toBeCloseTo(bigramEntropy + 0.15);
      else {
        expect(level.threshold.loss).toBeLessThan(contextBigramEntropy);
        expect(final.loss).toBeLessThan(contextBigramEntropy);
      }
      const generate = async () => {
        const random = new Random(4900),
          tokens = encode('the ');
        const context = level.id === '4-3' ? 1 : 2;
        for (let step = 0; step < 24; step++) {
          const values = Array.from({ length: level.interface.inputs.x[0] }, () =>
            tokens.slice(-context),
          ).flat();
          const x = T.tensor(
            values,
            context === 1 ? [level.interface.inputs.x[0]] : [level.interface.inputs.x[0], 2],
          );
          const output = plan.execute({ x });
          try {
            const probabilities = (await output.y.read()).slice(0, vocabularySize);
            expect([...probabilities].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
            tokens.push(sampleToken(probabilities, random));
          } finally {
            plan.releaseExecution(output);
            x.dispose();
          }
        }
        return decode(tokens);
      };
      const sample = await generate();
      expect(await generate()).toBe(sample);
      console.info(
        JSON.stringify({
          ...chapter4Stats.find((entry) => entry.id === level.id),
          corpusStats,
          initial,
          final,
          sample,
        }),
      );
    } finally {
      session.dispose();
      plan.dispose();
      for (const data of [batch, evaluation]) {
        Object.values(data.inputs).forEach((t) => t.dispose());
        data.targets?.dispose();
      }
    }
  }, 60000);
}

it('chapter contracts keep unlocks and interfaces in progression order', () => {
  expect(chapter4Levels.map((l) => l.id)).toEqual(['4-1', '4-2', '4-3', '4-4', '4-5']);
  expect(chapter4Levels.map((level) => level.unlock)).toEqual([
    ['4-2'],
    ['4-3'],
    ['4-4'],
    ['4-5'],
    [],
  ]);
  expect(chapter4Levels.map((level) => level.par)).toEqual([4, 8, 10, 5, 29]);
  for (const level of chapter4Levels) {
    expect(level.hintsKeys).toHaveLength(2);
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);
  }
  expect(chapter4ForwardLevels[0].interface.inputs).toEqual({ idx: [4], k: [5] });
  expect(chapter4ForwardLevels[0].shapeVariants![0].inputs).toEqual({ idx: [6], k: [7] });
  expect(chapter4ForwardLevels[1].interface.inputs).toEqual({ idx: [4], table: [5, 3] });
  expect(registry.get('embedding')!.inputs).toEqual(['idx', 'table']);
  expect(chapter4ForwardLevels[1].forbidden).toContain('gather');
  for (const level of chapter4ForwardLevels.slice(0, 2))
    expect(level.available).not.toContain('gather');
});
it('sampling reference returns each interior bin, including the last', async () => {
  const level = chapter4ForwardLevels[2];
  for (const fixture of level.fixed!) {
    const inputs = Object.fromEntries(
      Object.entries(fixture.inputs).map(([name, values]) => [
        name,
        T.tensor(values, level.interface.inputs[name]),
      ]),
    );
    const result = level.reference(inputs);
    try {
      expect([...(await result.read())]).toEqual([Number(fixture.id.split('-')[1]) % 5]);
    } finally {
      result.dispose();
      Object.values(inputs).forEach((t) => t.dispose());
    }
  }
});
