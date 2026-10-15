import { expect, it } from 'vitest';
import { evaluateLevel, trainingBatchSource, type TrainLevel } from './levels';
import { Random } from './random';
import * as T from './tensor';
import { type TrainingReport } from './training';
import { xorLevel, xorSolution, createExampleRegistry } from '../levels/examples';

it('resampling advances a reproducible seed+2 stream and requires held-out data', async () => {
  const level: TrainLevel = {
    ...xorLevel,
    resample: true,
    dataset: (random) => ({ inputs: { x: T.tensor([random.uniform()], [1]) } }),
    evaluationDataset: xorLevel.dataset,
  };
  const a = trainingBatchSource(level),
    b = trainingBatchSource(level);
  if (typeof a !== 'function' || typeof b !== 'function') throw new Error('expected factories');
  const random = new Random(level.seed + 2);
  const values: number[] = [];
  for (let step = 0; step < 3; step++) {
    const x = await a(step),
      y = await b(step);
    try {
      const value = (await x.inputs.x.read())[0];
      values.push(value);
      expect(value).toBe(Math.fround(random.uniform()));
      expect(await x.inputs.x.read()).toEqual(await y.inputs.x.read());
    } finally {
      x.inputs.x.dispose();
      y.inputs.x.dispose();
    }
  }
  expect(new Set(values).size).toBe(3);
  expect(() => trainingBatchSource({ ...level, evaluationDataset: undefined })).toThrow(
    'training.missing_evaluation_dataset',
  );
});

it('explicit opt-out preserves exact legacy loss and reports', async () => {
  const reportsA: TrainingReport[] = [],
    reportsB: TrainingReport[] = [];
  const level = { ...xorLevel, steps: 3 };
  const a = await evaluateLevel(level, xorSolution, {
    registry: createExampleRegistry(),
    onReport: (r) => reportsA.push(r),
  });
  const b = await evaluateLevel({ ...level, resample: false }, xorSolution, {
    registry: createExampleRegistry(),
    onReport: (r) => reportsB.push(r),
  });
  expect(a.errorKey).toBeUndefined();
  expect(a).toEqual(b);
  expect(reportsA).toHaveLength(3);
  expect(reportsA).toEqual(reportsB);
});

it.each([undefined, 25, 250])(
  'resampled reports use held-out evaluation at cadence %s and final',
  async (evaluationEvery) => {
    let batches = 0;
    const reports: TrainingReport[] = [];
    const result = await evaluateLevel(
      {
        ...xorLevel,
        steps: 51,
        resample: true,
        evaluationEvery,
        dataset: (random) => {
          batches++;
          return xorLevel.dataset(random);
        },
        evaluationDataset: () => ({
          inputs: { x: T.tensor([0, 0, 1, 1], [2, 2]) },
          targets: T.tensor([1, 0, 1, 0], [2, 2]),
        }),
      },
      xorSolution,
      { registry: createExampleRegistry(), onReport: (r) => reports.push(r) },
    );
    expect(result.errorKey).toBeUndefined();
    expect(batches).toBe(51);
    expect(reports.map((r) => r.step)).toEqual(
      evaluationEvery === 250 ? [51] : evaluationEvery === 25 ? [25, 50, 51] : [50, 51],
    );
    expect(reports.at(-1)).toEqual(result.training);
  },
);
