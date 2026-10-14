import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { tensor, mul, sum } from './tensor';
import { TrainingSession, type TrainingBatch, type TrainingModel } from './training';

function fixture() {
  const backend = new CpuBackend();
  const parameters = new Map([['weight', tensor([1], [1], backend)]]);
  const model: TrainingModel = {
    parameters,
    execute: (inputs) => ({ y: mul(inputs.x, parameters.get('weight')!) }),
  };
  const batch = (): TrainingBatch => {
    const x = tensor([2], [1], backend);
    return { inputs: { x }, targets: x }; // Shared input/target must be disposed exactly once.
  };
  const loss = (prediction: ReturnType<typeof tensor>) => sum(mul(prediction, prediction), [0]);
  return { backend, parameters, model, batch, loss };
}

it('owns fresh async batches, preserves fixed evaluation tensors and resumes factory step indices', async () => {
  const f = fixture(),
    evaluation = f.batch(),
    steps: number[] = [];
  const session = new TrainingSession(
    f.model,
    async (step) => {
      steps.push(step);
      return f.batch();
    },
    f.loss,
    { kind: 'adam', learningRate: 0.1 },
    'y',
    { accuracy: false, evaluationBatch: evaluation },
  );
  try {
    const initial = await session.evaluate();
    await session.run(2);
    const final = await session.run(1);
    expect(steps).toEqual([0, 1, 2]);
    expect(final.step).toBe(3);
    expect(final.loss).toBeLessThan(initial.loss);
    expect(await evaluation.targets.read()).toEqual(new Float32Array([2]));
    session.dispose();
    expect(f.backend.liveBuffers).toBe(2);
  } finally {
    evaluation.targets.dispose();
    f.parameters.forEach((t) => t.dispose());
  }
  expect(f.backend.liveBuffers).toBe(0);
});

it('uses and releases a factory batch for implicit evaluation, including zero-step runs', async () => {
  const f = fixture(),
    steps: number[] = [];
  const session = new TrainingSession(
    f.model,
    (step) => {
      steps.push(step);
      return f.batch();
    },
    f.loss,
    { kind: 'sgd', learningRate: 0.1 },
    'y',
    { accuracy: false },
  );
  try {
    await session.run(1);
    await session.run(0);
    expect(steps).toEqual([0, 1, 1]);
    const controller = new AbortController();
    controller.abort();
    await session.run(2, { signal: controller.signal });
    expect(steps).toEqual([0, 1, 1, 1]);
    expect(f.backend.liveBuffers).toBe(1);
  } finally {
    f.parameters.forEach((t) => t.dispose());
  }
  expect(f.backend.liveBuffers).toBe(0);
});

it.each(['execute', 'loss', 'nonfinite', 'factory'] as const)(
  'releases owned batches on %s failure',
  async (failure) => {
    const f = fixture();
    if (failure === 'execute')
      f.model.execute = () => {
        throw new Error('execute');
      };
    const session = new TrainingSession(
      f.model,
      () => {
        if (failure === 'factory') throw new Error('factory');
        return f.batch();
      },
      failure === 'loss'
        ? () => {
            throw new Error('loss');
          }
        : failure === 'nonfinite'
          ? () => tensor([NaN], [], f.backend)
          : f.loss,
      { kind: 'adam', learningRate: 0.1 },
      'y',
      { accuracy: false },
    );
    try {
      await expect(session.run(1)).rejects.toThrow();
      expect(f.backend.liveBuffers).toBe(1);
      await expect(session.evaluate()).rejects.toThrow();
      expect(f.backend.liveBuffers).toBe(1);
    } finally {
      f.parameters.forEach((t) => t.dispose());
    }
    expect(f.backend.liveBuffers).toBe(0);
  },
);

it('borrows fixed training and explicit evaluation batches', async () => {
  const f = fixture(),
    batch = f.batch();
  const session = new TrainingSession(
    f.model,
    batch,
    f.loss,
    { kind: 'sgd', learningRate: 0.1 },
    'y',
    { accuracy: false },
  );
  try {
    await session.run(2);
    await session.evaluate(batch);
    expect(await batch.targets.read()).toEqual(new Float32Array([2]));
    expect(f.backend.liveBuffers).toBe(2);
  } finally {
    batch.targets.dispose();
    f.parameters.forEach((t) => t.dispose());
  }
  expect(f.backend.liveBuffers).toBe(0);
});
