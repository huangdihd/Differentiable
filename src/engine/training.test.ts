// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { tensor, mul, sum } from './tensor';
import { Optimizer, TrainingSession, type TrainingBatch, type TrainingModel } from './training';
import { workerError } from '../worker/runtime';

it.each([0, -1, NaN, Infinity, -Infinity])(
  'reports the actual invalid learning rate %s and range',
  (actual) => {
    try {
      new Optimizer({ kind: 'sgd', learningRate: actual });
      expect.fail('expected invalid learning rate');
    } catch (error) {
      expect(workerError(error)).toEqual({
        key: 'training.learning_rate',
        node: undefined,
        port: undefined,
        params: {
          actual: String(actual),
          min: 0,
          max: Number.MAX_VALUE,
          minExclusive: true,
          rangeKey: 'training.learning_rate_range',
        },
      });
    }
  },
);

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
    expect(await evaluation.targets!.read()).toEqual(new Float32Array([2]));
    session.dispose();
    expect(f.backend.liveBuffers).toBe(2);
  } finally {
    evaluation.targets?.dispose();
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
    expect(await batch.targets!.read()).toEqual(new Float32Array([2]));
    expect(f.backend.liveBuffers).toBe(2);
  } finally {
    batch.targets?.dispose();
    f.parameters.forEach((t) => t.dispose());
  }
  expect(f.backend.liveBuffers).toBe(0);
});

it.each(['run', 'evaluate', 'trainStepTrace'] as const)(
  'replays only after %s fails, identifies the first probe, and releases the replay',
  async (method) => {
    const f = fixture();
    let executions = 0,
      batches = 0;
    f.model.execute = (inputs, probes) => {
      executions++;
      const y = mul(inputs.x, f.parameters.get('weight')!);
      probes?.set('first', y);
      probes?.set('later', y);
      return { y };
    };
    const session = new TrainingSession(
      f.model,
      () => {
        batches++;
        const x = tensor([Infinity], [1], f.backend);
        return { inputs: { x }, targets: x };
      },
      f.loss,
      { kind: 'sgd', learningRate: 0.1 },
      'y',
      { accuracy: false },
    );
    try {
      const operation = method === 'run' ? session.run(1) : session[method]();
      await expect(operation).rejects.toMatchObject({
        key: 'training.nonfinite',
        node: 'first',
        message: 'training.nonfinite',
        params: { actual: 'Infinity', index: 0, stageKey: 'training.stage.forward' },
      });
      expect(executions).toBe(2);
      expect(batches).toBe(1);
      expect(session.optimizer.step).toBe(0);
      expect(f.backend.liveBuffers).toBe(1);
    } finally {
      session.dispose();
      f.parameters.forEach((t) => t.dispose());
    }
    expect(f.backend.liveBuffers).toBe(0);
  },
);

it('does not replay successful executions or mask failures with replay exceptions', async () => {
  const f = fixture();
  let executions = 0;
  const execute = f.model.execute;
  f.model.execute = (inputs, trace) => {
    executions++;
    if (trace) throw new Error('replay failed');
    return execute(inputs);
  };
  const batch = f.batch();
  const session = new TrainingSession(
    f.model,
    batch,
    f.loss,
    { kind: 'sgd', learningRate: 0.1 },
    'y',
    { accuracy: false },
  );
  try {
    await session.evaluate();
    expect(executions).toBe(1);
    const broken = new TrainingSession(
      f.model,
      batch,
      () => tensor([NaN], [], f.backend),
      { kind: 'sgd', learningRate: 0.1 },
      'y',
      { accuracy: false },
    );
    await expect(broken.evaluate()).rejects.toMatchObject({ key: 'training.nonfinite' });
    expect(executions).toBe(3);
    expect(f.backend.liveBuffers).toBe(2);
  } finally {
    session.dispose();
    batch.targets?.dispose();
    f.parameters.forEach((t) => t.dispose());
  }
  expect(f.backend.liveBuffers).toBe(0);
});
