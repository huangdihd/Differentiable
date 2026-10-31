// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { afterEach, expect, it, vi } from 'vitest';
import { CpuBackend } from '../engine/cpu';
import * as T from '../engine/tensor';
import { TrainingSession, type TrainingBatch } from '../engine/training';
import { VOCAB } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import {
  CharacterRecurrentModel,
  recurrentCharacterPredictionBatch,
  releaseRecurrentBatch,
} from './character-recurrent-model';
import type { FixedWindowResponse } from './fixed-window-protocol';
import { RecurrentCharacterExperiment } from './recurrent-character-experiment';

import {
  recurrentCharacterParameterTestProgress,
  recurrentCharacterTestProgress,
} from './recurrent-character-test-fixture';

afterEach(() => vi.restoreAllMocks());

it('really trains 320 then 640 steps once, updates every parameter and performs full/reset inference with the same weights', async () => {
  const defaultBackend = T.defaultBackend as CpuBackend;
  const defaultBuffers = defaultBackend.liveBuffers;
  const initialBackend = new CpuBackend();
  const initialModel = new CharacterRecurrentModel(initialBackend, {
    progress: recurrentCharacterTestProgress(),
  });
  const initialParameters = new Map<string, Float32Array>();
  for (const [name, value] of initialModel.parameters)
    initialParameters.set(name, await value.read());
  initialModel.dispose();
  expect(initialBackend.liveBuffers).toBe(0);

  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  const training = vi.spyOn(TrainingSession.prototype, 'run');
  const evaluation = vi.spyOn(TrainingSession.prototype, 'evaluate');
  const progress: Extract<FixedWindowResponse, { type: 'progress' }>[] = [];
  // Immediate start and throttled step-only updates (no loss readback).
  const stepOnly: number[] = [];
  let backend: CpuBackend | undefined;
  try {
    const pending = experiment.run((report) => {
      if (report.loss !== undefined) progress.push(report);
      else stepOnly.push(report.step);
    });
    await expect(experiment.run()).rejects.toThrow('worker.training_busy');
    const first = await pending;
    const session = training.mock.contexts[0] as TrainingSession;
    const model = session.model as CharacterRecurrentModel;
    backend = [...model.parameters.values()][0].backend as CpuBackend;
    const buffersAfterFirst = backend.liveBuffers;
    expect(backend).toBeInstanceOf(CpuBackend);
    expect(backend).not.toBe(defaultBackend);
    expect(T.defaultBackend).toBe(defaultBackend);
    expect(defaultBackend.liveBuffers).toBe(defaultBuffers);
    expect(first).toMatchObject({
      model: 'rnn',
      parameterCount: 1683,
      sampleCount: 1495,
      step: 320,
      stepsPerRun: 320,
      seed: 4105,
      learningRate: 0.02,
    });
    const initial = await evaluation.mock.results[0].value;
    expect(initial.step).toBe(0);
    expect(first.loss).toBeGreaterThan(0);
    expect(first.loss).toBeLessThan(initial.loss);
    const serializedFirst = JSON.stringify(first);
    const second = await experiment.run((report) => {
      if (report.loss !== undefined) progress.push(report);
      else stepOnly.push(report.step);
    });
    expect(JSON.stringify(first)).toBe(serializedFirst);
    expect(second).toMatchObject({ step: 640, stepsPerRun: 320, learningRate: 0.02 });
    expect(second.loss).toBeGreaterThan(0);
    expect(second.loss).toBeLessThan(initial.loss);
    expect(training.mock.contexts).toEqual([session, session]);
    expect(training.mock.calls.map(([steps]) => steps)).toEqual([320, 320]);
    expect(session.optimizer.step).toBe(640);
    expect(backend.liveBuffers).toBe(buffersAfterFirst);
    expect(progress.map(({ step }) => step)).toEqual(
      Array.from({ length: 64 }, (_, index) => (index + 1) * 10),
    );
    expect(progress.every(({ loss }) => Number.isFinite(loss))).toBe(true);
    expect(progress.slice(0, 32).every(({ totalSteps }) => totalSteps === 320)).toBe(true);
    expect(progress.slice(32).every(({ totalSteps }) => totalSteps === 640)).toBe(true);
    for (const [name, parameter] of model.parameters) {
      const values = await parameter.read();
      expect(values.every(Number.isFinite)).toBe(true);
      expect(
        values.some((value, index) => value !== initialParameters.get(name)![index]),
        name,
      ).toBe(true);
    }

    for (const snapshot of [first, second]) {
      expect(snapshot.predictions.map(({ history }) => history)).toEqual(
        fixedWindowExamples.map(({ history }) => history),
      );
      expect(snapshot.resetPredictions?.map(({ history }) => history)).toEqual(
        fixedWindowExamples.map(({ history }) => history),
      );
      expect(snapshot.predictions.map(({ input }) => input)).toEqual(
        fixedWindowExamples.map(({ history }) => history),
      );
      expect(snapshot.resetPredictions?.map(({ input }) => input)).toEqual(['th', 'th']);
      expect(snapshot.predictions[0].probabilities).not.toEqual(
        snapshot.predictions[1].probabilities,
      );
      expect(snapshot.resetPredictions![0].probabilities).toEqual(
        snapshot.resetPredictions![1].probabilities,
      );
      expect(snapshot.resetPredictions![0].probabilities).not.toBe(
        snapshot.resetPredictions![1].probabilities,
      );
      snapshot.predictions.forEach((prediction, index) => {
        expect(prediction.probabilities).not.toEqual(
          snapshot.resetPredictions![index].probabilities,
        );
      });
      for (const prediction of [...snapshot.predictions, ...snapshot.resetPredictions!]) {
        expect(prediction.probabilities).toHaveLength(VOCAB.length);
        expect(
          prediction.probabilities.every((value) => Number.isFinite(value) && value >= 0),
        ).toBe(true);
        expect(prediction.probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 5);
        expect(prediction.prediction).toBe(
          VOCAB[prediction.probabilities.indexOf(Math.max(...prediction.probabilities))],
        );
      }
    }
    expect(second.predictions[0].probabilities).not.toEqual(first.predictions[0].probabilities);
    // Read the live model independently: the snapshot is not a canned lesson answer.
    for (const prediction of [...second.predictions, ...second.resetPredictions!]) {
      const batch = recurrentCharacterPredictionBatch(prediction.input, backend);
      let outputs: Record<string, T.Tensor> | undefined;
      try {
        outputs = model.execute(batch.inputs);
        expect(Array.from(await outputs.probabilities.read())).toEqual(prediction.probabilities);
      } finally {
        if (outputs) model.releaseExecution(outputs, Object.values(batch.inputs));
        releaseRecurrentBatch(batch);
      }
    }
    expect(backend.liveBuffers).toBe(buffersAfterFirst);
    await expect(experiment.run()).rejects.toThrow('worker.rnn_complete');
    expect(training).toHaveBeenCalledTimes(2);
    console.info(
      JSON.stringify({
        recurrentCharacter: [first, second].map(({ step, loss, predictions }) => ({
          step,
          loss,
          predictions: predictions.map(({ prediction }) => prediction),
        })),
      }),
    );
    const batch = session.batch as TrainingBatch;
    experiment.dispose();
    for (const value of Object.values(batch.inputs))
      expect(() => value.assertLive()).toThrow('engine.tensor.disposed');
    await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
  } finally {
    experiment.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
  expect(defaultBackend.liveBuffers).toBe(defaultBuffers);
  // Fully expanded player circuits intentionally do not use the native matrix shortcut.
}, 900000);

it('supports local calibration options and keeps Adam state across the two configured runs', async () => {
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress(), {
    learningRate: 0.02,
    stepsPerRun: 1,
  });
  const training = vi.spyOn(TrainingSession.prototype, 'run');
  let backend: CpuBackend | undefined;
  try {
    expect(await experiment.run()).toMatchObject({ learningRate: 0.02, stepsPerRun: 1, step: 1 });
    const session = training.mock.contexts[0] as TrainingSession;
    backend = [...session.model.parameters.values()][0].backend as CpuBackend;
    const buffers = backend.liveBuffers;
    expect(await experiment.run()).toMatchObject({ learningRate: 0.02, stepsPerRun: 1, step: 2 });
    expect(session.optimizer.options).toEqual({ kind: 'adam', learningRate: 0.02, clipNorm: 1 });
    expect(training.mock.contexts).toEqual([session, session]);
    expect(session.optimizer.step).toBe(2);
    expect(backend.liveBuffers).toBe(buffers);
  } finally {
    experiment.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
});

it.each([
  { learningRate: 0 },
  { learningRate: -1 },
  { learningRate: NaN },
  { learningRate: Infinity },
  { stepsPerRun: 0 },
  { stepsPerRun: -1 },
  { stepsPerRun: 1.5 },
  { stepsPerRun: Infinity },
])('rejects invalid options %j without allocating', (options) => {
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  expect(
    () => new RecurrentCharacterExperiment(recurrentCharacterTestProgress(), options),
  ).toThrow();
  expect(allocation).not.toHaveBeenCalled();
});

it('trains a legal player parameter and reports it in the experiment parameter count', async () => {
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterParameterTestProgress(), {
    stepsPerRun: 1,
  });
  const training = vi.spyOn(TrainingSession.prototype, 'run');
  let backend: CpuBackend | undefined;
  try {
    const first = await experiment.run();
    expect(first).toMatchObject({ step: 1, parameterCount: 1684 });
    const session = training.mock.contexts[0] as TrainingSession;
    const parameter = session.model.parameters.get('cell/embedding')!;
    backend = parameter.backend as CpuBackend;
    const value = (await parameter.read())[0];
    expect(Number.isFinite(value)).toBe(true);
    expect(value).not.toBe(1);
    const baseline = backend.liveBuffers;
    const second = await experiment.run();
    expect(second).toMatchObject({ step: 2, parameterCount: 1684 });
    expect(training.mock.contexts).toEqual([session, session]);
    expect((await session.model.parameters.get('cell/embedding')!.read())[0]).not.toBe(value);
    expect(backend.liveBuffers).toBe(baseline);
  } finally {
    experiment.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
}, 20000);

it('reports nonfinite player parameters and releases lazily allocated storage', async () => {
  const progress = recurrentCharacterParameterTestProgress();
  // A finite graph attribute can still overflow float32 storage at initialization.
  progress.passedSolutions['time-rnn'].nodes.find(({ id }) => id === 'embedding')!.attrs = {
    shape: [],
    init: 'constant',
    value: Number.MAX_VALUE,
  };
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  const experiment = new RecurrentCharacterExperiment(progress, { stepsPerRun: 1 });
  await expect(experiment.run()).rejects.toThrow('training.nonfinite');
  for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
    expect(backend.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
});

it('disposes unused experiments without allocating', async () => {
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  experiment.dispose();
  experiment.dispose();
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
  expect(allocation).not.toHaveBeenCalled();
});

it('waits for in-flight initialization to finish before releasing resources', async () => {
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  const pending = experiment.run();
  experiment.dispose();
  await expect(pending).rejects.toThrow('worker.rnn_disposed');
  for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
    expect(backend.liveBuffers).toBe(0);
});

it('cancels after an optimizer update and releases parameters, tapes, moments and batches', async () => {
  const training = vi.spyOn(TrainingSession.prototype, 'run');
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress(), {
    stepsPerRun: 1,
  });
  let backend: CpuBackend | undefined;
  await expect(
    experiment.run((report) => {
      if (report.loss === undefined) return;
      const session = training.mock.contexts[0] as TrainingSession;
      backend = [...session.model.parameters.values()][0].backend as CpuBackend;
      // The final step's loss is drained before its update; dispose still cancels the run.
      expect(report.step).toBe(1);
      experiment.dispose();
    }),
  ).rejects.toThrow('worker.rnn_disposed');
  expect(backend?.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
});

it('propagates initialization readback failures and releases all resources', async () => {
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  vi.spyOn(T.Tensor.prototype, 'read').mockRejectedValue(new Error('initial readback failure'));
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  await expect(experiment.run()).rejects.toThrow('initial readback failure');
  for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
    expect(backend.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
});

it('propagates training and progress failures without retrying partly trained state', async () => {
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress(), {
    stepsPerRun: 1,
  });
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  await expect(
    experiment.run((report) => {
      if (report.loss === undefined) return;
      throw new Error('progress callback failure');
    }),
  ).rejects.toThrow('progress callback failure');
  for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
    expect(backend.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');

  vi.spyOn(TrainingSession.prototype, 'run').mockRejectedValue(new Error('training failure'));
  const failed = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  await expect(failed.run()).rejects.toThrow('training failure');
  for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
    expect(backend.liveBuffers).toBe(0);
  await expect(failed.run()).rejects.toThrow('worker.rnn_disposed');
});

it('releases partial forward tapes and existing Adam moments when the second training step fails', async () => {
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress(), {
    stepsPerRun: 2,
  });
  const submit = CpuBackend.prototype.submit;
  let updated = false;
  let backend: CpuBackend | undefined;
  vi.spyOn(CpuBackend.prototype, 'submit').mockImplementation(function (
    this: CpuBackend,
    commands,
  ) {
    backend = this;
    if (updated && commands.some(({ op }) => op === 'exp'))
      throw new Error('second forward execution failure');
    submit.call(this, commands);
    if (commands.some(({ op }) => op === 'optimizer_adam')) updated = true;
  });
  await expect(experiment.run()).rejects.toMatchObject({
    key: 'component_error',
    params: { message: 'second forward execution failure' },
  });
  expect(updated).toBe(true);
  expect(backend?.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
});

it.each(['vjp', 'optimizer_adam'])(
  'propagates %s kernel failures and releases the execution and optimizer resources',
  async (operation) => {
    const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress(), {
      stepsPerRun: 1,
    });
    const submit = CpuBackend.prototype.submit;
    let backend: CpuBackend | undefined;
    let matchingCommands = 0;
    vi.spyOn(CpuBackend.prototype, 'submit').mockImplementation(function (
      this: CpuBackend,
      commands,
    ) {
      backend = this;
      if (commands.some(({ op }) => op === operation) && ++matchingCommands === 2)
        throw new Error(`injected ${operation} failure`);
      submit.call(this, commands);
    });
    await expect(experiment.run()).rejects.toThrow(`injected ${operation} failure`);
    expect(matchingCommands).toBe(2);
    expect(backend?.liveBuffers).toBe(0);
    await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
  },
);

it.each(['readback', 'nonfinite', 'normalization'] as const)(
  'cleans prediction resources on %s failure without hiding diagnostics',
  async (failure) => {
    const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
    vi.spyOn(TrainingSession.prototype, 'run').mockResolvedValue({ step: 320, loss: 1 });
    const read = T.Tensor.prototype.read;
    vi.spyOn(T.Tensor.prototype, 'read').mockImplementation(function (this: T.Tensor) {
      if (this.shape[0] !== 1 || this.shape[1] !== VOCAB.length) return read.call(this);
      if (failure === 'readback') return Promise.reject(new Error('prediction readback failure'));
      return Promise.resolve(
        new Float32Array(VOCAB.length).fill(failure === 'nonfinite' ? NaN : 1),
      );
    });
    const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
    const expected = {
      readback: 'prediction readback failure',
      nonfinite: 'training.nonfinite',
      normalization: 'worker.rnn_probabilities',
    }[failure];
    await expect(experiment.run()).rejects.toThrow(expected);
    for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
      expect(backend.liveBuffers).toBe(0);
  },
);

it('defers disposal while a prediction readback is pending and rejects the cancelled result', async () => {
  let reachedRead!: () => void;
  let finishRead!: () => void;
  const reached = new Promise<void>((resolve) => {
    reachedRead = resolve;
  });
  const waiting = new Promise<void>((resolve) => {
    finishRead = resolve;
  });
  let model: CharacterRecurrentModel | undefined;
  let backend: CpuBackend | undefined;
  vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(async function (
    this: TrainingSession,
  ) {
    model = this.model as CharacterRecurrentModel;
    backend = [...model.parameters.values()][0].backend as CpuBackend;
    return { step: 320, loss: 1 };
  });
  const read = T.Tensor.prototype.read;
  vi.spyOn(T.Tensor.prototype, 'read').mockImplementation(async function (this: T.Tensor) {
    if (this.shape[0] === 1 && this.shape[1] === VOCAB.length) {
      reachedRead();
      await waiting;
    }
    return read.call(this);
  });
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  const pending = experiment.run();
  try {
    await reached;
    experiment.dispose();
    expect(backend!.liveBuffers).toBeGreaterThan(0);
    for (const value of model!.parameters.values()) expect(() => value.assertLive()).not.toThrow();
  } finally {
    finishRead();
    await expect(pending).rejects.toThrow('worker.rnn_disposed');
    experiment.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
});

it('rejects a progress snapshot without the passed cell and cleans the allocated batch', async () => {
  const progress = recurrentCharacterTestProgress();
  delete progress.passedSolutions['time-rnn'];
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  const experiment = new RecurrentCharacterExperiment(progress);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_player_cell_required');
  for (const backend of new Set(allocation.mock.contexts as CpuBackend[]))
    expect(backend.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.rnn_disposed');
});
