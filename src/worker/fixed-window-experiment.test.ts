import { afterEach, expect, it, onTestFailed, vi } from 'vitest';
import { CpuBackend } from '../engine/cpu';
import { WebGpuBackend } from '../engine/webgpu';
import { GraphError, GraphPlan } from '../engine/graph';
import * as T from '../engine/tensor';
import { TrainingSession } from '../engine/training';
import { encode, VOCAB } from '../levels/corpus';
import { characterComparisonPositions } from '../levels/character-comparison-data';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import { FixedWindowExperiment, FixedWindowWorkerHandler } from './fixed-window-experiment';
import { RecurrentCharacterExperiment } from './recurrent-character-experiment';
import { recurrentCharacterTestProgress } from './recurrent-character-test-fixture';
import type { FixedWindowRequest, FixedWindowResponse } from './fixed-window-protocol';

afterEach(() => vi.restoreAllMocks());

it('really trains 160 then 320 steps in one isolated session, preserves borrowed parameters and reproduces the seed', async () => {
  const defaultBackend = T.defaultBackend as CpuBackend;
  const initialBuffers = defaultBackend.liveBuffers;
  const experiment = new FixedWindowExperiment();
  const replay = new FixedWindowExperiment();
  onTestFailed(() => {
    experiment.dispose();
    replay.dispose();
  });
  const training = vi.spyOn(TrainingSession.prototype, 'run');
  const evaluation = vi.spyOn(TrainingSession.prototype, 'evaluate');
  const disposal = vi.spyOn(GraphPlan.prototype, 'dispose');
  const progress: Extract<FixedWindowResponse, { type: 'progress' }>[] = [];
  // Immediate start and throttled step-only updates (no loss readback).
  const stepOnly: number[] = [];
  let backend: CpuBackend | undefined;
  let replayBackend: CpuBackend | undefined;
  try {
    const pending = experiment.run((report) => {
      if (report.loss !== undefined) progress.push(report);
      else stepOnly.push(report.step);
    });
    await expect(experiment.run()).rejects.toThrow('worker.training_busy');
    const first = await pending;
    const session = training.mock.contexts[0] as TrainingSession;
    const plan = session.model as GraphPlan;
    backend = [...plan.parameters.values()][0].backend as CpuBackend;
    const buffersAfterFirst = backend.liveBuffers;
    expect(backend).toBeInstanceOf(CpuBackend);
    expect(backend).not.toBe(defaultBackend);
    expect(T.defaultBackend).toBe(defaultBackend);
    expect(defaultBackend.liveBuffers).toBe(initialBuffers);
    expect(buffersAfterFirst).toBeGreaterThan(0);
    expect(first).toMatchObject({
      model: 'fixed-window',
      parameterCount: 1299,
      sampleCount: 512,
      step: 160,
      stepsPerRun: 160,
      seed: 4105,
      learningRate: 0.03,
    });
    expect(Number.isFinite(first.loss)).toBe(true);
    expect(first.loss).toBeGreaterThan(0);
    const initial = await evaluation.mock.results[0].value;
    expect(initial.step).toBe(0);
    expect(first.loss).toBeLessThan(initial.loss);
    const firstSerialized = JSON.stringify(first);
    const second = await experiment.run((report) => {
      if (report.loss !== undefined) progress.push(report);
      else stepOnly.push(report.step);
    });
    expect(JSON.stringify(first)).toBe(firstSerialized);
    expect(second).toMatchObject({ step: 320, stepsPerRun: 160, seed: 4105, learningRate: 0.03 });
    expect(second.loss).toBeGreaterThan(0);
    expect(second.loss).toBeLessThan(first.loss);
    expect(training.mock.contexts).toEqual([session, session]);
    expect(training.mock.calls.map(([steps]) => steps)).toEqual([160, 160]);
    expect(session.optimizer.step).toBe(320);
    // Each run reports its starting step before any training work.
    expect(stepOnly[0]).toBe(0);
    expect(stepOnly).toContain(160);
    expect(backend.liveBuffers).toBe(buffersAfterFirst);
    expect(progress.map(({ step }) => step)).toEqual(
      Array.from({ length: 32 }, (_, index) => (index + 1) * 10),
    );
    expect(progress.every(({ loss }) => Number.isFinite(loss))).toBe(true);
    expect(progress.slice(0, 16).every(({ totalSteps }) => totalSteps === 160)).toBe(true);
    expect(progress.slice(16).every(({ totalSteps }) => totalSteps === 320)).toBe(true);

    for (const snapshot of [first, second]) {
      expect(snapshot.predictions.map(({ history }) => history)).toEqual(
        fixedWindowExamples.map(({ history }) => history),
      );
      expect(snapshot.predictions).toHaveLength(2);
      expect(snapshot.predictions[0].probabilities).toEqual(snapshot.predictions[1].probabilities);
      expect(snapshot.predictions[0].probabilities).not.toBe(snapshot.predictions[1].probabilities);
      expect(
        snapshot.predictions.map(
          ({ prediction }, index) => prediction === fixedWindowExamples[index].target,
        ),
      ).toEqual([false, true]);
      for (const prediction of snapshot.predictions) {
        expect(prediction.input).toBe('th');
        expect(prediction.prediction).toBe('e');
        expect(prediction.probabilities).toHaveLength(VOCAB.length);
        expect(
          prediction.probabilities.every((value) => Number.isFinite(value) && value >= 0),
        ).toBe(true);
        expect(prediction.probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 5);
        const best = prediction.probabilities.indexOf(Math.max(...prediction.probabilities));
        expect(prediction.prediction).toBe(VOCAB[best]);
      }
    }
    const probabilityIndex = encode('e')[0];
    const firstProbability = first.predictions[0].probabilities[probabilityIndex];
    const secondProbability = second.predictions[0].probabilities[probabilityIndex];
    expect(firstProbability).toBeGreaterThan(0.5);
    expect(secondProbability).toBeGreaterThan(0.5);
    expect(second.predictions[0].probabilities).not.toEqual(first.predictions[0].probabilities);

    // Predictions run on the trained plan itself; no separate batch-1 plan is compiled.
    expect(
      (disposal.mock.contexts as GraphPlan[]).filter((value) => value.inputShapes.x[0] === 1),
    ).toHaveLength(0);
    await expect(experiment.run()).rejects.toThrow('worker.fixed_window_complete');
    expect(training).toHaveBeenCalledTimes(2);
    expect(backend.liveBuffers).toBe(buffersAfterFirst);

    expect(await replay.run()).toEqual(first);
    const replaySession = training.mock.contexts[2] as TrainingSession;
    replayBackend = [...replaySession.model.parameters.values()][0].backend as CpuBackend;
    expect(replayBackend).not.toBe(backend);
    expect(T.defaultBackend).toBe(defaultBackend);
    expect(defaultBackend.liveBuffers).toBe(initialBuffers);

    const parameters = [...plan.parameters.values()];
    if (typeof session.batch === 'function') throw new Error('Expected fixed training batch');
    const batches = [session.batch];
    expect(session.batch.inputs.x.shape).toEqual([512, 2]);
    experiment.dispose();
    experiment.dispose();
    expect(session.optimizer.step).toBe(0);
    expect(plan.parameters.size).toBe(0);
    expect(backend.liveBuffers).toBe(0);
    for (const parameter of parameters)
      expect(() => parameter.assertLive()).toThrow('engine.tensor.disposed');
    for (const batch of batches) {
      for (const value of [...Object.values(batch.inputs), batch.targets])
        expect(() => value!.assertLive()).toThrow('engine.tensor.disposed');
    }
    await expect(experiment.run()).rejects.toThrow('worker.fixed_window_disposed');
    console.info(
      JSON.stringify({
        fixedWindow: [
          { step: first.step, loss: first.loss, probabilityE: firstProbability },
          { step: second.step, loss: second.loss, probabilityE: secondProbability },
        ],
      }),
    );
  } finally {
    experiment.dispose();
    replay.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
  expect(replayBackend?.liveBuffers).toBe(0);
  expect(defaultBackend.liveBuffers).toBe(initialBuffers);
}, 90000);

it('trains the expanded player graph reproducibly and continues its optimizer on all 1495 targets', async () => {
  // Full 320/640 budgets are exercised by the headless acceptance; this test bounds
  // engine work while verifying that production requests those budgets unchanged.
  const run = TrainingSession.prototype.run;
  const training = vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(function (
    this: TrainingSession,
    steps,
    options,
  ) {
    expect(steps).toBe(320);
    return run.call(this, 10, options);
  });
  const experiment = new FixedWindowExperiment(recurrentCharacterTestProgress());
  const replay = new FixedWindowExperiment(recurrentCharacterTestProgress());
  onTestFailed(() => {
    experiment.dispose();
    replay.dispose();
  });
  let backend: CpuBackend | undefined;
  try {
    const first = await experiment.run();
    const session = training.mock.contexts[0] as TrainingSession;
    backend = [...session.model.parameters.values()][0].backend as CpuBackend;
    const buffers = backend.liveBuffers;
    const second = await experiment.run();
    expect(first).toMatchObject({
      step: 10,
      stepsPerRun: 320,
      sampleCount: 1495,
      learningRate: 0.02,
    });
    expect(second).toMatchObject({
      step: 20,
      stepsPerRun: 320,
      sampleCount: 1495,
      learningRate: 0.02,
    });
    expect(training.mock.contexts).toEqual([session, session]);
    expect(second.loss).toBeLessThan(first.loss);
    expect(backend.liveBuffers).toBe(buffers);
    for (const snapshot of [first, second])
      expect(snapshot.predictions[0].probabilities).toEqual(snapshot.predictions[1].probabilities);
    expect(await replay.run()).toEqual(first);
  } finally {
    experiment.dispose();
    replay.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
}, 60000);

it('trains and predicts with the original fixed-batch flatten without changing the saved graph', async () => {
  const progress = recurrentCharacterTestProgress();
  const flatten = progress.passedSolutions['4-5'].nodes.find((node) => node.id === 'flatten')!;
  flatten.attrs = { ...flatten.attrs, shape: [512, 16] };
  const before = JSON.stringify(progress);
  const run = TrainingSession.prototype.run;
  const training = vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(function (
    this: TrainingSession,
    steps,
    options,
  ) {
    expect(steps).toBe(320);
    return run.call(this, 1, options);
  });
  const experiment = new FixedWindowExperiment(progress);
  const dynamic = new FixedWindowExperiment(recurrentCharacterTestProgress());
  const backends = new Set<CpuBackend>();
  try {
    const first = await experiment.run();
    const expected = await dynamic.run();
    for (const session of training.mock.contexts as TrainingSession[])
      backends.add([...session.model.parameters.values()][0].backend as CpuBackend);
    expect(first).toEqual(expected);
    expect(first).toMatchObject({ step: 1, sampleCount: 1495, stepsPerRun: 320 });
    expect(first.predictions).toHaveLength(2);
    expect(first.predictions[0].probabilities).toEqual(first.predictions[1].probabilities);
    expect(JSON.stringify(progress)).toBe(before);
  } finally {
    experiment.dispose();
    dynamic.dispose();
  }
  for (const backend of backends) expect(backend.liveBuffers).toBe(0);
});

it('normalizes player output weights like generation without changing training, top-1 or the saved graph', async () => {
  const progress = recurrentCharacterTestProgress();
  const graph = progress.passedSolutions['4-5'];
  const output = graph.nodes.find((node) => node.component === 'output')!;
  const edge = graph.edges.find((connection) => connection.to.node === output.id)!;
  const source = edge.from;
  graph.nodes.push(
    { id: 'player_scale', component: 'const', attrs: { value: 1.0002 } },
    { id: 'player_scaled_output', component: 'mul' },
  );
  edge.from = { node: 'player_scaled_output', port: 'out' };
  graph.edges.push(
    { from: source, to: { node: 'player_scaled_output', port: 'a' } },
    {
      from: { node: 'player_scale', port: 'out' },
      to: { node: 'player_scaled_output', port: 'b' },
    },
  );
  const saved = JSON.stringify(progress);
  const run = TrainingSession.prototype.run;
  let backend: CpuBackend | undefined;
  vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(function (
    this: TrainingSession,
    steps,
    options,
  ) {
    expect(steps).toBe(320);
    backend = [...this.model.parameters.values()][0].backend as CpuBackend;
    return run.call(this, 1, options).then((report) => {
      predictionsStarted = true;
      return report;
    });
  });
  let predictionsStarted = false;
  const rawOutputs: number[][] = [];
  const read = T.Tensor.prototype.read;
  vi.spyOn(T.Tensor.prototype, 'read').mockImplementation(async function (this: T.Tensor) {
    const values = await read.call(this);
    if (predictionsStarted && this.shape.length === 2 && this.shape[1] === VOCAB.length)
      rawOutputs.push(Array.from(values).slice(0, VOCAB.length));
    return values;
  });
  const experiment = new FixedWindowExperiment(progress);
  try {
    const snapshot = await experiment.run();
    expect(snapshot.step).toBe(1);
    expect(rawOutputs).toHaveLength(2);
    for (const [index, prediction] of snapshot.predictions.entries()) {
      const raw = rawOutputs[index];
      const total = raw.reduce((sum, value) => sum + value, 0);
      expect(Math.abs(total - 1)).toBeGreaterThan(1e-5);
      expect(prediction.probabilities).toEqual(raw.map((value) => value / total));
      expect(prediction.prediction).toBe(VOCAB[raw.indexOf(Math.max(...raw))]);
      expect(prediction.probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    }
    expect(snapshot.predictions[0].probabilities).toEqual(snapshot.predictions[1].probabilities);
    expect(JSON.stringify(progress)).toBe(saved);
  } finally {
    experiment.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
});

it('compiles the saved fixed-window circuit rather than replacing it with the reference', async () => {
  const progress = recurrentCharacterTestProgress();
  const graph = progress.passedSolutions['4-5'];
  const parameter = graph.nodes.find((node) => node.component === 'param')!;
  const originalId = parameter.id;
  parameter.id = 'player_owned_embedding';
  for (const edge of graph.edges) {
    if (edge.from.node === originalId) edge.from.node = parameter.id;
    if (edge.to.node === originalId) edge.to.node = parameter.id;
  }
  const before = JSON.stringify(progress);
  let backend: CpuBackend | undefined;
  vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(async function (
    this: TrainingSession,
  ) {
    expect(this.model.parameters.has('player_owned_embedding')).toBe(true);
    backend = [...this.model.parameters.values()][0].backend as CpuBackend;
    expect(typeof this.batch).toBe('object');
    if (typeof this.batch !== 'function') expect(this.batch.inputs.x.shape).toEqual([1495, 2]);
    return { step: 320, loss: 1 };
  });
  const experiment = new FixedWindowExperiment(progress);
  try {
    expect(await experiment.run()).toMatchObject({
      step: 320,
      sampleCount: 1495,
      learningRate: 0.02,
    });
    expect(JSON.stringify(progress)).toBe(before);
  } finally {
    experiment.dispose();
  }
  expect(backend?.liveBuffers).toBe(0);
});

it('executes saved nested matrix multiplication instead of the native shortcut', async () => {
  const progress = recurrentCharacterTestProgress();
  const graph = progress.passedSolutions['2-5'];
  const output = graph.nodes.find((node) => node.component === 'output')!;
  const edge = graph.edges.find((connection) => connection.to.node === output.id)!;
  const source = edge.from;
  graph.nodes.push(
    { id: 'player_zero', component: 'const', attrs: { value: 0 } },
    { id: 'player_multiply', component: 'mul' },
  );
  edge.from = { node: 'player_multiply', port: 'out' };
  graph.edges.push(
    { from: source, to: { node: 'player_multiply', port: 'a' } },
    { from: { node: 'player_zero', port: 'out' }, to: { node: 'player_multiply', port: 'b' } },
  );
  vi.spyOn(TrainingSession.prototype, 'run').mockResolvedValue({ step: 320, loss: 1 });
  const ordinary = new FixedWindowExperiment(recurrentCharacterTestProgress());
  const changed = new FixedWindowExperiment(progress);
  try {
    const first = await ordinary.run();
    const second = await changed.run();
    expect(second.predictions[0].probabilities).not.toEqual(first.predictions[0].probabilities);
  } finally {
    ordinary.dispose();
    changed.dispose();
  }
});

it('rejects a missing player fixed graph instead of using a reference fallback', () => {
  const progress = recurrentCharacterTestProgress();
  delete progress.passedSolutions['4-5'];
  expect(() => new FixedWindowExperiment(progress)).toThrow(
    'worker.player_experiment_graph_required',
  );
});

it('disposes an unused experiment without allocations and rejects future runs', async () => {
  const allocation = vi.spyOn(CpuBackend.prototype, 'allocate');
  const experiment = new FixedWindowExperiment();
  experiment.dispose();
  experiment.dispose();
  await expect(experiment.run()).rejects.toThrow('worker.fixed_window_disposed');
  expect(allocation).not.toHaveBeenCalled();
});

it('waits for in-flight initialization before releasing its owned batches', async () => {
  const defaultBackend = T.defaultBackend as CpuBackend;
  const initialBuffers = defaultBackend.liveBuffers;
  const backends = new Set<CpuBackend>();
  const allocate = CpuBackend.prototype.allocate;
  vi.spyOn(CpuBackend.prototype, 'allocate').mockImplementation(function (
    this: CpuBackend,
    ...args
  ) {
    backends.add(this);
    return allocate.apply(this, args);
  });
  const experiment = new FixedWindowExperiment();
  const pending = experiment.run();
  experiment.dispose();
  await expect(pending).rejects.toThrow('worker.fixed_window_disposed');
  expect(defaultBackend.liveBuffers).toBe(initialBuffers);
  for (const backend of backends)
    if (backend !== defaultBackend) expect(backend.liveBuffers).toBe(0);
});

it('cancels between optimizer steps and releases tapes, optimizer state, plans and batches', async () => {
  const training = vi.spyOn(TrainingSession.prototype, 'run');
  const experiment = new FixedWindowExperiment();
  let backend: CpuBackend | undefined;
  await expect(
    experiment.run((report) => {
      if (report.loss === undefined) return;
      const session = training.mock.contexts[0] as TrainingSession;
      backend = [...session.model.parameters.values()][0].backend as CpuBackend;
      expect(session.optimizer.step).toBe(10);
      experiment.dispose();
    }),
  ).rejects.toThrow('worker.fixed_window_disposed');
  expect(backend?.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.fixed_window_disposed');
});

it('propagates engine failure and cleans initialized resources without retrying a partly trained session', async () => {
  let backend: CpuBackend | undefined;
  vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(async function (
    this: TrainingSession,
  ) {
    backend = [...this.model.parameters.values()][0].backend as CpuBackend;
    throw new Error('private engine diagnostic');
  });
  const experiment = new FixedWindowExperiment();
  await expect(experiment.run()).rejects.toThrow('private engine diagnostic');
  expect(backend?.liveBuffers).toBe(0);
  await expect(experiment.run()).rejects.toThrow('worker.fixed_window_disposed');
});

it('releases prediction executions and borrowed plans when readback fails', async () => {
  let backend: CpuBackend | undefined;
  vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(async function (
    this: TrainingSession,
  ) {
    backend = [...this.model.parameters.values()][0].backend as CpuBackend;
    predicting = true;
    return { step: 160, loss: 1 };
  });
  let predicting = false;
  const read = T.Tensor.prototype.read;
  vi.spyOn(T.Tensor.prototype, 'read').mockImplementation(function (this: T.Tensor) {
    if (predicting && this.shape.length === 2 && this.shape[1] === VOCAB.length)
      return Promise.reject(new Error('private readback diagnostic'));
    return read.call(this);
  });
  const experiment = new FixedWindowExperiment();
  await expect(experiment.run()).rejects.toThrow('private readback diagnostic');
  expect(backend?.liveBuffers).toBe(0);
});

it('worker handler forwards progress/results and reports only the generic localized error', async () => {
  const snapshot = {
    model: 'fixed-window' as const,
    parameterCount: 1299,
    sampleCount: characterComparisonPositions.length,
    step: 160,
    loss: 1,
    stepsPerRun: 160,
    seed: 4105,
    learningRate: 0.01,
    predictions: [],
  };
  const progress = { type: 'progress' as const, step: 10, totalSteps: 160, loss: 2 };
  const run = vi
    .spyOn(FixedWindowExperiment.prototype, 'run')
    .mockImplementation(async (onProgress) => {
      onProgress?.(progress);
      return snapshot;
    });
  const messages: FixedWindowResponse[] = [];
  const handler = new FixedWindowWorkerHandler((message) => messages.push(message));
  try {
    await handler.handle({ type: 'train', model: 'fixed-window' });
    expect(messages).toEqual([progress, { type: 'result', snapshot }]);
    run.mockRejectedValueOnce(new Error('secret stack or diagnostic'));
    await handler.handle({ type: 'train', model: 'fixed-window' });
    expect(messages.at(-1)).toEqual({ type: 'error', key: 'worker.execution_error', params: {} });
    await handler.handle({ type: 'reset' } as unknown as FixedWindowRequest);
    expect(messages.at(-1)).toEqual({ type: 'error', key: 'worker.invalid_request', params: {} });
    expect(run).toHaveBeenCalledTimes(2);
    expect(JSON.stringify(messages)).not.toMatch(/secret|stack|diagnostic/);
  } finally {
    handler.dispose();
  }
});

it('preserves the failing node, port and shape diagnostic without forwarding exception prose', async () => {
  vi.spyOn(FixedWindowExperiment.prototype, 'run').mockRejectedValue(
    new GraphError('reshape_mismatch', 'player_flatten', 'x', {
      actual: [1495, 2, 8],
      expected: [512, 16],
      message: 'private exception prose',
    }),
  );
  const messages: FixedWindowResponse[] = [];
  const handler = new FixedWindowWorkerHandler((message) => messages.push(message));
  try {
    await handler.handle({ type: 'train', model: 'fixed-window' });
    expect(messages).toEqual([
      {
        type: 'error',
        key: 'reshape_mismatch',
        node: 'player_flatten',
        port: 'x',
        params: { actual: [1495, 2, 8], expected: [512, 16] },
      },
    ]);
  } finally {
    handler.dispose();
  }
});

it('binds one model per worker and refuses switching, invalid models, and calls after disposal', async () => {
  const run = vi.spyOn(RecurrentCharacterExperiment.prototype, 'run').mockResolvedValue({
    model: 'rnn',
    parameterCount: 1683,
    sampleCount: 1495,
    step: 320,
    stepsPerRun: 320,
    seed: 4105,
    learningRate: 0.01,
    loss: 1,
    predictions: [],
    resetPredictions: [],
  });
  const fixedRun = vi.spyOn(FixedWindowExperiment.prototype, 'run');
  const messages: FixedWindowResponse[] = [];
  const handler = new FixedWindowWorkerHandler((message) => messages.push(message));
  await handler.handle({ type: 'train', model: 'invalid' } as unknown as FixedWindowRequest);
  expect(messages.at(-1)?.type).toBe('error');
  await handler.handle({ type: 'train', model: 'rnn' });
  expect(messages.at(-1)?.type).toBe('error');
  await handler.handle({
    type: 'train',
    model: 'rnn',
    playerProgress: recurrentCharacterTestProgress(),
  });
  expect(messages.at(-1)?.type).toBe('result');
  await handler.handle({
    type: 'train',
    model: 'rnn',
    playerProgress: recurrentCharacterTestProgress(),
  });
  expect(messages.at(-1)?.type).toBe('error');
  await handler.handle({ type: 'train', model: 'fixed-window' });
  expect(messages.at(-1)?.type).toBe('error');
  expect(fixedRun).not.toHaveBeenCalled();
  handler.dispose();
  await handler.handle({ type: 'train', model: 'rnn' });
  expect(messages.at(-1)?.type).toBe('error');
  expect(run).toHaveBeenCalledOnce();
});

it('worker handler creates the requested backend type in its own Worker', async () => {
  const snapshot = {
    model: 'fixed-window' as const,
    parameterCount: 1299,
    sampleCount: 512,
    step: 160,
    loss: 1,
    stepsPerRun: 160,
    seed: 4105,
    learningRate: 0.03,
    predictions: [],
  };
  const run = vi.spyOn(FixedWindowExperiment.prototype, 'run').mockResolvedValue(snapshot);
  const create = vi.spyOn(WebGpuBackend, 'create');
  const messages: FixedWindowResponse[] = [];
  const cpuHandler = new FixedWindowWorkerHandler((message) => messages.push(message));
  await cpuHandler.handle({ type: 'train', model: 'fixed-window', backend: 'cpu' });
  expect(create).not.toHaveBeenCalled();
  expect((run.mock.contexts[0] as FixedWindowExperiment)['backend']).toBeInstanceOf(CpuBackend);
  cpuHandler.dispose();

  const webGpuBackend = { dispose: vi.fn() } as unknown as WebGpuBackend;
  create.mockResolvedValue(webGpuBackend);
  const webGpuHandler = new FixedWindowWorkerHandler((message) => messages.push(message));
  await webGpuHandler.handle({ type: 'train', model: 'fixed-window', backend: 'webgpu' });
  expect(create).toHaveBeenCalledOnce();
  expect((run.mock.contexts[1] as FixedWindowExperiment)['backend']).toBe(webGpuBackend);
  webGpuHandler.dispose();
  expect(webGpuBackend.dispose).toHaveBeenCalledOnce();
});

it('worker handler rejects duplicate training rather than queueing a second run', async () => {
  let finish!: () => void;
  const waiting = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const training = vi.spyOn(TrainingSession.prototype, 'run').mockImplementation(async () => {
    await waiting;
    throw new Error('private failure after pending work');
  });
  const messages: FixedWindowResponse[] = [];
  const handler = new FixedWindowWorkerHandler((message) => messages.push(message));
  const pending = handler.handle({ type: 'train', model: 'fixed-window' });
  try {
    await handler.handle({ type: 'train', model: 'fixed-window' });
    expect(messages).toEqual([{ type: 'error', key: 'worker.training_busy', params: {} }]);
  } finally {
    finish();
    await pending;
    handler.dispose();
  }
  expect(training).toHaveBeenCalledTimes(1);
  expect(messages.filter((message) => message.type !== 'progress')).toEqual([
    { type: 'error', key: 'worker.training_busy', params: {} },
    { type: 'error', key: 'worker.execution_error', params: {} },
  ]);
});
