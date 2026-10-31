// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { compileGraph, createDefaultRegistry, type GraphJSON } from './graph';
import { evaluateLevel, trainingOptimizer, type TrainLevel } from './levels';
import * as T from './tensor';
import { crossEntropy, TrainingSession } from './training';
import { WorkerRuntime } from '../worker/runtime';
import type { Command, ReplyData, WorkerResponse } from '../worker/protocol';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'w', component: 'param', attrs: { shape: [1, 2], init: 'constant', value: 0 } },
    { id: 'prob', component: 'softmax' },
    { id: 'out', component: 'output', attrs: { name: 'y' } },
  ],
  edges: [
    { from: { node: 'w', port: 'out' }, to: { node: 'prob', port: 'x' } },
    { from: { node: 'prob', port: 'out' }, to: { node: 'out', port: 'x' } },
  ],
};
const level: TrainLevel = {
  id: 'classification-test',
  kind: 'train',
  titleKey: 'test',
  descriptionKey: 'test',
  available: ['param', 'softmax'],
  unlock: [],
  interface: { inputs: {}, output: 'y', outputShape: [1, 2] },
  seed: 1,
  steps: 1,
  optimizer: { kind: 'sgd', learningRate: 0.1 },
  tunable: { learningRate: { min: 0.001, max: 1, default: 0.1 }, optimizer: ['adam', 'sgd'] },
  dataset: () => ({ inputs: {}, targets: T.tensor([0, 1], [1, 2]) }),
  threshold: { loss: 2, accuracy: 1 },
};

it('defaults to the first allowed optimizer, ignoring undeclared or invalid choices', () => {
  expect(trainingOptimizer(level)).toBe('adam');
  expect(trainingOptimizer(level, 'sgd')).toBe('sgd');
  expect(trainingOptimizer(level, 'invalid' as 'sgd')).toBe('adam');
  expect(trainingOptimizer({ ...level, tunable: undefined }, 'adam')).toBe('sgd');
  expect(trainingOptimizer({ ...level, tunable: { ...level.tunable!, optimizer: [] } })).toBe(
    'sgd',
  );
});

it('traces classification accuracy and nonzero gradients before updating, without leaks', async () => {
  const backend = new CpuBackend();
  const model = compileGraph(graph, { backend });
  const targets = T.tensor([0, 1], [1, 2], backend);
  const session = new TrainingSession(
    model,
    { inputs: {}, targets },
    crossEntropy,
    level.optimizer,
  );
  try {
    const trace = await session.stepTrace();
    expect(trace.accuracy).toBe(0);
    expect(trace.gradient.w.values[0]).toBeCloseTo(0.5);
    expect(trace.gradient.w.values[1]).toBeCloseTo(-0.5);
    expect(trace.gradient.prob.values[1]).toBeCloseTo(-2);
    expect(trace.parameters.w.after[1]).toBeCloseTo(0.05);
    expect((await session.stepTrace()).accuracy).toBe(1);
  } finally {
    session.dispose();
    model.dispose();
    targets.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('evaluates optimizer overrides and identifies accuracy-only failures', async () => {
  const adam = await evaluateLevel(level, graph);
  const sgd = await evaluateLevel(level, graph, { optimizer: 'sgd' });
  expect(adam.passed).toBe(true);
  expect(adam.training!.loss).toBeLessThan(sgd.training!.loss);
  const result = await evaluateLevel({ ...level, steps: 0 }, graph);
  expect(result.error).toEqual({ key: 'levels.train_accuracy', params: { accuracy: 0 } });
});

it('worker steps reset optimizer state on kind changes but preserve state on LR changes', async () => {
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime(
    (m) => messages.push(m),
    [level],
    undefined,
    createDefaultRegistry(),
  );
  let requestId = 0;
  const send = async (command: Command) => {
    await runtime.handle({ ...command, requestId: ++requestId, revision: 0 });
    const reply = messages.at(-1)!;
    expect(reply.type).toBe('reply');
    return reply.type === 'reply' ? (reply.data as ReplyData['trainStep']) : undefined!;
  };
  try {
    await send({ type: 'init', backend: 'cpu' });
    const command = { type: 'trainStep' as const, levelId: level.id, graph };
    const adam = await send(command);
    expect(adam.parameters.w.after[1]).toBeCloseTo(0.1);
    expect((await send({ ...command, learningRate: 0.2 })).step).toBe(2);
    const sgd = await send({ ...command, optimizer: 'sgd' });
    expect(sgd.step).toBe(1);
    expect(sgd.parameters.w.after[1]).toBeCloseTo(0.05);
  } finally {
    await runtime.dispose();
  }
});
