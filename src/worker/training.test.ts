// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { afterEach, expect, it } from 'vitest';
import type { GraphJSON } from '../engine/graph';
import { initialProgress, type Level, type Progress, type TrainLevel } from '../engine/levels';
import { tensor } from '../engine/tensor';
import { WorkerRuntime } from './runtime';
import { WorkerClient, type WorkerTransport } from './client';
import type { Command, ReplyData, WorkerRequest, WorkerResponse } from './protocol';

import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import { chapter2Levels } from '../levels/chapter2';
import { chapter3ForwardLevels, chapter3TrainLevel } from '../levels/chapter3';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'w', component: 'param', attrs: { shape: [], init: 'constant', value: 2 } },
    { id: 'square', component: 'mul' },
    { id: 'loss', component: 'output', attrs: { name: 'loss' } },
  ],
  edges: [
    { from: { node: 'w', port: 'out' }, to: { node: 'square', port: 'a' } },
    { from: { node: 'w', port: 'out' }, to: { node: 'square', port: 'b' } },
    { from: { node: 'square', port: 'out' }, to: { node: 'loss', port: 'x' } },
  ],
};
const level: TrainLevel = {
  id: 'scalar',
  kind: 'train',
  titleKey: 'test',
  descriptionKey: 'test',
  available: ['param', 'mul'],
  unlock: [],
  interface: { inputs: {}, output: 'loss', outputShape: [] },
  seed: 42,
  steps: 3,
  optimizer: { kind: 'sgd', learningRate: 0.1 },
  tunable: { learningRate: { min: 0.01, max: 0.2, default: 0.1 } },
  dataset: () => ({ inputs: {}, targets: tensor([0], []) }),
  lossOutput: 'loss',
  threshold: { loss: 0.01 },
};
const runtimes: WorkerRuntime[] = [];
afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((r) => r.dispose()));
});
function harness(levels: readonly Level[] = [level], progress: Progress = initialProgress([])) {
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime((m) => messages.push(structuredClone(m)), levels, progress);
  runtimes.push(runtime);
  let requestId = 0;
  return {
    messages,
    async send(command: Command, revision = 0) {
      await runtime.handle({ ...command, requestId: ++requestId, revision });
      return messages.at(-1)!;
    },
  };
}
function trace(message: WorkerResponse): ReplyData['trainStep'] {
  expect(message.type).toBe('reply');
  if (message.type !== 'reply' || message.command !== 'trainStep')
    throw new Error(JSON.stringify(message));
  return message.data as ReplyData['trainStep'];
}
it('keeps three SGD steps, returns pre-update traces and post-update parameters, and resets deterministically', async () => {
  const h = harness();
  await h.send({ type: 'init', backend: 'cpu' });
  let w = 2;
  let first: ReplyData['trainStep'] | undefined;
  for (let step = 1; step <= 3; step++) {
    const t = trace(await h.send({ type: 'trainStep', levelId: level.id, graph }));
    first ??= t;
    expect(t.step).toBe(step);
    expect(t.loss).toBeCloseTo(w * w);
    expect(t.forward.w).toMatchObject({ shape: [], size: 1 });
    expect(t.forward.w.values[0]).toBeCloseTo(w);
    expect(t.gradient.w.values[0]).toBeCloseTo(2 * w);
    expect(t.parameters.w.before[0]).toBeCloseTo(w);
    w *= 0.8;
    expect(t.parameters.w.after[0]).toBeCloseTo(w);
  }
  expect(await h.send({ type: 'trainReset' })).toMatchObject({ data: { state: 'reset' } });
  expect(trace(await h.send({ type: 'trainStep', levelId: level.id, graph }))).toEqual(first);
  expect(h.messages.some((m) => m.type === 'result')).toBe(false);
});
it('rebuilds on graph, level and revision changes but retains parameters on rate changes', async () => {
  const h = harness([level, { ...level, id: 'other' }]);
  await h.send({ type: 'init', backend: 'cpu' });
  const command = { type: 'trainStep' as const, levelId: level.id, graph };
  await h.send(command);
  const changed = structuredClone(graph);
  changed.nodes[0].attrs!.value = 3;
  expect(trace(await h.send({ ...command, graph: changed })).loss).toBe(9);
  expect(trace(await h.send({ ...command, levelId: 'other' })).step).toBe(1);
  const resumed = trace(await h.send({ ...command, levelId: 'other', learningRate: 0.2 }));
  expect(resumed.step).toBe(2);
  expect(resumed.parameters.w.before[0]).toBeCloseTo(1.6);
  expect(resumed.parameters.w.after[0]).toBeCloseTo(0.96);
  expect(trace(await h.send(command, 1)).step).toBe(1);
});
it('rejects step/reset while continuous training owns a session, including paused sessions', async () => {
  const h = harness();
  await h.send({ type: 'init', backend: 'cpu' });
  await h.send({ type: 'start', levelId: level.id, graph });
  expect(await h.send({ type: 'trainStep', levelId: level.id, graph })).toMatchObject({
    error: { key: 'worker.training_busy' },
  });
  await h.send({ type: 'pause' });
  expect(await h.send({ type: 'trainReset' })).toMatchObject({
    error: { key: 'worker.training_busy' },
  });
  await h.send({ type: 'stop' });
  expect(trace(await h.send({ type: 'trainStep', levelId: level.id, graph })).step).toBe(1);
});
it('applies declared learning-rate bounds/defaults and ignores overrides on non-tunable levels', async () => {
  const h = harness([level, { ...level, id: 'fixed', tunable: undefined }]);
  await h.send({ type: 'init', backend: 'cpu' });
  for (const [learningRate, expected] of [
    [10, 1.2],
    [-1, 1.96],
    [NaN, 1.6],
    [Infinity, 1.6],
  ]) {
    await h.send({ type: 'trainReset' });
    const t = trace(await h.send({ type: 'trainStep', levelId: level.id, graph, learningRate }));
    expect(t.parameters.w.after[0]).toBeCloseTo(expected);
  }
  const fixed = trace(
    await h.send({ type: 'trainStep', levelId: 'fixed', graph, learningRate: 0.2 }),
  );
  expect(fixed.parameters.w.after[0]).toBeCloseTo(1.6);
  const evaluated = await h.send({ type: 'evaluate', levelId: level.id, graph, learningRate: 0.2 });
  expect(evaluated).toMatchObject({
    type: 'reply',
    data: { training: { step: 3, loss: expect.any(Number) } },
  });
  if (evaluated.type !== 'reply') return;
  const result = evaluated.data as ReplyData['evaluate'];
  expect(result.training!.loss).toBeCloseTo((2 * 0.6 ** 3) ** 2);
  await h.send({ type: 'start', levelId: level.id, graph, learningRate: 0.2 });
  for (let i = 0; i < 100 && !h.messages.some((m) => m.type === 'result'); i++)
    await new Promise((r) => setTimeout(r, 5));
  const finished = h.messages.find((m) => m.type === 'result');
  expect(finished?.type).toBe('result');
  if (finished?.type === 'result')
    expect(finished.result.training!.loss).toBeCloseTo(result.training!.loss);
});
it('bounds trace arrays to 256 values while retaining full sizes and shapes', async () => {
  const wide = structuredClone(graph);
  wide.nodes[0].attrs!.shape = [300];
  wide.nodes.push({ id: 'sum', component: 'sum', attrs: { axes: [0] } });
  wide.edges[2] = { from: { node: 'square', port: 'out' }, to: { node: 'sum', port: 'x' } };
  wide.edges.push({ from: { node: 'sum', port: 'out' }, to: { node: 'loss', port: 'x' } });
  const h = harness([{ ...level, available: [...level.available, 'sum'] }]);
  await h.send({ type: 'init', backend: 'cpu' });
  const t = trace(await h.send({ type: 'trainStep', levelId: level.id, graph: wide }));
  for (const values of [t.forward.w, t.gradient.w]) {
    expect(values.shape).toEqual([300]);
    expect(values.size).toBe(300);
    expect(values.values).toHaveLength(256);
  }
  expect(t.parameters.w.before).toHaveLength(300);
  expect(t.parameters.w.after).toHaveLength(300);
});
it('client forwards optional rates and exposes step/reset RPCs', async () => {
  const requests: WorkerRequest[] = [];
  let listener: (event: MessageEvent<WorkerResponse>) => void;
  const transport: WorkerTransport = {
    postMessage(request) {
      requests.push(request);
      listener({
        data: {
          type: 'reply',
          requestId: request.requestId,
          revision: request.revision,
          command: request.type,
          data: { state: 'reset' },
        },
      } as MessageEvent<WorkerResponse>);
    },
    addEventListener(_type, fn) {
      listener = fn;
    },
    removeEventListener() {},
    terminate() {},
  };
  const client = new WorkerClient(transport);
  await client.trainStep(level.id, graph, 0.05);
  await client.trainReset();
  await client.start(level.id, graph, 0.06);
  await client.evaluate(level.id, graph, 0.07);
  expect(
    requests.map(({ type, learningRate }: WorkerRequest & { learningRate?: number }) => [
      type,
      learningRate,
    ]),
  ).toEqual([
    ['trainStep', 0.05],
    ['trainReset', undefined],
    ['start', 0.06],
    ['evaluate', 0.07],
  ]);
  client.dispose();
});

async function finalResult(h: ReturnType<typeof harness>) {
  for (let i = 0; i < 200 && !h.messages.some((m) => m.type === 'result'); i++)
    await new Promise((resolve) => setTimeout(resolve, 5));
  const message = h.messages.find((m) => m.type === 'result');
  if (message?.type !== 'result') throw new Error('Missing training result');
  return message.result;
}
function chapterHarness() {
  const previous = [
    ...chapter0Levels,
    ...chapter1Levels,
    ...chapter2Levels,
    ...chapter3ForwardLevels,
  ];
  const progress = initialProgress([]);
  for (const item of previous) {
    progress.passed.push(item.id);
    progress.unlockedComponents.push(...(item.unlockComponents ?? []));
    if (item.referenceGraph) progress.passedSolutions[item.id] = item.referenceGraph;
  }
  return harness([...previous, chapter3TrainLevel], progress);
}
it('3-4 reference retains SGD updates for three decreasing losses and resets from its seed', async () => {
  const h = chapterHarness();
  await h.send({ type: 'init', backend: 'cpu' });
  const command = {
    type: 'trainStep' as const,
    levelId: '3-4',
    graph: chapter3TrainLevel.referenceGraph!,
  };
  const traces: ReplyData['trainStep'][] = [];
  for (let i = 0; i < 3; i++) {
    const t = trace(await h.send(command));
    expect(t.step).toBe(i + 1);
    if (i) expect(t.loss).toBeLessThan(traces[i - 1].loss);
    expect(Object.keys(t.parameters).sort()).toEqual(['b', 'w']);
    for (const [id, parameter] of Object.entries(t.parameters)) {
      if (i) expect(parameter.before).toEqual(traces[i - 1].parameters[id].after);
      parameter.after.forEach((value, j) =>
        expect(value).toBeCloseTo(parameter.before[j] - 0.01 * t.gradient[id].values[j], 6),
      );
    }
    traces.push(t);
  }
  await h.send({ type: 'trainReset' });
  expect(trace(await h.send(command))).toEqual(traces[0]);
});
it('3-4 continuous training classifies high-rate divergence and slow 60-step training', async () => {
  for (const [learningRate, key] of [
    [2, 'levels.train_diverged'],
    [0.001, 'levels.train_slow'],
  ] as const) {
    const h = chapterHarness();
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({
      type: 'start',
      levelId: '3-4',
      graph: chapter3TrainLevel.referenceGraph!,
      learningRate,
    });
    const result = await finalResult(h);
    expect(result.passed).toBe(false);
    expect(result.errorKey).toBe(key);
    expect(result.error).toEqual({ key, params: { loss: result.training!.loss } });
    if (key === 'levels.train_slow') expect(result.training!.step).toBe(60);
    else expect(result.training!.step).toBeLessThan(60);
  }
});
it('continuous training classifies nonfinite loss but not stopped or fixed runs', async () => {
  const h = harness([level, { ...level, id: 'fixed', tunable: undefined }]);
  await h.send({ type: 'init', backend: 'cpu' });
  const overflowing = structuredClone(graph);
  overflowing.nodes[0].attrs!.value = 1e30;
  await h.send({ type: 'start', levelId: level.id, graph: overflowing });
  expect(await finalResult(h)).toMatchObject({
    passed: false,
    error: { key: 'levels.train_diverged', params: { loss: Infinity } },
  });
  h.messages.length = 0;
  await h.send({ type: 'start', levelId: 'fixed', graph });
  expect((await finalResult(h)).error).toBeUndefined();
  h.messages.length = 0;
  await h.send({ type: 'start', levelId: level.id, graph });
  await h.send({ type: 'stop' });
  expect(await finalResult(h)).toMatchObject({ passed: false });
  expect((await finalResult(h)).error).toBeUndefined();
});
