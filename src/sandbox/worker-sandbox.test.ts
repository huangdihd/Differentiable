// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { afterEach, expect, it } from 'vitest';
import type { GraphJSON } from '../engine/graph';
import { initialProgress } from '../engine/levels';
import { WorkerRuntime } from '../worker/runtime';
import type { Command, ReplyData, WorkerResponse } from '../worker/protocol';
import { gameLevels } from '../levels/catalog';
import { exportWeights, parseWeights, sandboxLevelId } from './slots';

const edge = (from: string, to: string, port: string) => ({
  from: { node: from, port: 'out' },
  to: { node: to, port },
});
const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 't', component: 'input', attrs: { name: 't' } },
    { id: 'w', component: 'param', attrs: { shape: [], init: 'constant', value: 2 } },
    { id: 'p', component: 'mul' },
    { id: 'minus', component: 'const', attrs: { value: -1 } },
    { id: 'negT', component: 'mul' },
    { id: 'd', component: 'add' },
    { id: 'sq', component: 'mul' },
    { id: 'error', component: 'sum' },
    { id: 'loss', component: 'output', attrs: { name: 'loss' } },
  ],
  edges: [
    edge('w', 'p', 'a'),
    edge('x', 'p', 'b'),
    edge('t', 'negT', 'a'),
    edge('minus', 'negT', 'b'),
    edge('p', 'd', 'a'),
    edge('negT', 'd', 'b'),
    edge('d', 'sq', 'a'),
    edge('d', 'sq', 'b'),
    edge('sq', 'error', 'x'),
    edge('error', 'loss', 'x'),
  ],
};
const runtimes: WorkerRuntime[] = [];
afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((r) => r.dispose()));
});

it('trains a sandbox slot and round-trips its weights through a file', async () => {
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime(
    (m) => messages.push(structuredClone(m)),
    gameLevels,
    initialProgress([]),
  );
  runtimes.push(runtime);
  let requestId = 0;
  const send = async (command: Command) => {
    await runtime.handle({ ...command, requestId: ++requestId, revision: 0 });
    const last = messages.at(-1)!;
    if (last.type !== 'reply') throw new Error(JSON.stringify(last));
    return last.data;
  };
  const levelId = sandboxLevelId('a');
  await send({ type: 'init', backend: 'cpu' });
  const dataset = { kind: 'builtin' as const, id: 'noisy-line' as const };
  await send({ type: 'setSandbox', sandbox: { slotId: 'a', dataset } });
  const step = (await send({ type: 'trainStep', levelId, graph })) as ReplyData['trainStep'];
  expect(step.parameters.w.before[0]).toBeCloseTo(2);
  expect(Number.isFinite(step.loss)).toBe(true);
  const exported = (await send({
    type: 'exportWeights',
    levelId,
    graph,
  })) as ReplyData['exportWeights'];
  expect(exported.parameters.w.shape).toEqual([]);
  expect(exported.parameters.w.values[0]).toBeCloseTo(step.parameters.w.after[0]);
  const file = parseWeights(JSON.stringify(exportWeights(exported.parameters)));
  file.parameters.w.values = [5];
  await send({ type: 'setSandbox', sandbox: { slotId: 'a', dataset, weights: file } });
  const seeded = (await send({ type: 'trainStep', levelId, graph })) as ReplyData['trainStep'];
  expect(seeded.parameters.w.before[0]).toBeCloseTo(5);
});
