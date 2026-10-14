import { buildBackwardGraph, traceBackward } from '../engine/backward-graph';
import type { GraphJSON } from '../engine/graph';
import { gameLevels } from '../levels/catalog';
import { initialProgress } from '../engine/levels';
import { afterEach, describe, expect, it } from 'vitest';
import {
  createExampleRegistry,
  exampleLevels,
  derivativeLevel,
  derivativeSolution,
  sigmoidSolution,
  sigmoidLevel,
  xorLevel,
  xorSolution,
} from '../levels/examples';
import { WorkerRuntime, workerError } from './runtime';
import { WorkerClient, type WorkerTransport } from './client';
import type { Command, WorkerRequest, WorkerResponse } from './protocol';

const runtimes: WorkerRuntime[] = [];
afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((r) => r.dispose()));
});
function harness(levels: ConstructorParameters<typeof WorkerRuntime>[1] = exampleLevels) {
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime(
    (m) => messages.push(structuredClone(m)),
    levels,
    initialProgress([]),
    createExampleRegistry(),
  );
  runtimes.push(runtime);
  let requestId = 0;
  return {
    runtime,
    messages,
    async send(command: Command, revision = 0) {
      await runtime.handle({ ...command, revision, requestId: ++requestId });
      return messages.at(-1)!;
    },
  };
}
async function until(check: () => boolean) {
  for (let i = 0; i < 300 && !check(); i++) await new Promise((r) => setTimeout(r, 5));
  expect(check()).toBe(true);
}
describe('worker runtime protocol', () => {
  it('uses the formal catalog by default', async () => {
    const messages: WorkerResponse[] = [];
    const runtime = new WorkerRuntime((message) => messages.push(message));
    runtimes.push(runtime);
    await runtime.handle({ type: 'init', backend: 'cpu', requestId: 1, revision: 0 });
    const level = gameLevels.find((item) => item.referenceGraph)!;
    expect(level).toBeDefined();
    await runtime.handle({
      type: 'compile',
      levelId: level.id,
      graph: level.referenceGraph!,
      requestId: 2,
      revision: 0,
    });
    expect(messages.at(-1)).toMatchObject({ type: 'reply', data: { diagnostics: [] } });
  });
  it('requires init and falls back to CPU in Node', async () => {
    const h = harness();
    expect(await h.send({ type: 'pause' })).toMatchObject({
      type: 'error',
      error: { key: 'worker.not_initialized', params: {} },
    });
    expect(await h.send({ type: 'init' })).toMatchObject({
      type: 'reply',
      data: { backend: 'cpu', fallback: { key: 'worker.cpu_fallback' } },
    });
  });
  it('compiles with structured diagnostics and probes augmented graphs without mutation', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    expect(
      await h.send({ type: 'compile', levelId: 'sigmoid', graph: sigmoidSolution }),
    ).toMatchObject({ data: { diagnostics: [], outputShapes: { y: [5] } } });
    const graph = structuredClone(sigmoidSolution);
    graph.edges = [];
    expect(await h.send({ type: 'compile', levelId: 'sigmoid', graph })).toMatchObject({
      data: { diagnostics: [{ key: expect.any(String), params: expect.any(Object) }] },
    });
    const before = structuredClone(sigmoidSolution);
    for (let i = 0; i < 3; i++) {
      expect(
        await h.send({
          type: 'probe',
          levelId: 'sigmoid',
          graph: sigmoidSolution,
          endpoint: { node: 'neg', port: 'out' },
          inputs: { x: [-4, -1, 0, 1, 4] },
        }),
      ).toMatchObject({ data: { shape: [5], values: [4, 1, -0, -1, -4] } });
    }
    expect(sigmoidSolution).toEqual(before);
  });
  it('probes the wired part of a half-built graph', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    // Only x -> neg is wired; the output is still unconnected, so a full compile would fail.
    const graph = structuredClone(sigmoidSolution);
    graph.edges = graph.edges.filter((e) => e.to.node === 'neg');
    expect(
      await h.send({
        type: 'probe',
        levelId: 'sigmoid',
        graph,
        endpoint: { node: 'neg', port: 'out' },
        inputs: { x: [-4, -1, 0, 1, 4] },
      }),
    ).toMatchObject({ data: { shape: [5], values: [4, 1, -0, -1, -4] } });
  });
  it('returns serializable backward graphs and optional traces without mutating the graph', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    const before = structuredClone(derivativeSolution);
    const options = {
      registry: createExampleRegistry(),
      availableComponents: derivativeLevel.available,
      inputShapes: derivativeLevel.interface.inputs,
      output: derivativeLevel.interface.output,
      wrt: ['x'],
    };
    const backward = buildBackwardGraph(derivativeSolution, options);
    expect(
      await h.send({ type: 'backward', levelId: 'derivative', graph: derivativeSolution }),
    ).toMatchObject({ type: 'reply', command: 'backward', data: { backward } });
    const inputs = { x: [-2, -1, 0, 1, 2] };
    const trace = await traceBackward(derivativeSolution, { ...options, inputs });
    for (let i = 0; i < 3; i++)
      expect(
        await h.send({
          type: 'backward',
          levelId: 'derivative',
          graph: derivativeSolution,
          inputs,
        }),
      ).toMatchObject({ type: 'reply', data: { backward, trace } });
    expect(derivativeSolution).toEqual(before);
  });
  it('selects derivativeInput only for derivative levels and all inputs otherwise', async () => {
    const graph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'a', component: 'input', attrs: { name: 'a' } },
        { id: 'b', component: 'input', attrs: { name: 'b' } },
        { id: 'sum', component: 'add' },
        { id: 'out', component: 'output', attrs: { name: 'chosen' } },
      ],
      edges: [
        { from: { node: 'a', port: 'out' }, to: { node: 'sum', port: 'a' } },
        { from: { node: 'b', port: 'out' }, to: { node: 'sum', port: 'b' } },
        { from: { node: 'sum', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    for (const kind of ['forward', 'derivative'] as const) {
      const level = {
        ...derivativeLevel,
        kind,
        derivativeInput: 'b',
        interface: { inputs: { a: [], b: [] }, output: 'chosen', outputShape: [] },
      };
      const h = harness([level]);
      await h.send({ type: 'init', backend: 'cpu' });
      const expected = buildBackwardGraph(graph, {
        registry: createExampleRegistry(),
        availableComponents: level.available,
        inputShapes: level.interface.inputs,
        output: 'chosen',
        wrt: kind === 'derivative' ? ['b'] : ['a', 'b'],
      });
      expect(await h.send({ type: 'backward', levelId: level.id, graph })).toMatchObject({
        type: 'reply',
        data: { backward: expected },
      });
    }
  });
  it('rejects invalid backward graphs, missing trace inputs and stale revisions', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    for (const command of [
      { type: 'backward', levelId: 'missing', graph: derivativeSolution },
      { type: 'backward', levelId: 'derivative', graph: { ...derivativeSolution, edges: [] } },
      { type: 'backward', levelId: 'derivative', graph: derivativeSolution, inputs: {} },
    ] satisfies Command[])
      expect(await h.send(command, 1)).toMatchObject({
        type: 'error',
        error: { key: expect.any(String), params: expect.any(Object) },
      });
    expect(
      await h.send({ type: 'backward', levelId: 'derivative', graph: derivativeSolution }, 0),
    ).toMatchObject({ type: 'error', error: { key: 'worker.stale_revision' } });
  });
  it('evaluates actual forward and derivative graphs and rejects old revisions', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    for (const [levelId, graph] of [
      ['sigmoid', sigmoidSolution],
      ['derivative', derivativeSolution],
    ] as const) {
      expect(await h.send({ type: 'evaluate', levelId, graph }, 2)).toMatchObject({
        revision: 2,
        data: { levelId, passed: true },
      });
    }
    expect(await h.send({ type: 'pause' }, 1)).toMatchObject({
      type: 'error',
      error: { key: 'worker.stale_revision' },
    });
  });
  it('evaluates shape variants while compile and probe keep the primary interface', async () => {
    const level = {
      ...sigmoidLevel,
      shapeVariants: [
        { inputs: { x: [3] }, fixed: [{ id: 'fixed-0', inputs: { x: [-1, 0, 1] } }] },
      ],
    };
    const h = harness([level]);
    await h.send({ type: 'init', backend: 'cpu' });
    expect(
      await h.send({ type: 'compile', levelId: level.id, graph: sigmoidSolution }),
    ).toMatchObject({ data: { outputShapes: { y: [5] } } });
    expect(
      await h.send({ type: 'evaluate', levelId: level.id, graph: sigmoidSolution }),
    ).toMatchObject({
      data: {
        passed: true,
        tests: expect.arrayContaining([
          expect.objectContaining({ id: 'shape-1/fixed-0', passed: true }),
        ]),
      },
    });
    expect(
      await h.send({
        type: 'probe',
        levelId: level.id,
        graph: sigmoidSolution,
        endpoint: { node: 'neg', port: 'out' },
        inputs: { x: [-4, -1, 0, 1, 4] },
      }),
    ).toMatchObject({ data: { shape: [5] } });
  });
  it('pauses and resumes the same optimizer, emits progress and final LevelResult', async () => {
    const h = harness([{ ...xorLevel, steps: 5 }]);
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({ type: 'start', levelId: 'xor', graph: xorSolution });
    await until(() => h.messages.some((m) => m.type === 'progress'));
    await h.send({ type: 'pause' });
    const count = h.messages.filter((m) => m.type === 'progress').length;
    await new Promise((r) => setTimeout(r, 20));
    expect(h.messages.filter((m) => m.type === 'progress')).toHaveLength(count);
    await h.send({ type: 'resume' });
    await until(() => h.messages.some((m) => m.type === 'result'));
    const reports = h.messages.filter((m) => m.type === 'progress');
    expect(reports.map((m) => m.report.step)).toEqual([1, 2, 3, 4, 5]);
    expect(h.messages.find((m) => m.type === 'result')).toMatchObject({
      requestId: 2,
      revision: 0,
      result: { levelId: 'xor', training: { step: 5 }, tests: [{ id: 'evaluation' }] },
    });
  });
  it('stop returns a nonpassing final result and releases training', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({ type: 'start', levelId: 'xor', graph: xorSolution });
    await h.send({ type: 'pause' });
    expect(await h.send({ type: 'stop' })).toMatchObject({ data: { state: 'stopped' } });
    expect(h.messages.find((m) => m.type === 'result')).toMatchObject({
      result: { passed: false, training: { step: 0 } },
    });
    expect(await h.send({ type: 'resume' })).toMatchObject({
      type: 'error',
      error: { key: 'worker.no_training' },
    });
  });
  it('cancels paused training on revision change without evaluating it', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({ type: 'start', levelId: 'xor', graph: xorSolution });
    await h.send({ type: 'pause' });
    expect(await h.send({ type: 'cancel' }, 1)).toMatchObject({
      type: 'reply',
      data: { state: 'stopped' },
      revision: 1,
    });
    expect(await h.send({ type: 'resume' }, 1)).toMatchObject({
      type: 'error',
      error: { key: 'worker.no_training' },
    });
    expect(h.messages.some((m) => m.type === 'result')).toBe(false);
  });
  it('releases training when stop evaluation fails', async () => {
    const h = harness([{ ...xorLevel, lossOutput: 'missing' }]);
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({ type: 'start', levelId: 'xor', graph: xorSolution });
    await h.send({ type: 'pause' });
    expect(await h.send({ type: 'stop' })).toMatchObject({
      type: 'error',
      error: { key: 'training.missing_loss' },
    });
    expect(await h.send({ type: 'resume' })).toMatchObject({
      type: 'error',
      error: { key: 'worker.no_training' },
    });
  });
  it('reports failed training and rejects requests after disposal', async () => {
    const h = harness([{ ...xorLevel, lossOutput: 'missing' }]);
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({ type: 'start', levelId: 'xor', graph: xorSolution });
    await until(() => h.messages.some((m) => m.type === 'result'));
    expect(h.messages.find((m) => m.type === 'result')).toMatchObject({
      result: { passed: false, errorKey: 'training.missing_loss' },
    });
    await h.runtime.dispose();
    expect(await h.send({ type: 'resume' })).toMatchObject({
      type: 'error',
      error: { key: 'worker.disposed' },
    });
  });
  it('sync rebuilds the registry from passed solutions and applies dynamic availability', async () => {
    const source = {
      ...derivativeLevel,
      id: 'earned',
      kind: 'forward' as const,
      unlockComponents: ['earnedComponent'],
    };
    const target = { ...source, id: 'consumer', available: [], unlockComponents: [] };
    const h = harness([source, target]);
    const graph = {
      version: 1 as const,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 'f', component: 'earnedComponent' },
        { id: 'y', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'f', port: 'x' } },
        { from: { node: 'f', port: 'out' }, to: { node: 'y', port: 'x' } },
      ],
    };
    await h.send({ type: 'init', backend: 'cpu' });
    const progress = initialProgress(['consumer'], ['earnedComponent']);
    progress.passed = ['earned'];
    progress.passedSolutions.earned = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 'y', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'y', port: 'x' } }],
    };
    expect(await h.send({ type: 'syncProgress', progress }, 1)).toMatchObject({
      data: { state: 'synced' },
    });
    expect(await h.send({ type: 'compile', levelId: 'consumer', graph }, 1)).toMatchObject({
      data: { diagnostics: [], outputShapes: { y: [5] } },
    });
    expect(await h.send({ type: 'evaluate', levelId: 'consumer', graph }, 1)).toMatchObject({
      data: {
        tests: expect.arrayContaining([expect.objectContaining({ actual: [-2, -1, 0, 1, 2] })]),
      },
    });
    await h.send({ type: 'syncProgress', progress: initialProgress([]) }, 2);
    expect(await h.send({ type: 'compile', levelId: 'consumer', graph }, 2)).toMatchObject({
      data: { diagnostics: [expect.objectContaining({ key: expect.any(String) })] },
    });
    expect(await h.send({ type: 'compile', levelId: 'consumer', graph }, 1)).toMatchObject({
      error: { key: 'worker.stale_revision' },
    });
  });
  it('sync releases paused training without emitting an old result', async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    await h.send({ type: 'start', levelId: 'xor', graph: xorSolution });
    await h.send({ type: 'pause' });
    await h.send({ type: 'syncProgress', progress: initialProgress([]) }, 1);
    expect(await h.send({ type: 'resume' }, 1)).toMatchObject({
      error: { key: 'worker.no_training' },
    });
    expect(h.messages.some((m) => m.type === 'result')).toBe(false);
  });
  it('does not leak arbitrary exception prose', () => {
    expect(workerError(new Error('private exception text'))).toEqual({
      key: 'worker.execution_error',
      params: {},
    });
  });
});
it('typed client matches IDs, rejects stale pending calls and filters training events', async () => {
  const requests: WorkerRequest[] = [];
  let listener: ((event: MessageEvent<WorkerResponse>) => void) | undefined;
  const transport: WorkerTransport = {
    postMessage: (m) => {
      requests.push(m);
    },
    addEventListener: (_, fn) => {
      listener = fn;
    },
    removeEventListener: () => {
      listener = undefined;
    },
    terminate: () => {},
  };
  const client = new WorkerClient(transport);
  const pending = client.backward('sigmoid', sigmoidSolution, { x: [-4, -1, 0, 1, 4] });
  expect(requests.at(-1)).toMatchObject({
    type: 'backward',
    levelId: 'sigmoid',
    graph: sigmoidSolution,
    inputs: { x: [-4, -1, 0, 1, 4] },
    revision: 0,
  });
  const rejected = expect(pending).rejects.toMatchObject({
    detail: { key: 'worker.stale_revision' },
  });
  client.setRevision(1);
  expect(requests.at(-1)).toMatchObject({ type: 'cancel', revision: 1 });
  await rejected;
  const events: WorkerResponse[] = [];
  client.subscribe((e) => events.push(e));
  const start = client.start('xor', xorSolution);
  const request = requests.at(-1)!;
  const emit = (data: WorkerResponse) => listener!({ data } as MessageEvent<WorkerResponse>);
  emit({
    type: 'reply',
    requestId: request.requestId,
    revision: 1,
    command: 'start',
    data: { state: 'running' },
  });
  await expect(start).resolves.toEqual({ state: 'running' });
  emit({
    type: 'progress',
    requestId: request.requestId,
    revision: 0,
    report: { step: 1, loss: 1 },
  });
  emit({
    type: 'progress',
    requestId: request.requestId + 1,
    revision: 1,
    report: { step: 1, loss: 1 },
  });
  expect(events).toHaveLength(0);
  emit({
    type: 'progress',
    requestId: request.requestId,
    revision: 1,
    report: { step: 1, loss: 1 },
  });
  expect(events).toHaveLength(1);
  const replacement = client.start('xor', xorSolution);
  const replacementRejected = expect(replacement).rejects.toMatchObject({
    detail: { key: 'worker.execution_error' },
  });
  emit({
    type: 'error',
    requestId: requests.at(-1)!.requestId,
    revision: 1,
    error: { key: 'worker.execution_error', params: {} },
  });
  await replacementRejected;
  emit({
    type: 'progress',
    requestId: request.requestId,
    revision: 1,
    report: { step: 2, loss: 0.5 },
  });
  expect(events).toHaveLength(2);
  const compile = client.compile('sigmoid', sigmoidSolution);
  const invalidReply = expect(compile).rejects.toMatchObject({
    detail: { key: 'worker.invalid_reply' },
  });
  emit({
    type: 'reply',
    requestId: requests.at(-1)!.requestId,
    revision: 1,
    command: 'pause',
    data: { state: 'paused' },
  });
  await invalidReply;
  const obsolete = client.compile('sigmoid', sigmoidSolution);
  const obsoleteRejected = expect(obsolete).rejects.toMatchObject({
    detail: { key: 'worker.stale_revision' },
  });
  const progress = initialProgress([]);
  const synced = client.syncProgress(progress);
  await obsoleteRejected;
  expect(requests.at(-2)).toMatchObject({ type: 'cancel', revision: 2 });
  expect(requests.at(-1)).toMatchObject({ type: 'syncProgress', revision: 2, progress });
  emit({
    type: 'progress',
    requestId: request.requestId,
    revision: 1,
    report: { step: 3, loss: 0.1 },
  });
  expect(events).toHaveLength(2);
  emit({
    type: 'reply',
    requestId: requests.at(-1)!.requestId,
    revision: 2,
    command: 'syncProgress',
    data: { state: 'synced' },
  });
  await expect(synced).resolves.toEqual({ state: 'synced' });
  expect(() => client.syncProgress(progress, 2)).toThrow('worker.invalid_revision');
  client.dispose();
  await expect(client.pause()).rejects.toMatchObject({ detail: { key: 'worker.disposed' } });
});
