import { afterEach, expect, it } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { initialProgress, type Level } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import { chapter2Levels } from '../levels/chapter2';
import { chapter3ForwardLevels } from '../levels/chapter3';
import { chapter4TrainLevels } from '../levels/chapter4';
import { VOCAB, encode, decode } from '../levels/corpus';
import { referenceRegistry } from '../levels/level-helpers';
import { WorkerClient, type WorkerTransport } from './client';
import type { Command, ReplyData, WorkerRequest, WorkerResponse } from './protocol';
import { WorkerRuntime } from './runtime';

const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
];
const registry = referenceRegistry(preceding);
const progress = initialProgress([]);
progress.unlockedComponents = preceding.flatMap((level) => level.unlockComponents ?? []);
const runtimes: WorkerRuntime[] = [];
afterEach(async () => {
  await Promise.all(runtimes.splice(0).map((runtime) => runtime.dispose()));
});
function harness(levels: readonly Level[] = chapter4TrainLevels) {
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime(
    (m) => messages.push(structuredClone(m)),
    levels,
    progress,
    registry,
  );
  runtimes.push(runtime);
  let requestId = 0;
  return {
    messages,
    async send(command: Command, revision = 0) {
      await runtime.handle({ ...command, revision, requestId: ++requestId });
      return messages.at(-1)!;
    },
  };
}
const level = chapter4TrainLevels[0];
function generate(
  graph: GraphJSON = level.referenceGraph!,
): Extract<Command, { type: 'generate' }> {
  return { type: 'generate', levelId: level.id, graph, prompt: 'the ', length: 100, seed: 4900 };
}
function text(message: WorkerResponse) {
  expect(message).toMatchObject({ type: 'reply', command: 'generate' });
  return (message as Extract<WorkerResponse, { type: 'reply' }>).data as ReplyData['generate'];
}

it('untrained generation is seeded, filters unknown characters, pads context only and validates options', async () => {
  const h = harness();
  await h.send({ type: 'init', backend: 'cpu' });
  const command = generate();
  const first = text(await h.send(command));
  expect(text(await h.send(command))).toEqual(first);
  expect(text(await h.send({ ...command, prompt: '💥the 🙂' }))).toEqual(first);
  expect(text(await h.send({ ...command, seed: 4901 }))).not.toEqual(first);
  expect(first.text).toHaveLength(104);
  expect(text(await h.send({ ...command, prompt: '💥', length: 0 })).text).toBe('');
  expect(text(await h.send({ ...command, prompt: '', length: 400 })).text).toHaveLength(400);
  for (const length of [-1, 401, 1.5, NaN, Infinity])
    expect(await h.send({ ...command, length })).toMatchObject({
      error: { key: 'worker.invalid_request' },
    });
  for (const temperature of [0, -1, NaN, Infinity])
    expect(await h.send({ ...command, temperature })).toMatchObject({
      error: { key: 'worker.invalid_request' },
    });
  expect(text(await h.send({ ...command, temperature: Number.MIN_VALUE })).text).toHaveLength(104);
  expect(await h.send({ ...command, seed: 1.5 })).toMatchObject({
    error: { key: 'worker.invalid_request' },
  });
});

it('only explicitly generating training levels are eligible and active/paused runs are busy', async () => {
  const h = harness([{ ...level, generates: false }, chapter0Levels[0]]);
  await h.send({ type: 'init', backend: 'cpu' });
  expect(await h.send(generate())).toMatchObject({
    error: { key: 'worker.generation_unavailable' },
  });
  expect(await h.send({ ...generate(), levelId: chapter0Levels[0].id })).toMatchObject({
    error: { key: 'worker.generation_unavailable' },
  });
  const active = harness();
  await active.send({ type: 'init', backend: 'cpu' });
  await active.send({ type: 'start', levelId: level.id, graph: level.referenceGraph! });
  expect(await active.send(generate())).toMatchObject({ error: { key: 'worker.training_busy' } });
  await active.send({ type: 'pause' });
  expect(await active.send(generate())).toMatchObject({ error: { key: 'worker.training_busy' } });
  await active.send({ type: 'stop' });
  text(await active.send(generate()));
});

it('snapshots exact post-step parameters, survive plan release, and invalidate on identity/revision/reset/registry changes', async () => {
  const h = harness([level, { ...level, id: 'other' }]);
  await h.send({ type: 'init', backend: 'cpu' });
  const command = generate();
  const baseline = text(await h.send(command));
  const step = await h.send({ type: 'trainStep', levelId: level.id, graph: level.referenceGraph! });
  expect(step).toMatchObject({ type: 'reply', command: 'trainStep' });
  const trace = (step as Extract<WorkerResponse, { type: 'reply' }>).data as ReplyData['trainStep'];
  // Independently execute a fresh inference plan with the exact returned post-update values.
  const parameters = new Map(
    Object.entries(trace.parameters).map(([id, value]) => [
      id,
      T.tensor(value.after, trace.forward[id].shape),
    ]),
  );
  const plan = compileGraph(command.graph, {
    registry,
    inputShapes: { x: [1] },
    parameters,
    random: new Random(level.seed),
  });
  const tokens = encode(command.prompt);
  const random = new Random(command.seed);
  try {
    for (let i = 0; i < command.length; i++) {
      const x = T.tensor(tokens.slice(-1), [1]);
      const outputs = plan.execute({ x });
      try {
        const probabilities = await outputs.y.read();
        const total = probabilities.reduce((sum, p) => sum + p, 0);
        let draw = random.uniform() * total;
        let token = VOCAB.length - 1;
        for (let j = 0; j < probabilities.length; j++) {
          draw -= probabilities[j];
          if (draw < 0) {
            token = j;
            break;
          }
        }
        tokens.push(token);
      } finally {
        plan.releaseExecution(outputs);
        x.dispose();
      }
    }
    const trained = text(await h.send(command));
    expect(trained.text).toBe(decode(tokens));
    expect(trained).not.toEqual(baseline);
    // A failed replacement disposes the old step plan, but not its successful snapshot.
    const invalid = { ...command.graph, edges: [] };
    expect(await h.send({ type: 'trainStep', levelId: level.id, graph: invalid })).toMatchObject({
      type: 'error',
    });
    expect(text(await h.send(command))).toEqual(trained);
    expect(text(await h.send({ ...command, levelId: 'other' }))).toEqual(baseline);
    const renamed = structuredClone(command.graph);
    renamed.nodes.find((node) => node.id === 'table')!.attrs!.scale = 0.150001;
    expect(text(await h.send({ ...command, graph: renamed }))).not.toEqual(trained);
    // start releases the step plan; its successful stop replaces the snapshot (latest wins).
    await h.send({ type: 'start', levelId: level.id, graph: command.graph });
    await h.send({ type: 'pause' });
    await h.send({ type: 'stop' });
    expect(text(await h.send(command))).toEqual(baseline);
    await h.send({ type: 'trainStep', levelId: level.id, graph: command.graph });
    expect(text(await h.send(command))).toEqual(trained);
    await h.send({ type: 'trainReset' });
    expect(text(await h.send(command))).toEqual(baseline);
    await h.send({ type: 'trainStep', levelId: level.id, graph: command.graph });
    expect(text(await h.send(command, 1))).toEqual(baseline);
    await h.send({ type: 'trainStep', levelId: level.id, graph: command.graph }, 1);
    await h.send({ type: 'syncProgress', progress }, 1);
    expect(text(await h.send(command, 1))).toEqual(baseline);
  } finally {
    plan.dispose();
    parameters.forEach((value) => value.dispose());
  }
});

for (const trainedLevel of chapter4TrainLevels) {
  it(`${trainedLevel.id}: actual continuous reference training retains parameters after completion and samples single-context inference`, async () => {
    const h = harness();
    await h.send({ type: 'init', backend: 'cpu' });
    const command = { ...generate(trainedLevel.referenceGraph!), levelId: trainedLevel.id };
    const baseline = text(await h.send(command));
    await h.send({ type: 'start', levelId: trainedLevel.id, graph: trainedLevel.referenceGraph! });
    for (let i = 0; i < 6000 && !h.messages.some((m) => m.type === 'result'); i++)
      await new Promise((resolve) => setTimeout(resolve, 10));
    const result = h.messages.find((m) => m.type === 'result');
    expect(result).toMatchObject({
      result: { passed: true, training: { step: trainedLevel.steps } },
    });
    const sample = text(await h.send(command));
    expect(sample).not.toEqual(baseline);
    expect(sample.text).toHaveLength(104);
    expect(text(await h.send(command))).toEqual(sample);
    expect(text(await h.send({ ...command, prompt: '', length: 2 })).text).toHaveLength(2);
    console.info(
      JSON.stringify({ levelId: trainedLevel.id, seed: command.seed, sample: sample.text }),
    );
  }, 90000);
}

it('client exposes typed generate RPC', async () => {
  const requests: WorkerRequest[] = [];
  let listener: (event: MessageEvent<WorkerResponse>) => void;
  const transport: WorkerTransport = {
    postMessage(request) {
      requests.push(request);
      listener({
        data: {
          type: 'reply',
          command: 'generate',
          requestId: request.requestId,
          revision: request.revision,
          data: { text: 'the cat' },
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
  expect(await client.generate(level.id, level.referenceGraph!, 'the ', 3, 42, 0.7)).toEqual({
    text: 'the cat',
  });
  expect(requests[0]).toMatchObject({
    type: 'generate',
    prompt: 'the ',
    length: 3,
    seed: 42,
    temperature: 0.7,
  });
  client.dispose();
});
