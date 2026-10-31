// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { afterEach, expect, it } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { initialProgress, type ForwardLevel, type Level } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import { chapter2Levels } from '../levels/chapter2';
import { chapter3ForwardLevels } from '../levels/chapter3';
import { chapter4TrainLevels } from '../levels/chapter4';
import { gameLevels } from '../levels/catalog';
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
function harness(levels: readonly Level[] = chapter4TrainLevels, sequence = false) {
  const prerequisites = sequence
    ? gameLevels.filter((level): level is ForwardLevel => level.kind === 'forward')
    : preceding;
  const testProgress = sequence ? initialProgress([]) : progress;
  if (sequence)
    testProgress.unlockedComponents = prerequisites.flatMap(
      (level) => level.unlockComponents ?? [],
    );
  const messages: WorkerResponse[] = [];
  const runtime = new WorkerRuntime(
    (m) => messages.push(structuredClone(m)),
    levels,
    testProgress,
    sequence ? referenceRegistry(prerequisites) : registry,
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

it('sequence generation uses the declared context, samples the last position and hides left padding', async () => {
  const context = 4;
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      {
        id: 'table',
        component: 'const',
        attrs: {
          shape: [VOCAB.length, VOCAB.length],
          value: Array.from({ length: VOCAB.length ** 2 }, (_, i) =>
            Number(Math.floor(i / VOCAB.length) === i % VOCAB.length),
          ),
        },
      },
      { id: 'lookup', component: 'gather' },
      { id: 'sequence', component: 'reshape', attrs: { shape: [1, context, VOCAB.length] } },
      { id: 'y', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [
      { from: { node: 'x', port: 'out' }, to: { node: 'lookup', port: 'idx' } },
      { from: { node: 'table', port: 'out' }, to: { node: 'lookup', port: 'table' } },
      { from: { node: 'lookup', port: 'out' }, to: { node: 'sequence', port: 'x' } },
      { from: { node: 'sequence', port: 'out' }, to: { node: 'y', port: 'x' } },
    ],
  };
  const sequenceLevel = {
    ...level,
    available: ['const', 'gather', 'reshape'],
    interface: {
      inputs: { x: [8, context] },
      output: 'y',
      outputShape: [8, context, VOCAB.length],
    },
  };
  const h = harness([sequenceLevel]);
  await h.send({ type: 'init', backend: 'cpu' });
  for (const prompt of ['', 'a', 'the cat', '💥a']) {
    const filtered = [...prompt].filter((char) => VOCAB.includes(char)).join('');
    expect(text(await h.send({ ...generate(graph), prompt, length: 3 })).text).toBe(
      filtered + (filtered.at(-1) ?? ' ').repeat(3),
    );
  }
  expect(text(await h.send({ ...generate(graph), prompt: '', length: 0 })).text).toBe('');
  // A vocabulary-sized trailing axis alone must not accept malformed sequence layouts.
  for (const shape of [
    [context, VOCAB.length],
    [1, 2, 2, VOCAB.length],
    [1, context, VOCAB.length - 1],
  ]) {
    const malformed = structuredClone(graph);
    const sequence = malformed.nodes.find((node) => node.id === 'sequence')!;
    sequence.component = 'const';
    sequence.attrs = { shape, value: 1 };
    malformed.edges = malformed.edges.filter((edge) => edge.to.node !== 'sequence');
    expect(await h.send({ ...generate(malformed), length: 0 })).toMatchObject({
      error: { key: 'levels.output_shape' },
    });
  }
  for (const context of [0, -1, 1.5]) {
    const invalid = harness([
      {
        ...sequenceLevel,
        interface: { ...sequenceLevel.interface, inputs: { x: [8, context] } },
      },
    ]);
    await invalid.send({ type: 'init', backend: 'cpu' });
    expect(await invalid.send(generate(graph))).toMatchObject({
      error: { key: 'worker.generation_unavailable' },
    });
  }
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

it('snapshots exact post-step parameters, survive plan release, and invalidate on identity/reset but survive revision and progress sync', async () => {
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
        const probabilities = await outputs[level.interface.output].read();
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
    // Passing a level bumps the revision and syncs progress; the trained model must survive.
    expect(text(await h.send(command, 1))).toEqual(trained);
    await h.send({ type: 'syncProgress', progress }, 1);
    expect(text(await h.send(command, 1))).toEqual(trained);
    // The client cancels on every revision bump; that must not forget a finished model.
    await h.send({ type: 'cancel' }, 2);
    expect(text(await h.send(command, 2))).toEqual(trained);
  } finally {
    plan.dispose();
    parameters.forEach((value) => value.dispose());
  }
});

const sequenceTrainLevels = gameLevels.filter(
  (level) => level.kind === 'train' && level.id === '5-5',
);
it('registers the Attention capstone under its stable ID', () => {
  expect(sequenceTrainLevels).toHaveLength(1);
  expect(sequenceTrainLevels[0]).toMatchObject({ generates: true });
});
for (const trainedLevel of [
  ...chapter4TrainLevels,
  ...sequenceTrainLevels,
  // Only exercise completion and retained inference parameters here, not convergence.
  ...gameLevels.filter((l) => l.id === '6-4').map((level) => ({ ...level, steps: 2 })),
]) {
  if (trainedLevel.kind !== 'train') continue;
  it(`${trainedLevel.id}: actual continuous reference training retains parameters after completion and samples single-context inference`, async () => {
    const h = harness([trainedLevel], ['5-5', '6-4'].includes(trainedLevel.id));
    await h.send({ type: 'init', backend: 'cpu' });
    const command = { ...generate(trainedLevel.referenceGraph!), levelId: trainedLevel.id };
    const baseline = text(await h.send(command));
    await h.send({ type: 'start', levelId: trainedLevel.id, graph: trainedLevel.referenceGraph! });
    for (let i = 0; i < 6000 && !h.messages.some((m) => m.type === 'result'); i++)
      await new Promise((resolve) => setTimeout(resolve, 10));
    const result = h.messages.find((m) => m.type === 'result');
    expect(result).toMatchObject({
      result: {
        ...(trainedLevel.id === '6-4' ? {} : { passed: true }),
        training: { step: trainedLevel.steps },
      },
    });
    if (trainedLevel.id === '6-4') {
      for (const prompt of ['little moth ', 'round the pond ', 'penny wren ']) {
        const request = { ...command, prompt, length: 90, temperature: 0.2 };
        const generated = text(await h.send(request));
        expect(text(await h.send(request))).toEqual(generated);
        expect(generated.text).toHaveLength(prompt.length + 90);
        expect([...generated.text].every((char) => VOCAB.includes(char))).toBe(true);
        console.info(
          JSON.stringify({
            levelId: trainedLevel.id,
            prompt,
            temperature: 0.2,
            seed: request.seed,
            sample: generated.text,
          }),
        );
      }
    }
    const sample = text(await h.send(command));
    expect(sample).not.toEqual(baseline);
    expect(sample.text).toHaveLength(104);
    expect([...sample.text].every((char) => VOCAB.includes(char))).toBe(true);
    expect(text(await h.send(command))).toEqual(sample);
    expect(text(await h.send({ ...command, seed: 4901 }))).not.toEqual(sample);
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
