import { expect, it, vi } from 'vitest';
import { TrainingPlotRefresh, trainingPlot } from './training-plot';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { tensor } from '../engine/tensor';
import { WorkerRuntime } from './runtime';
import { buildPlayerRegistry } from '../levels/registry';
import { chapter3ForwardLevels } from '../levels/chapter3';
import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import { chapter2Levels } from '../levels/chapter2';
import { initialProgress, type TrainLevel } from '../engine/levels';
import type { WorkerResponse } from './protocol';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x', shape: [2] } },
    { id: 't', component: 'input', attrs: { name: 't', shape: [2] } },
    { id: 'w', component: 'param', attrs: { shape: [], init: 'constant', value: 1 } },
    { id: 'p', component: 'mul' },
    { id: 'mse', component: 'mse' },
    { id: 'out', component: 'output', attrs: { name: 'loss' } },
  ],
  edges: [
    { from: { node: 'x', port: 'out' }, to: { node: 'p', port: 'a' } },
    { from: { node: 'w', port: 'out' }, to: { node: 'p', port: 'b' } },
    { from: { node: 'p', port: 'out' }, to: { node: 'mse', port: 'p' } },
    { from: { node: 't', port: 'out' }, to: { node: 'mse', port: 't' } },
    { from: { node: 'mse', port: 'out' }, to: { node: 'out', port: 'x' } },
  ],
};
const prerequisites = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
];
const progressState = initialProgress([]);
for (const lesson of prerequisites) {
  progressState.passed.push(lesson.id);
  progressState.unlockedComponents.push(...(lesson.unlockComponents ?? []));
  if (lesson.referenceGraph) progressState.passedSolutions[lesson.id] = lesson.referenceGraph;
}
const dataset = () => ({ inputs: { x: tensor([1, 2], [2]), t: tensor([0, 0], [2]) } });
it('throttles snapshots at 250ms, with immediate first and forced final snapshots', async () => {
  let now = 0;
  const refresh = new TrainingPlotRefresh(() => now);
  const capture = vi.fn(async () => undefined);
  await refresh.refresh(capture);
  now = 249;
  await refresh.refresh(capture);
  expect(capture).toHaveBeenCalledTimes(1);
  now = 250;
  await refresh.refresh(capture);
  expect(capture).toHaveBeenCalledTimes(2);
  now = 251;
  await refresh.refresh(capture, true);
  expect(capture).toHaveBeenCalledTimes(3);
});
it('reads full live predictions without updating or disposing parameters and borrowed data', async () => {
  const plan = compileGraph(graph, { registry: buildPlayerRegistry(prerequisites, progressState) });
  const batch = dataset();
  try {
    expect(await trainingPlot(graph, plan, batch, 'loss')).toEqual({
      kind: 'regression',
      x: [1, 2],
      t: [0, 0],
      p: [1, 2],
    });
    const [key, old] = [...plan.parameters][0];
    plan.parameters.set(key, tensor([2], []));
    old.dispose();
    expect(await trainingPlot(graph, plan, batch, 'loss')).toMatchObject({ p: [2, 4] });
    expect(Array.from(await batch.inputs.x.read())).toEqual([1, 2]);
  } finally {
    plan.dispose();
    Object.values(batch.inputs).forEach((t) => t.dispose());
  }
});
it('maps classification rows to labels and excludes resampled batches', async () => {
  const classification: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x', shape: [2, 2] } },
      { id: 'out', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'out', port: 'x' } }],
  };
  const plan = compileGraph(classification);
  const batch = {
    inputs: { x: tensor([0, 1, 2, 0], [2, 2]) },
    targets: tensor([1, 0, 0, 1], [2, 2]),
  };
  try {
    expect(await trainingPlot(classification, plan, batch, 'y')).toEqual({
      kind: 'classes',
      points: [
        [0, 1],
        [2, 0],
      ],
      label: [0, 1],
      predicted: [1, 0],
    });
    expect(await trainingPlot(classification, plan, () => batch, 'y')).toBeUndefined();
  } finally {
    plan.dispose();
    batch.inputs.x.dispose();
    batch.targets.dispose();
  }
});
it('delivers final trained predictions through progress before result, and serializes stop', async () => {
  const level: TrainLevel = {
    id: 'plot',
    kind: 'train',
    titleKey: 'test',
    descriptionKey: 'test',
    available: ['param', 'mul', 'mse'],
    unlock: [],
    interface: { inputs: { x: [2], t: [2] }, output: 'loss', outputShape: [1] },
    seed: 1,
    steps: 3,
    optimizer: { kind: 'sgd', learningRate: 0.1 },
    dataset,
    lossOutput: 'loss',
    threshold: { loss: 1 },
  };
  const messages: WorkerResponse[] = [];
  let complete!: () => void;
  const done = new Promise<void>((resolve) => {
    complete = resolve;
  });
  const runtime = new WorkerRuntime(
    (m) => {
      messages.push(m);
      if (m.type === 'result') complete();
    },
    [...prerequisites, level],
    progressState,
  );
  try {
    await runtime.handle({ type: 'init', backend: 'cpu', requestId: 1, revision: 0 });
    await runtime.handle({ type: 'start', levelId: level.id, graph, requestId: 2, revision: 0 });
    expect(messages.at(-1)).toMatchObject({ type: 'reply', command: 'start' });
    await done;
    const progress = messages.filter((m) => m.type === 'progress');
    expect(progress).toHaveLength(3);
    expect(progress[0].plot).toMatchObject({ p: [0.5, 1] });
    expect(progress[2].plot).toMatchObject({ p: [0.125, 0.25] });
    expect(messages.at(-1)?.type).toBe('result');
    await runtime.handle({ type: 'start', levelId: level.id, graph, requestId: 3, revision: 0 });
    await runtime.handle({ type: 'stop', requestId: 4, revision: 0 });
    const count = messages.length;
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(messages).toHaveLength(count);
  } finally {
    await runtime.dispose();
  }
});
