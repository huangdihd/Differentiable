import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { traceBackward } from './backward-graph';
import { compileGraph, createDefaultRegistry, type GraphJSON } from './graph';
import { evaluateLevel, trainingLearningRate, type TrainLevel } from './levels';
import * as T from './tensor';
import { TrainingSession } from './training';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'w', component: 'param', attrs: { shape: [], init: 'constant', value: 2 } },
    { id: 'square', component: 'mul' },
    { id: 'out', component: 'output', attrs: { name: 'loss' } },
  ],
  edges: [
    { from: { node: 'w', port: 'out' }, to: { node: 'square', port: 'a' } },
    { from: { node: 'w', port: 'out' }, to: { node: 'square', port: 'b' } },
    { from: { node: 'square', port: 'out' }, to: { node: 'out', port: 'x' } },
  ],
};
const level: TrainLevel = {
  id: 'trace-test',
  titleKey: 'test',
  descriptionKey: 'test',
  kind: 'train',
  available: ['mul', 'param'],
  unlock: [],
  interface: { inputs: {}, output: 'loss', outputShape: [] },
  seed: 1,
  steps: 1,
  optimizer: { kind: 'sgd', learningRate: 0.1 },
  tunable: { learningRate: { min: 0.001, max: 10, default: 0.1 } },
  dataset: () => ({ inputs: {} }),
  lossOutput: 'loss',
  threshold: { loss: 3 },
};
it('clamps declared learning rates and uses default for nonfinite requests', () => {
  expect(trainingLearningRate(level, -1)).toBe(0.001);
  expect(trainingLearningRate(level, 100)).toBe(10);
  expect(trainingLearningRate(level, NaN)).toBe(0.1);
  expect(trainingLearningRate(level)).toBe(0.1);
  expect(trainingLearningRate({ ...level, tunable: undefined }, 10)).toBe(0.1);
});
it('evaluates target-free loss and distinguishes slow from divergent training', async () => {
  expect((await evaluateLevel(level, graph)).passed).toBe(true);
  const slow = await evaluateLevel(level, graph, { learningRate: 0.001 });
  expect(slow.errorKey).toBe('levels.train_slow');
  expect(slow.error?.params.loss).toBe(slow.training?.loss);
  expect((await evaluateLevel(level, graph, { learningRate: 10 })).errorKey).toBe(
    'levels.train_diverged',
  );
  const invalid = structuredClone(graph);
  invalid.nodes[0].attrs!.value = 1e30;
  expect((await evaluateLevel(level, invalid)).errorKey).toBe('levels.train_diverged');
});
it('traces live target-free training, agrees with symbolic backward and retains no tape allocations', async () => {
  const backend = new CpuBackend();
  const model = compileGraph(graph, { backend });
  const session = new TrainingSession(
    model,
    () => ({ inputs: {} }),
    undefined,
    { kind: 'sgd', learningRate: 0.1 },
    'loss',
    { lossOutput: 'loss' },
  );
  try {
    const symbolic = await traceBackward(graph, {
      inputShapes: {},
      output: 'loss',
      wrt: ['w'],
      inputs: {},
    });
    const first = await session.stepTrace();
    expect(first.step).toBe(1);
    expect(first.loss).toBe(4);
    expect(first.forward.w).toEqual({ shape: [], size: 1, values: [2] });
    expect(first.gradient.w.values).toEqual(symbolic.gradient.w.values);
    expect(first.gradient.square.values).toEqual(symbolic.gradient.square.values);
    expect(first.parameters.w.before).toEqual([2]);
    expect(first.parameters.w.after[0]).toBeCloseTo(1.6);
    expect((await session.trainStepTrace()).loss).toBeCloseTo(2.56);
    expect(backend.liveBuffers).toBe(1);
    expect(JSON.parse(JSON.stringify(first))).toEqual(first);
  } finally {
    session.dispose();
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});
it('caps node snapshots at 256 entries and reports full sizes', async () => {
  const backend = new CpuBackend();
  const x = T.tensor(new Float32Array(300).fill(1), [300], backend);
  const model = compileGraph(
    {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 'sum', component: 'sum', attrs: { axes: [0] } },
        { id: 'out', component: 'output', attrs: { name: 'loss' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'sum', port: 'x' } },
        { from: { node: 'sum', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    },
    { backend, inputShapes: { x: [300] } },
  );
  const session = new TrainingSession(
    model,
    { inputs: { x } },
    undefined,
    { kind: 'sgd', learningRate: 0.1 },
    'loss',
    { lossOutput: 'loss' },
  );
  try {
    const trace = await session.stepTrace();
    expect(trace.forward.x.values).toHaveLength(256);
    expect(trace.gradient.x.values).toHaveLength(256);
    expect(trace.forward.x.size).toBe(300);
    expect(trace.gradient.x.shape).toEqual([300]);
  } finally {
    session.dispose();
    model.dispose();
    x.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('traces only canvas composites, aliases and storage; omits disconnected gradients', async () => {
  const backend = new CpuBackend();
  const registry = createDefaultRegistry().register({
    name: 'square',
    inputs: ['x'],
    outputs: ['out'],
    subgraph: {
      version: 1,
      nodes: [
        { id: 'input', component: 'input', attrs: { name: 'x' } },
        { id: 'internal', component: 'mul' },
        { id: 'output', component: 'output', attrs: { name: 'out' } },
      ],
      edges: [
        { from: { node: 'input', port: 'out' }, to: { node: 'internal', port: 'a' } },
        { from: { node: 'input', port: 'out' }, to: { node: 'internal', port: 'b' } },
        { from: { node: 'internal', port: 'out' }, to: { node: 'output', port: 'x' } },
      ],
    },
  });
  const canvas: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'input', component: 'input', attrs: { name: 'x' } },
      { id: 'w space', component: 'param', attrs: { shape: [1], init: 'constant', value: 2 } },
      { id: 'unused', component: 'const', attrs: { shape: [1], value: 7 } },
      { id: 'disconnected', component: 'exp' },
      { id: 'square', component: 'square' },
      { id: 'output', component: 'output', attrs: { name: 'loss' } },
    ],
    edges: [
      { from: { node: 'input', port: 'out' }, to: { node: 'disconnected', port: 'x' } },
      { from: { node: 'w space', port: 'out' }, to: { node: 'square', port: 'x' } },
      { from: { node: 'square', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  };
  const model = compileGraph(canvas, { registry, backend, inputShapes: { x: [1] } });
  const x = T.tensor([3], [1], backend);
  const session = new TrainingSession(
    model,
    { inputs: { x } },
    undefined,
    { kind: 'sgd', learningRate: 0.1 },
    'loss',
    { lossOutput: 'loss' },
  );
  try {
    const trace = await session.stepTrace();
    expect(Object.keys(trace.forward).sort()).toEqual(canvas.nodes.map((n) => n.id).sort());
    expect(Object.keys(trace.gradient).sort()).toEqual(['output', 'square', 'w space']);
    expect(trace.forward.output).toEqual(trace.forward.square);
    expect(trace.gradient['w space'].values).toEqual([4]);
    expect(trace.parameters['w space'].before).toEqual([2]);
    expect(trace.parameters['w space'].after[0]).toBeCloseTo(1.6);
    expect((await session.run(1)).loss).toBeCloseTo(1.6384);
    expect((await session.evaluate()).loss).toBeCloseTo(1.6384);
    expect(backend.liveBuffers).toBe(3);
  } finally {
    session.dispose();
    model.dispose();
    x.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});
