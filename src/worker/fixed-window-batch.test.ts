import { afterEach, expect, it, vi } from 'vitest';
import { CpuBackend } from '../engine/cpu';
import {
  compileGraph,
  createDefaultRegistry,
  GraphError,
  GraphPlan,
  type GraphJSON,
  type Shape,
  type TensorPorts,
} from '../engine/graph';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { chapter4TrainLevels } from '../levels/chapter4';
import { VOCAB } from '../levels/corpus';
import { availableComponents, buildPlayerRegistry } from '../levels/registry';
import { gameLevels } from '../levels/catalog';
import { fixedWindowBatchRegistry } from './fixed-window-batch';
import { recurrentCharacterTestProgress } from './recurrent-character-test-fixture';

afterEach(() => vi.restoreAllMocks());

function flattenGraph(shape: number[] = [512, 16], component = 'reshape'): GraphJSON {
  return {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'flatten', component, attrs: { shape } },
      { id: 'output', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [
      { from: { node: 'x', port: 'out' }, to: { node: 'flatten', port: 'x' } },
      { from: { node: 'flatten', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  };
}

async function compareExecutions(plans: GraphPlan[], shape: Shape, backend: CpuBackend) {
  const data = Float32Array.from(
    { length: shape.reduce((size, width) => size * width, 1) },
    (_, index) => index % VOCAB.length,
  );
  const x = T.tensor(data, [...shape], backend);
  let expected: Record<string, { shape: Shape; data: Float32Array }> | undefined;
  try {
    for (const plan of plans) {
      let outputs: TensorPorts | undefined;
      try {
        outputs = plan.execute({ x });
        const actual = Object.fromEntries(
          await Promise.all(
            Object.entries(outputs).map(async ([port, tensor]) => [
              port,
              { shape: tensor.shape, data: await tensor.read() },
            ]),
          ),
        );
        if (expected) expect(actual).toEqual(expected);
        else expected = actual;
      } finally {
        if (outputs) plan.releaseExecution(outputs);
      }
    }
  } finally {
    x.dispose();
  }
}

it('executes the saved 512-row circuit and 1495/1-row variants exactly like dynamic flatten without changing saves or parameters', async () => {
  const progress = recurrentCharacterTestProgress();
  const graph = progress.passedSolutions['4-5'];
  graph.nodes.find(({ id }) => id === 'flatten')!.attrs!.shape = [512, 16];
  const saved = JSON.stringify(progress);
  const level = chapter4TrainLevels.find(({ id }) => id === '4-5')!;
  const components = availableComponents(level, progress);
  const registry = buildPlayerRegistry(gameLevels, progress);
  const adapted = fixedWindowBatchRegistry(graph, registry, level.interface.inputs, components);
  const dynamic = structuredClone(graph);
  dynamic.nodes.find(({ id }) => id === 'flatten')!.attrs!.shape = [-1, 16];
  const backend = new CpuBackend();
  const options = { backend, registry, preferNative: false, availableComponents: components };
  const original = compileGraph(graph, {
    ...options,
    inputShapes: level.interface.inputs,
    random: new Random(level.seed),
  });
  const reference = compileGraph(dynamic, {
    ...options,
    inputShapes: level.interface.inputs,
    random: new Random(level.seed),
  });
  const parameterShapes = (plan: GraphPlan) =>
    [...plan.parameters].map(([key, tensor]) => [key, tensor.shape]);
  try {
    await compareExecutions([original, reference], [512, 2], backend);
    const expectedParameters = parameterShapes(original);
    expect(expectedParameters).toEqual(parameterShapes(reference));
    expect(expectedParameters.map(([key]) => key)).toEqual(['embedding', 'w1', 'b1', 'w2', 'b2']);
    const buffers = backend.liveBuffers;
    for (const rows of [512, 1495, 1]) {
      const inputShapes = { x: [rows, 2] };
      if (rows !== 512)
        expect(() => compileGraph(graph, { ...options, inputShapes })).toThrow(GraphError);
      const plan = compileGraph(graph, {
        ...options,
        registry: adapted,
        inputShapes,
        parameters: original.parameters,
      });
      const dynamicPlan = compileGraph(dynamic, {
        ...options,
        inputShapes,
        parameters: reference.parameters,
      });
      try {
        expect(plan.outputShapes.probabilities).toEqual([rows, VOCAB.length]);
        await compareExecutions([plan, dynamicPlan], inputShapes.x, backend);
        expect(parameterShapes(plan)).toEqual(expectedParameters);
        expect(parameterShapes(dynamicPlan)).toEqual(expectedParameters);
      } finally {
        plan.dispose();
        dynamicPlan.dispose();
      }
      expect(backend.liveBuffers).toBe(buffers);
    }
    expect(JSON.stringify(progress)).toBe(saved);
    expect(registry.get('reshape')!.primitive).not.toBe(adapted.get('reshape')!.primitive);
  } finally {
    original.dispose();
    reference.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('observes nested player definitions, preserving every other primitive, native and subgraph', async () => {
  const registry = createDefaultRegistry();
  const nested = flattenGraph();
  nested.nodes.push(
    { id: 'keep_parameter', component: 'param', attrs: { shape: [512], init: 'ones' } },
    { id: 'keep_constant', component: 'const', attrs: { shape: [512], value: 2 } },
  );
  const native = { infer: vi.fn(), execute: vi.fn() };
  registry.register({
    name: 'player_flatten',
    inputs: ['x'],
    outputs: ['out'],
    native,
    subgraph: () => nested,
  });
  registry.register({
    name: 'player_block',
    inputs: ['x'],
    outputs: ['out'],
    subgraph: flattenGraph([], 'player_flatten'),
  });
  const graph = flattenGraph([], 'player_block');
  const before = JSON.stringify([graph, nested]);
  const adapted = fixedWindowBatchRegistry(graph, registry, { x: [512, 2, 8] });
  for (const definition of registry.list()) {
    const actual = adapted.get(definition.name)!;
    if (definition.name !== 'reshape') expect(actual).toBe(definition);
    expect(actual.native).toBe(definition.native);
    expect(actual.subgraph).toBe(definition.subgraph);
    expect(actual.primitive?.execute).toBe(definition.primitive?.execute);
  }
  const backend = new CpuBackend();
  for (const rows of [1495, 1]) {
    const options = { inputShapes: { x: [rows, 2, 8] }, backend, preferNative: false };
    const plan = compileGraph(graph, { ...options, registry: adapted });
    const reference = compileGraph(flattenGraph([-1, 16]), options);
    try {
      await compareExecutions([plan, reference], options.inputShapes.x, backend);
      expect([...plan.parameters.keys()]).toEqual(['flatten/flatten/keep_parameter']);
      expect([...plan.parameters.values()][0].shape).toEqual([512]);
    } finally {
      plan.dispose();
      reference.dispose();
    }
    expect(backend.liveBuffers).toBe(0);
  }
  expect(JSON.stringify([graph, nested])).toBe(before);
  expect(native.infer).not.toHaveBeenCalled();
  expect(native.execute).not.toHaveBeenCalled();
});

it.each([false, true])(
  'adapts after dimension-port resolution (wired row count: %s)',
  async (wiredRows) => {
    const graph = flattenGraph([512, 1]);
    graph.nodes.push(
      { id: 'width', component: 'const', attrs: { value: 16 } },
      { id: 'batch', component: 'const', attrs: { value: 512 } },
    );
    graph.edges.push({ from: { node: 'width', port: 'out' }, to: { node: 'flatten', port: 'd1' } });
    if (wiredRows)
      graph.edges.push({
        from: { node: 'batch', port: 'out' },
        to: { node: 'flatten', port: 'd0' },
      });
    const before = JSON.stringify(graph);
    const registry = fixedWindowBatchRegistry(graph, createDefaultRegistry(), { x: [512, 2, 8] });
    const backend = new CpuBackend();
    for (const rows of [1495, 1]) {
      const options = { inputShapes: { x: [rows, 2, 8] }, backend };
      const plan = compileGraph(graph, { ...options, registry });
      const reference = compileGraph(flattenGraph([-1, 16]), options);
      try {
        expect(plan.outputShapes.out).toEqual([rows, 16]);
        await compareExecutions([plan, reference], options.inputShapes.x, backend);
      } finally {
        plan.dispose();
        reference.dispose();
      }
    }
    expect(JSON.stringify(graph)).toBe(before);
    expect(backend.liveBuffers).toBe(0);
  },
);

it('leaves inferred batch dimensions and non-shape attributes untouched and does not execute during observation', () => {
  const graph = flattenGraph([-1, 16]);
  graph.nodes[1].attrs!.label = 'keep 512 literally';
  const saved = JSON.stringify(graph);
  const registry = createDefaultRegistry();
  const infer = vi.spyOn(registry.get('reshape')!.primitive!, 'infer');
  const execution = vi.spyOn(GraphPlan.prototype, 'execute');
  const disposal = vi.spyOn(GraphPlan.prototype, 'dispose');
  const backend = T.defaultBackend as CpuBackend;
  const buffers = backend.liveBuffers;
  const adapted = fixedWindowBatchRegistry(graph, registry, { x: [512, 2, 8] });
  expect(execution).not.toHaveBeenCalled();
  expect(disposal).toHaveBeenCalledTimes(1);
  expect(backend.liveBuffers).toBe(buffers);
  for (const rows of [512, 1495, 1]) {
    const plan = compileGraph(graph, { registry: adapted, inputShapes: { x: [rows, 2, 8] } });
    try {
      expect(plan.outputShapes.out).toEqual([rows, 16]);
      expect(infer.mock.calls.at(-1)![1]).toEqual({ shape: [-1, 16], label: 'keep 512 literally' });
    } finally {
      plan.dispose();
    }
  }
  expect(execution).not.toHaveBeenCalled();
  expect(JSON.stringify(graph)).toBe(saved);
  expect(backend.liveBuffers).toBe(buffers);
});

it.each([
  { original: [512, 2, 8], changed: [1495, 4, 4], target: [512, 16] },
  { original: [512, 2, 8], changed: [1495, 2, 9], target: [512, 16] },
  { original: [512, 2, 8], changed: [1495, 2, 8], target: [256, 32] },
  { original: [512, 2, 8], changed: [1495, 2, 8], target: [16, 512] },
])(
  'preserves the original GraphError for unsupported reshape $target / $changed',
  ({ original, changed, target }) => {
    const graph = flattenGraph(target);
    const registry = createDefaultRegistry();
    const adapted = fixedWindowBatchRegistry(graph, registry, { x: original });
    const failure = (candidate: typeof registry) => {
      try {
        const plan = compileGraph(graph, { registry: candidate, inputShapes: { x: changed } });
        plan.dispose();
      } catch (error) {
        return error;
      }
    };
    const expected = failure(registry);
    expect(expected).toBeInstanceOf(GraphError);
    expect(failure(adapted)).toEqual(expected);
  },
);

it('requires the complete resolved target signature, not just the batch number', () => {
  const registry = createDefaultRegistry();
  const graph = flattenGraph();
  const adapted = fixedWindowBatchRegistry(graph, registry, { x: [512, 2, 8] });
  expect(() =>
    compileGraph(flattenGraph([512, 8, 2]), {
      registry: adapted,
      inputShapes: { x: [1495, 2, 8] },
    }),
  ).toThrow(expect.objectContaining({ key: 'reshape_mismatch', node: 'flatten' }));
});

it('rejects invalid original graphs and unavailable components without execution or allocation', () => {
  const registry = createDefaultRegistry();
  const graph = flattenGraph();
  graph.nodes[0] = {
    id: 'x',
    component: 'param',
    attrs: { shape: [511, 2, 8], init: 'ones' },
  };
  const execution = vi.spyOn(GraphPlan.prototype, 'execute');
  const backend = T.defaultBackend as CpuBackend;
  const buffers = backend.liveBuffers;
  expect(() => fixedWindowBatchRegistry(graph, registry, { x: [512, 2, 8] })).toThrow(
    expect.objectContaining({ key: 'reshape_mismatch', node: 'flatten' }),
  );
  expect(() => fixedWindowBatchRegistry(flattenGraph(), registry, { x: [512, 2, 8] }, [])).toThrow(
    expect.objectContaining({ key: 'component_unavailable', node: 'flatten' }),
  );
  expect(execution).not.toHaveBeenCalled();
  expect(backend.liveBuffers).toBe(buffers);
});

it('does not accept an observed reshape until the rest of the original parameterized graph validates', () => {
  const progress = recurrentCharacterTestProgress();
  const graph = progress.passedSolutions['4-5'];
  graph.nodes.find(({ id }) => id === 'flatten')!.attrs!.shape = [512, 16];
  graph.nodes.find(({ id }) => id === 'w1')!.attrs!.shape = [512, 24];
  const registry = buildPlayerRegistry(gameLevels, progress);
  const inputShapes = { x: [512, 2] };
  const before = JSON.stringify(graph);
  const execution = vi.spyOn(GraphPlan.prototype, 'execute');
  let originalError: unknown;
  try {
    compileGraph(graph, { registry, inputShapes, preferNative: false }).dispose();
  } catch (error) {
    originalError = error;
  }
  expect(originalError).toBeInstanceOf(GraphError);
  expect(originalError).toMatchObject({ node: expect.stringMatching(/^product(?:\/|$)/) });
  expect(() => fixedWindowBatchRegistry(graph, registry, inputShapes)).toThrow(
    originalError as Error,
  );
  expect(execution).not.toHaveBeenCalled();
  expect(JSON.stringify(graph)).toBe(before);
});

it('still reports borrowed parameter shape mismatches rather than resizing learned tensors', () => {
  const graph = flattenGraph();
  graph.nodes.push({ id: 'weight', component: 'param', attrs: { shape: [512], init: 'ones' } });
  const registry = fixedWindowBatchRegistry(graph, createDefaultRegistry(), { x: [512, 2, 8] });
  const backend = new CpuBackend();
  const weight = T.tensor([1], [1], backend);
  const x = T.tensor(new Float32Array(16), [1, 2, 8], backend);
  const plan = compileGraph(graph, {
    registry,
    backend,
    inputShapes: { x: x.shape },
    parameters: new Map([['weight', weight]]),
  });
  try {
    expect(() => plan.execute({ x })).toThrow(
      expect.objectContaining({
        key: 'parameter_shape_mismatch',
        node: 'weight',
        params: { expected: [512], actual: [1] },
      }),
    );
    expect(plan.parameters.get('weight')).toBe(weight);
  } finally {
    plan.dispose();
    weight.dispose();
    x.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});
