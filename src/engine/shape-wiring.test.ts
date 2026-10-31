// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { compileGraph, type Attributes, type GraphJSON } from './graph';
import { CpuBackend } from './cpu';
import * as T from './tensor';
import { traceBackward } from './backward-graph';
import { TrainingSession } from './training';
import { evaluateLevel } from './levels';
import { chapter2Levels } from '../levels/chapter2';
import { chapter3ForwardLevels } from '../levels/chapter3';
import { chapter4ForwardLevels, chapter4TrainLevels } from '../levels/chapter4';
import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import { referenceRegistry } from '../levels/level-helpers';

function builder() {
  const graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  const node = (
    id: string,
    component: string,
    args: Record<string, string> = {},
    attrs?: Attributes,
  ) => {
    graph.nodes.push({ id, component, attrs });
    for (const [port, source] of Object.entries(args))
      graph.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  };
  const output = (x: string) => node('output', 'output', { x }, { name: 'y' });
  return { graph, node, output };
}
function flatten(inferred = false) {
  const b = builder();
  b.node('x', 'input', {}, { name: 'x' });
  b.node('batch', 'size', { x: 'x' }, { axis: 0 });
  b.node('middle', 'size', { x: 'x' }, { axis: 1 });
  b.node('last', 'size', { x: 'x' });
  b.node('width', 'mul', { a: 'middle', b: 'last' });
  b.node(
    'flat',
    'reshape',
    { x: 'x', d0: 'batch', ...(inferred ? {} : { d1: 'width' }) },
    { shape: [1, -1] },
  );
  b.output('flat');
  return b.graph;
}
const values = Array.from({ length: 24 }, (_, i) => i - 12);
const registry = referenceRegistry([...chapter0Levels, ...chapter1Levels]);

describe('shape scalar wiring (step10)', () => {
  it.each([
    [0, 2],
    [1, 3],
    [-1, 4],
    [-2, 3],
    [undefined, 4],
  ] as const)('size axis %s infers a scalar and executes as %s', async (axis, expected) => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    b.output(b.node('size', 'size', { x: 'x' }, axis === undefined ? {} : { axis }));
    const backend = new CpuBackend();
    const plan = compileGraph(b.graph, { inputShapes: { x: [2, 3, 4] }, backend });
    expect(backend.liveBuffers).toBe(0);
    expect(plan.outputShapes.y).toEqual([]);
    const x = T.tensor(values, [2, 3, 4], backend);
    const outputs = plan.execute({ x });
    expect(Array.from(await outputs.y.read())).toEqual([expected]);
    plan.releaseExecution(outputs);
    plan.dispose();
    x.dispose();
    expect(backend.liveBuffers).toBe(0);
  });
  it.each([3, -4, 0.5, '0'])('rejects invalid size axis %s', (axis) => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    b.output(b.node('size', 'size', { x: 'x' }, { axis }));
    expect(() => compileGraph(b.graph, { inputShapes: { x: [2, 3, 4] } })).toThrow('size_axis');
  });
  it('rejects size of a rank-zero tensor', () => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    b.output(b.node('size', 'size', { x: 'x' }));
    expect(() => compileGraph(b.graph, { inputShapes: { x: [] } })).toThrow('size_axis');
  });
  it.each([false, true])(
    'reshapes with wired size/mul and inferred dimension=%s',
    async (inferred) => {
      const graph = flatten(inferred);
      const before = JSON.stringify(graph);
      const plan = compileGraph(graph, { inputShapes: { x: [2, 3, 4] } });
      expect(plan.outputShapes.y).toEqual([2, 12]);
      expect(JSON.stringify(graph)).toBe(before);
      const x = T.tensor(values, [2, 3, 4]);
      const outputs = plan.execute({ x });
      expect(Array.from(await outputs.y.read())).toEqual(values);
      const [gradient] = await T.backward(outputs.y, [x]);
      expect(Array.from(await gradient.read())).toEqual(Array(24).fill(1));
      gradient.dispose();
      plan.releaseExecution(outputs);
      plan.dispose();
      x.dispose();
    },
  );
  it.each([
    ['input', undefined, 'd0', 'dim_not_static'],
    ['param', undefined, 'd0', 'dim_not_static'],
    ['const', 2.5, 'd0', 'dim_not_integer'],
    ['const', 2, 'd2', 'dim_out_of_range'],
  ] as const)('reports structured %s dimension errors (%s, %s)', (component, value, port, key) => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    b.node(
      'dim',
      component,
      {},
      { name: 'dim', shape: [], ...(value === undefined ? {} : { value }) },
    );
    b.output(b.node('reshape', 'reshape', { x: 'x', [port]: 'dim' }, { shape: [2, 12] }));
    try {
      compileGraph(b.graph, { inputShapes: { x: [2, 3, 4], dim: [] } });
      expect.fail('expected dimension validation failure');
    } catch (error) {
      expect(error).toMatchObject({ key, node: 'reshape', port });
    }
  });
  it.each([false, true])('arange accepts a wired scalar (composite div=%s)', async (divide) => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    let n = b.node('size', 'size', { x: 'x' });
    if (divide) {
      b.node('two', 'const', {}, { value: 2 });
      n = b.node('half', 'div', { a: n, b: 'two' });
    }
    b.output(b.node('range', 'arange', { n }, { n: 99 }));
    const plan = compileGraph(b.graph, { registry, inputShapes: { x: [8] } });
    const expected = Array.from({ length: divide ? 4 : 8 }, (_, i) => i);
    expect(plan.outputShapes.y).toEqual([expected.length]);
    const x = T.tensor(Array(8).fill(0), [8]);
    const outputs = plan.execute({ x });
    expect(Array.from(await outputs.y.read())).toEqual(expected);
    plan.releaseExecution(outputs);
    plan.dispose();
    x.dispose();
  });
  it('symbolic backward ignores shape dependencies but preserves reshape data gradients', async () => {
    const trace = await traceBackward(flatten(), {
      inputShapes: { x: [2, 3, 4] },
      output: 'y',
      wrt: ['x'],
      inputs: { x: values },
    });
    expect(trace.gradient.x.values).toEqual(Array(24).fill(1));
  });
  it('size has zero eager and symbolic derivative', async () => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    b.output(b.node('size', 'size', { x: 'x' }));
    const plan = compileGraph(b.graph, { inputShapes: { x: [3] } });
    const x = T.tensor([1, 2, 3], [3]);
    const outputs = plan.execute({ x });
    const [gradient] = await T.backward(outputs.y, [x]);
    expect(Array.from(await gradient.read())).toEqual([0, 0, 0]);
    gradient.dispose();
    plan.releaseExecution(outputs);
    plan.dispose();
    x.dispose();
    const trace = await traceBackward(b.graph, {
      inputShapes: { x: [3] },
      output: 'y',
      wrt: ['x'],
      inputs: { x: [1, 2, 3] },
    });
    expect(trace.gradient.x.values).toEqual([0, 0, 0]);
  });
  it('trains one step through a wired reshape without leaking buffers', async () => {
    const b = builder();
    b.node('x', 'input', {}, { name: 'x' });
    b.node('weight', 'param', {}, { shape: [], init: 'ones' });
    b.node('scaled', 'mul', { a: 'x', b: 'weight' });
    b.node('batch', 'size', { x: 'x' }, { axis: 0 });
    b.output(b.node('flat', 'reshape', { x: 'scaled', d0: 'batch' }, { shape: [1, -1] }));
    const backend = new CpuBackend();
    const plan = compileGraph(b.graph, { backend, inputShapes: { x: [2, 3, 4] } });
    const session = new TrainingSession(
      plan,
      () => {
        const x = T.tensor(Array(24).fill(1), [2, 3, 4], backend);
        return { inputs: { x }, targets: x };
      },
      (prediction) => T.sum(T.mul(prediction, prediction), [0, 1]),
      { kind: 'sgd', learningRate: 0.01 },
      'y',
      { accuracy: false },
    );
    try {
      const initial = await session.evaluate();
      const report = await session.run(1);
      expect(report.step).toBe(1);
      expect(report.loss).toBeLessThan(initial.loss);
      expect((await plan.parameters.get('weight')!.read())[0]).toBeCloseTo(0.52);
    } finally {
      session.dispose();
      plan.dispose();
    }
    expect(backend.liveBuffers).toBe(0);
  });
  it('rewrites level 4-5 with wired dimensions and passes actual training acceptance', async () => {
    const level = chapter4TrainLevels.find((level) => level.id === '4-5')!;
    const before = JSON.stringify(level);
    const graph = structuredClone(level.referenceGraph!);
    const flat = graph.nodes.find((node) => node.id === 'flatten')!;
    flat.attrs = { shape: [1, 1] };
    graph.nodes.push(
      { id: 'batchSize', component: 'size', attrs: { axis: 0 } },
      { id: 'contextSize', component: 'size', attrs: { axis: 1 } },
      { id: 'embeddingSize', component: 'size', attrs: { axis: 2 } },
      { id: 'flatWidth', component: 'mul' },
    );
    for (const id of ['batchSize', 'contextSize', 'embeddingSize'])
      graph.edges.push({ from: { node: 'lookup', port: 'out' }, to: { node: id, port: 'x' } });
    graph.edges.push(
      { from: { node: 'contextSize', port: 'out' }, to: { node: 'flatWidth', port: 'a' } },
      { from: { node: 'embeddingSize', port: 'out' }, to: { node: 'flatWidth', port: 'b' } },
      { from: { node: 'batchSize', port: 'out' }, to: { node: 'flatten', port: 'd0' } },
      { from: { node: 'flatWidth', port: 'out' }, to: { node: 'flatten', port: 'd1' } },
    );
    const preceding = [
      ...chapter0Levels,
      ...chapter1Levels,
      ...chapter2Levels,
      ...chapter3ForwardLevels,
      ...chapter4ForwardLevels,
    ];
    const registry = referenceRegistry(preceding);
    // As in the chapter tests, include components earned on the route to this lesson.
    const earned = preceding.flatMap((lesson) => lesson.unlockComponents ?? []);
    const result = await evaluateLevel(
      { ...level, available: [...level.available, ...earned] },
      graph,
      { registry },
    );
    expect(result.passed, JSON.stringify(result)).toBe(true);
    expect(result.training?.step).toBe(level.steps);
    expect(result.training!.loss).toBeLessThan(level.threshold.loss);
    expect(result.tests).toEqual([expect.objectContaining({ id: 'evaluation', passed: true })]);
    expect(JSON.stringify(level)).toBe(before);
  }, 60000);
});
