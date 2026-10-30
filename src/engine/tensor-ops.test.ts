import { describe, expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { backward, disposeGraph, mul, sum, tensor, type Tensor } from './tensor';
import { compileGraph, GraphError, type Attributes, type GraphJSON, type GraphPlan } from './graph';
import { buildBackwardGraph } from './backward-graph';
import { createDefaultRegistry } from './graph';
import { createEditorState, editorReducer } from '../editor/state';
import { sigmoidLevel } from '../levels/examples';
import { nodePorts } from '../ui/graph-geometry';
import { buildPlayerRegistry } from '../levels/registry';
import { gameLevels } from '../levels/catalog';
import { createInitialProgress } from '../storage';

type Spec = { data: number[]; shape: number[] };
const iota = (shape: number[], scale = 1): Spec => {
  const n = shape.reduce((a, b) => a * b, 1);
  return { shape, data: Array.from({ length: n }, (_, i) => (i + 1) * scale) };
};

function opGraph(
  component: string,
  attrs: Attributes,
  ports: string[],
  shapes: number[][],
  output = 'out',
): GraphJSON {
  return {
    version: 1,
    nodes: [
      ...ports.map((port, i) => ({
        id: `in_${port}`,
        component: 'input',
        attrs: { name: port, shape: shapes[i] },
      })),
      { id: 'op', component, attrs },
      { id: 'output', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [
      ...ports.map((port) => ({
        from: { node: `in_${port}`, port: 'out' },
        to: { node: 'op', port },
      })),
      { from: { node: 'op', port: output }, to: { node: 'output', port: 'x' } },
    ],
  };
}

const plans: GraphPlan[] = [];
async function run(
  component: string,
  attrs: Attributes,
  ports: string[],
  specs: Spec[],
  output = 'out',
  withGradients = false,
) {
  const backend = new CpuBackend();
  const inputs = specs.map((s) => tensor(s.data, s.shape, backend));
  const plan = compileGraph(
    opGraph(
      component,
      attrs,
      ports,
      specs.map((s) => s.shape),
      output,
    ),
    {
      backend,
    },
  );
  plans.push(plan);
  const roots: Tensor[] = [...inputs];
  try {
    const out = plan.execute(Object.fromEntries(ports.map((p, i) => [p, inputs[i]]))).out;
    roots.push(out);
    const value = Array.from(await out.read());
    const shape = [...out.shape];
    if (!withGradients) return { value, shape, gradients: [] as number[][] };
    // Position-dependent weights so the gradient is not a trivial all-ones pattern.
    const n = value.length;
    const w = tensor(
      Array.from({ length: n }, (_, i) => ((i * 7) % 5) - 2),
      out.shape,
      backend,
    );
    const loss = sum(
      mul(out, w),
      out.shape.map((_, i) => i),
    );
    roots.push(w, loss);
    const grads = await backward(loss, inputs);
    roots.push(...grads);
    return {
      value,
      shape,
      gradients: await Promise.all(grads.map(async (g) => Array.from(await g.read()))),
    };
  } finally {
    disposeGraph(roots);
    plans.splice(0).forEach((p) => p.dispose());
  }
}

async function gradientCheck(
  component: string,
  attrs: Attributes,
  ports: string[],
  specs: Spec[],
  output = 'out',
) {
  const analytic = await run(component, attrs, ports, specs, output, true);
  const weights = analytic.value.map((_, i) => ((i * 7) % 5) - 2);
  const objective = (v: number[]) => v.reduce((t, x, i) => t + x * weights[i], 0);
  const h = 1e-2;
  for (let p = 0; p < specs.length; p++)
    for (let j = 0; j < specs[p].data.length; j++) {
      const shifted = (d: number) =>
        specs.map((s, i) => ({
          shape: s.shape,
          data: s.data.map((v, k) => v + (i === p && k === j ? d : 0)),
        }));
      const plus = objective((await run(component, attrs, ports, shifted(h), output)).value);
      const minus = objective((await run(component, attrs, ports, shifted(-h), output)).value);
      expect(analytic.gradients[p][j]).toBeCloseTo((plus - minus) / (2 * h), 2);
    }
}

describe('slice', () => {
  it('takes one index and drops the axis, or a range and keeps it', async () => {
    const x = iota([2, 3, 2]);
    expect(await run('slice', { axis: 1, index: 2, length: 1 }, ['x'], [x])).toMatchObject({
      shape: [2, 2],
      value: [5, 6, 11, 12],
    });
    expect(
      await run('slice', { axis: -1, index: -1, length: 1, keepdims: true }, ['x'], [x]),
    ).toMatchObject({
      shape: [2, 3, 1],
      value: [2, 4, 6, 8, 10, 12],
    });
    expect(await run('slice', { axis: 1, index: 1, length: 2 }, ['x'], [x])).toMatchObject({
      shape: [2, 2, 2],
      value: [3, 4, 5, 6, 9, 10, 11, 12],
    });
    expect(await run('slice', { axis: 0, index: 0, length: 1 }, ['x'], [iota([4])])).toMatchObject({
      shape: [],
      value: [1],
    });
  });
  it('matches numerical gradients', async () => {
    await gradientCheck('slice', { axis: 1, index: 1, length: 2 }, ['x'], [iota([2, 3, 2], 0.1)]);
  });
  it('rejects out-of-range slices and axes', () => {
    const compile = (attrs: Attributes) => () =>
      compileGraph(opGraph('slice', attrs, ['x'], [[2, 6]])).dispose();
    expect(compile({ axis: 1, index: 5, length: 2 })).toThrowError(/slice_range/);
    expect(compile({ axis: 2, index: 0, length: 1 })).toThrowError(/tensor_axis/);
  });
});

describe('stack and concat', () => {
  it('stacks same-shape inputs into a new axis', async () => {
    const r = await run(
      'stack',
      { count: 3, axis: 1 },
      ['a', 'b', 'c'],
      [iota([2, 2]), iota([2, 2], 10), iota([2, 2], 100)],
    );
    expect(r.shape).toEqual([2, 3, 2]);
    expect(r.value).toEqual([1, 2, 10, 20, 100, 200, 3, 4, 30, 40, 300, 400]);
    const last = await run('stack', { count: 2, axis: -1 }, ['a', 'b'], [iota([3]), iota([3], 10)]);
    expect(last).toMatchObject({ shape: [3, 2], value: [1, 10, 2, 20, 3, 30] });
  });
  it('concatenates along an axis with different lengths there', async () => {
    const r = await run(
      'concat',
      { count: 2, axis: 1 },
      ['a', 'b'],
      [iota([2, 1]), iota([2, 2], 10)],
    );
    expect(r).toMatchObject({ shape: [2, 3], value: [1, 10, 20, 2, 30, 40] });
  });
  it('match numerical gradients', async () => {
    await gradientCheck(
      'stack',
      { count: 2, axis: 0 },
      ['a', 'b'],
      [iota([2, 2], 0.1), iota([2, 2], 0.2)],
    );
    await gradientCheck(
      'concat',
      { count: 2, axis: -1 },
      ['a', 'b'],
      [iota([2, 1], 0.1), iota([2, 2], 0.2)],
    );
  });
  it('reports mismatched shapes and missing live inputs', () => {
    expect(() =>
      compileGraph(
        opGraph(
          'stack',
          { count: 2, axis: 1 },
          ['a', 'b'],
          [
            [2, 2],
            [2, 3],
          ],
        ),
      ).dispose(),
    ).toThrowError(/stack_shape/);
    expect(() =>
      compileGraph(
        opGraph(
          'concat',
          { count: 2, axis: 0 },
          ['a', 'b'],
          [
            [2, 2],
            [2, 3],
          ],
        ),
      ).dispose(),
    ).toThrowError(/concat_shape/);
    expect(() =>
      compileGraph(opGraph('stack', { count: 3, axis: 0 }, ['a', 'b'], [[2], [2]])).dispose(),
    ).toThrowError(GraphError);
  });
});

describe('split', () => {
  it('splits into equal parts with one output each', async () => {
    const x = iota([2, 4]);
    expect(await run('split', { parts: 2, axis: -1 }, ['x'], [x], 'out1')).toMatchObject({
      shape: [2, 2],
      value: [3, 4, 7, 8],
    });
    expect(await run('split', { parts: 2, axis: 0 }, ['x'], [x], 'out0')).toMatchObject({
      shape: [1, 4],
      value: [1, 2, 3, 4],
    });
    await gradientCheck('split', { parts: 2, axis: -1 }, ['x'], [iota([2, 4], 0.1)], 'out1');
    expect(() =>
      compileGraph(opGraph('split', { parts: 3, axis: -1 }, ['x'], [[2, 4]], 'out0')).dispose(),
    ).toThrowError(/split_uneven/);
  });
});

it('lowers to the symbolic training graph', () => {
  const graph = opGraph('slice', { axis: 1, index: 1, length: 1 }, ['x'], [[2, 3]]);
  expect(() =>
    buildBackwardGraph(graph, { output: 'out', inputShapes: { x: [2, 3] }, wrt: ['x'] }),
  ).not.toThrow();
});

it('shows count/parts ports and drops wires to ports that disappear', () => {
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'stack', component: 'stack', attrs: { count: 3, axis: 1 } },
      { id: 'split', component: 'split', attrs: { parts: 3, axis: -1 } },
      { id: 'out', component: 'output', attrs: { name: 'y' } },
    ],
    edges: ['a', 'b', 'c'].map((port) => ({
      from: { node: 'x', port: 'out' },
      to: { node: 'stack', port },
    })),
  };
  graph.edges.push({ from: { node: 'split', port: 'out2' }, to: { node: 'out', port: 'x' } });
  const registry = createDefaultRegistry();
  expect(nodePorts(graph.nodes[1], registry, 'inputs')).toEqual(['a', 'b', 'c']);
  expect(nodePorts(graph.nodes[2], registry, 'outputs')).toEqual(['out0', 'out1', 'out2']);
  let state = createEditorState(sigmoidLevel, graph);
  state = editorReducer(state, { type: 'setAttrs', id: 'stack', attrs: { count: 2 } }, registry);
  state = editorReducer(state, { type: 'setAttrs', id: 'split', attrs: { parts: 2 } }, registry);
  expect(state.graph.edges.map((e) => e.to.port).sort()).toEqual(['a', 'b']);
});

it('keeps the built-in tensor ops in the player registry', () => {
  const registry = buildPlayerRegistry(gameLevels, createInitialProgress());
  for (const name of ['slice', 'stack', 'concat', 'split'])
    expect(registry.get(name)).toBeDefined();
});
