import { describe, expect, it } from 'vitest';
import { buildBackwardGraph, traceBackward } from './backward-graph';
import {
  compileGraph,
  createDefaultRegistry,
  GraphError,
  type ShapePorts,
  type GraphJSON,
} from './graph';
import { CpuBackend } from './cpu';
import { Random } from './random';
import * as T from './tensor';
import { gameLevels } from '../levels/catalog';
import { referenceRegistry } from '../levels/level-helpers';

const registry = referenceRegistry(gameLevels.filter((l) => l.kind !== 'train'));
const square: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x', shape: [] } },
    { id: 'square', component: 'mul' },
    { id: 'out', component: 'output', attrs: { name: 'y' } },
  ],
  edges: [
    { from: { node: 'x', port: 'out' }, to: { node: 'square', port: 'a' } },
    { from: { node: 'x', port: 'out' }, to: { node: 'square', port: 'b' } },
    { from: { node: 'square', port: 'out' }, to: { node: 'out', port: 'x' } },
  ],
};
function close(actual: number[], expected: ArrayLike<number>) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, i) =>
    expect(Math.abs(value - expected[i])).toBeLessThanOrEqual(1e-5 + 1e-4 * Math.abs(expected[i])),
  );
}
const options = { registry, inputShapes: { x: [] }, output: 'y', wrt: ['x'] };
describe('backward graph', () => {
  it('preserves IDs and serializability, uses independent accumulation adds', async () => {
    const before = JSON.stringify(square);
    const built = buildBackwardGraph(square, options);
    expect(JSON.stringify(square)).toBe(before);
    expect(JSON.parse(JSON.stringify(built))).toEqual(built);
    expect(square.nodes.every((n) => built.graph.nodes.some((b) => b.id === n.id))).toBe(true);
    const accum = built.graph.nodes.filter((n) => built.roles[n.id].role === 'accumulate');
    expect(accum).toHaveLength(1);
    expect(accum[0].component).toBe('add');
    expect(built.roles[accum[0].id].mirrorOf).toBe('x');
    const plan = compileGraph(built.graph, options);
    expect(plan.outputShapes.grad_x).toEqual([]);
    plan.dispose();
    const trace = await traceBackward(square, { ...options, inputs: { x: [3] } });
    expect(trace.gradient.x.values).toEqual([6]);
    expect(trace.forward.square.values).toEqual([9]);
  });
  it('validates output, graph and input data', async () => {
    expect(() => buildBackwardGraph(square, { ...options, output: 'missing' })).toThrow(
      'missing_output',
    );
    expect(() => buildBackwardGraph({ ...square, edges: [] }, options)).toThrow('missing_port');
    await expect(traceBackward(square, { ...options, inputs: {} })).rejects.toThrow(
      'missing_input',
    );
    await expect(traceBackward(square, { ...options, inputs: { x: [1, 2] } })).rejects.toThrow(
      'engine.data.length',
    );
  });
  for (const level of gameLevels.filter((l) => /^[012]-/.test(l.id))) {
    it(`matches eager backward for all fixed/generated/variant cases: ${level.id}`, async () => {
      const backend = new CpuBackend();
      const groups =
        level.kind === 'train'
          ? [
              {
                inputs: level.interface.inputs,
                fixed: [{ id: 'train', inputs: {} as Record<string, number[]> }],
              },
            ]
          : [
              { inputs: level.interface.inputs, fixed: level.fixed, generated: level.generated },
              ...(level.shapeVariants ?? []),
            ];
      for (const group of groups) {
        const random = new Random(
          'generated' in group && group.generated ? group.generated.seed : 1,
        );
        const cases = [
          ...group.fixed,
          ...('generated' in group && group.generated
            ? Array.from({ length: group.generated.count }, (_, i) =>
                group.generated!.generate(random, i),
              )
            : []),
        ];
        for (const test of cases) {
          let data = test.inputs;
          if (level.kind === 'train') {
            const batch = level.dataset(new Random(level.seed));
            data = Object.fromEntries(
              await Promise.all(
                Object.entries(batch.inputs).map(async ([name, tensor]) => [
                  name,
                  Array.from(await tensor.read()),
                ]),
              ),
            );
            new Set([...Object.values(batch.inputs), batch.targets]).forEach((x) => x.dispose());
          }
          for (const preferNative of [true, false]) {
            const options = {
              backend,
              registry,
              inputShapes: group.inputs,
              output: level.interface.output,
              preferNative,
              wrt: Object.keys(group.inputs),
            };
            const plan = compileGraph(level.referenceGraph!, {
              ...options,
              random: new Random(42),
            });
            const inputs = Object.fromEntries(
              Object.entries(group.inputs).map(([name, shape]) => [
                name,
                T.tensor(data[name], shape, backend),
              ]),
            );
            const outputs = plan.execute(inputs);
            const targets = [...Object.values(inputs), ...plan.parameters.values()];
            const gradients = await T.backward(outputs[level.interface.output], targets);
            const trace = await traceBackward(level.referenceGraph!, { ...options, inputs: data });
            for (const [i, name] of Object.keys(inputs).entries()) {
              const node = level.referenceGraph!.nodes.find(
                (n) => n.component === 'input' && n.attrs?.name === name,
              )!;
              close(trace.gradient[node.id].values, await gradients[i].read());
            }
            for (const [i, path] of [...plan.parameters.keys()].entries()) {
              const node = level.referenceGraph!.nodes.find(
                (n) => encodeURIComponent(n.id) === path,
              );
              if (node)
                close(
                  trace.gradient[node.id].values,
                  await gradients[Object.keys(inputs).length + i].read(),
                );
            }
            gradients.forEach((x) => x.dispose());
            plan.releaseExecution(outputs);
            plan.dispose();
            Object.values(inputs).forEach((x) => x.dispose());
            expect(backend.liveBuffers).toBe(0);
          }
        }
      }
    });
  }
});

function primitive(
  component: string,
  shapes: ShapePorts,
  attrs: Record<string, unknown> = {},
): GraphJSON {
  const ports = createDefaultRegistry().get(component)!.inputs;
  return {
    version: 1,
    nodes: [
      ...Object.keys(shapes).map((name) => ({ id: name, component: 'input', attrs: { name } })),
      { id: 'op', component, attrs },
      { id: 'out', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [
      ...Object.keys(shapes).map((name, i) => ({
        from: { node: name, port: 'out' },
        to: { node: 'op', port: ports[i] },
      })),
      { from: { node: 'op', port: 'out' }, to: { node: 'out', port: 'x' } },
    ],
  };
}
async function checkPrimitive(
  component: string,
  shapes: ShapePorts,
  data: Record<string, number[]>,
  attrs: Record<string, unknown> = {},
) {
  const graph = primitive(component, shapes, attrs);
  const opts = { registry, inputShapes: shapes, output: 'y', wrt: Object.keys(shapes) };
  const backend = new CpuBackend();
  const plan = compileGraph(graph, { ...opts, backend });
  const inputs = Object.fromEntries(
    Object.entries(shapes).map(([name, shape]) => [name, T.tensor(data[name], shape, backend)]),
  );
  const outputs = plan.execute(inputs);
  const gradients = await T.backward(outputs.y, Object.values(inputs));
  try {
    const trace = await traceBackward(graph, { ...opts, inputs: data });
    for (const [i, name] of Object.keys(shapes).entries())
      close(trace.gradient[name].values, await gradients[i].read());
    const built = buildBackwardGraph(graph, opts);
    const checked = compileGraph(built.graph, opts);
    checked.dispose();
    expect(JSON.parse(JSON.stringify(built))).toEqual(built);
  } finally {
    gradients.forEach((t) => t.dispose());
    plan.releaseExecution(outputs);
    plan.dispose();
    Object.values(inputs).forEach((t) => t.dispose());
  }
  expect(backend.liveBuffers).toBe(0);
}
describe('primitive VJPs and structural contracts', () => {
  for (const op of ['add', 'mul', 'max', 'ge'])
    it(`${op}: broadcast/ties`, () =>
      checkPrimitive(op, { a: [2, 3], b: [3] }, { a: [1, 2, 3, 3, 2, 1], b: [1, 3, 1] }));
  for (const op of ['exp', 'log']) it(op, () => checkPrimitive(op, { x: [3] }, { x: [0.2, 1, 2] }));
  for (const op of ['sum', 'reduce_max'])
    for (const axes of [undefined, [0], [-1], [0, 2], []])
      for (const keepdims of [false, true])
        it(`${op}: ${axes} keep=${keepdims}`, () =>
          checkPrimitive(
            op,
            { x: [2, 2, 2] },
            { x: [3, 3, 1, 2, 3, 1, 2, 2] },
            axes ? { axes, keepdims } : { keepdims },
          ));
  for (const [op, attrs] of [
    ['reshape', { shape: [3, 2] }],
    ['unsqueeze', { axis: -1 }],
    ['transpose', { perm: [1, 0] }],
  ] as const)
    it(op, () => checkPrimitive(op, { x: [2, 3] }, { x: [1, 2, 3, 4, 5, 6] }, attrs));
  for (const op of ['softmax', 'layernorm'])
    it(`${op}: native composite expansion`, async () => {
      await checkPrimitive(op, { x: [2, 3] }, { x: [1, 2, 3, -2, 1, 4] });
      const g = primitive(op, { x: [2, 3] });
      // Force a collision between a public ID and the preferred nested ID.
      g.nodes.push({ id: 'op/result', component: 'const', attrs: { value: 2 } });
      const built = buildBackwardGraph(g, { inputShapes: { x: [2, 3] }, output: 'y', wrt: ['x'] });
      expect(built.graph.nodes.some((n) => n.component === op)).toBe(false);
      expect(built.graph.nodes.some((n) => n.id === 'op')).toBe(true);
      expect(Object.values(built.roles).some((r) => r.origin?.includes('op'))).toBe(true);
      const compiled = compileGraph(built.graph, { inputShapes: { x: [2, 3] } });
      compiled.dispose();
    });
  it('matmul', () =>
    checkPrimitive(
      'matmul',
      { a: [2, 3], b: [3, 2] },
      { a: [1, 2, 3, 4, 5, 6], b: [2, 1, 3, 2, 4, 3] },
    ));
  it('build never executes implementations', () => {
    const local = createDefaultRegistry();
    // Only alter this registry’s primitive implementation, not shared native definitions.
    local.get('mul')!.primitive!.execute = () => {
      throw Error('execution forbidden');
    };
    buildBackwardGraph(square, { ...options, registry: local });
  });
  it('unsupported gather/arange errors', () => {
    for (const op of ['gather', 'arange']) {
      const shapes: ShapePorts = op === 'gather' ? { table: [2, 2], idx: [1] } : {};
      try {
        buildBackwardGraph(primitive(op, shapes, { n: 2 }), {
          inputShapes: shapes,
          output: 'y',
          wrt: [],
        });
        throw Error('missing error');
      } catch (error) {
        expect(error).toBeInstanceOf(GraphError);
        expect((error as GraphError).key).toBe('backward_unsupported');
        expect((error as GraphError).params).toEqual({ component: op });
      }
    }
  });
  it('param and const leaves; seed and disconnected gradients', async () => {
    const g = primitive('mul', { a: [2], b: [2] });
    g.nodes[1] = { id: 'b', component: 'param', attrs: { shape: [2], init: 'constant', value: 3 } };
    g.nodes.push({ id: 'c', component: 'const', attrs: { value: 5 } });
    const trace = await traceBackward(g, {
      inputShapes: { a: [2] },
      output: 'y',
      wrt: ['a', 'b'],
      inputs: { a: [2, 4] },
    });
    expect(trace.gradient.a.values).toEqual([3, 3]);
    expect(trace.gradient.b.values).toEqual([2, 4]);
    expect(trace.gradient.c.values).toEqual([0]);
    expect(trace.gradient.op.values).toEqual([1, 1]);
  });
});
