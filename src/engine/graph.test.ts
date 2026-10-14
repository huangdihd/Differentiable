import { describe, expect, it } from 'vitest';
import {
  compileGraph,
  safeCompileGraph,
  createDefaultRegistry,
  GraphError,
  type GraphJSON,
  type GraphEdge,
} from './graph';
import { tensor, exp, backward, mul, sum } from './tensor';

const edge = (from: string, to: string, port = 'x', output = 'out'): GraphEdge => ({
  from: { node: from, port: output },
  to: { node: to, port },
});
const identity: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'input', component: 'input', attrs: { name: 'x', shape: [2] } },
    { id: 'output', component: 'output', attrs: { name: 'out' } },
  ],
  edges: [edge('input', 'output')],
};

describe('graph compiler', () => {
  it('registers only the designed primitive vocabulary and canonical standard components', () => {
    const registry = createDefaultRegistry();
    expect(
      registry
        .list()
        .filter((def) => def.primitive)
        .map((def) => def.name)
        .sort(),
    ).toEqual(
      [
        'input',
        'output',
        'const',
        'param',
        'add',
        'mul',
        'exp',
        'log',
        'max',
        'ge',
        'sum',
        'reduce_max',
        'reshape',
        'unsqueeze',
        'transpose',
        'gather',
        'arange',
      ].sort(),
    );
    for (const name of ['matmul', 'softmax', 'layernorm']) {
      const definition = registry.get(name)!;
      expect(definition.primitive).toBeUndefined();
      expect(definition.native).toBeDefined();
      const graph =
        typeof definition.subgraph === 'function' ? definition.subgraph({}) : definition.subgraph!;
      expect(graph.nodes.every((node) => registry.get(node.component)?.primitive)).toBe(true);
    }
    expect(registry.get('reduceMax')).toBeUndefined();
  });
  it('infers without allocating and executes an identity graph', async () => {
    const plan = compileGraph(identity);
    expect(plan.inputShapes).toEqual({ x: [2] });
    expect(plan.outputShapes).toEqual({ out: [2] });
    expect(plan.parameters.size).toBe(0);
    const x = tensor([2, 3], [2]);
    expect(plan.execute({ x }).out).toBe(x);
    expect(Array.from(await x.read())).toEqual([2, 3]);
    plan.dispose();
    x.dispose();
  });
  it('reports a missing port with structured context', () => {
    try {
      compileGraph({ ...identity, edges: [] });
      throw new Error('expected failure');
    } catch (error) {
      expect(error).toBeInstanceOf(GraphError);
      expect(error).toMatchObject({ key: 'missing_port', node: 'output', port: 'x' });
    }
  });
  it('rejects invalid source ports and unsupported versions', () => {
    expect(() =>
      compileGraph({ ...identity, edges: [edge('input', 'output', 'x', 'bad')] }),
    ).toThrow('unknown_output_port');
    expect(() => compileGraph({ ...identity, version: 2 } as unknown as GraphJSON)).toThrow(
      'unsupported_version',
    );
  });
  it('detects cycles even in disconnected nodes', () => {
    expect(() =>
      compileGraph({
        version: 1,
        nodes: [
          { id: 'a', component: 'exp' },
          { id: 'b', component: 'exp' },
        ],
        edges: [edge('a', 'b'), edge('b', 'a')],
      }),
    ).toThrow('cycle');
  });
  it('validates implementation availability and recursive components', () => {
    const registry = createDefaultRegistry().register({
      name: 'missing',
      inputs: [],
      outputs: ['out'],
    });
    expect(() =>
      compileGraph(
        { version: 1, nodes: [{ id: 'm', component: 'missing' }], edges: [] },
        { registry },
      ),
    ).toThrow('implementation_unavailable');
    registry.register({
      name: 'loop',
      inputs: [],
      outputs: [],
      subgraph: { version: 1, nodes: [{ id: 'self', component: 'loop' }], edges: [] },
    });
    expect(() =>
      compileGraph(
        { version: 1, nodes: [{ id: 'm', component: 'loop' }], edges: [] },
        { registry },
      ),
    ).toThrow('recursive_component');
  });
  it('flattens nested multi-output components with distinct persistent parameters', async () => {
    const subgraph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'p', component: 'param', attrs: { shape: [2], init: 'ones' } },
        { id: 'o1', component: 'output', attrs: { name: 'first' } },
        { id: 'o2', component: 'output', attrs: { name: 'second' } },
      ],
      edges: [edge('p', 'o1'), edge('p', 'o2')],
    };
    const registry = createDefaultRegistry().register({
      name: 'pair',
      inputs: [],
      outputs: ['first', 'second'],
      subgraph,
    });
    const graph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'a', component: 'pair' },
        { id: 'b', component: 'pair' },
        { id: 'o1', component: 'output', attrs: { name: 'left' } },
        { id: 'o2', component: 'output', attrs: { name: 'right' } },
      ],
      edges: [edge('a', 'o1', 'x', 'first'), edge('b', 'o2', 'x', 'second')],
    };
    const plan = compileGraph(graph, { registry });
    expect(plan.outputShapes).toEqual({ left: [2], right: [2] });
    const result = plan.execute({});
    expect([...plan.parameters.keys()]).toEqual(['a/p', 'b/p']);
    expect(result.left).not.toBe(result.right);
    expect(plan.execute({}).left).toBe(result.left);
    expect(Array.from(await result.right.read())).toEqual([1, 1]);
    plan.dispose();
    expect(() => plan.execute({})).toThrow('disposed_plan');
  });
  it('prefers native implementations but allows selecting primitives', () => {
    const registry = createDefaultRegistry().register({
      name: 'choice',
      inputs: [],
      outputs: ['out'],
      primitive: { infer: () => ({ out: [1] }), execute: () => ({ out: tensor([1], [1]) }) },
      native: { infer: () => ({ out: [2] }), execute: () => ({ out: tensor([1, 2], [2]) }) },
    });
    const g: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'c', component: 'choice' },
        { id: 'o', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [edge('c', 'o')],
    };
    expect(compileGraph(g, { registry }).outputShapes.y).toEqual([2]);
    expect(compileGraph(g, { registry, preferNative: false }).outputShapes.y).toEqual([1]);
  });
  it('rejects broadcast and reshape mismatches during compilation', () => {
    const g: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x', shape: [2] } },
        { id: 'r', component: 'reshape', attrs: { shape: [3] } },
      ],
      edges: [edge('x', 'r')],
    };
    expect(() => compileGraph(g)).toThrow('reshape_mismatch');
    expect(() => compileGraph(identity).execute({ x: tensor([1], [1]) })).toThrow(
      'input_shape_mismatch',
    );
  });
});

describe('graph safety and execution ownership', () => {
  it('checks only top-level availability, including disconnected nodes', () => {
    expect(safeCompileGraph(identity, { availableComponents: [] }).ok).toBe(true);
    const nested: GraphJSON = {
      version: 1,
      nodes: [{ id: 'p', component: 'param', attrs: { shape: [] } }],
      edges: [],
    };
    const registry = createDefaultRegistry().register({
      name: 'wrapped',
      inputs: [],
      outputs: [],
      subgraph: nested,
      native: { infer: () => ({}), execute: () => ({}) },
    });
    const result = safeCompileGraph(
      { version: 1, nodes: [{ id: 'w', component: 'wrapped' }], edges: [] },
      { registry, availableComponents: ['wrapped'] },
    );
    expect(result.ok).toBe(true);
    if (result.ok) result.plan.dispose();
    const denied = safeCompileGraph(nested, { registry, availableComponents: ['wrapped'] });
    expect(denied).toMatchObject({
      ok: false,
      errors: [{ key: 'component_unavailable', node: 'p' }],
    });
  });
  it.each([true, false])(
    'uses unlocked softmax without exposing its internal exp (native=%s)',
    async (preferNative) => {
      const graph: GraphJSON = {
        ...identity,
        nodes: [...identity.nodes, { id: 's', component: 'softmax' }],
        edges: [edge('input', 's'), edge('s', 'output')],
      };
      const plan = compileGraph(graph, { preferNative, availableComponents: ['softmax'] });
      const x = tensor([0, 0], [2]);
      const outputs = plan.execute({ x });
      expect(Array.from(await outputs.out.read())).toEqual([0.5, 0.5]);
      plan.releaseExecution(outputs);
      plan.dispose();
      x.dispose();
      expect(() =>
        compileGraph(
          { ...graph, nodes: [...graph.nodes, { id: 'forbidden', component: 'exp' }] },
          { availableComponents: ['softmax'] },
        ),
      ).toThrow('component_unavailable');
    },
  );
  it('initializes constants and bypasses native shortcuts for nested parameters', async () => {
    const subgraph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'p', component: 'param', attrs: { shape: [2], init: 'constant', value: 3 } },
        { id: 'o', component: 'output', attrs: { name: 'out' } },
      ],
      edges: [edge('p', 'o')],
    };
    const registry = createDefaultRegistry().register({
      name: 'wrapped',
      inputs: [],
      outputs: ['out'],
      subgraph,
      native: {
        infer: () => {
          throw new Error('must not use native');
        },
        execute: () => ({}),
      },
    });
    const plan = compileGraph(
      {
        version: 1,
        nodes: [
          { id: 'w', component: 'wrapped' },
          { id: 'o', component: 'output', attrs: { name: 'out' } },
        ],
        edges: [edge('w', 'o')],
      },
      { registry },
    );
    const outputs = plan.execute({});
    plan.releaseExecution(outputs);
    expect(Array.from(await outputs.out.read())).toEqual([3, 3]);
    expect(plan.parameters.get('w/p')).toBe(outputs.out);
    plan.dispose();
    for (const attrs of [
      { init: 'constant', value: Infinity },
      { init: 'constant' },
      { scale: NaN },
      { scale: -1 },
    ]) {
      expect(
        safeCompileGraph({
          version: 1,
          nodes: [{ id: 'p', component: 'param', attrs: { shape: [], ...attrs } }],
          edges: [],
        }).ok,
      ).toBe(false);
    }
  });
  it('releases disconnected results and their tapes while retaining inputs and constants', () => {
    const seen: ReturnType<typeof tensor>[] = [];
    const registry = createDefaultRegistry().register({
      name: 'capture',
      inputs: ['x'],
      outputs: ['out'],
      primitive: {
        infer: (i) => ({ out: i.x }),
        execute: (i) => {
          const hidden = exp(i.x),
            out = exp(hidden);
          seen.push(hidden, out);
          return { out };
        },
      },
    });
    const graph: GraphJSON = {
      ...identity,
      nodes: [
        ...identity.nodes,
        { id: 'c', component: 'const', attrs: { shape: [2], value: 1 } },
        { id: 'd', component: 'capture' },
      ],
      edges: [...identity.edges, edge('c', 'd')],
    };
    const plan = compileGraph(graph, { registry });
    const x = tensor([1, 2], [2]);
    const outputs = plan.execute({ x });
    plan.releaseExecution(outputs, { x });
    seen.forEach((value) => expect(() => value.assertLive()).toThrow('disposed'));
    expect(() => x.assertLive()).not.toThrow();
    const again = plan.execute({ x });
    plan.releaseExecution(again);
    plan.dispose();
    x.dispose();
  });
  it('cleans up prior step results when execution fails', () => {
    let intermediate: ReturnType<typeof tensor>;
    const registry = createDefaultRegistry()
      .register({
        name: 'capture',
        inputs: ['x'],
        outputs: ['out'],
        primitive: {
          infer: (i) => ({ out: i.x }),
          execute: (i) => ({ out: (intermediate = exp(i.x)) }),
        },
      })
      .register({
        name: 'fail',
        inputs: ['x'],
        outputs: ['out'],
        primitive: {
          infer: (i) => ({ out: i.x }),
          execute: () => {
            throw new Error('failure');
          },
        },
      });
    const graph: GraphJSON = {
      ...identity,
      nodes: [...identity.nodes, { id: 'a', component: 'capture' }, { id: 'b', component: 'fail' }],
      edges: [...identity.edges, edge('input', 'a'), edge('a', 'b')],
    };
    const plan = compileGraph(graph, { registry });
    const x = tensor([1, 2], [2]);
    expect(() => plan.execute({ x })).toThrow('component_error');
    expect(() => intermediate.assertLive()).toThrow('disposed');
    expect(() => x.assertLive()).not.toThrow();
    plan.dispose();
    x.dispose();
  });
});

describe('standard graph native equivalence', () => {
  it.each([
    ['matmul', { a: [2, 3], b: [3, 4] }],
    ['matmul', { a: [2, 1, 2, 3], b: [1, 3, 3, 4] }],
    ['matmul', { a: [2, 3], b: [2, 3, 4] }],
    ['softmax', { x: [2, 3, 4] }],
    ['layernorm', { x: [2, 3, 4] }],
    ['layernorm', { x: [2, 1] }],
  ] as [string, Record<string, number[]>][])(
    '%s matches values and weighted gradients for %j',
    async (component, shapes) => {
      const graph: GraphJSON = {
        version: 1,
        nodes: [
          ...Object.keys(shapes).map((name) => ({ id: name, component: 'input', attrs: { name } })),
          { id: 'op', component, attrs: { eps: 0.03 } },
          { id: 'out', component: 'output', attrs: { name: 'out' } },
        ],
        edges: [...Object.keys(shapes).map((name) => edge(name, 'op', name)), edge('op', 'out')],
      };
      const inputs = Object.fromEntries(
        Object.entries(shapes).map(([name, shape]) => [
          name,
          tensor(
            Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => Math.sin(i + 1)),
            shape,
          ),
        ]),
      );
      const results: number[][][] = [];
      for (const preferNative of [true, false]) {
        const plan = compileGraph(graph, { inputShapes: shapes, preferNative });
        const outputs = plan.execute(inputs);
        const weights = tensor(
          Array.from({ length: outputs.out.shape.reduce((a, b) => a * b, 1) }, (_, i) =>
            Math.cos(i),
          ),
          outputs.out.shape,
        );
        const weighted = mul(outputs.out, weights);
        const loss = sum(
          weighted,
          outputs.out.shape.map((_, i) => i),
        );
        const gradients = await backward(loss, Object.values(inputs));
        results.push(
          await Promise.all(
            [outputs.out, ...gradients].map(async (x) => Array.from(await x.read())),
          ),
        );
        gradients.forEach((x) => x.dispose());
        loss.dispose();
        weighted.dispose();
        weights.dispose();
        plan.releaseExecution(outputs);
        plan.dispose();
      }
      results[0].forEach((values, i) =>
        values.forEach((value, j) => expect(results[1][i][j]).toBeCloseTo(value, 4)),
      );
      Object.values(inputs).forEach((x) => x.dispose());
    },
  );
});

describe('unsqueeze primitive', () => {
  const graph = (axis: unknown, shape: number[]): GraphJSON => ({
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x', shape } },
      { id: 'u', component: 'unsqueeze', attrs: { axis } },
      { id: 'o', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [edge('x', 'u'), edge('u', 'o')],
  });
  it.each([
    [0, [1, 2, 3]],
    [1, [2, 1, 3]],
    [2, [2, 3, 1]],
    [-3, [1, 2, 3]],
    [-2, [2, 1, 3]],
    [-1, [2, 3, 1]],
  ] as [number, number[]][])(
    'inserts axis %s and preserves values and gradients',
    async (axis, shape) => {
      const plan = compileGraph(graph(axis, [2, 3]));
      expect(plan.outputShapes.out).toEqual(shape);
      const x = tensor([1, 2, 3, 4, 5, 6], [2, 3]);
      const outputs = plan.execute({ x });
      expect(Array.from(await outputs.out.read())).toEqual([1, 2, 3, 4, 5, 6]);
      const weighted = mul(outputs.out, outputs.out);
      const loss = sum(weighted, [0, 1, 2]);
      const [gradient] = await backward(loss, [x]);
      expect(gradient.shape).toEqual([2, 3]);
      expect(Array.from(await gradient.read())).toEqual([2, 4, 6, 8, 10, 12]);
      gradient.dispose();
      loss.dispose();
      weighted.dispose();
      plan.releaseExecution(outputs);
      plan.dispose();
      x.dispose();
    },
  );
  it.each([0, -1])('supports scalar axis %s', (axis) => {
    const plan = compileGraph(graph(axis, []));
    expect(plan.outputShapes.out).toEqual([1]);
    plan.dispose();
  });
  it.each([undefined, null, '0', 0.5, NaN, Infinity, -4, 3])(
    'rejects invalid axis %s with node context',
    (axis) => {
      expect(safeCompileGraph(graph(axis, [2, 3]))).toMatchObject({
        ok: false,
        errors: [{ key: 'invalid_axes', node: 'u' }],
      });
    },
  );
  it('preserves zero-sized dimensions', () => {
    const plan = compileGraph(graph(-1, [0, 3]));
    expect(plan.outputShapes.out).toEqual([0, 3, 1]);
    plan.dispose();
  });
});

describe('shape-independent standard answers', () => {
  it.each(['matmul', 'softmax', 'layernorm'])(
    '%s can be serialized once and reused across shapes',
    (name) => {
      const definition = createDefaultRegistry().get(name)!;
      if (typeof definition.subgraph !== 'function') throw new Error('expected answer factory');
      const answer = definition.subgraph({});
      const shapes: Record<string, number[]> =
        name === 'matmul' ? { a: [2, 1, 3, 4], b: [1, 5, 4, 6] } : { x: [2, 3, 7] };
      expect(definition.subgraph({}, shapes)).toEqual(answer);
      const plan = compileGraph(JSON.parse(JSON.stringify(answer)) as GraphJSON, {
        inputShapes: shapes,
      });
      expect(plan.outputShapes.out).toEqual(name === 'matmul' ? [2, 5, 3, 6] : [2, 3, 7]);
      plan.dispose();
    },
  );
});

describe('transpose axis pairs', () => {
  const graph = (attrs: Record<string, unknown>, shape = [2, 3, 4]): GraphJSON => ({
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x', shape } },
      { id: 't', component: 'transpose', attrs },
      { id: 'o', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [edge('x', 't'), edge('t', 'o')],
  });
  it.each([
    [1, 2],
    [-2, -1],
    [1, -1],
  ])('swaps axes %j with perm priority and gradients', async (...axes) => {
    const plan = compileGraph(graph({ axes, perm: 'ignored' }));
    expect(plan.outputShapes.out).toEqual([2, 4, 3]);
    const data = Array.from({ length: 24 }, (_, i) => i + 1);
    const x = tensor(data, [2, 3, 4]);
    const outputs = plan.execute({ x });
    const expected = [0, 1].flatMap((b) =>
      [0, 1, 2, 3].flatMap((k) => [0, 1, 2].map((j) => data[b * 12 + j * 4 + k])),
    );
    expect(Array.from(await outputs.out.read())).toEqual(expected);
    const weighted = mul(outputs.out, outputs.out);
    const loss = sum(weighted, [0, 1, 2]);
    const [gradient] = await backward(loss, [x]);
    expect(gradient.shape).toEqual([2, 3, 4]);
    expect(Array.from(await gradient.read())).toEqual(data.map((v) => 2 * v));
    gradient.dispose();
    loss.dispose();
    weighted.dispose();
    plan.releaseExecution(outputs);
    plan.dispose();
    x.dispose();
  });
  it.each([
    undefined,
    null,
    '0,1',
    [],
    [0],
    [0, 1, 2],
    [0, 0],
    [0, -3],
    [0, 3],
    [-4, 0],
    [0, 0.5],
    [0, NaN],
    [0, Infinity],
    [0, '1'],
  ])('rejects invalid axes %j', (axes) => {
    expect(safeCompileGraph(graph({ axes, perm: [0, 1, 2] }))).toMatchObject({
      ok: false,
      errors: [{ key: 'invalid_axes', node: 't' }],
    });
  });
  it('preserves default reversal, legacy permutations, and zero-sized dimensions', () => {
    for (const [attrs, shape, expected] of [
      [{}, [2, 3, 4], [4, 3, 2]],
      [{ perm: [1, 2, 0] }, [2, 3, 4], [3, 4, 2]],
      [{ axes: [-2, -1] }, [2, 0, 4], [2, 4, 0]],
    ] as [Record<string, unknown>, number[], number[]][]) {
      const plan = compileGraph(graph(attrs, shape));
      expect(plan.outputShapes.out).toEqual(expected);
      plan.dispose();
    }
    for (const shape of [[], [3]])
      expect(safeCompileGraph(graph({ axes: [-2, -1] }, shape))).toMatchObject({
        ok: false,
        errors: [{ key: 'invalid_axes' }],
      });
  });
});
