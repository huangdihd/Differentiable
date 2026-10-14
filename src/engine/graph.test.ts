import { describe, expect, it } from 'vitest';
import {
  compileGraph,
  safeCompileGraph,
  createDefaultRegistry,
  GraphError,
  type GraphJSON,
  type GraphEdge,
} from './graph';
import { tensor, exp } from './tensor';

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
