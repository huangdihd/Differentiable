import { describe, expect, it } from 'vitest';
import { traceBackward } from './backward-graph';
import { graphMetrics, type ForwardLevel } from './levels';
import { configurableModule } from './parameterized-components';
import {
  ComponentRegistry,
  compileGraph,
  createDefaultRegistry,
  safeCompileGraph,
  type Attributes,
  type ComponentDefinition,
  type GraphJSON,
  type ShapePorts,
} from './graph';
import { Random } from './random';
import { tensor } from './tensor';
import { TrainingSession } from './training';
import { gameLevels } from '../levels/catalog';
import { referenceRegistry } from '../levels/level-helpers';
import { translateDiagnostic } from '../i18n';
import { createEditorState, editorReducer } from '../editor/state';
import { sigmoidLevel } from '../levels/examples';

const graph = (
  inputs: readonly string[],
  nodes: GraphJSON['nodes'],
  edges: GraphJSON['edges'],
): GraphJSON => ({
  version: 1,
  nodes: [
    ...inputs.map((name) => ({ id: name, component: 'input', attrs: { name } })),
    ...nodes,
    { id: 'output', component: 'output', attrs: { name: 'out' } },
  ],
  edges: [
    ...edges,
    { from: { node: nodes.at(-1)!.id, port: 'out' }, to: { node: 'output', port: 'x' } },
  ],
});
const passthrough = (name: string, inputs: readonly string[]): ComponentDefinition => ({
  name,
  inputs,
  outputs: ['out'],
  subgraph: {
    version: 1,
    nodes: [
      ...inputs.map((input) => ({ id: input, component: 'input', attrs: { name: input } })),
      { id: 'output', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [{ from: { node: inputs[0], port: 'out' }, to: { node: 'output', port: 'x' } }],
  },
});
const recurrentWindowDefinition = (
  name: string,
  cell: string,
  stateInput: string,
  initialShape: number[],
): ComponentDefinition => ({
  name,
  inputs: ['x', 'wx', 'wh', 'b'],
  outputs: ['out'],
  subgraph: {
    version: 1,
    nodes: [
      ...['x', 'wx', 'wh', 'b'].map((input) => ({
        id: input,
        component: 'input',
        attrs: { name: input },
      })),
      { id: 'initial_width', component: 'const', attrs: { shape: initialShape, value: 0 } },
      { id: 'step_0', component: cell },
      { id: 'step_1', component: cell },
      { id: 'output', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [
      { from: { node: 'x', port: 'out' }, to: { node: 'step_0', port: 'x' } },
      {
        from: { node: 'initial_width', port: 'out' },
        to: { node: 'step_0', port: stateInput },
      },
      { from: { node: 'step_0', port: 'out' }, to: { node: 'step_1', port: stateInput } },
      { from: { node: 'step_1', port: 'out' }, to: { node: 'output', port: 'x' } },
      ...['wx', 'wh', 'b'].flatMap((parameter) =>
        ['step_0', 'step_1'].map((step) => ({
          from: { node: parameter, port: 'out' },
          to: { node: step, port: parameter },
        })),
      ),
    ],
  },
});
const linearDefinition = (): ComponentDefinition => ({
  name: 'linear',
  inputs: ['x', 'w', 'b'],
  outputs: ['out'],
  subgraph: graph(
    ['x', 'w', 'b'],
    [
      { id: 'product', component: 'matmul' },
      { id: 'affine', component: 'add' },
    ],
    [
      { from: { node: 'x', port: 'out' }, to: { node: 'product', port: 'a' } },
      { from: { node: 'w', port: 'out' }, to: { node: 'product', port: 'b' } },
      { from: { node: 'product', port: 'out' }, to: { node: 'affine', port: 'a' } },
      { from: { node: 'b', port: 'out' }, to: { node: 'affine', port: 'b' } },
    ],
  ),
});
const registryWith = (definition: ComponentDefinition) => {
  const registry = new ComponentRegistry();
  for (const component of createDefaultRegistry().list()) registry.register(component);
  registry.register(configurableModule(definition));
  return registry;
};
const expand = (
  definition: ComponentDefinition,
  attrs: Attributes,
  inputs: ShapePorts,
): GraphJSON => (configurableModule(definition).subgraph as Function)(attrs, inputs) as GraphJSON;
const generated = (expanded: GraphJSON, id: string) =>
  expanded.nodes.find((node) => node.id === id);

describe('configurable parameterized components', () => {
  it('gives each Linear instance generated weight and bias with configurable feature sizes', () => {
    const registry = registryWith(linearDefinition());
    const linear = registry.get('linear')!;
    expect(linear.defaultAttributes).toEqual({
      automatic_parameters: true,
      in_features: 2,
      out_features: 2,
      bias: true,
    });
    expect(linear.generatedInputs).toEqual(['w', 'b']);
    const model: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'projection',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 3, out_features: 2, bias: true },
        },
        {
          id: 'other',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 3, out_features: 2, bias: true },
        },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'x', port: 'out' }, to: { node: 'other', port: 'x' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    expect(graphMetrics(model, registry).parameterCount).toBe(16);
    const plan = compileGraph(model, {
      registry,
      inputShapes: { x: [4, 3] },
      random: new Random(7),
    });
    const x = tensor(new Float32Array(12), [4, 3]);
    try {
      expect(plan.outputShapes.y).toEqual([4, 2]);
      const output = plan.execute({ x });
      try {
        expect(plan.parameters.get('projection/w')?.shape).toEqual([3, 2]);
        expect(plan.parameters.get('projection/b')?.shape).toEqual([2]);
        expect(plan.parameters.get('other/w')?.shape).toEqual([3, 2]);
        expect(plan.parameters.get('other/b')?.shape).toEqual([2]);
        expect(plan.parameters.get('projection/w')).not.toBe(plan.parameters.get('other/w'));
      } finally {
        plan.releaseExecution(output);
      }
    } finally {
      plan.dispose();
      x.dispose();
    }
  });

  it('treats rank-one input as a scalar batch for one-feature Linear', () => {
    const registry = registryWith(linearDefinition());
    const model = (outFeatures: number): GraphJSON => ({
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'projection',
          component: 'linear',
          attrs: {
            automatic_parameters: true,
            in_features: 1,
            out_features: outFeatures,
            bias: true,
          },
        },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    });
    const scalar = compileGraph(model(1), {
      registry,
      inputShapes: { x: [16] },
      random: new Random(8),
    });
    const vector = compileGraph(model(2), {
      registry,
      inputShapes: { x: [16] },
      random: new Random(8),
    });
    const x = tensor(new Float32Array(16), [16]);
    try {
      const scalarOutput = scalar.execute({ x });
      const vectorOutput = vector.execute({ x });
      try {
        expect(scalar.outputShapes.y).toEqual([16]);
        expect(scalar.parameters.get('projection/w')?.shape).toEqual([1, 1]);
        expect(scalar.parameters.get('projection/b')?.shape).toEqual([1]);
        expect(vector.outputShapes.y).toEqual([16, 2]);
      } finally {
        scalar.releaseExecution(scalarOutput);
        vector.releaseExecution(vectorOutput);
      }
    } finally {
      scalar.dispose();
      vector.dispose();
      x.dispose();
    }
  });

  it('maps an automatic Linear shape failure to its public x/w ports', () => {
    const registry = registryWith(linearDefinition());
    const model: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'projection',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 2, out_features: 2, bias: true },
        },
      ],
      edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } }],
    };
    const result = safeCompileGraph(model, {
      registry,
      inputShapes: { x: [4, 3] },
      random: new Random(8),
    });
    expect(result).toMatchObject({
      ok: false,
      errors: [
        {
          key: 'matmul_shape',
          node: 'projection/product',
          params: { a: [4, 3], b: [2, 2], input_names: { a: ['x'], b: ['w'] } },
        },
      ],
    });
    if (result.ok) throw new Error('expected automatic Linear to reject mismatched features');
    for (const locale of ['en', 'zh-CN'] as const) {
      const message = translateDiagnostic(locale, result.errors[0]);
      expect(message).toContain(locale === 'en' ? 'x is [4×3]' : 'x 是 [4×3]');
      expect(message).toContain(locale === 'en' ? 'w is [2×2]' : 'w 是 [2×2]');
      expect(message).not.toContain(locale === 'en' ? 'a is' : 'a 是');
      expect(message).not.toContain(locale === 'en' ? 'b is' : 'b 是');
    }
  });

  it('trains generated parameters and exposes their grouped module update', async () => {
    const registry = registryWith(linearDefinition());
    const model: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'projection',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 1, out_features: 1, bias: true },
        },
        { id: 'total', component: 'sum', attrs: { axes: [0, 1] } },
        { id: 'out', component: 'output', attrs: { name: 'loss' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'total', port: 'x' } },
        { from: { node: 'total', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    const plan = compileGraph(model, {
      registry,
      inputShapes: { x: [1, 1] },
      random: new Random(11),
    });
    const x = tensor([2], [1, 1]);
    const session = new TrainingSession(
      plan,
      { inputs: { x } },
      undefined,
      { kind: 'sgd', learningRate: 0.1 },
      'loss',
      { lossOutput: 'loss' },
    );
    try {
      const trace = await session.stepTrace();
      expect(trace.parameters.projection.before).toHaveLength(2);
      expect(trace.parameters.projection.after).toHaveLength(2);
      expect(trace.parameters.projection.after).not.toEqual(trace.parameters.projection.before);
      expect([...plan.parameters.keys()]).toEqual(['projection/w', 'projection/b']);
    } finally {
      session.dispose();
      plan.dispose();
      x.dispose();
    }
  });

  it('traces gradients for generated parameter paths using the live parameter state', async () => {
    const registry = registryWith(linearDefinition());
    const model: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'projection',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 2, out_features: 1, bias: true },
        },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    const plan = compileGraph(model, {
      registry,
      inputShapes: { x: [1, 2] },
      random: new Random(12),
    });
    const x = tensor([2, 3], [1, 2]);
    try {
      const output = plan.execute({ x });
      plan.releaseExecution(output);
      const weight = Array.from(await plan.parameters.get('projection/w')!.read());
      const trace = await traceBackward(model, {
        registry,
        inputShapes: { x: [1, 2] },
        output: 'y',
        wrt: ['projection/w', 'projection/b'],
        inputs: { x: [2, 3] },
        parameters: plan.parameters,
      });
      expect(trace.forward['projection/w'].values).toEqual(weight);
      expect(trace.gradient['projection/w'].values).toEqual([2, 3]);
      expect(trace.gradient['projection/b'].values).toEqual([1]);
    } finally {
      plan.dispose();
      x.dispose();
    }
  });

  it('omits the trainable Linear bias when bias is disabled', () => {
    const registry = registryWith(linearDefinition());
    const model: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'projection',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 3, out_features: 2, bias: false },
        },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    expect(graphMetrics(model, registry).parameterCount).toBe(6);
    const plan = compileGraph(model, {
      registry,
      inputShapes: { x: [1, 3] },
      random: new Random(8),
    });
    const x = tensor([1, 2, 3], [1, 3]);
    try {
      const output = plan.execute({ x });
      try {
        expect([...plan.parameters.keys()]).toEqual(['projection/w']);
      } finally {
        plan.releaseExecution(output);
      }
    } finally {
      plan.dispose();
      x.dispose();
    }
  });

  it('uses the same mechanism for Embedding and leaves explicit legacy parameters usable', () => {
    const definition: ComponentDefinition = {
      name: 'embedding',
      inputs: ['idx', 'table'],
      outputs: ['out'],
      subgraph: graph(
        ['idx', 'table'],
        [{ id: 'lookup', component: 'gather' }],
        [
          { from: { node: 'table', port: 'out' }, to: { node: 'lookup', port: 'table' } },
          { from: { node: 'idx', port: 'out' }, to: { node: 'lookup', port: 'idx' } },
        ],
      ),
    };
    const configurable = configurableModule(definition);
    expect(configurable.generatedInputs).toEqual(['table']);
    expect(configurable.defaultAttributes).toEqual({
      automatic_parameters: true,
      num_embeddings: 8,
      embedding_dim: 3,
    });
    const expanded = expand(
      definition,
      { automatic_parameters: true, num_embeddings: 11, embedding_dim: 5 },
      { idx: [4] },
    );
    expect(generated(expanded, 'table')).toMatchObject({
      component: 'param',
      attrs: { shape: [11, 5] },
    });
    const legacy = expand(
      definition,
      { automatic_parameters: false, num_embeddings: 8, embedding_dim: 3 },
      { idx: [4], table: [7, 2] },
    );
    expect(generated(legacy, 'table')).toMatchObject({
      component: 'input',
      attrs: { name: 'table' },
    });
  });

  it.each([
    {
      component: 'embedding',
      attrs: { num_embeddings: 7, embedding_dim: 4 },
      inputShape: [2],
      input: [0, 1],
      outputShape: [2, 4],
      parameters: { table: [7, 4] },
    },
    {
      component: 'embed_pos',
      attrs: { num_embeddings: 7, max_length: 3, embedding_dim: 4 },
      inputShape: [2, 3],
      input: [0, 1, 2, 3, 4, 5],
      outputShape: [2, 3, 4],
      parameters: { tokens: [7, 4], positions: [3, 4] },
    },
    {
      component: 'feed_forward',
      attrs: { features: 4, hidden_features: 6 },
      inputShape: [2, 3, 4],
      input: Array(24).fill(0.5),
      outputShape: [2, 3, 4],
      parameters: { w1: [4, 6], b1: [6], w2: [6, 4], b2: [4] },
    },
    {
      component: 'transformer_block',
      attrs: { features: 4, hidden_features: 6 },
      inputShape: [2, 3, 4],
      input: Array(24).fill(0.5),
      outputShape: [2, 3, 4],
      parameters: {
        wq: [4, 4],
        wk: [4, 4],
        wv: [4, 4],
        wo: [4, 4],
        w1: [4, 6],
        b1: [6],
        w2: [6, 4],
        b2: [4],
      },
    },
  ])('compiles the earned $component implementation with automatic parameters', (fixture) => {
    const registry = referenceRegistry(
      gameLevels.filter(
        (level): level is ForwardLevel => level.kind === 'forward' || level.kind === 'derivative',
      ),
    );
    const model: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        {
          id: 'module',
          component: fixture.component,
          attrs: { automatic_parameters: true, ...fixture.attrs },
        },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        {
          from: { node: 'x', port: 'out' },
          to: {
            node: 'module',
            port:
              fixture.component === 'embedding' || fixture.component === 'embed_pos' ? 'idx' : 'x',
          },
        },
        { from: { node: 'module', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    expect(graphMetrics(model, registry).parameterCount).toBe(
      (Object.values(fixture.parameters) as number[][]).reduce(
        (total, shape) => total + shape.reduce((size, dimension) => size * dimension, 1),
        0,
      ),
    );
    const plan = compileGraph(model, {
      registry,
      inputShapes: { x: fixture.inputShape },
      random: new Random(15),
    });
    const x = tensor(fixture.input, fixture.inputShape);
    try {
      expect(plan.outputShapes.y).toEqual(fixture.outputShape);
      const output = plan.execute({ x });
      try {
        for (const [name, shape] of Object.entries(fixture.parameters))
          expect(plan.parameters.get(`module/${name}`)?.shape).toEqual(shape);
      } finally {
        plan.releaseExecution(output);
      }
    } finally {
      plan.dispose();
      x.dispose();
    }
  });

  it.each([
    {
      component: 'rnn_cell',
      inputs: ['x', 'h', 'wx', 'wh', 'b'],
      attrs: { input_features: 3, hidden_features: 5 },
      parameters: { wx: [3, 5], wh: [5, 5], b: [5] },
    },
    {
      component: 'rnn_window',
      inputs: ['x', 'wx', 'wh', 'b'],
      attrs: { input_features: 1, hidden_features: 5 },
      parameters: { wx: [1, 5], wh: [5, 5], b: [5] },
    },
  ])('generates configured parameters for $component', (fixture) => {
    const expanded = expand(
      passthrough(fixture.component, fixture.inputs),
      { automatic_parameters: true, ...fixture.attrs },
      { x: [2, 3] },
    );
    for (const [name, shape] of Object.entries(fixture.parameters))
      expect(generated(expanded, name)?.attrs?.shape).toEqual(shape);
  });

  it.each(['rnn_window'])('keeps %s input width fixed to one scalar per time step', (component) => {
    const configurable = configurableModule(passthrough(component, ['x', 'wx', 'wh', 'b']));
    expect(
      configurable.attributes?.find((attribute) => attribute.name === 'input_features'),
    ).toMatchObject({
      min: 1,
      max: 1,
      default: 1,
    });
  });

  it.each([
    {
      component: 'rnn_window',
      cell: 'rnn_cell',
      stateInput: 'h',
      initialShape: [8],
      expectedInitialShape: [5],
    },
  ])(
    'shares $component parameters across unrolled cells and adapts its initial width',
    (fixture) => {
      const expanded = expand(
        recurrentWindowDefinition(
          fixture.component,
          fixture.cell,
          fixture.stateInput,
          fixture.initialShape,
        ),
        { automatic_parameters: true, input_features: 1, hidden_features: 5 },
        { x: [2, 6] },
      );

      expect(generated(expanded, 'initial_width')?.attrs).toEqual({
        shape: fixture.expectedInitialShape,
        value: 0,
      });
      expect(expanded.nodes.filter((node) => node.component === 'param')).toHaveLength(3);
      for (const parameter of ['wx', 'wh', 'b']) {
        expect(
          expanded.edges.filter((edge) => edge.from.node === parameter).map((edge) => edge.to.node),
        ).toEqual(['step_0', 'step_1']);
      }
    },
  );

  it('generates configured parameters for positional embeddings, FFNs, and Transformer blocks', () => {
    const positional = expand(
      passthrough('embed_pos', ['idx', 'tokens', 'positions']),
      {
        automatic_parameters: true,
        num_embeddings: 11,
        max_length: 7,
        embedding_dim: 5,
      },
      { idx: [2, 4] },
    );
    expect(generated(positional, 'tokens')).toMatchObject({
      component: 'param',
      attrs: { shape: [11, 5] },
    });
    expect(generated(positional, 'positions')).toMatchObject({
      component: 'param',
      attrs: { shape: [7, 5] },
    });

    const feedForward = expand(
      passthrough('feed_forward', ['x', 'w1', 'b1', 'w2', 'b2']),
      { automatic_parameters: true, features: 6, hidden_features: 10 },
      { x: [2, 4, 6] },
    );
    expect(generated(feedForward, 'w1')?.attrs?.shape).toEqual([6, 10]);
    expect(generated(feedForward, 'b1')?.attrs?.shape).toEqual([10]);
    expect(generated(feedForward, 'w2')?.attrs?.shape).toEqual([10, 6]);
    expect(generated(feedForward, 'b2')?.attrs?.shape).toEqual([6]);

    const transformer = expand(
      passthrough('transformer_block', ['x', 'wq', 'wk', 'wv', 'wo', 'w1', 'b1', 'w2', 'b2']),
      { automatic_parameters: true, features: 8, hidden_features: 12 },
      { x: [2, 4, 8] },
    );
    for (const name of ['wq', 'wk', 'wv', 'wo'])
      expect(generated(transformer, name)?.attrs?.shape).toEqual([8, 8]);
    expect(generated(transformer, 'w1')?.attrs?.shape).toEqual([8, 12]);
    expect(generated(transformer, 'b1')?.attrs?.shape).toEqual([12]);
    expect(generated(transformer, 'w2')?.attrs?.shape).toEqual([12, 8]);
    expect(generated(transformer, 'b2')?.attrs?.shape).toEqual([8]);
  });

  it('uses compiler defaults consistently and rejects explicit generated-input overrides', () => {
    const registry = registryWith(linearDefinition());
    const automatic: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 'projection', component: 'linear', attrs: { automatic_parameters: true } },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    expect(graphMetrics(automatic, registry).parameterCount).toBe(6);
    const plan = compileGraph(automatic, {
      registry,
      inputShapes: { x: [1, 2] },
      random: new Random(13),
    });
    plan.dispose();

    const overridden = structuredClone(automatic);
    overridden.nodes.splice(1, 0, {
      id: 'weight',
      component: 'param',
      attrs: { shape: [2, 2] },
    });
    overridden.edges.push({
      from: { node: 'weight', port: 'out' },
      to: { node: 'projection', port: 'w' },
    });
    expect(() => graphMetrics(overridden, registry)).toThrow('generated_input_connected');
    expect(() =>
      compileGraph(overridden, {
        registry,
        inputShapes: { x: [1, 2] },
        random: new Random(13),
      }),
    ).toThrow('generated_input_connected at projection:w');
  });

  it('rejects malformed and unsupported module dimensions at the owning node', () => {
    const registry = registryWith(linearDefinition());
    const compile = (inFeatures: unknown) =>
      compileGraph(
        {
          version: 1,
          nodes: [
            { id: 'x', component: 'input', attrs: { name: 'x' } },
            {
              id: 'projection',
              component: 'linear',
              attrs: {
                automatic_parameters: true,
                in_features: inFeatures,
                out_features: 2,
                bias: true,
              },
            },
            { id: 'out', component: 'output', attrs: { name: 'y' } },
          ],
          edges: [
            { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
            { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
          ],
        },
        { registry, inputShapes: { x: [1, 2] }, random: new Random(14) },
      );
    expect(() => compile(null)).toThrow('invalid_attribute at projection');
    expect(() => compile(1025)).toThrow('invalid_attribute at projection');
  });

  it('rejects odd Transformer widths before expanding fixed two-head attention', () => {
    const definition = passthrough('transformer_block', [
      'x',
      'wq',
      'wk',
      'wv',
      'wo',
      'w1',
      'b1',
      'w2',
      'b2',
    ]);
    const registry = registryWith(definition);
    expect(() =>
      compileGraph(
        {
          version: 1,
          nodes: [
            { id: 'x', component: 'input', attrs: { name: 'x' } },
            {
              id: 'block',
              component: 'transformer_block',
              attrs: { automatic_parameters: true, features: 7, hidden_features: 16 },
            },
            { id: 'out', component: 'output', attrs: { name: 'y' } },
          ],
          edges: [
            { from: { node: 'x', port: 'out' }, to: { node: 'block', port: 'x' } },
            { from: { node: 'block', port: 'out' }, to: { node: 'out', port: 'x' } },
          ],
        },
        { registry, inputShapes: { x: [2, 4, 7] }, random: new Random(9) },
      ),
    ).toThrow('invalid_attribute at block');
  });
});

describe('automatic parameters toggle', () => {
  it('is a visible boolean on every parameterized module, missing still meaning off', () => {
    const registry = referenceRegistry(
      gameLevels.filter(
        (level): level is ForwardLevel => level.kind === 'forward' || level.kind === 'derivative',
      ),
    );
    const defs = registry
      .list()
      .filter((def) => def.attributes?.some((a) => a.name === 'automatic_parameters'));
    expect(defs.length).toBeGreaterThanOrEqual(4);
    for (const def of defs) {
      const attribute = def.attributes!.find((a) => a.name === 'automatic_parameters')!;
      expect(attribute.hidden).not.toBe(true);
      expect(attribute.default).toBe(false);
    }
  });
  it('drops wires into weight ports that disappear when turned on, and keeps them when off', () => {
    const graph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 'weight', component: 'param', attrs: { shape: [2, 2] } },
        {
          id: 'projection',
          component: 'linear',
          attrs: { automatic_parameters: false, in_features: 2, out_features: 2, bias: false },
        },
        { id: 'out', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'projection', port: 'x' } },
        { from: { node: 'weight', port: 'out' }, to: { node: 'projection', port: 'w' } },
        { from: { node: 'projection', port: 'out' }, to: { node: 'out', port: 'x' } },
      ],
    };
    const registry = registryWith(linearDefinition());
    const state = createEditorState(sigmoidLevel, graph);
    const on = editorReducer(
      state,
      { type: 'setAttrs', id: 'projection', attrs: { automatic_parameters: true } },
      registry,
    );
    expect(on.error).toBeFalsy();
    expect(on.graph.edges.some((e) => e.to.port === 'w')).toBe(false);
    expect(on.graph.edges.some((e) => e.to.port === 'x' && e.to.node === 'projection')).toBe(true);
    const off = editorReducer(
      on,
      { type: 'setAttrs', id: 'projection', attrs: { automatic_parameters: false } },
      registry,
    );
    expect(off.graph.edges).toHaveLength(2);
  });
});
