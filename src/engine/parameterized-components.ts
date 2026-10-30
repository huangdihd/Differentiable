import {
  COMPONENT_INPUT_ATTRIBUTE,
  GraphError,
  type Attributes,
  type ComponentAttributeDefinition,
  type ComponentDefinition,
  type GraphJSON,
  type ShapePorts,
} from './graph';

type ParameterSpec = {
  input: string;
  shape: (attrs: Attributes) => number[];
  init?: 'zeros' | 'ones' | 'normal' | 'uniform';
  scale?: number;
  enabled?: (attrs: Attributes) => boolean;
};
type ModuleSpec = {
  attributes: readonly ComponentAttributeDefinition[];
  defaults: Attributes;
  parameters: readonly ParameterSpec[];
  optionalInputs?: readonly string[];
  adapt?: (graph: GraphJSON, attrs: Attributes, inputs?: ShapePorts) => void;
};

const integer = (
  name: string,
  defaultValue: number,
  constraints: Pick<ComponentAttributeDefinition, 'min' | 'max' | 'multipleOf'> = {},
): ComponentAttributeDefinition => ({
  name,
  kind: 'integer',
  default: defaultValue,
  min: 1,
  max: 1024,
  ...constraints,
});
const automatic: ComponentAttributeDefinition = {
  name: 'automatic_parameters',
  kind: 'boolean',
  // Missing means a legacy component whose explicit parameter wires remain authoritative.
  default: false,
};
const normal = (input: string, shape: ParameterSpec['shape']): ParameterSpec => ({
  input,
  shape,
  init: 'normal',
  scale: 0.15,
});
const zeros = (input: string, shape: ParameterSpec['shape']): ParameterSpec => ({
  input,
  shape,
  init: 'zeros',
});
const number = (attrs: Attributes, name: string) => attrs[name] as number;
const freshId = (graph: GraphJSON, base: string) => {
  const ids = new Set(graph.nodes.map((node) => node.id));
  let id = base;
  for (let suffix = 2; ids.has(id); suffix++) id = `${base}_${suffix}`;
  return id;
};

/** Keep the expanded window's zero state aligned with its configured hidden width. */
function adaptRecurrentWindow(graph: GraphJSON, attrs: Attributes): void {
  const initialWidth = graph.nodes.find((node) => node.id === 'initial_width');
  if (!initialWidth) return;
  const shape = initialWidth.attrs?.shape;
  initialWidth.attrs = {
    ...initialWidth.attrs,
    shape: Array.isArray(shape)
      ? [...shape.slice(0, -1), number(attrs, 'hidden_features')]
      : [number(attrs, 'hidden_features')],
  };
}

/** A start state without the batch axis would surface as a confusing matmul error deep inside. */
function requireBatchedState(graph: GraphJSON, attrs: Attributes, inputs?: ShapePorts): void {
  const h = inputs?.h;
  const x = inputs?.x;
  const width = inputs?.wh?.[0] ?? number(attrs, 'hidden_features');
  if (x && !h) {
    // An unconnected h starts the sequence from zeros, one state per sentence: [B, H].
    const shape = [...x.slice(0, -1), width];
    graph.nodes = graph.nodes.map((node) =>
      node.component === 'input' && node.attrs?.name === 'h'
        ? { id: node.id, component: 'const', attrs: { value: 0, shape } }
        : node,
    );
    return;
  }
  if (!h || !x || h.length >= Math.max(2, x.length)) return;
  throw new GraphError('rnn_state_batch', undefined, 'h', { h, hidden: width });
}

/**
 * Position vectors are looked up by position 0…T−1, so the table needs at least T rows. Catch a
 * too-small 「最大序列长度」 while shapes are inferred, on the node the player placed, instead of
 * as an out-of-range lookup deep inside it at run time.
 */
function requireSequenceFits(_graph: GraphJSON, attrs: Attributes, inputs?: ShapePorts): void {
  const length = inputs?.idx?.at(-1);
  const rows =
    inputs?.positions?.[0] ??
    (attrs.automatic_parameters === true ? number(attrs, 'max_length') : undefined);
  if (length === undefined || rows === undefined || length <= rows) return;
  throw new GraphError('embed_pos_length', undefined, 'idx', { length, max_length: rows });
}

/** Treat rank-one x as a batch of scalars when Linear declares one input feature. */
function adaptScalarBatchLinear(graph: GraphJSON, attrs: Attributes, inputs?: ShapePorts): void {
  if (attrs.automatic_parameters !== true || attrs.in_features !== 1 || inputs?.x?.length !== 1)
    return;
  const input = graph.nodes.find((node) => node.component === 'input' && node.attrs?.name === 'x');
  if (!input) return;
  const expanded = freshId(graph, '__scalar_batch_input');
  for (const edge of graph.edges) if (edge.from.node === input.id) edge.from.node = expanded;
  graph.nodes.push({ id: expanded, component: 'unsqueeze', attrs: { axis: -1 } });
  graph.edges.push({
    from: { node: input.id, port: 'out' },
    to: { node: expanded, port: 'x' },
  });
  if (attrs.out_features !== 1) return;
  for (const output of graph.nodes.filter((node) => node.component === 'output')) {
    const incoming = graph.edges.find((edge) => edge.to.node === output.id && edge.to.port === 'x');
    if (!incoming) continue;
    const source = { ...incoming.from };
    const flattened = freshId(graph, `__scalar_batch_${output.id}`);
    graph.nodes.push({ id: flattened, component: 'reshape', attrs: { shape: [-1] } });
    incoming.from = { node: flattened, port: 'out' };
    graph.edges.push({ from: source, to: { node: flattened, port: 'x' } });
  }
}

/** Parameterized learned components share one declarative nn.Module-style contract. */
const MODULES: Readonly<Record<string, ModuleSpec>> = {
  linear: {
    attributes: [
      automatic,
      integer('in_features', 2),
      integer('out_features', 2),
      {
        name: 'bias',
        kind: 'boolean',
        default: true,
      },
    ],
    defaults: { automatic_parameters: true, in_features: 2, out_features: 2, bias: true },
    parameters: [
      normal('w', (a) => [number(a, 'in_features'), number(a, 'out_features')]),
      {
        ...zeros('b', (a) => [number(a, 'out_features')]),
        enabled: (a) => Boolean(a.bias),
      },
    ],
    adapt: adaptScalarBatchLinear,
  },
  embedding: {
    attributes: [automatic, integer('num_embeddings', 8), integer('embedding_dim', 3)],
    defaults: { automatic_parameters: true, num_embeddings: 8, embedding_dim: 3 },
    parameters: [normal('table', (a) => [number(a, 'num_embeddings'), number(a, 'embedding_dim')])],
  },
  embed_pos: {
    attributes: [
      automatic,
      integer('num_embeddings', 8),
      integer('max_length', 8),
      integer('embedding_dim', 3),
    ],
    defaults: {
      automatic_parameters: true,
      num_embeddings: 8,
      max_length: 8,
      embedding_dim: 3,
    },
    parameters: [
      normal('tokens', (a) => [number(a, 'num_embeddings'), number(a, 'embedding_dim')]),
      normal('positions', (a) => [number(a, 'max_length'), number(a, 'embedding_dim')]),
    ],
    adapt: requireSequenceFits,
  },
  rnn_cell: {
    attributes: [automatic, integer('input_features', 1), integer('hidden_features', 8)],
    defaults: { automatic_parameters: true, input_features: 1, hidden_features: 8 },
    parameters: [
      normal('wx', (a) => [number(a, 'input_features'), number(a, 'hidden_features')]),
      normal('wh', (a) => [number(a, 'hidden_features'), number(a, 'hidden_features')]),
      zeros('b', (a) => [number(a, 'hidden_features')]),
    ],
    optionalInputs: ['h'],
    adapt: requireBatchedState,
  },
  rnn_window: {
    attributes: [
      automatic,
      integer('input_features', 1, { max: 1 }),
      integer('hidden_features', 8),
    ],
    defaults: { automatic_parameters: true, input_features: 1, hidden_features: 8 },
    parameters: [
      normal('wx', (a) => [number(a, 'input_features'), number(a, 'hidden_features')]),
      normal('wh', (a) => [number(a, 'hidden_features'), number(a, 'hidden_features')]),
      zeros('b', (a) => [number(a, 'hidden_features')]),
    ],
    adapt: adaptRecurrentWindow,
  },
  feed_forward: {
    attributes: [automatic, integer('features', 8), integer('hidden_features', 16)],
    defaults: { automatic_parameters: true, features: 8, hidden_features: 16 },
    parameters: [
      normal('w1', (a) => [number(a, 'features'), number(a, 'hidden_features')]),
      zeros('b1', (a) => [number(a, 'hidden_features')]),
      normal('w2', (a) => [number(a, 'hidden_features'), number(a, 'features')]),
      zeros('b2', (a) => [number(a, 'features')]),
    ],
  },
  transformer_block: {
    attributes: [
      automatic,
      integer('features', 8, { min: 1, multipleOf: 2 }),
      integer('hidden_features', 16),
    ],
    defaults: { automatic_parameters: true, features: 8, hidden_features: 16 },
    parameters: [
      ...['wq', 'wk', 'wv', 'wo'].map((input) =>
        normal(input, (a) => [number(a, 'features'), number(a, 'features')]),
      ),
      normal('w1', (a) => [number(a, 'features'), number(a, 'hidden_features')]),
      zeros('b1', (a) => [number(a, 'hidden_features')]),
      normal('w2', (a) => [number(a, 'hidden_features'), number(a, 'features')]),
      zeros('b2', (a) => [number(a, 'features')]),
    ],
  },
};

function parameterize(
  graph: GraphJSON,
  attrs: Attributes,
  parameters: readonly ParameterSpec[],
): GraphJSON {
  const copy = structuredClone(graph);
  if (!attrs.automatic_parameters) return copy;
  for (const parameter of parameters) {
    const input = copy.nodes.find(
      (node) => node.component === 'input' && node.attrs?.name === parameter.input,
    );
    if (!input) continue;
    const enabled = parameter.enabled?.(attrs) ?? true;
    input.component = enabled ? 'param' : 'const';
    input.attrs = enabled
      ? {
          shape: parameter.shape(attrs),
          init: parameter.init ?? 'normal',
          ...(parameter.scale === undefined ? {} : { scale: parameter.scale }),
          [COMPONENT_INPUT_ATTRIBUTE]: parameter.input,
        }
      : {
          shape: parameter.shape(attrs),
          value: 0,
          [COMPONENT_INPUT_ATTRIBUTE]: parameter.input,
        };
  }
  return copy;
}

/**
 * Adds automatic, instance-scoped parameters to a learned player component without changing its
 * saved implementation. Legacy graphs keep explicit parameter wires with automatic mode off;
 * fresh nodes use automatic parameters and expose only their data inputs.
 */
export function configurableModule(definition: ComponentDefinition): ComponentDefinition {
  const spec = MODULES[definition.name];
  if (!spec?.parameters.every((parameter) => definition.inputs.includes(parameter.input)))
    return definition;
  const source = definition.subgraph;
  if (!source) return definition;
  const moduleAttributes = new Set(spec.attributes.map((attribute) => attribute.name));
  return {
    ...definition,
    optionalInputs: [
      ...new Set([
        ...(definition.optionalInputs ?? []),
        ...(spec.optionalInputs ?? []),
        ...spec.parameters.map((p) => p.input),
      ]),
    ],
    generatedInputs: [
      ...new Set([
        ...(definition.generatedInputs ?? []),
        ...spec.parameters.map((parameter) => parameter.input),
      ]),
    ],
    attributes: [
      ...(definition.attributes ?? []).filter((attribute) => !moduleAttributes.has(attribute.name)),
      ...spec.attributes,
    ],
    defaultAttributes: { ...definition.defaultAttributes, ...spec.defaults },
    subgraph: (attrs, inputs) => {
      const graph = parameterize(
        typeof source === 'function' ? source(attrs, inputs) : source,
        attrs,
        spec.parameters,
      );
      spec.adapt?.(graph, attrs, inputs);
      return graph;
    },
  };
}
