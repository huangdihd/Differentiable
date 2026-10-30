import {
  GraphError,
  type Attributes,
  type ComponentDefinition,
  type GraphJSON,
  type ShapePorts,
} from './graph';
import { infer } from './backend';
import * as T from './tensor';
import { tensorOpComponents } from './tensor-ops';

/** Canonical primitive-only answers with shape-independent topology and attributes. */
function answer(name: string, attrs: Attributes, inputs?: ShapePorts): GraphJSON {
  const graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  const node = (
    id: string,
    component: string,
    args: Record<string, string> = {},
    attributes?: Attributes,
  ) => {
    graph.nodes.push({ id, component, attrs: attributes });
    for (const [port, source] of Object.entries(args))
      graph.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  };
  const constant = (id: string, value: number) => node(id, 'const', {}, { shape: [], value });
  const unary = (id: string, op: string, x: string, a?: Attributes) => node(id, op, { x }, a);
  const binary = (id: string, op: string, a: string, b: string) => node(id, op, { a, b });
  let result: string;
  if (name === 'matmul') {
    const a = inputs?.a ?? [1, 1],
      b = inputs?.b ?? [1, 1];
    if (a.length < 2 || b.length < 2 || a.at(-1) !== b.at(-2))
      throw new GraphError('matmul_shape', undefined, undefined, { a, b });
    infer('matmul', [a, b]);
    node('a', 'input', {}, { name: 'a' });
    node('b', 'input', {}, { name: 'b' });
    const left = unary('left', 'unsqueeze', 'a', { axis: -1 });
    const right = unary('right', 'unsqueeze', 'b', { axis: -3 });
    result = unary('result', 'sum', binary('products', 'mul', left, right), { axes: [-2] });
  } else {
    const shape = inputs?.x ?? [1];
    infer(name as 'softmax' | 'layernorm', [shape], attrs);
    node('x', 'input', {}, { name: 'x' });
    const minusOne = constant('minus_one', -1);
    const negate = (id: string, x: string) => binary(id, 'mul', x, minusOne);
    const reduce = (id: string, op: string, x: string) =>
      unary(id, op, x, { axes: [-1], keepdims: true });
    if (name === 'softmax') {
      const maximum = reduce('maximum', 'reduce_max', 'x');
      const shifted = binary('shifted', 'add', 'x', negate('negative_maximum', maximum));
      const weights = unary('weights', 'exp', shifted);
      const total = reduce('total', 'sum', weights);
      const inverse = unary(
        'inverse',
        'exp',
        negate('negative_log_total', unary('log_total', 'log', total)),
      );
      result = binary('result', 'mul', weights, inverse);
    } else {
      const zeros = binary('zeros', 'mul', 'x', constant('zero', 0));
      const ones = binary('ones', 'add', zeros, constant('one', 1));
      const count = reduce('count', 'sum', ones);
      const scale = unary(
        'scale',
        'exp',
        negate('negative_log_count', unary('log_count', 'log', count)),
      );
      const mean = binary('mean', 'mul', reduce('total', 'sum', 'x'), scale);
      const centered = binary('centered', 'add', 'x', negate('negative_mean', mean));
      const squares = binary('squares', 'mul', centered, centered);
      const variance = binary('variance', 'mul', reduce('square_total', 'sum', squares), scale);
      const stabilized = binary(
        'stabilized',
        'add',
        variance,
        constant('epsilon', (attrs.eps as number | undefined) ?? 1e-5),
      );
      const exponent = binary(
        'exponent',
        'mul',
        unary('log_variance', 'log', stabilized),
        constant('negative_half', -0.5),
      );
      result = binary('result', 'mul', centered, unary('inverse_std', 'exp', exponent));
    }
  }
  node('output', 'output', { x: result }, { name: 'out' });
  return graph;
}

class RecurrentGraphBuilder {
  readonly graph: GraphJSON = { version: 1, nodes: [], edges: [] };

  node(
    id: string,
    component: string,
    inputs: Record<string, string> = {},
    attrs?: Attributes,
  ): string {
    this.graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, source] of Object.entries(inputs))
      this.graph.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  }

  input(name: string): string {
    return this.node(name, 'input', {}, { name });
  }

  constant(id: string, value: number | number[], shape: number[] = []): string {
    return this.node(id, 'const', {}, { value, shape });
  }

  unary(id: string, component: string, x: string, attrs?: Attributes): string {
    return this.node(id, component, { x }, attrs);
  }

  binary(id: string, component: string, a: string, b: string): string {
    return this.node(id, component, { a, b });
  }

  output(value: string): void {
    this.node('output', 'output', { x: value }, { name: 'out' });
  }
}

const sameShape = (actual: readonly number[], expected: readonly number[]) =>
  actual.length === expected.length &&
  actual.every((dimension, index) => dimension === expected[index]);

function rnnWindowShape(
  inputs?: ShapePorts,
  attrs: Attributes = {},
): { batch: number; steps: number; width: number } {
  if (!inputs)
    return {
      batch: 1,
      steps: 1,
      width: typeof attrs.hidden_features === 'number' ? attrs.hidden_features : 1,
    };
  const x = inputs.x;
  if (!x || x.length !== 2)
    throw new GraphError('input_shape_mismatch', undefined, 'x', {
      actual: x,
      expected: ['B', 'T'],
    });
  const wh = inputs.wh;
  if (wh && wh.length !== 2)
    throw new GraphError('input_shape_mismatch', undefined, 'wh', {
      actual: wh,
      expected: ['H', 'H'],
    });
  const width = wh?.[0] ?? attrs.hidden_features;
  if (typeof width !== 'number')
    throw new GraphError('input_shape_mismatch', undefined, 'wh', {
      actual: wh,
      expected: ['H', 'H'],
    });
  const expected: ShapePorts = { wx: [1, width], wh: [width, width], b: [width] };
  for (const port of ['wx', 'wh', 'b'] as const) {
    const actual = inputs[port];
    if (actual && !sameShape(actual, expected[port]))
      throw new GraphError('input_shape_mismatch', undefined, port, {
        actual,
        expected: expected[port],
      });
  }
  return { batch: x[0], steps: x[1], width };
}

function primitiveMatmul(builder: RecurrentGraphBuilder, id: string, a: string, b: string): string {
  const left = builder.unary(`${id}_left`, 'unsqueeze', a, { axis: -1 });
  const right = builder.unary(`${id}_right`, 'unsqueeze', b, { axis: -3 });
  const products = builder.binary(`${id}_products`, 'mul', left, right);
  return builder.unary(id, 'sum', products, { axes: [-2] });
}

function sigmoid(builder: RecurrentGraphBuilder, id: string, x: string): string {
  const negative = builder.binary(
    `${id}_negative`,
    'mul',
    x,
    builder.constant(`${id}_minus_one`, -1),
  );
  const denominator = builder.binary(
    `${id}_denominator`,
    'add',
    builder.constant(`${id}_one`, 1),
    builder.unary(`${id}_exponential`, 'exp', negative),
  );
  return builder.unary(id, 'recip', denominator);
}

function tanh(builder: RecurrentGraphBuilder, id: string, x: string): string {
  const doubled = builder.binary(`${id}_doubled`, 'mul', x, builder.constant(`${id}_two`, 2));
  const activated = sigmoid(builder, `${id}_sigmoid`, doubled);
  const scaled = builder.binary(
    `${id}_scaled`,
    'mul',
    activated,
    builder.constant(`${id}_scale`, 2),
  );
  return builder.binary(id, 'add', scaled, builder.constant(`${id}_offset`, -1));
}

function rnnWindow(inputs?: ShapePorts, attrs: Attributes = {}): GraphJSON {
  const { batch, steps, width } = rnnWindowShape(inputs, attrs);
  const builder = new RecurrentGraphBuilder();
  const x = builder.input('x');
  const wx = builder.input('wx');
  const wh = builder.input('wh');
  const bias = builder.input('b');
  let hidden = builder.constant('initial_hidden', 0, [batch, width]);
  let history = builder.constant('history', 0, [batch, steps, width]);

  for (let step = 0; step < steps; step++) {
    const prefix = `step_${step}`;
    const selector = Array.from({ length: steps }, (_, index) => (index === step ? 1 : 0));
    const selected = builder.binary(
      `${prefix}_selected`,
      'mul',
      x,
      builder.constant(`${prefix}_selector`, selector, [steps]),
    );
    const value = builder.unary(
      `${prefix}_value`,
      'unsqueeze',
      builder.unary(`${prefix}_sum`, 'sum', selected, { axes: [-1] }),
      { axis: -1 },
    );
    const inputProjection = primitiveMatmul(builder, `${prefix}_x_wx`, value, wx);
    const inputAffine = builder.binary(`${prefix}_input_projection`, 'add', inputProjection, bias);
    const recurrentProjection = primitiveMatmul(builder, `${prefix}_h_wh`, hidden, wh);
    const affine = builder.binary(`${prefix}_affine`, 'add', inputAffine, recurrentProjection);
    hidden = tanh(builder, `${prefix}_hidden`, affine);
    const row = builder.unary(`${prefix}_row`, 'unsqueeze', hidden, { axis: 1 });
    const rowMask = Array.from({ length: steps }, (_, index) => (index === step ? 1 : 0));
    const placed = builder.binary(
      `${prefix}_placed`,
      'mul',
      row,
      builder.constant(`${prefix}_row_mask`, rowMask, [steps, 1]),
    );
    history = builder.binary(`${prefix}_history`, 'add', history, placed);
  }

  builder.output(history);
  return builder.graph;
}

const algebraComponents: ComponentDefinition[] = (['matmul', 'softmax', 'layernorm'] as const).map(
  (name) => ({
    name,
    inputs: name === 'matmul' ? ['a', 'b'] : ['x'],
    outputs: ['out'],
    subgraph: (attrs, inputs) => answer(name, attrs, inputs),
    native: {
      infer: (inputs, attrs) => {
        if (
          name === 'matmul' &&
          (inputs.a.length < 2 || inputs.b.length < 2 || inputs.a.at(-1) !== inputs.b.at(-2))
        )
          throw new GraphError('matmul_shape', undefined, undefined, { a: inputs.a, b: inputs.b });
        return { out: infer(name, name === 'matmul' ? [inputs.a, inputs.b] : [inputs.x], attrs) };
      },
      execute: (inputs, attrs) => ({
        out:
          name === 'matmul'
            ? T.matmul(inputs.a, inputs.b)
            : name === 'softmax'
              ? T.softmax(inputs.x)
              : T.layernorm(inputs.x, (attrs.eps as number | undefined) ?? 1e-5),
      }),
    },
  }),
);

const rnnWindowComponent: ComponentDefinition = {
  name: 'rnn_window',
  inputs: ['x', 'wx', 'wh', 'b'],
  outputs: ['out'],
  subgraph: (attrs, inputs) => rnnWindow(inputs, attrs),
};

export const standardComponents: ComponentDefinition[] = [
  ...algebraComponents,
  rnnWindowComponent,
  ...tensorOpComponents,
];
