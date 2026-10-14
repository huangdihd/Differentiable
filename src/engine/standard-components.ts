import {
  GraphError,
  type Attributes,
  type ComponentDefinition,
  type GraphJSON,
  type ShapePorts,
} from './graph';
import { infer } from './backend';
import * as T from './tensor';

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

export const standardComponents: ComponentDefinition[] = (
  ['matmul', 'softmax', 'layernorm'] as const
).map((name) => ({
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
}));
