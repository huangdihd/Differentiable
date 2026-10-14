import type { Attributes, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { scalarPrimitives, tensorCases, withPar } from './level-helpers';

/** Answers use rank-relative axes, so saved components remain reusable at new widths. */
function answer(kind: string): GraphJSON {
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
  const unary = (id: string, op: string, x: string, attrs?: Attributes) =>
    node(id, op, { x }, attrs);
  const binary = (id: string, op: string, a: string, b: string) => node(id, op, { a, b });
  const constant = (id: string, value: number) => node(id, 'const', {}, { value });
  let result: string;
  if (['dot', 'outer', 'matmul'].includes(kind)) {
    node('a', 'input', {}, { name: 'a' });
    node('b', 'input', {}, { name: 'b' });
    if (kind === 'dot')
      result = unary('result', 'sum', binary('products', 'mul', 'a', 'b'), { axes: [-1] });
    else if (kind === 'outer')
      result = binary(
        'result',
        'mul',
        unary('left', 'unsqueeze', 'a', { axis: -1 }),
        unary('right', 'unsqueeze', 'b', { axis: 0 }),
      );
    else {
      // [.., M, K, 1] * [.., 1, K, N], then contract K. No fixed reshape or transpose.
      const left = unary('left', 'unsqueeze', 'a', { axis: -1 });
      const right = unary('right', 'unsqueeze', 'b', { axis: -3 });
      result = unary('result', 'sum', binary('products', 'mul', left, right), { axes: [-2] });
    }
  } else {
    node('x', 'input', {}, { name: 'x' });
    const reduce = (id: string, op: string, x: string) =>
      unary(id, op, x, { axes: [-1], keepdims: true });
    if (kind === 'mean') {
      const ones = binary(
        'ones',
        'add',
        binary('zeros', 'mul', 'x', constant('zero', 0)),
        constant('one', 1),
      );
      result = binary('result', 'div', reduce('total', 'sum', 'x'), reduce('count', 'sum', ones));
    } else if (kind === 'softmax') {
      const shifted = binary('shifted', 'sub', 'x', reduce('maximum', 'reduce_max', 'x'));
      const weights = unary('weights', 'exp', shifted);
      result = binary('result', 'div', weights, reduce('total', 'sum', weights));
    } else {
      const centered = binary('centered', 'sub', 'x', unary('mean', 'mean', 'x'));
      const variance = unary('variance', 'mean', binary('squares', 'mul', centered, centered));
      const stabilized = binary('stabilized', 'add', variance, constant('epsilon', 1e-5));
      result = binary('result', 'div', centered, unary('std', 'sqrt', stabilized));
    }
  }
  node('output', 'output', { x: result }, { name: 'y' });
  return graph;
}

const definitions: {
  name: string;
  shapes: ShapePorts[];
  output: number[];
  reference: ForwardLevel['reference'];
}[] = [
  {
    name: 'dot',
    shapes: [
      { a: [6], b: [6] },
      { a: [3], b: [3] },
    ],
    output: [],
    reference: ({ a, b }) => T.sum(T.mul(a, b), [-1]),
  },
  {
    name: 'mean',
    shapes: [{ x: [4, 6] }, { x: [3, 5] }, { x: [2, 3, 4] }],
    output: [4, 1],
    reference: ({ x }) =>
      T.mul(T.sum(x, [-1], true), T.tensor([1 / x.shape.at(-1)!], [], x.backend)),
  },
  {
    name: 'outer',
    shapes: [
      { a: [4], b: [3] },
      { a: [2], b: [5] },
    ],
    output: [4, 3],
    reference: ({ a, b }) => T.mul(T.reshape(a, [a.shape[0], 1]), T.reshape(b, [1, b.shape[0]])),
  },
  {
    name: 'matmul',
    shapes: [
      { a: [4, 3], b: [3, 5] },
      { a: [2, 6], b: [6, 3] },
      { a: [2, 4, 3], b: [3, 5] },
    ],
    output: [4, 5],
    reference: ({ a, b }) => T.matmul(a, b),
  },
  {
    name: 'softmax',
    shapes: [{ x: [4, 6] }, { x: [3, 5] }],
    output: [4, 6],
    reference: ({ x }) => T.softmax(x),
  },
  {
    name: 'layernorm',
    shapes: [{ x: [4, 8] }, { x: [3, 5] }],
    output: [4, 8],
    reference: ({ x }) => T.layernorm(x),
  },
];

function fixtures(index: number, inputs: ShapePorts, name: string) {
  const cases = tensorCases(index, inputs);
  if (name === 'softmax') {
    const width = inputs.x.at(-1)!;
    cases.fixed.push({
      id: 'large-logits',
      inputs: {
        x: Array.from(
          { length: inputs.x.reduce((a, b) => a * b, 1) },
          (_, i) => 90 + (20 * (i % width)) / (width - 1),
        ),
      },
    });
  }
  return cases;
}

export const chapter2Levels: ForwardLevel[] = [];
for (const [index, definition] of definitions.entries()) {
  const id = `2-${index + 1}`;
  chapter2Levels.push(
    withPar(
      {
        id,
        chapter: '2',
        kind: 'forward',
        titleKey: `levels.${id}.title`,
        descriptionKey: `levels.${id}.description`,
        hintsKeys: [`levels.${id}.hint1`, `levels.${id}.hint2`],
        available: [...scalarPrimitives, 'max', 'sum', 'unsqueeze', 'reduce_max'],
        forbidden: [definition.name],
        interface: { inputs: definition.shapes[0], output: 'y', outputShape: definition.output },
        unlock: [index === 5 ? 'xor' : `2-${index + 2}`],
        unlockComponents: [definition.name],
        referenceGraph: answer(definition.name),
        ...fixtures(index, definition.shapes[0], definition.name),
        shapeVariants: definition.shapes.slice(1).map((inputs, variant) => ({
          inputs,
          ...fixtures(20 + index * 3 + variant, inputs, definition.name),
        })),
        reference: definition.reference,
        // Expanded mean/div use log/exp; cancellation on constant rows amplifies f32 error.
        atol: definition.name === 'layernorm' ? 1e-3 : 1e-4,
        rtol: 1e-4,
      },
      [...chapter0Levels, ...chapter1Levels, ...chapter2Levels],
    ),
  );
}
