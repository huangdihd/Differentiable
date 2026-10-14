import type { Attributes, GraphJSON } from '../engine/graph';
import type { ForwardLevel, TrainLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { scalarPrimitives, tensorCases, withPar } from './level-helpers';

const preceding = [...chapter0Levels, ...chapter1Levels, ...chapter2Levels];
const available = [...scalarPrimitives, 'max', 'sum', 'reduce_max', 'unsqueeze', 'transpose'];

function answer(index: number): GraphJSON {
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
  const binary = (id: string, op: string, a: string, b: string) => node(id, op, { a, b });
  const constant = (id: string, value: number) => node(id, 'const', {}, { value });
  for (const name of index === 3 ? ['p', 't'] : index === 4 ? ['x', 't'] : ['w', 'x', 't'])
    node(name, 'input', {}, { name });
  let result: string;
  if (index === 4) {
    for (const name of ['w', 'b']) node(name, 'param', {}, { shape: [], init: 'zeros' });
    const p = binary('prediction', 'add', binary('weighted', 'mul', 'w', 'x'), 'b');
    result = node('loss', 'mse', { p, t: 't' });
  } else {
    const prediction =
      index === 3 ? 'p' : binary('prediction', index === 2 ? 'dot' : 'mul', 'w', 'x');
    const residual = binary(
      'residual',
      'add',
      prediction,
      binary('negativeTarget', 'mul', 't', constant('minus', -1)),
    );
    if (index === 3) {
      result = node('mean', 'mean', { x: binary('squares', 'mul', residual, residual) });
    } else {
      const gradient = binary(
        'gradient',
        'mul',
        binary('twiceResidual', 'mul', constant('two', 2), residual),
        'x',
      );
      result =
        index === 2
          ? gradient
          : binary(
              'updated',
              'add',
              'w',
              binary('step', 'mul', constant('negativeRate', -0.1), gradient),
            );
    }
  }
  node('output', 'output', { x: result }, { name: index === 4 ? 'loss' : 'y' });
  return graph;
}

const scalar = (value: number, like: T.Tensor) => T.tensor([value], [], like.backend);
const residual = (p: T.Tensor, t: T.Tensor) => T.add(p, T.mul(t, scalar(-1, t)));
const metadata = (index: number) => ({
  id: `3-${index}`,
  chapter: '3',
  titleKey: `levels.3-${index}.title`,
  descriptionKey: `levels.3-${index}.description`,
  hintsKeys: [`levels.3-${index}.hint1`, `levels.3-${index}.hint2`],
  available,
  unlock: [index === 4 ? 'xor' : `3-${index + 1}`],
  referenceGraph: answer(index),
});

export const chapter3ForwardLevels: ForwardLevel[] = [];
chapter3ForwardLevels.push(
  withPar<ForwardLevel>(
    {
      ...metadata(1),
      kind: 'forward',
      interface: { inputs: { w: [], x: [], t: [] }, output: 'y', outputShape: [] },
      fixed: [
        [0, 2, 3],
        [2, 1, -1],
        [-1, 2, 2],
        [3, -1, 1],
        [1, 3, -2],
        [-2, -2, 1],
      ].map(([w, x, t], i) => ({
        id: `step-${i}`,
        inputs: { w: [w], x: [x], t: [t] },
      })),
      reference: ({ w, x, t }) =>
        T.add(w, T.mul(scalar(-0.1, w), T.mul(T.mul(scalar(2, w), residual(T.mul(w, x), t)), x))),
    },
    preceding,
  ),
);
chapter3ForwardLevels.push(
  withPar<ForwardLevel>(
    {
      ...metadata(2),
      kind: 'forward',
      interface: { inputs: { w: [3], x: [3], t: [] }, output: 'y', outputShape: [3] },
      ...tensorCases(301, { w: [3], x: [3], t: [] }),
      shapeVariants: [
        { inputs: { w: [5], x: [5], t: [] }, ...tensorCases(302, { w: [5], x: [5], t: [] }) },
      ],
      reference: ({ w, x, t }) =>
        T.mul(T.mul(scalar(2, w), residual(T.sum(T.mul(w, x), [-1]), t)), x),
    },
    [...preceding, ...chapter3ForwardLevels],
  ),
);
chapter3ForwardLevels.push(
  withPar<ForwardLevel>(
    {
      ...metadata(3),
      kind: 'forward',
      interface: { inputs: { p: [8], t: [8] }, output: 'y', outputShape: [1] },
      unlockComponents: ['mse'],
      forbidden: ['mse'],
      ...tensorCases(303, { p: [8], t: [8] }),
      shapeVariants: [[5], [2, 4]].map((shape, i) => ({
        inputs: { p: shape, t: shape },
        ...tensorCases(304 + i, { p: shape, t: shape }),
      })),
      reference: ({ p, t }) => {
        const error = residual(p, t);
        return T.mul(T.sum(T.mul(error, error), [-1], true), scalar(1 / p.shape.at(-1)!, p));
      },
    },
    [...preceding, ...chapter3ForwardLevels],
  ),
);

export const chapter3TrainLevel = withPar<TrainLevel>(
  {
    ...metadata(4),
    kind: 'train',
    available: [...available, 'param'],
    interface: { inputs: { x: [16], t: [16] }, output: 'loss', outputShape: [1] },
    seed: 3104,
    steps: 60,
    optimizer: { kind: 'sgd', learningRate: 0.01 },
    tunable: { learningRate: { min: 0.001, max: 2, default: 0.01 } },
    lossOutput: 'loss',
    threshold: { loss: 0.05 },
    dataset: (random) => {
      const x = Array.from({ length: 16 }, (_, i) => -2 + (4 * i) / 15);
      const t = x.map((value) => 1.5 * value - 0.5 + (random.uniform() - 0.5) * 0.1);
      return { inputs: { x: T.tensor(x, [16]), t: T.tensor(t, [16]) } };
    },
  },
  [...preceding, ...chapter3ForwardLevels],
);

export const chapter3Levels = [...chapter3ForwardLevels, chapter3TrainLevel];
