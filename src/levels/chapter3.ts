import type { Attributes, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel, TrainLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { formulaGraph, scalarPrimitives, tensorCases, withPar } from './level-helpers';

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
  unlock: [`3-${index + 1}`],
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

/** Ordinary player circuits; no training-only or loss kernels in the reference graph. */
function classificationAnswer(hidden = false): GraphJSON {
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
  node('x', 'input', {}, { name: 'x' });
  let x = 'x';
  if (hidden) {
    node('w1', 'param', {}, { shape: [2, 16], init: 'normal', scale: 1 });
    node('b1', 'param', {}, { shape: [16], init: 'zeros' });
    node('hiddenLinear', 'matmul', { a: x, b: 'w1' });
    node('hiddenBias', 'add', { a: 'hiddenLinear', b: 'b1' });
    x = node('hidden', 'sigmoid', { x: 'hiddenBias' });
  }
  node(
    'w',
    'param',
    {},
    { shape: [hidden ? 16 : 2, 2], init: hidden ? 'normal' : 'zeros', scale: 0.5 },
  );
  node('b', 'param', {}, { shape: [2], init: 'zeros' });
  node('linear', 'matmul', { a: x, b: 'w' });
  node('biased', 'add', { a: 'linear', b: 'b' });
  node('probabilities', 'softmax', { x: 'biased' });
  node('output', 'output', { x: 'probabilities' }, { name: 'y' });
  return graph;
}

function crossEntropyAnswer(): GraphJSON {
  return {
    version: 1,
    nodes: [
      { id: 'p', component: 'input', attrs: { name: 'p' } },
      { id: 't', component: 'input', attrs: { name: 't' } },
      { id: 'log', component: 'log' },
      { id: 'selected', component: 'mul' },
      { id: 'rows', component: 'sum', attrs: { axes: [-1] } },
      { id: 'average', component: 'mean' },
      { id: 'minus', component: 'const', attrs: { value: -1 } },
      { id: 'negative', component: 'mul' },
      { id: 'output', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [
      ['p', 'log', 'x'],
      ['log', 'selected', 'a'],
      ['t', 'selected', 'b'],
      ['selected', 'rows', 'x'],
      ['rows', 'average', 'x'],
      ['average', 'negative', 'a'],
      ['minus', 'negative', 'b'],
      ['negative', 'output', 'x'],
    ].map(([from, to, port]) => ({ from: { node: from, port: 'out' }, to: { node: to, port } })),
  };
}
function probabilityCases(rows: number, classes: number) {
  return {
    fixed: Array.from({ length: 5 }, (_, k) => {
      const p: number[] = [],
        t: number[] = [];
      for (let i = 0; i < rows; i++) {
        const weights = Array.from({ length: classes }, (_, j) =>
          Math.exp(((i * 3 + j * 7 + k) % 11) / 3),
        );
        const total = weights.reduce((a, b) => a + b, 0);
        p.push(...weights.map((w) => w / total));
        t.push(...weights.map((_, j) => Number(j === (i + k) % classes)));
      }
      return { id: `probabilities-${k}`, inputs: { p, t } };
    }),
  };
}
const crossEntropyLevel = withPar<ForwardLevel>(
  {
    ...metadata(5),
    kind: 'forward',
    unlock: ['neuron'],
    referenceGraph: crossEntropyAnswer(),
    interface: { inputs: { p: [4, 3], t: [4, 3] }, output: 'y', outputShape: [1] },
    unlockComponents: ['cross_entropy'],
    forbidden: ['cross_entropy'],
    ...probabilityCases(4, 3),
    shapeVariants: [
      [2, 5],
      [6, 2],
    ].map(([rows, classes]) => ({
      inputs: { p: [rows, classes], t: [rows, classes] },
      ...probabilityCases(rows, classes),
    })),
    reference: ({ p, t }) =>
      T.mul(T.sum(T.mul(t, T.log(p)), [-2, -1]), T.tensor([-1 / p.shape[0]], [1], p.backend)),
  },
  [...preceding, ...chapter3ForwardLevels],
);

/** A point and a batch share the same circuit; only the input shapes change. */
function affineAnswer(component: 'dot' | 'matmul'): GraphJSON {
  return {
    version: 1,
    nodes: [
      ...['x', 'w', 'b'].map((name) => ({ id: name, component: 'input', attrs: { name } })),
      { id: 'weighted', component },
      { id: 'biased', component: 'add' },
      { id: 'output', component: 'output', attrs: { name: 'y' } },
    ],
    edges: [
      ['x', 'weighted', 'a'],
      ['w', 'weighted', 'b'],
      ['weighted', 'biased', 'a'],
      ['b', 'biased', 'b'],
      ['biased', 'output', 'x'],
    ].map(([from, to, port]) => ({ from: { node: from, port: 'out' }, to: { node: to, port } })),
  };
}

const affineDefinitions: {
  id: string;
  component: 'dot' | 'matmul';
  shapes: ShapePorts[];
  output: number[];
}[] = [
  {
    id: 'neuron',
    component: 'dot',
    shapes: [
      { x: [2], w: [2], b: [] },
      { x: [5, 2], w: [2], b: [] },
      { x: [3], w: [3], b: [] },
      { x: [4, 3], w: [3], b: [] },
    ],
    output: [],
  },
  {
    id: 'layer',
    component: 'matmul',
    shapes: [
      { x: [4, 2], w: [2, 3], b: [3] },
      { x: [6, 2], w: [2, 3], b: [3] },
      { x: [4, 5], w: [5, 2], b: [2] },
      { x: [1, 3], w: [3, 4], b: [4] },
    ],
    output: [4, 3],
  },
];
const affineLevels = affineDefinitions.map(({ id, component, shapes, output }, index) =>
  withPar<ForwardLevel>(
    {
      id,
      chapter: '3',
      kind: 'forward',
      titleKey: `levels.${id}.title`,
      descriptionKey: `levels.${id}.description`,
      hintsKeys: [`levels.${id}.hint1`, `levels.${id}.hint2`],
      available: crossEntropyLevel.available,
      unlock: [id === 'neuron' ? 'layer' : '3-6'],
      ...(id === 'layer' ? { unlockComponents: ['linear'], forbidden: ['linear'] } : {}),
      interface: { inputs: shapes[0], output: 'y', outputShape: output },
      referenceGraph: affineAnswer(component),
      // Lead with the varied case so case 1 shows real numbers, not all zeros.
      ...((cases) => ({
        ...cases,
        fixed: [cases.fixed[2], ...cases.fixed.filter((_, k) => k !== 2)],
      }))(tensorCases(320 + index * 4, shapes[0])),
      shapeVariants: shapes.slice(1).map((inputs, variant) => ({
        inputs,
        ...tensorCases(321 + index * 4 + variant, inputs),
      })),
      reference: ({ x, w, b }) =>
        T.add(component === 'dot' ? T.sum(T.mul(x, w), [-1]) : T.matmul(x, w), b),
    },
    [...preceding, ...chapter3ForwardLevels, crossEntropyLevel],
  ),
);

const momentumLevel = withPar<ForwardLevel>(
  {
    ...metadata(7),
    kind: 'forward',
    interface: { inputs: { w: [], g: [], m: [] }, output: 'y', outputShape: [] },
    referenceGraph: formulaGraph('w + (-0.1) * (0.9 * m + g)', ['w', 'g', 'm']),
    ...tensorCases(307, { w: [], g: [], m: [] }),
    reference: ({ w, g, m }) =>
      T.add(w, T.mul(scalar(-0.1, w), T.add(T.mul(scalar(0.9, m), m), g))),
  },
  [...preceding, ...chapter3ForwardLevels, crossEntropyLevel],
);
chapter3ForwardLevels.push(crossEntropyLevel, ...affineLevels, momentumLevel);

export const chapter3ClassificationLevels: TrainLevel[] = [6, 8, 9].map((index) => {
  const count = index === 9 ? 64 : 40;
  return withPar<TrainLevel>(
    {
      ...metadata(index),
      kind: 'train',
      unlock: [index === 9 ? 'xor' : `3-${index + 1}`],
      available: [...available, 'param'],
      referenceGraph: classificationAnswer(index === 9),
      interface: { inputs: { x: [count, 2] }, output: 'y', outputShape: [count, 2] },
      seed: 3100 + index,
      steps: index === 9 ? 300 : index === 8 ? 600 : 80,
      optimizer: { kind: index === 9 ? 'adam' : 'sgd', learningRate: index === 6 ? 0.001 : 0.05 },
      tunable: {
        learningRate: {
          min: 0.001,
          max: index === 6 ? 1 : 0.1,
          default: index === 6 ? 0.001 : 0.05,
        },
        ...(index === 6
          ? {}
          : {
              optimizer:
                index === 9
                  ? (['adam', 'sgd'] as ('adam' | 'sgd')[])
                  : (['sgd', 'adam'] as ('adam' | 'sgd')[]),
            }),
      },
      threshold: { loss: 0.15, accuracy: index === 9 ? 0.95 : 1 },
      dataset: (random) => {
        const x: number[] = [],
          targets: number[] = [];
        for (let i = 0; i < count; i++) {
          const label = i % 2,
            sign = label ? 1 : -1;
          if (index === 9) {
            const angle =
              (2 * Math.PI * (Math.floor(i / 2) + random.uniform() * 0.2)) / (count / 2);
            const radius = (label ? 2 : 0.6) + (random.uniform() - 0.5) * 0.1;
            x.push(radius * Math.cos(angle), radius * Math.sin(angle));
          } else if (index === 8) {
            // The small feature determines the label; the large nuisance feature is balanced.
            x.push(
              sign * (0.8 + random.uniform() * 0.4) * 0.05,
              (Math.floor(i / 2) % 2 ? 1 : -1) * (0.8 + random.uniform() * 0.4) * 20,
            );
          } else {
            x.push(sign * (1 + random.uniform() * 0.5), sign * (1 + random.uniform() * 0.5));
          }
          targets.push(1 - label, label);
        }
        return { inputs: { x: T.tensor(x, [count, 2]) }, targets: T.tensor(targets, [count, 2]) };
      },
    },
    [...preceding, ...chapter3ForwardLevels],
  );
});
export const chapter3Levels = [
  ...chapter3ForwardLevels.slice(0, 3),
  chapter3TrainLevel,
  crossEntropyLevel,
  ...affineLevels,
  chapter3ClassificationLevels[0],
  momentumLevel,
  ...chapter3ClassificationLevels.slice(1),
];
