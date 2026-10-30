import { mseLock } from './level-helpers';
import { noisyLine, noisyLineBatch, rings, smallAndLarge, trainOn, twoBlobs } from '../datasets';
import { Random } from '../engine/random';
import { axisLabels, matrixAxes } from './output-axes';
import type { Attributes, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel, TrainLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { formulaGraph, scalarPrimitives, tensorCases, withPar } from './level-helpers';

const preceding = [...chapter0Levels, ...chapter1Levels, ...chapter2Levels];
const available = [...scalarPrimitives, 'max', 'sum', 'reduce_max', 'unsqueeze', 'transpose'];
const outputNames: Readonly<Record<number, string>> = {
  1: 'updatedWeight',
  2: 'gradient',
  3: 'loss',
  4: 'loss',
  5: 'loss',
  6: 'probabilities',
  7: 'updatedWeight',
  8: 'probabilities',
  9: 'probabilities',
};
const outputName = (index: number) => {
  const output = outputNames[index];
  if (!output) throw new Error(`Unknown chapter 3 output: ${index}`);
  return output;
};
const lossGradient = '∂L/∂p';

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
  for (const name of index === 3
    ? ['p', 't']
    : index === 4
      ? ['x', 't']
      : index === 2
        ? ['x', lossGradient]
        : ['w', 'x', lossGradient])
    node(name, 'input', {}, { name });
  let result: string;
  if (index === 4) {
    for (const name of ['w', 'b']) node(name, 'param', {}, { shape: [], init: 'zeros' });
    const p = binary('prediction', 'add', binary('weighted', 'mul', 'w', 'x'), 'b');
    result = node('loss', 'mse', { p, t: 't' });
  } else if (index === 3) {
    const residual = binary(
      'residual',
      'add',
      'p',
      binary('negativeTarget', 'mul', 't', constant('minus', -1)),
    );
    result = node('mean', 'mean', { x: binary('squares', 'mul', residual, residual) });
  } else {
    // The loss side has already produced ∂L/∂p. Reuse the local reverse-pass contract from 1-11
    // instead of expanding the squared-loss derivative again inside the model-side circuit.
    const gradient = binary('gradient', 'mul', lossGradient, 'x');
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
  node('output', 'output', { x: result }, { name: outputName(index) });
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

const lossIntroGraph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'p', component: 'input', attrs: { name: 'p' } },
    { id: 't', component: 'input', attrs: { name: 't' } },
    { id: 'error', component: 'sub' },
    { id: 'loss', component: 'square' },
    { id: 'output', component: 'output', attrs: { name: 'loss' } },
  ],
  edges: [
    ['p', 'error', 'a'],
    ['t', 'error', 'b'],
    ['error', 'loss', 'x'],
    ['loss', 'output', 'x'],
  ].map(([from, to, port]) => ({ from: { node: from, port: 'out' }, to: { node: to, port } })),
};

/** Loss gets its own playable introduction before gradients or parameter updates appear. */
export const lossIntroLevel = withPar<ForwardLevel>(
  {
    id: 'loss-intro',
    chapter: '3',
    kind: 'forward',
    titleKey: 'levels.loss-intro.title',
    descriptionKey: 'levels.loss-intro.description',
    hintsKeys: ['levels.loss-intro.hint1', 'levels.loss-intro.hint2'],
    available,
    interface: { inputs: { p: [], t: [] }, output: 'loss', outputShape: [] },
    unlock: ['3-1'],
    referenceGraph: lossIntroGraph,
    fixed: [
      { id: 'exact', inputs: { p: [3], t: [3] } },
      { id: 'near', inputs: { p: [4], t: [3] } },
      { id: 'far', inputs: { p: [5], t: [3] } },
      { id: 'below', inputs: { p: [-2], t: [1] } },
      { id: 'negative-target', inputs: { p: [2], t: [-1] } },
      { id: 'fractional', inputs: { p: [0.5], t: [-0.5] } },
    ],
    reference: ({ p, t }) => {
      const error = residual(p, t);
      return T.mul(error, error);
    },
  },
  preceding,
);

export const chapter3ForwardLevels: ForwardLevel[] = [lossIntroLevel];
chapter3ForwardLevels.push(
  withPar<ForwardLevel>(
    {
      ...metadata(1),
      unlock: ['neuron'],
      kind: 'forward',
      interface: {
        inputs: { w: [], x: [], [lossGradient]: [] },
        output: 'updatedWeight',
        outputShape: [],
      },
      revalidateOnLoad: true,
      fixed: [
        [0, 2, 3],
        [2, 1, -1],
        [-1, 2, 2],
        [3, -1, 1],
        [1, 3, -2],
        [-2, -2, 1],
      ].map(([w, x, t], i) => ({
        id: `step-${i}`,
        inputs: { w: [w], x: [x], [lossGradient]: [2 * (w * x - t)] },
      })),
      reference: ({ w, x, [lossGradient]: upstream }) =>
        T.add(w, T.mul(scalar(-0.1, w), T.mul(upstream, x))),
    },
    [...preceding, ...chapter3ForwardLevels],
  ),
);
chapter3ForwardLevels.push(
  withPar<ForwardLevel>(
    {
      ...metadata(2),
      unlock: ['layer'],
      kind: 'forward',
      interface: {
        inputs: { x: [3], [lossGradient]: [] },
        output: 'gradient',
        outputShape: [3],
      },
      revalidateOnLoad: true,
      ...tensorCases(301, { x: [3], [lossGradient]: [] }),
      shapeVariants: [
        {
          inputs: { x: [5], [lossGradient]: [] },
          ...tensorCases(302, { x: [5], [lossGradient]: [] }),
        },
      ],
      reference: ({ x, [lossGradient]: upstream }) => T.mul(upstream, x),
    },
    [...preceding, ...chapter3ForwardLevels],
  ),
);
chapter3ForwardLevels.push(
  withPar<ForwardLevel>(
    {
      ...metadata(3),
      kind: 'forward',
      interface: { inputs: { p: [8], t: [8] }, output: 'loss', outputShape: [1] },
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
    componentDefaults: {
      linear: { automatic_parameters: true, in_features: 1, out_features: 1, bias: true },
    },
    interface: { inputs: { x: [16], t: [16] }, output: 'loss', outputShape: [1] },
    seed: 3104,
    steps: 60,
    optimizer: { kind: 'sgd', learningRate: 0.01 },
    tunable: { learningRate: { min: 0.001, max: 2, default: 0.01 } },
    lossOutput: 'loss',
    lockedLoss: mseLock,
    threshold: { loss: 0.05 },
    ...trainOn(noisyLine),
    // Same 16 noisy points, shuffled for grading: a [16] Parameter fitted row by row fails.
    evaluationDataset: (random) => noisyLineBatch(new Random(3104), random),
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
  node('output', 'output', { x: 'probabilities' }, { name: 'probabilities' });
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
      { id: 'output', component: 'output', attrs: { name: 'loss' } },
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
    referenceGraph: crossEntropyAnswer(),
    interface: { inputs: { p: [4, 3], t: [4, 3] }, output: 'loss', outputShape: [1] },
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
function affineAnswer(component: 'dot' | 'matmul', output: string): GraphJSON {
  return {
    version: 1,
    nodes: [
      ...['x', 'w', 'b'].map((name) => ({ id: name, component: 'input', attrs: { name } })),
      { id: 'weighted', component },
      { id: 'biased', component: 'add' },
      { id: 'output', component: 'output', attrs: { name: output } },
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
      available,
      unlock: [id === 'neuron' ? '3-2' : '3-3'],
      ...(id === 'layer' ? { unlockComponents: ['linear'], forbidden: ['linear'] } : {}),
      interface: {
        inputs: shapes[0],
        output: id === 'neuron' ? 'score' : 'scores',
        outputShape: output,
      },
      ...(id === 'layer' && {
        outputNoteKey: 'levels.layer.expectedNote',
        outputAxes: matrixAxes(shapes, (_, inputs) => ({
          rows: axisLabels('x', inputs.x[0]),
          cols: axisLabels('n', inputs.b[0]),
        })),
      }),
      referenceGraph: affineAnswer(component, id === 'neuron' ? 'score' : 'scores'),
      ...tensorCases(320 + index * 4, shapes[0]),
      shapeVariants: shapes.slice(1).map((inputs, variant) => ({
        inputs,
        ...tensorCases(321 + index * 4 + variant, inputs),
      })),
      reference: ({ x, w, b }) =>
        T.add(component === 'dot' ? T.sum(T.mul(x, w), [-1]) : T.matmul(x, w), b),
    },
    [...preceding, ...chapter3ForwardLevels.slice(0, 3)],
  ),
);

function momentumAnswer(): GraphJSON {
  return {
    version: 1,
    nodes: [
      ...['w', 'g', 'm'].map((name) => ({ id: name, component: 'input', attrs: { name } })),
      { id: 'decay', component: 'const', attrs: { value: 0.9 } },
      { id: 'decayedMomentum', component: 'mul' },
      { id: 'updatedMomentum', component: 'add' },
      { id: 'negativeRate', component: 'const', attrs: { value: -0.1 } },
      { id: 'step', component: 'mul' },
      { id: 'updatedWeight', component: 'add' },
      { id: 'weightOutput', component: 'output', attrs: { name: 'updatedWeight' } },
      { id: 'momentumOutput', component: 'output', attrs: { name: 'updatedMomentum' } },
    ],
    edges: [
      ['m', 'decayedMomentum', 'a'],
      ['decay', 'decayedMomentum', 'b'],
      ['decayedMomentum', 'updatedMomentum', 'a'],
      ['g', 'updatedMomentum', 'b'],
      ['updatedMomentum', 'step', 'a'],
      ['negativeRate', 'step', 'b'],
      ['w', 'updatedWeight', 'a'],
      ['step', 'updatedWeight', 'b'],
      ['updatedWeight', 'weightOutput', 'x'],
      ['updatedMomentum', 'momentumOutput', 'x'],
    ].map(([from, to, port]) => ({ from: { node: from, port: 'out' }, to: { node: to, port } })),
  };
}

const momentumLevel = withPar<ForwardLevel>(
  {
    ...metadata(7),
    kind: 'forward',
    interface: {
      inputs: { w: [], g: [], m: [] },
      output: 'updatedWeight',
      outputShape: [],
      outputs: { updatedWeight: [], updatedMomentum: [] },
    },
    referenceGraph: momentumAnswer(),
    revalidateOnLoad: true,
    ...tensorCases(307, { w: [], g: [], m: [] }),
    referenceOutputs: ({ w, g, m }) => {
      const updatedMomentum = T.add(T.mul(scalar(0.9, m), m), g);
      return {
        updatedWeight: T.add(w, T.mul(scalar(-0.1, w), updatedMomentum)),
        updatedMomentum,
      };
    },
    reference: ({ w, g, m }) =>
      T.add(w, T.mul(scalar(-0.1, w), T.add(T.mul(scalar(0.9, m), m), g))),
  },
  [...preceding, ...chapter3ForwardLevels, crossEntropyLevel],
);
chapter3ForwardLevels.push(crossEntropyLevel, ...affineLevels, momentumLevel);

export const chapter3ClassificationLevels: TrainLevel[] = [6, 8, 9].map((index) => {
  const count = index === 9 ? 64 : 40;
  // Same builder as train, different seed at evaluate (seed + 1) so param-table
  // memorizers that ignore x fail hold-out — mirrors chapter 4+ evaluationDataset.
  const points = index === 9 ? rings : index === 8 ? smallAndLarge : twoBlobs;
  return withPar<TrainLevel>(
    {
      ...metadata(index),
      kind: 'train',
      unlock: [index === 9 ? 'xor' : `3-${index + 1}`],
      available: [...available, 'param'],
      referenceGraph: classificationAnswer(index === 9),
      interface: {
        inputs: { x: [count, 2] },
        output: 'probabilities',
        outputShape: [count, 2],
      },
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
      ...trainOn(points),
      evaluationDataset: points.build,
    },
    [...preceding, ...chapter3ForwardLevels],
  );
});
export const chapter3Levels = [
  ...chapter3ForwardLevels.slice(0, 2),
  affineLevels[0],
  chapter3ForwardLevels[2],
  affineLevels[1],
  chapter3ForwardLevels[3],
  chapter3TrainLevel,
  crossEntropyLevel,
  chapter3ClassificationLevels[0],
  momentumLevel,
  ...chapter3ClassificationLevels.slice(1),
];
