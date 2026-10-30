import { mseLock } from './level-helpers';
import type { Attributes, GraphEndpoint, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel, ObservationLevel, TrainLevel } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels, chapter4TrainLevels } from './chapter4';
import { withPar } from './level-helpers';

const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
];
/** Slice/Stack/Concat/Split arrive with chapter 5, so sequence plumbing is one node each. */
export const tensorOps = ['slice', 'stack', 'concat', 'split'];
const available = [...chapter4TrainLevels[0].available, ...tensorOps];
const levelIds = [
  'time-window-limit',
  'time-rnn',
  'time-rnn-window',
  'time-rnn-delay',
  'time-attention',
  'time-forecast',
] as const;
const windowLength = 6;
const delayLength = windowLength;
const hiddenFeatures = 8;

const oneHot = (length: number, selectedIndex: number) =>
  Array.from({ length }, (_, index) => (index === selectedIndex ? 1 : 0));

class GraphBuilder {
  readonly graph: GraphJSON = { version: 1, nodes: [], edges: [] };

  node(
    id: string,
    component: string,
    inputs: Record<string, GraphEndpoint> = {},
    attrs?: Attributes,
  ): GraphEndpoint {
    this.graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, from] of Object.entries(inputs))
      this.graph.edges.push({ from, to: { node: id, port } });
    return { node: id, port: 'out' };
  }

  input(name: string): GraphEndpoint {
    return this.node(name, 'input', {}, { name });
  }

  constant(id: string, value: number | number[], shape?: number[]): GraphEndpoint {
    return this.node(id, 'const', {}, { value, ...(shape ? { shape } : {}) });
  }

  output(name: string, value: GraphEndpoint): void {
    this.node(`output_${name}`, 'output', { x: value }, { name });
  }
}

const add = (builder: GraphBuilder, id: string, a: GraphEndpoint, b: GraphEndpoint) =>
  builder.node(id, 'add', { a, b });
const mul = (builder: GraphBuilder, id: string, a: GraphEndpoint, b: GraphEndpoint) =>
  builder.node(id, 'mul', { a, b });
const matmul = (builder: GraphBuilder, id: string, a: GraphEndpoint, b: GraphEndpoint) =>
  builder.node(id, 'matmul', { a, b });

function rnnAnswer(): GraphJSON {
  const builder = new GraphBuilder();
  const x = builder.input('x');
  const hidden = builder.input('h');
  const wx = builder.input('wx');
  const wh = builder.input('wh');
  const bias = builder.input('b');
  const affine = add(
    builder,
    'affine',
    add(builder, 'input_projection', matmul(builder, 'x_wx', x, wx), bias),
    matmul(builder, 'h_wh', hidden, wh),
  );
  builder.output('hiddenState', builder.node('next', 'tanh', { x: affine }));
  return builder.graph;
}

export function rnnWindowAnswer(): GraphJSON {
  const builder = new GraphBuilder();
  const x = builder.input('x');
  const wx = builder.input('wx');
  const wh = builder.input('wh');
  const bias = builder.input('b');
  // Slice character k as [B,1]; the first cell's h is left empty, so it starts from zeros.
  let hidden: GraphEndpoint | undefined;
  const states: Record<string, GraphEndpoint> = {};
  for (let step = 0; step < windowLength; step++) {
    const character = builder.node(
      `char_${step}`,
      'slice',
      { x },
      { axis: 1, index: step, length: 1, keepdims: true },
    );
    hidden = builder.node(
      `step_${step}_hidden`,
      'rnn_cell',
      { x: character, ...(hidden ? { h: hidden } : {}), wx, wh, b: bias },
      { automatic_parameters: false, input_features: 1, hidden_features: hiddenFeatures },
    );
    states[String.fromCharCode(97 + step)] = hidden;
  }
  builder.output(
    'hiddenStates',
    builder.node('history', 'stack', states, { count: windowLength, axis: 1 }),
  );
  return builder.graph;
}

function selectFinalState(
  builder: GraphBuilder,
  id: string,
  history: GraphEndpoint,
): GraphEndpoint {
  const length = builder.node(`${id}_length`, 'size', { x: history }, { axis: 1 });
  const last = add(builder, `${id}_last`, length, builder.constant(`${id}_minus_one`, -1));
  const positions = builder.node(`${id}_positions`, 'arange', { n: length });
  const selector = mul(
    builder,
    `${id}_selector`,
    builder.node(`${id}_at_or_after`, 'ge', { a: positions, b: last }),
    builder.node(`${id}_at_or_before`, 'ge', { a: last, b: positions }),
  );
  return builder.node(
    id,
    'sum',
    {
      x: mul(
        builder,
        `${id}_selected`,
        history,
        builder.node(`${id}_selector_column`, 'unsqueeze', { x: selector }, { axis: -1 }),
      ),
    },
    { axes: [1] },
  );
}

export function rnnDelayAnswer(): GraphJSON {
  const builder = new GraphBuilder();
  const x = builder.input('x');
  const target = builder.input('t');
  const history = builder.node(
    'history',
    'rnn_window',
    { x },
    { automatic_parameters: true, input_features: 1, hidden_features: hiddenFeatures },
  );
  const finalState = builder.node(
    'final_state',
    'slice',
    { x: history },
    { axis: 1, index: -1, length: 1, keepdims: false },
  );
  const projected = builder.node(
    'prediction_column',
    'linear',
    { x: finalState },
    {
      automatic_parameters: true,
      in_features: hiddenFeatures,
      out_features: 1,
      bias: true,
    },
  );
  const prediction = builder.node('prediction', 'reshape', { x: projected }, { shape: [-1] });
  builder.output('loss', builder.node('loss', 'mse', { p: prediction, t: target }));
  return builder.graph;
}

export function temporalAttentionAnswer(): GraphJSON {
  const builder = new GraphBuilder();
  const history = builder.input('history');
  const query = builder.input('query');
  const width = builder.node('width', 'size', { x: history }, { axis: -1 });
  const rawScores = matmul(
    builder,
    'raw_scores',
    builder.node('query_row', 'unsqueeze', { x: query }, { axis: 1 }),
    builder.node('history_transposed', 'transpose', { x: history }, { axes: [-2, -1] }),
  );
  const scores = builder.node('scaled_scores', 'div', {
    a: rawScores,
    b: builder.node('root_width', 'sqrt', { x: width }),
  });
  const weights = builder.node('weights', 'softmax', { x: scores });
  const attended = matmul(builder, 'attended', weights, history);
  const batch = builder.node('batch', 'size', { x: history }, { axis: 0 });
  builder.output(
    'context',
    builder.node('context', 'reshape', { x: attended, d0: batch, d1: width }, { shape: [1, 1] }),
  );
  return builder.graph;
}

export function forecastAnswer(): GraphJSON {
  const builder = new GraphBuilder();
  const x = builder.input('x');
  const target = builder.input('t');
  const history = builder.node(
    'history',
    'rnn_window',
    { x },
    {
      automatic_parameters: true,
      input_features: 1,
      hidden_features: hiddenFeatures,
    },
  );
  const query = selectFinalState(builder, 'query', history);
  const context = builder.node('context', 'temporal_attention', { history, query });
  const projected = builder.node(
    'prediction_column',
    'linear',
    { x: context },
    {
      automatic_parameters: true,
      in_features: hiddenFeatures,
      out_features: 1,
      bias: true,
    },
  );
  const prediction = builder.node('prediction', 'reshape', { x: projected }, { shape: [-1] });
  builder.output('loss', builder.node('loss', 'mse', { p: prediction, t: target }));
  return builder.graph;
}

function rnnReference(args: Record<string, T.Tensor>): T.Tensor {
  return tanhTensor(T.add(T.add(T.matmul(args.x, args.wx), T.matmul(args.h, args.wh)), args.b));
}

function sigmoidTensor(x: T.Tensor): T.Tensor {
  return T.recip(
    T.add(T.tensor([1], [], x.backend), T.exp(T.mul(x, T.tensor([-1], [], x.backend)))),
  );
}

function tanhTensor(x: T.Tensor): T.Tensor {
  const doubled = T.mul(x, T.tensor([2], [], x.backend));
  return T.add(
    T.mul(sigmoidTensor(doubled), T.tensor([2], [], x.backend)),
    T.tensor([-1], [], x.backend),
  );
}

function attentionReference(history: T.Tensor, query: T.Tensor): T.Tensor {
  const [batch, , width] = history.shape;
  const scores = T.mul(
    T.matmul(T.reshape(query, [batch, 1, width]), T.transpose(history, [0, 2, 1])),
    T.tensor([1 / Math.sqrt(width)], [], history.backend),
  );
  return T.reshape(T.matmul(T.softmax(scores), history), [batch, width]);
}

function fixtureValue(random: Random, caseIndex: number, port: number, index: number) {
  if (caseIndex === 0) return 0;
  if (caseIndex === 1) return (((index + port) % 7) - 3) / 4;
  return random.uniform() * 2 - 1;
}

function fixtures(index: number, inputs: ShapePorts) {
  const make = (random: Random, caseIndex: number) => ({
    id: `case-${caseIndex}`,
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([name, shape], port) => [
        name,
        Array.from({ length: shape.reduce((size, dimension) => size * dimension, 1) }, (_, index) =>
          fixtureValue(random, caseIndex, port, index),
        ),
      ]),
    ),
  });
  return {
    fixed: Array.from({ length: 4 }, (_, i) => make(new Random(7500 + index * 10 + i), i)),
    generated: { seed: 7900 + index, count: 4, generate: make },
  };
}

function recurrentInputs(batch = 4, inputFeatures = 2, hidden = 3): ShapePorts {
  return {
    x: [batch, inputFeatures],
    h: [batch, hidden],
    wx: [inputFeatures, hidden],
    wh: [hidden, hidden],
    b: [hidden],
  };
}

export const windowLimitLevel: ObservationLevel = {
  id: levelIds[0],
  chapter: '5',
  kind: 'observation',
  titleKey: `levels.${levelIds[0]}.title`,
  descriptionKey: `levels.${levelIds[0]}.description`,
  unlock: [levelIds[1]],
};

export const temporalForwardLevels: ForwardLevel[] = [];

const rnnInputs = recurrentInputs();
const rnnLevel = withPar<ForwardLevel>(
  {
    id: levelIds[1],
    chapter: '5',
    kind: 'forward',
    titleKey: `levels.${levelIds[1]}.title`,
    descriptionKey: `levels.${levelIds[1]}.description`,
    hintsKeys: [`levels.${levelIds[1]}.hint1`, `levels.${levelIds[1]}.hint2`],
    available,
    unlock: [levelIds[2]],
    unlockComponents: ['rnn_cell'],
    forbidden: ['rnn_cell'],
    referenceGraph: rnnAnswer(),
    interface: { inputs: rnnInputs, output: 'hiddenState', outputShape: [4, 3] },
    ...fixtures(1, rnnInputs),
    shapeVariants: [recurrentInputs(2, 4, 5)].map((inputs) => ({
      inputs,
      ...fixtures(11, inputs),
    })),
    reference: (args) => rnnReference(args),
  },
  [...preceding, ...temporalForwardLevels],
);
temporalForwardLevels.push(rnnLevel);

export function delayedCopyBatch(random: Random, batch = 32) {
  const inputs: number[] = [];
  const targets: number[] = [];
  for (let row = 0; row < batch; row++) {
    const remembered = random.uniform() * 1.6 - 0.8;
    targets.push(remembered);
    inputs.push(remembered);
    for (let step = 1; step < delayLength; step++) inputs.push(random.uniform() * 0.4 - 0.2);
  }
  return {
    inputs: {
      x: T.tensor(inputs, [batch, delayLength]),
      t: T.tensor(targets, [batch]),
    },
  };
}

const windowInputs: ShapePorts = {
  x: [4, windowLength],
  wx: [1, hiddenFeatures],
  wh: [hiddenFeatures, hiddenFeatures],
  b: [hiddenFeatures],
};
const rnnWindowLevel = withPar<ForwardLevel>(
  {
    id: levelIds[2],
    chapter: '5',
    kind: 'forward',
    titleKey: `levels.${levelIds[2]}.title`,
    descriptionKey: `levels.${levelIds[2]}.description`,
    hintsKeys: [`levels.${levelIds[2]}.hint1`, `levels.${levelIds[2]}.hint2`],
    available,
    unlock: [levelIds[3]],
    unlockComponents: ['rnn_window'],
    forbidden: ['rnn_window'],
    // The lesson wires one shared wx/wh/b into every step, so fresh cells expose those ports.
    componentDefaults: {
      rnn_cell: { automatic_parameters: false, input_features: 1, hidden_features: hiddenFeatures },
    },
    referenceGraph: rnnWindowAnswer(),
    interface: { inputs: windowInputs, output: 'hiddenStates', outputShape: [4, windowLength, 8] },
    ...fixtures(3, windowInputs),
    shapeVariants: [{ x: [2, windowLength], wx: [1, 8], wh: [8, 8], b: [8] }].map((inputs) => ({
      inputs,
      ...fixtures(13, inputs),
    })),
    reference: ({ x, wx, wh, b }) => {
      const batch = x.shape[0];
      let hidden = T.tensor(
        new Float32Array(batch * hiddenFeatures),
        [batch, hiddenFeatures],
        x.backend,
      );
      let history: T.Tensor | undefined;
      for (let step = 0; step < windowLength; step++) {
        const mask = T.tensor(oneHot(windowLength, step), [windowLength], x.backend);
        const value = T.reshape(T.sum(T.mul(x, mask), [-1]), [batch, 1]);
        hidden = rnnReference({ x: value, h: hidden, wx, wh, b });
        const row = T.reshape(hidden, [batch, 1, hiddenFeatures]);
        const rowMask = T.tensor(oneHot(windowLength, step), [windowLength, 1], x.backend);
        const placed = T.mul(row, rowMask);
        history = history ? T.add(history, placed) : placed;
      }
      return history!;
    },
  },
  [...preceding, ...temporalForwardLevels],
);
temporalForwardLevels.push(rnnWindowLevel);

export const temporalRnnDelayLevel = withPar<TrainLevel>(
  {
    id: levelIds[3],
    chapter: '5',
    kind: 'train',
    titleKey: `levels.${levelIds[3]}.title`,
    descriptionKey: `levels.${levelIds[3]}.description`,
    hintsKeys: [`levels.${levelIds[3]}.hint1`, `levels.${levelIds[3]}.hint2`],
    available: [...available, 'param'],
    presets: ['rnn_window'],
    componentDefaults: {
      rnn_window: {
        automatic_parameters: true,
        input_features: 1,
        hidden_features: hiddenFeatures,
      },
      linear: {
        automatic_parameters: true,
        in_features: hiddenFeatures,
        out_features: 1,
        bias: true,
      },
    },
    unlock: [levelIds[4]],
    referenceGraph: rnnDelayAnswer(),
    interface: {
      inputs: { x: [32, delayLength], t: [32] },
      output: 'loss',
      outputShape: [1],
    },
    seed: 7503,
    steps: 300,
    optimizer: { kind: 'adam', learningRate: 0.02 },
    tunable: { learningRate: { min: 0.001, max: 0.1, default: 0.02 }, optimizer: ['adam'] },
    lossOutput: 'loss',
    lockedLoss: mseLock,
    threshold: { loss: 0.015 },
    dataset: delayedCopyBatch,
    evaluationDataset: delayedCopyBatch,
    sequenceProbe: 'delay',
  },
  [...preceding, ...temporalForwardLevels],
);

const attentionInputs: ShapePorts = { history: [3, 5, 4], query: [3, 4] };
const attentionLevel = withPar<ForwardLevel>(
  {
    id: levelIds[4],
    chapter: '5',
    kind: 'forward',
    titleKey: `levels.${levelIds[4]}.title`,
    descriptionKey: `levels.${levelIds[4]}.description`,
    hintsKeys: [`levels.${levelIds[4]}.hint1`, `levels.${levelIds[4]}.hint2`],
    available,
    unlock: [levelIds[5]],
    unlockComponents: ['temporal_attention'],
    forbidden: ['temporal_attention'],
    revalidateOnLoad: true,
    referenceGraph: temporalAttentionAnswer(),
    interface: { inputs: attentionInputs, output: 'context', outputShape: [3, 4] },
    ...fixtures(7, attentionInputs),
    shapeVariants: [{ history: [2, 7, 6], query: [2, 6] }].map((inputs) => ({
      inputs,
      ...fixtures(17, inputs),
    })),
    reference: ({ history, query }) => attentionReference(history, query),
  },
  [...preceding, ...temporalForwardLevels],
);
temporalForwardLevels.push(attentionLevel);

export function waveBatch(random: Random) {
  const batch = 16;
  const inputs: number[] = [];
  const targets: number[] = [];
  for (let row = 0; row < batch; row++) {
    const phase = random.uniform() * Math.PI * 2;
    const frequency = 0.35 + random.uniform() * 0.25;
    const amplitude = 0.6 + random.uniform() * 0.4;
    for (let step = 0; step < windowLength; step++)
      inputs.push(amplitude * Math.sin(phase + step * frequency));
    targets.push(amplitude * Math.sin(phase + windowLength * frequency));
  }
  return {
    inputs: {
      x: T.tensor(inputs, [batch, windowLength]),
      t: T.tensor(targets, [batch]),
    },
  };
}

export const temporalTrainLevel = withPar<TrainLevel>(
  {
    id: levelIds[5],
    chapter: '5',
    kind: 'train',
    titleKey: `levels.${levelIds[5]}.title`,
    descriptionKey: `levels.${levelIds[5]}.description`,
    hintsKeys: [`levels.${levelIds[5]}.hint1`, `levels.${levelIds[5]}.hint2`],
    available: [...available, 'param'],
    componentDefaults: {
      rnn_window: {
        automatic_parameters: true,
        input_features: 1,
        hidden_features: hiddenFeatures,
      },
      linear: {
        automatic_parameters: true,
        in_features: hiddenFeatures,
        out_features: 1,
        bias: true,
      },
    },
    unlock: ['5-1'],
    referenceGraph: forecastAnswer(),
    interface: { inputs: { x: [16, windowLength], t: [16] }, output: 'loss', outputShape: [1] },
    seed: 7606,
    steps: 400,
    optimizer: { kind: 'adam', learningRate: 0.02 },
    tunable: { learningRate: { min: 0.001, max: 0.1, default: 0.02 }, optimizer: ['adam'] },
    lossOutput: 'loss',
    lockedLoss: mseLock,
    threshold: { loss: 0.02 },
    dataset: waveBatch,
    evaluationDataset: waveBatch,
    sequenceProbe: 'forecast',
  },
  [...preceding, ...temporalForwardLevels],
);

export const temporalLevels = [
  windowLimitLevel,
  ...temporalForwardLevels.slice(0, 2),
  temporalRnnDelayLevel,
  ...temporalForwardLevels.slice(2),
  temporalTrainLevel,
];
