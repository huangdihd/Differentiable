import type { Attributes, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel, TrainLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import {
  bigramEntropy,
  contextBatch,
  contextBigramEntropy,
  contextEntropy,
  corpusPairs,
  corpusStats,
  vocabularySize,
} from './corpus';
import { Random } from '../engine/random';
import { scalarPrimitives, withPar } from './level-helpers';

const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
];
const available = [
  ...scalarPrimitives,
  'max',
  'sum',
  'reduce_max',
  'unsqueeze',
  'transpose',
  'reshape',
  'size',
  'ge',
];
const N = 512,
  D = 8,
  H = 24;
function answer(index: number): GraphJSON {
  const graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  const node = (
    id: string,
    component: string,
    args: Record<string, string> = {},
    attrs?: Attributes,
  ) => {
    graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, source] of Object.entries(args))
      graph.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  };
  const input = (name: string) => node(name, 'input', {}, { name });
  const param = (name: string, shape: number[]) =>
    node(name, 'param', {}, { shape, init: 'normal', scale: 0.15 });
  let result: string;
  if (index === 1) {
    const idx = node('column', 'unsqueeze', { x: input('idx') }, { axis: -1 });
    const k = input('k');
    result = node('equal', 'mul', {
      a: node('left', 'ge', { a: idx, b: k }),
      b: node('right', 'ge', { a: k, b: idx }),
    });
  } else if (index === 2) {
    const idx = input('idx'),
      table = input('table');
    const k = node('classes', 'const', {}, { value: [0, 1, 2, 3, 4], shape: [5] });
    result = node('lookup', 'matmul', { a: node('encoded', 'one_hot', { idx, k }), b: table });
  } else if (index === 4) {
    const p = input('p'),
      u = input('u'),
      triangle = input('triangle');
    const column = node('column', 'unsqueeze', { x: p }, { axis: -1 });
    const cumulative = node(
      'cumulative',
      'sum',
      {
        x: node('weighted', 'mul', { a: column, b: triangle }),
      },
      { axes: [0] },
    );
    result = node(
      'index',
      'sum',
      { x: node('compare', 'ge', { a: u, b: cumulative }) },
      { axes: [-1] },
    );
  } else {
    const x = input('x');
    if (index === 3) {
      result = node('logits', 'gather', {
        table: param('table', [vocabularySize, vocabularySize]),
        idx: x,
      });
    } else {
      const lookup = node('lookup', 'gather', {
        table: param('embedding', [vocabularySize, D]),
        idx: x,
      });
      const flat = node('flatten', 'reshape', { x: lookup }, { shape: [-1, 2 * D] });
      const hidden = node('hidden', 'tanh', {
        x: node('affine', 'add', {
          a: node('product', 'matmul', { a: flat, b: param('w1', [2 * D, H]) }),
          b: param('b1', [H]),
        }),
      });
      result = node('logits', 'add', {
        a: node('projection', 'matmul', { a: hidden, b: param('w2', [H, vocabularySize]) }),
        b: param('b2', [vocabularySize]),
      });
    }
    result = node('probability', 'softmax', { x: result });
  }
  node('output', 'output', { x: result }, { name: 'y' });
  return graph;
}
const metadata = (index: number) => ({
  id: `4-${index}`,
  chapter: '4',
  titleKey: `levels.4-${index}.title`,
  descriptionKey: `levels.4-${index}.description`,
  hintsKeys: [`levels.4-${index}.hint1`, `levels.4-${index}.hint2`],
  available,
  unlock: index === 5 ? [] : [`4-${index + 1}`],
  referenceGraph: answer(index),
});
function forwardCases(index: number, inputs: ShapePorts) {
  const make = (random: Random, i: number) => {
    const values: Record<string, number[]> = {};
    if (index === 4) {
      const n = inputs.p[0];
      const weights = Array.from({ length: n }, () => 0.2 + random.uniform());
      const total = weights.reduce((a, b) => a + b, 0);
      values.p = weights.map((v) => v / total);
      // Deliberately inside a bin, never exactly on a cumulative boundary.
      const bin = i % n;
      values.u = [values.p.slice(0, bin).reduce((a, b) => a + b, 0) + values.p[bin] * 0.37];
      values.triangle = Array.from({ length: n * n }, (_, j) =>
        Math.floor(j / n) <= j % n ? 1 : 0,
      );
    } else {
      const classes = inputs.k?.[0] ?? inputs.table[0];
      values.idx = Array.from({ length: inputs.idx[0] }, (_, j) => (i + j * 2) % classes);
      if (index === 1) values.k = Array.from({ length: classes }, (_, j) => j);
      else values.table = Array.from({ length: classes * 3 }, () => random.uniform() * 8 - 4);
    }
    return { id: `case-${i}`, inputs: values };
  };
  return {
    fixed: Array.from({ length: 8 }, (_, i) => make(new Random(400 + i), i)),
    generated: { seed: 4400 + index, count: 8, generate: make },
  };
}
export const chapter4ForwardLevels: ForwardLevel[] = [];
for (const index of [1, 2, 4]) {
  const inputs: ShapePorts =
    index === 1
      ? { idx: [4], k: [5] }
      : index === 2
        ? { idx: [4], table: [5, 3] }
        : { p: [5], u: [], triangle: [5, 5] };
  const variant: ShapePorts | undefined =
    index === 1
      ? { idx: [6], k: [7] }
      : index === 4
        ? { p: [8], u: [], triangle: [8, 8] }
        : undefined;
  chapter4ForwardLevels.push(
    withPar<ForwardLevel>(
      {
        ...metadata(index),
        kind: 'forward',
        unlockComponents:
          index === 1 ? ['one_hot'] : index === 2 ? ['embedding', 'gather'] : ['sample'],
        forbidden: index === 1 ? ['one_hot'] : index === 2 ? ['embedding', 'gather'] : ['sample'],
        interface: {
          inputs,
          output: 'y',
          outputShape: index === 1 ? [4, 5] : index === 2 ? [4, 3] : [],
        },
        ...forwardCases(index, inputs),
        shapeVariants: variant ? [{ inputs: variant, ...forwardCases(index, variant) }] : [],
        reference: (args) => {
          if (index === 2) return T.gather(args.table, args.idx);
          if (index === 4) {
            const cumulative = T.sum(
              T.mul(T.reshape(args.p, [args.p.shape[0], 1]), args.triangle),
              [0],
            );
            return T.sum(T.ge(args.u, cumulative), [-1]);
          }
          const column = T.reshape(args.idx, [args.idx.shape[0], 1]);
          return T.mul(T.ge(column, args.k), T.ge(args.k, column));
        },
      },
      [...preceding, ...chapter4ForwardLevels],
    ),
  );
}
export const chapter4TrainLevels: (TrainLevel & { generates: true })[] = [3, 5].map((index) =>
  withPar(
    {
      ...metadata(index),
      kind: 'train' as const,
      generates: true as const,
      available: [...available, 'gather', 'param'],
      interface: {
        inputs: { x: index === 3 ? [corpusStats.bigrams] : [N, 2] },
        output: 'y',
        outputShape: [index === 3 ? corpusStats.bigrams : N, vocabularySize],
      },
      seed: 4100 + index,
      steps: index === 3 ? 180 : 160,
      optimizer: { kind: 'adam' as const, learningRate: 0.03 },
      tunable: {
        learningRate: { min: 0.001, max: 0.1, default: 0.03 },
        optimizer: ['adam', 'sgd'] as const,
      },
      threshold: { loss: index === 3 ? bigramEntropy + 0.15 : contextBigramEntropy - 0.15 },
      dataset: index === 3 ? corpusPairs : contextBatch,
      evaluationDataset: index === 3 ? corpusPairs : contextBatch,
    },
    [...preceding, ...chapter4ForwardLevels.filter((level) => Number(level.id.slice(2)) < index)],
  ),
);
export const chapter4Levels = [
  chapter4ForwardLevels[0],
  chapter4ForwardLevels[1],
  chapter4TrainLevels[0],
  chapter4ForwardLevels[2],
  chapter4TrainLevels[1],
];
/** Static calibration inputs; measured losses and generated text are emitted by tests. */
export const chapter4Stats = chapter4Levels.map((level) => ({
  id: level.id,
  par: level.par,
  ...(level.kind === 'train'
    ? {
        seed: level.seed,
        steps: level.steps,
        threshold: level.threshold,
        learningRate: level.tunable!.learningRate,
        batchSize: level.interface.inputs.x[0],
        vocabularySize,
        corpusLength: corpusStats.characters,
        bigramEntropy,
        contextBigramEntropy,
        contextEntropy,
        embeddingWidth: D,
        hiddenWidth: H,
      }
    : {}),
}));
