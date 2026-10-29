import { gardenRhyme, shuffledOrder } from '../datasets';
import { axisLabels, matrixAxes } from './output-axes';
import type { Attributes, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel, TrainLevel } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels, chapter4TrainLevels } from './chapter4';
import { temporalForwardLevels } from './chapter-temporal';
import { corpusTokens, vocabularySize } from './corpus';
import { withPar } from './level-helpers';

const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
  ...temporalForwardLevels,
];
const available = [...chapter4TrainLevels[0].available, 'arange'];
export const attentionConfig = { N: 128, T: 8, D: 12 };
const { N, T: context, D } = attentionConfig;
const outputNames = [
  'attentionScores',
  'maskedScores',
  'context',
  'embeddings',
  'probabilities',
] as const;
/** Fixed seeded corpus windows: both optimization and grading use this empirical distribution. */
export const attentionSequences = (() => {
  const random = new Random(5505);
  return Array.from({ length: N }, () => {
    const start = Math.floor(random.uniform() * (corpusTokens.length - context));
    return corpusTokens.slice(start, start + context + 1);
  });
})();
/** With `order`, the same windows shuffled: grading uses them so a [N,T,V] Parameter table
 * fitted to the training rows cannot pass without reading x.
 */
export function attentionBatch(order?: Random) {
  const sequences = order
    ? shuffledOrder(order, N).map((i) => attentionSequences[i])
    : attentionSequences;
  const targets = new Float32Array(N * context * vocabularySize);
  sequences.forEach((sequence, row) => {
    for (let i = 0; i < context; i++)
      targets[(row * context + i) * vocabularySize + sequence[i + 1]] = 1;
  });
  return {
    inputs: {
      x: T.tensor(
        sequences.flatMap((s) => s.slice(0, context)),
        [N, context],
      ),
    },
    targets: T.tensor(targets, [N, context, vocabularySize]),
  };
}
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
  const constant = (id: string, value: number) => node(id, 'const', {}, { value });
  const param = (id: string, shape: number[]) =>
    node(id, 'param', {}, { shape, init: 'normal', scale: 0.15 });
  let result: string;
  if (index === 1) {
    const q = input('q'),
      k = input('k');
    const width = node('width', 'size', { x: q }, { axis: -1 });
    result = node('scaled', 'div', {
      a: node('scores', 'matmul', {
        a: q,
        b: node('transposed', 'transpose', { x: k }, { axes: [-2, -1] }),
      }),
      b: node('root', 'sqrt', { x: width }),
    });
  } else if (index === 2) {
    const scores = input('scores');
    const range = node('range', 'arange', {
      n: node('length', 'size', { x: scores }, { axis: -1 }),
    });
    const mask = node('mask', 'ge', {
      a: node('rows', 'unsqueeze', { x: range }, { axis: -1 }),
      b: node('columns', 'unsqueeze', { x: range }, { axis: 0 }),
    });
    result = node('masked', 'add', {
      a: node('kept', 'mul', { a: scores, b: mask }),
      b: node('blocked', 'mul', {
        a: node('negative', 'add', { a: mask, b: constant('minusOne', -1) }),
        b: constant('large', 1e9),
      }),
    });
  } else if (index === 3) {
    const q = input('q'),
      k = input('k'),
      v = input('v');
    result = node('attended', 'matmul', {
      a: node('weights', 'softmax', {
        x: node('masked', 'causal_mask', { scores: node('scores', 'attn_scores', { q, k }) }),
      }),
      b: v,
    });
  } else if (index === 4) {
    const idx = input('idx'),
      tokens = input('tokens'),
      positions = input('positions');
    const range = node('range', 'arange', { n: node('length', 'size', { x: idx }, { axis: -1 }) });
    result = node('embedded', 'add', {
      a: node('tokenVectors', 'gather', { table: tokens, idx }),
      b: node('positionVectors', 'gather', { table: positions, idx: range }),
    });
  } else {
    const x = input('x');
    const embedded = node('embedded', 'embed_pos', {
      idx: x,
      tokens: param('tokens', [vocabularySize, D]),
      positions: param('positions', [context, D]),
    });
    const project = (id: string) =>
      node(id, 'matmul', { a: embedded, b: param(`${id}Weight`, [D, D]) });
    const attended = node('attended', 'attention', {
      q: project('q'),
      k: project('k'),
      v: project('v'),
    });
    result = node('probability', 'softmax', {
      x: node('logits', 'matmul', { a: attended, b: param('head', [D, vocabularySize]) }),
    });
  }
  node('output', 'output', { x: result }, { name: outputNames[index - 1] });
  return graph;
}
const names = ['attn_scores', 'causal_mask', 'attention', 'embed_pos'];
const shapes: ShapePorts[][] = [
  [
    { q: [4, 3], k: [4, 3] },
    { q: [6, 5], k: [6, 5] },
    { q: [2, 3, 4], k: [2, 3, 4] },
  ],
  [{ scores: [4, 4] }, { scores: [6, 6] }, { scores: [2, 3, 3] }],
  [
    { q: [4, 3], k: [4, 3], v: [4, 3] },
    { q: [6, 5], k: [6, 5], v: [6, 2] },
    { q: [2, 3, 4], k: [2, 3, 4], v: [2, 3, 5] },
  ],
  [
    { idx: [4], tokens: [7, 3], positions: [8, 3] },
    { idx: [6], tokens: [9, 5], positions: [10, 5] },
    { idx: [2, 3], tokens: [7, 4], positions: [6, 4] },
  ],
];
function fixtures(index: number, inputs: ShapePorts) {
  const make = (random: Random, i: number) => ({
    id: `case-${i}`,
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([name, shape]) => [
        name,
        Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, j) =>
          name === 'idx' ? (j + i) % inputs.tokens[0] : i === 0 ? 0 : random.uniform() * 4 - 2,
        ),
      ]),
    ),
  });
  return {
    // All-zero case-0 goes last: the probe panel should open on real numbers.
    fixed: [1, 2, 3, 4, 0].map((i) => make(new Random(5100 + i), i)),
    generated: { seed: 5500 + index, count: 5, generate: make },
  };
}
function scores(q: T.Tensor, k: T.Tensor) {
  const perm = k.shape.map((_, i) => i);
  [perm[perm.length - 2], perm[perm.length - 1]] = [perm[perm.length - 1], perm[perm.length - 2]];
  return T.mul(
    T.matmul(q, T.transpose(k, perm)),
    T.tensor([1 / Math.sqrt(q.shape.at(-1)!)], [], q.backend),
  );
}
function masked(x: T.Tensor) {
  const n = x.shape.at(-1)!;
  const m = T.tensor(
    Array.from({ length: n * n }, (_, i) => (i % n <= Math.floor(i / n) ? 1 : 0)),
    [n, n],
    x.backend,
  );
  return T.add(
    T.mul(x, m),
    T.mul(T.add(m, T.tensor([-1], [], x.backend)), T.tensor([1e9], [], x.backend)),
  );
}
const metadata = (index: number) => ({
  id: `5-${index}`,
  chapter: '6',
  titleKey: `levels.5-${index}.title`,
  descriptionKey: `levels.5-${index}.description`,
  hintsKeys: [`levels.5-${index}.hint1`, `levels.5-${index}.hint2`],
  available,
  unlock: index === 5 ? ['6-1'] : [`5-${index + 1}`],
  referenceGraph: answer(index),
});
export const chapter5ForwardLevels: ForwardLevel[] = [];
for (let i = 0; i < 4; i++) {
  const inputs = shapes[i][0];
  chapter5ForwardLevels.push(
    withPar<ForwardLevel>(
      {
        ...metadata(i + 1),
        kind: 'forward',
        unlockComponents: [names[i]],
        forbidden: [names[i]],
        outputNoteKey: `levels.5-${i + 1}.expectedNote`,
        outputAxes: matrixAxes(shapes[i], (args, inputs) => {
          if (i === 3)
            return inputs.idx.length === 1
              ? { rows: args.idx.map((id, position) => `p${position}:${id}`) }
              : {};
          const shape = i === 1 ? inputs.scores : inputs.q;
          if (shape.length !== 2) return {};
          return {
            rows: axisLabels('q', shape[0]),
            ...(i < 2 && { cols: axisLabels('k', i === 1 ? shape[1] : inputs.k[0]) }),
          };
        }),
        interface: {
          inputs,
          output: outputNames[i],
          outputShape: [
            [4, 4],
            [4, 4],
            [4, 3],
            [4, 3],
          ][i],
        },
        ...fixtures(i, inputs),
        shapeVariants: shapes[i].slice(1).map((inputs) => ({ inputs, ...fixtures(i, inputs) })),
        // Absolute tolerance keeps visible entries strict; masked entries must be exactly -1e9 in f32.
        atol: 1e-4,
        rtol: i === 1 ? 0 : 1e-4,
        reference: (args) => {
          if (i === 0) return scores(args.q, args.k);
          if (i === 1) return masked(args.scores);
          if (i === 2) return T.matmul(T.softmax(masked(scores(args.q, args.k))), args.v);
          return T.add(
            T.gather(args.tokens, args.idx),
            T.gather(args.positions, T.arange(args.idx.shape.at(-1)!, args.idx.backend)),
          );
        },
      },
      [...preceding, ...chapter5ForwardLevels],
    ),
  );
}
export const chapter5TrainLevels: TrainLevel[] = [
  withPar<TrainLevel>(
    {
      ...metadata(5),
      kind: 'train',
      generates: true,
      interface: {
        inputs: { x: [N, context] },
        output: outputNames[4],
        outputShape: [N, context, vocabularySize],
      },
      seed: 5505,
      steps: 200,
      optimizer: { kind: 'adam', learningRate: 0.03 },
      tunable: { learningRate: { min: 0.001, max: 0.1, default: 0.03 }, optimizer: ['adam'] },
      threshold: { loss: 1.75 },
      dataset: () => attentionBatch(),
      data: { dataset: gardenRhyme, context },
      evaluationDataset: (random) => attentionBatch(random),
    },
    [...preceding, ...chapter5ForwardLevels],
  ),
];
export const chapter5Levels = [...chapter5ForwardLevels, ...chapter5TrainLevels];
