import { gardenRhyme } from '../datasets';
import type { Attributes, GraphJSON } from '../engine/graph';
import type { Random } from '../engine/random';
import type { TrainLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels } from './chapter4';
import { chapter5ForwardLevels, chapter5TrainLevels } from './chapter5';
import { chapter6ForwardLevels } from './chapter6-forward';
import { corpusTokens, vocabularySize } from './corpus';
import { withPar } from './level-helpers';

export { chapter6ForwardLevels } from './chapter6-forward';
export const gptPreceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
  ...chapter5ForwardLevels,
  ...chapter6ForwardLevels,
];
export const gptConfig = { T: 16, D: 24, H: 48, stride: 1, N: 8 };
const { T: context, D, H, stride, N: batchSize } = gptConfig;
/** Every alignment, including the final next-character target. */
export const gptSequences = (() => {
  const starts: number[] = [];
  for (let start = 0; start + context < corpusTokens.length; start += stride) starts.push(start);
  const last = corpusTokens.length - context - 1;
  if (starts.at(-1) !== last) starts.push(last);
  return starts.map((start) => corpusTokens.slice(start, start + context + 1));
})();
/** 400 evenly spaced starts, including both corpus endpoints; no random evaluation sampling. */
export const gptEvaluationIndices = Array.from({ length: 400 }, (_, i) =>
  Math.floor((i * (gptSequences.length - 1)) / 399),
);
export function gptBatch(random?: Random) {
  const sequences = random
    ? Array.from(
        { length: batchSize },
        () => gptSequences[Math.floor(random.uniform() * gptSequences.length)],
      )
    : gptEvaluationIndices.map((index) => gptSequences[index]);
  const N = sequences.length;
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
function answer(): GraphJSON {
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
  const param = (id: string, shape: number[]) =>
    node(id, 'param', {}, { shape, init: 'normal', scale: 0.15 });
  let x = node('embedded', 'embed_pos', {
    idx: node('x', 'input', {}, { name: 'x' }),
    tokens: param('tokens', [vocabularySize, D]),
    positions: param('positions', [context, D]),
  });
  for (let i = 0; i < 2; i++) {
    const args: Record<string, string> = { x };
    for (const [name, shape] of Object.entries({
      wq: [D, D],
      wk: [D, D],
      wv: [D, D],
      wo: [D, D],
      w1: [D, H],
      b1: [H],
      w2: [H, D],
      b2: [D],
    }))
      args[name] = param(`block${i}_${name}`, shape);
    x = node(`block${i}`, 'transformer_block', args);
  }
  const normalized = node('normalized', 'layernorm', { x });
  const logits = node('logits', 'matmul', { a: normalized, b: param('head', [D, vocabularySize]) });
  node(
    'output',
    'output',
    { x: node('probability', 'softmax', { x: logits }) },
    { name: 'probabilities' },
  );
  return graph;
}
export const chapter6TrainLevels: TrainLevel[] = [
  withPar<TrainLevel>(
    {
      id: '6-4',
      chapter: '6',
      kind: 'train',
      generates: true,
      resample: true,
      titleKey: 'levels.6-4.title',
      descriptionKey: 'levels.6-4.description',
      hintsKeys: ['levels.6-4.hint1', 'levels.6-4.hint2'],
      available: chapter5TrainLevels[0].available,
      unlock: [],
      referenceGraph: answer(),
      interface: {
        inputs: { x: [batchSize, context] },
        output: 'probabilities',
        outputShape: [batchSize, context, vocabularySize],
      },
      seed: 6404,
      steps: 1500,
      evaluationEvery: 250,
      optimizer: { kind: 'adam', learningRate: 0.01 },
      tunable: { learningRate: { min: 0.001, max: 0.05, default: 0.01 }, optimizer: ['adam'] },
      threshold: { loss: 1.1 },
      dataset: gptBatch,
      data: { dataset: gardenRhyme, context },
      evaluationDataset: () => gptBatch(),
    },
    gptPreceding,
  ),
];
export const chapter6Levels = [...chapter6ForwardLevels, ...chapter6TrainLevels];
