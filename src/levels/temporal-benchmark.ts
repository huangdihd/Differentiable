import { createDefaultRegistry, type GraphEndpoint, type GraphJSON } from '../engine/graph';
import { evaluateLevel, graphMetrics, type TrainLevel } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';

export const temporalBenchmarkModels = ['fixed-window', 'rnn', 'rnn-attention'] as const;
export type TemporalBenchmarkModel = (typeof temporalBenchmarkModels)[number];

export const temporalBenchmarkDelays = [2, 6, 16, 32] as const;
const benchmarkSeeds = [811] as const;
const hiddenFeatures = 8;
const trainingSteps = 300;
const trainingBatchSize = 32;
const evaluationBatchSize = 128;
const learningRate = 0.02;

export interface TemporalBenchmarkRun {
  seed: number;
  mse: number;
}

export interface TemporalBenchmarkResult {
  model: TemporalBenchmarkModel;
  parameterCount: number;
  delays: Array<{
    delay: number;
    runs: TemporalBenchmarkRun[];
    meanMse: number;
  }>;
}

export interface TemporalBenchmarkArtifact {
  schemaVersion: 1;
  task: 'delayed-copy';
  models: readonly TemporalBenchmarkModel[];
  delays: readonly number[];
  seeds: readonly number[];
  training: {
    backend: 'cpu';
    optimizer: 'adam';
    learningRate: number;
    steps: number;
    batchSize: number;
    evaluationBatchSize: number;
    hiddenFeatures: number;
  };
  results: TemporalBenchmarkResult[];
  attentionProbe: {
    history: number[][];
    query: number[];
    weights: number[];
    context: number[];
  };
}

class BenchmarkGraphBuilder {
  readonly graph: GraphJSON = { version: 1, nodes: [], edges: [] };

  node(
    id: string,
    component: string,
    inputs: Record<string, GraphEndpoint> = {},
    attrs?: Record<string, unknown>,
  ): GraphEndpoint {
    this.graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, from] of Object.entries(inputs))
      this.graph.edges.push({ from, to: { node: id, port } });
    return { node: id, port: 'out' };
  }

  input(name: string): GraphEndpoint {
    return this.node(name, 'input', {}, { name });
  }

  constant(id: string, value: number | number[], shape: number[] = []): GraphEndpoint {
    return this.node(id, 'const', {}, { value, shape });
  }

  parameter(id: string, shape: number[], init: 'zeros' | 'normal' = 'normal'): GraphEndpoint {
    return this.node(
      id,
      'param',
      {},
      { shape, init, ...(init === 'normal' ? { scale: 0.15 } : {}) },
    );
  }

  output(name: string, value: GraphEndpoint): void {
    this.node(`output_${name}`, 'output', { x: value }, { name });
  }
}

function selectFinalState(
  builder: BenchmarkGraphBuilder,
  history: GraphEndpoint,
  sequenceLength: number,
): GraphEndpoint {
  const transposed = builder.node(
    'history_transposed',
    'transpose',
    { x: history },
    {
      axes: [-2, -1],
    },
  );
  const selector = builder.constant(
    'final_selector',
    Array.from({ length: sequenceLength }, (_, index) => (index === sequenceLength - 1 ? 1 : 0)),
    [sequenceLength, 1],
  );
  const selected = builder.node('selected_final', 'matmul', { a: transposed, b: selector });
  return builder.node('final_state', 'reshape', { x: selected }, { shape: [-1, hiddenFeatures] });
}

function attend(
  builder: BenchmarkGraphBuilder,
  history: GraphEndpoint,
  query: GraphEndpoint,
): GraphEndpoint {
  const queryRow = builder.node('query_row', 'unsqueeze', { x: query }, { axis: 1 });
  const transposed = builder.node(
    'attention_history_transposed',
    'transpose',
    { x: history },
    {
      axes: [-2, -1],
    },
  );
  const rawScores = builder.node('attention_raw_scores', 'matmul', {
    a: queryRow,
    b: transposed,
  });
  const scores = builder.node('attention_scores', 'mul', {
    a: rawScores,
    b: builder.constant('attention_scale', 1 / Math.sqrt(hiddenFeatures)),
  });
  const weights = builder.node('attention_weights', 'softmax', { x: scores });
  const attended = builder.node('attended_history', 'matmul', { a: weights, b: history });
  return builder.node(
    'attention_context',
    'reshape',
    { x: attended },
    {
      shape: [-1, hiddenFeatures],
    },
  );
}

export function temporalBenchmarkGraph(model: TemporalBenchmarkModel, delay: number): GraphJSON {
  const sequenceLength = delay + 1;
  const builder = new BenchmarkGraphBuilder();
  const x = builder.input('x');
  const target = builder.input('t');
  let features: GraphEndpoint;
  let featureCount: number;

  if (model === 'fixed-window') {
    const selector = Array.from({ length: sequenceLength * 2 }, (_, index) => {
      const row = Math.floor(index / 2);
      const column = index % 2;
      return row === sequenceLength - 2 + column ? 1 : 0;
    });
    features = builder.node('visible_suffix', 'matmul', {
      a: x,
      b: builder.constant('suffix_selector', selector, [sequenceLength, 2]),
    });
    featureCount = 2;
  } else {
    const history = builder.node('rnn_history', 'rnn_window', {
      x,
      wx: builder.parameter('wx', [1, hiddenFeatures]),
      wh: builder.parameter('wh', [hiddenFeatures, hiddenFeatures]),
      b: builder.parameter('b', [hiddenFeatures], 'zeros'),
    });
    const finalState = selectFinalState(builder, history, sequenceLength);
    if (model === 'rnn-attention') {
      const zeroBatch = builder.node('zero_query_batch', 'mul', {
        a: finalState,
        b: builder.constant('query_zero', 0),
      });
      const query = builder.node('attention_query', 'add', {
        a: zeroBatch,
        b: builder.parameter('query', [hiddenFeatures]),
      });
      features = attend(builder, history, query);
    } else {
      features = finalState;
    }
    featureCount = hiddenFeatures;
  }

  const predictionColumn = builder.node('prediction_column', 'matmul', {
    a: features,
    b: builder.parameter('output_weight', [featureCount, 1]),
  });
  const biased = builder.node('biased_prediction', 'add', {
    a: predictionColumn,
    b: builder.parameter('output_bias', [1], 'zeros'),
  });
  const prediction = builder.node('prediction', 'reshape', { x: biased }, { shape: [-1] });
  const negativeTarget = builder.node('negative_target', 'mul', {
    a: target,
    b: builder.constant('minus_one', -1),
  });
  const difference = builder.node('difference', 'add', { a: prediction, b: negativeTarget });
  const squared = builder.node('squared_error', 'mul', { a: difference, b: difference });
  const total = builder.node('error_total', 'sum', { x: squared }, { axes: [0] });
  const batchSize = builder.node('batch_size', 'size', { x: prediction }, { axis: 0 });
  const mean = builder.node('mean_error', 'mul', {
    a: total,
    b: builder.node('inverse_batch', 'recip', { x: batchSize }),
  });
  builder.output('loss', builder.node('loss', 'reshape', { x: mean }, { shape: [1] }));
  return builder.graph;
}

export function delayedCopyBenchmarkBatch(random: Random, delay: number, batchSize: number) {
  const sequenceLength = delay + 1;
  const inputs: number[] = [];
  const targets: number[] = [];
  for (let row = 0; row < batchSize; row++) {
    // The remembered event is visibly marked by amplitude; later distractors stay near zero.
    const remembered = 0.5 + random.uniform();
    targets.push(remembered);
    inputs.push(remembered);
    for (let step = 1; step < sequenceLength; step++) inputs.push(random.uniform() * 0.4 - 0.2);
  }
  return {
    inputs: {
      x: T.tensor(inputs, [batchSize, sequenceLength]),
      t: T.tensor(targets, [batchSize]),
    },
  };
}

function benchmarkLevel(model: TemporalBenchmarkModel, delay: number, seed: number): TrainLevel {
  const graph = temporalBenchmarkGraph(model, delay);
  const sequenceLength = delay + 1;
  return {
    id: `temporal-benchmark-${model}-${delay}-${seed}`,
    chapter: '5',
    kind: 'train',
    titleKey: '',
    descriptionKey: '',
    available: createDefaultRegistry()
      .list()
      .map((component) => component.name),
    unlock: [],
    referenceGraph: graph,
    interface: {
      inputs: { x: [trainingBatchSize, sequenceLength], t: [trainingBatchSize] },
      output: 'loss',
      outputShape: [1],
    },
    seed,
    steps: trainingSteps,
    optimizer: { kind: 'adam', learningRate },
    resample: true,
    evaluationEvery: trainingSteps,
    lossOutput: 'loss',
    threshold: { loss: Number.MAX_VALUE },
    dataset: (random) => delayedCopyBenchmarkBatch(random, delay, trainingBatchSize),
    evaluationDataset: (random) => delayedCopyBenchmarkBatch(random, delay, evaluationBatchSize),
  };
}

function fixedAttentionProbe(): TemporalBenchmarkArtifact['attentionProbe'] {
  const history = [
    [1, 0, 0, 0, 3],
    [0, 1, 0, 0, 8],
    [0, 0, 1, 0, 2],
    [0, 0, 0, 1, 6],
  ];
  const query = [0, 8, 0, 0, 0];
  const scores = history.map(
    (row) => row.reduce((sum, value, index) => sum + value * query[index], 0) / Math.sqrt(5),
  );
  const maximum = Math.max(...scores);
  const unnormalized = scores.map((score) => Math.exp(score - maximum));
  const total = unnormalized.reduce((sum, value) => sum + value, 0);
  const weights = unnormalized.map((value) => value / total);
  const context = history[0].map((_, feature) =>
    history.reduce((sum, row, index) => sum + weights[index] * row[feature], 0),
  );
  return { history, query, weights, context };
}

export async function runTemporalBenchmarkCase(
  model: TemporalBenchmarkModel,
  delay: number,
  seed: number,
): Promise<{ parameterCount: number; run: TemporalBenchmarkRun }> {
  const registry = createDefaultRegistry();
  const level = benchmarkLevel(model, delay, seed);
  const parameterCount = graphMetrics(level.referenceGraph!, registry).parameterCount;
  const result = await evaluateLevel(level, level.referenceGraph!, { registry });
  if (!result.training || result.errorKey)
    throw new Error(
      `temporal benchmark failed for ${model}, delay ${delay}: ${result.errorKey ?? 'missing report'}`,
    );
  return { parameterCount, run: { seed, mse: result.training.loss } };
}

export async function generateTemporalBenchmark(): Promise<TemporalBenchmarkArtifact> {
  const results: TemporalBenchmarkResult[] = [];

  for (const model of temporalBenchmarkModels) {
    const delays: TemporalBenchmarkResult['delays'] = [];
    let parameterCount = 0;
    for (const delay of temporalBenchmarkDelays) {
      const runs: TemporalBenchmarkRun[] = [];
      for (const seed of benchmarkSeeds) {
        const result = await runTemporalBenchmarkCase(model, delay, seed);
        parameterCount = result.parameterCount;
        runs.push(result.run);
      }
      delays.push({
        delay,
        runs,
        meanMse: runs.reduce((sum, run) => sum + run.mse, 0) / runs.length,
      });
    }
    results.push({ model, parameterCount, delays });
  }

  return {
    schemaVersion: 1,
    task: 'delayed-copy',
    models: temporalBenchmarkModels,
    delays: temporalBenchmarkDelays,
    seeds: benchmarkSeeds,
    training: {
      backend: 'cpu',
      optimizer: 'adam',
      learningRate,
      steps: trainingSteps,
      batchSize: trainingBatchSize,
      evaluationBatchSize,
      hiddenFeatures,
    },
    results,
    attentionProbe: fixedAttentionProbe(),
  };
}
