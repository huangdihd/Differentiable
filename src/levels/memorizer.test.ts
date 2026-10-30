import { TRAIN_DEPENDENCY_ERROR } from '../engine/train-dependency';
import { expect, it } from 'vitest';
import type { GraphJSON } from '../engine/graph';
import { evaluateLevel, type ForwardLevel, type TrainLevel } from '../engine/levels';
import { Random } from '../engine/random';
import { gameLevels } from './catalog';
import { referenceRegistry } from './level-helpers';

// A Parameter shaped like the batch output can fit every training row without reading x.
// Grading must use a held-out or reordered batch so that lookup table fails.
const forwards = gameLevels.filter(
  (l): l is ForwardLevel => l.kind === 'forward' || l.kind === 'derivative',
);
const registry = referenceRegistry(forwards);
const earned = forwards.flatMap((l) => l.unlockComponents ?? []);
const trainLevels = gameLevels.filter((l): l is TrainLevel => l.kind === 'train');
const playable = (level: TrainLevel) => ({
  ...level,
  available: [...level.available, ...earned, 'mse'],
});
const edge = (from: string, to: string, port: string) => ({
  from: { node: from, port: 'out' },
  to: { node: to, port },
});

function memorizer(level: TrainLevel): GraphJSON {
  if (level.lossOutput)
    return {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x' } },
        { id: 't', component: 'input', attrs: { name: 't' } },
        {
          id: 'table',
          component: 'param',
          attrs: { shape: [...level.interface.inputs.t], init: 'zeros' },
        },
        { id: 'error', component: 'mse' },
        { id: 'output', component: 'output', attrs: { name: 'loss' } },
      ],
      edges: [edge('table', 'error', 'p'), edge('t', 'error', 't'), edge('error', 'output', 'x')],
    };
  return {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      {
        id: 'table',
        component: 'param',
        attrs: { shape: [...level.interface.outputShape], init: 'zeros' },
      },
      { id: 'probability', component: 'softmax' },
      { id: 'output', component: 'output', attrs: { name: level.interface.output } },
    ],
    edges: [edge('table', 'probability', 'x'), edge('probability', 'output', 'x')],
  };
}

// xor (3-13): its 4 rows are the whole input domain and the table reshuffles per seed, but it
// has no separate grading batch; left as a truth-table lesson. 6-4 resamples minibatches.
const guarded = [
  '3-4',
  '3-6',
  '3-8',
  '3-9',
  '4-3',
  '4-5',
  'time-rnn-delay',
  'time-forecast',
  '5-5',
];

it('every fixed-batch training level grades on a separate batch, except the xor truth table', () => {
  const fixed = trainLevels.filter((l) => !l.resample).map((l) => l.id);
  expect(fixed.filter((id) => id !== 'xor').sort()).toEqual([...guarded].sort());
  for (const id of guarded) {
    const level = trainLevels.find((l) => l.id === id)!;
    expect(level.evaluationDataset, id).toBeTypeOf('function');
  }
});

it.each(guarded)('%s graded batch reorders or replaces the training rows', async (id) => {
  const level = trainLevels.find((l) => l.id === id)!;
  const train = level.dataset(new Random(level.seed));
  const graded = level.evaluationDataset!(new Random(level.seed + 1));
  try {
    const tensors = (b: typeof train) => [
      ...Object.values(b.inputs),
      ...(b.targets ? [b.targets] : []),
    ];
    const a = await Promise.all(tensors(train).map((t) => t.read()));
    const b = await Promise.all(tensors(graded).map((t) => t.read()));
    expect(b.map((v) => v.length)).toEqual(a.map((v) => v.length));
    expect(a.some((values, i) => values.some((v, j) => v !== b[i][j]))).toBe(true);
  } finally {
    for (const batch of [train, graded]) {
      Object.values(batch.inputs).forEach((t) => t.dispose());
      batch.targets?.dispose();
    }
  }
});

it.each(guarded)(
  '%s rejects a batch-shaped Parameter memorizer that ignores x',
  async (id) => {
    const level = trainLevels.find((l) => l.id === id)!;
    const graph = memorizer(level);
    // The structural anti-cheat already rejects it: the prediction never reads x.
    const flagged = await evaluateLevel(playable(level), graph, { registry });
    expect(flagged.passed).toBe(false);
    expect(flagged.errorKey ?? flagged.error?.key).toBe(TRAIN_DEPENDENCY_ERROR);
    // Without that check, the held-out evaluation batch must still catch it.
    const options = {
      registry,
      learningRate: level.lossOutput ? 0.5 : 0.1,
      optimizer: 'adam' as const,
      checkDependency: false,
    };
    // Control: on the old same-batch grading the memorizer passes, so the test bites.
    const ungraded = await evaluateLevel(
      playable({ ...level, evaluationDataset: undefined }),
      graph,
      options,
    );
    // 3-6 is SGD-only for 80 steps; the table never reached the line there even before.
    if (id !== '3-6') expect(ungraded.passed, `${id} control`).toBe(true);
    const result = await evaluateLevel(playable(level), graph, options);
    expect(result.passed, JSON.stringify(result.training)).toBe(false);
    expect(result.training!.loss).toBeGreaterThan(level.threshold.loss);
  },
  60000,
);
