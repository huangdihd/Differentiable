import { expect, it } from 'vitest';
import { compileGraph, type Attributes, type GraphJSON, type ShapePorts } from '../engine/graph';
import { evaluateLevel, graphMetrics, type CircuitLevel } from '../engine/levels';
import { Random } from '../engine/random';
import { tensor } from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels } from './chapter4';
import {
  rnnWindowAnswer,
  temporalForwardLevels,
  temporalLevels,
  temporalRnnDelayLevel,
  temporalTrainLevel,
  windowLimitLevel,
} from './chapter-temporal';
import { referenceRegistry } from './level-helpers';

const preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
];
const registry = referenceRegistry([...preceding, ...temporalForwardLevels]);
const earned = [...preceding, ...temporalForwardLevels].flatMap(
  (level) => level.unlockComponents ?? [],
);
const playable = <Level extends CircuitLevel>(level: Level): Level => ({
  ...level,
  available: [...level.available, ...earned, ...(level.presets ?? [])],
});

for (const level of temporalForwardLevels) {
  it(`${level.id} passes primary, generated and shape-variant fixtures`, async () => {
    const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
    expect(result.passed, JSON.stringify(result)).toBe(true);
    expect(result.tests).toHaveLength(16);
    expect(level.shapeVariants).toHaveLength(1);
    expect(level.par).toBe(graphMetrics(level.referenceGraph!, registry).expandedPrimitives);

    const wrong = structuredClone(level.referenceGraph!);
    const output = wrong.nodes.find((node) => node.component === 'output')!;
    wrong.edges.find((edge) => edge.to.node === output.id)!.from = {
      node: Object.keys(level.interface.inputs)[0],
      port: 'out',
    };
    expect((await evaluateLevel(playable(level), wrong, { registry })).passed).toBe(false);
  });
}

it('keeps the seven-level RNN causal order and exact unlock chain', () => {
  expect(temporalLevels.map((level) => level.id)).toEqual([
    'time-window-limit',
    'time-rnn',
    'time-rnn-window',
    'time-rnn-delay',
    'time-recall-bottleneck',
    'time-attention',
    'time-forecast',
  ]);
  expect(temporalLevels.map((level) => level.chapter)).toEqual(Array(7).fill('5'));
  expect(temporalLevels.map((level) => level.unlock)).toEqual([
    ['time-rnn'],
    ['time-rnn-window'],
    ['time-rnn-delay'],
    ['time-recall-bottleneck'],
    ['time-attention'],
    ['time-forecast'],
    ['5-1'],
  ]);
  expect(temporalForwardLevels.flatMap((level) => level.unlockComponents ?? [])).toEqual([
    'rnn_cell',
    'rnn_window',
    'temporal_attention',
  ]);
});

it('starts chapter 5 with an observation page without a circuit, score or fixtures', () => {
  expect(temporalLevels[0]).toBe(windowLimitLevel);
  expect(windowLimitLevel).toEqual({
    id: 'time-window-limit',
    chapter: '5',
    kind: 'observation',
    titleKey: 'levels.time-window-limit.title',
    descriptionKey: 'levels.time-window-limit.description',
    unlock: ['time-rnn'],
  });
  expect(temporalForwardLevels.some((level) => level.id === windowLimitLevel.id)).toBe(false);
});

it('time-rnn-window unrolls six steps with one shared parameter set', () => {
  const graph = rnnWindowAnswer();
  const cells = graph.nodes.filter((node) => node.component === 'rnn_cell');
  expect(cells).toHaveLength(6);
  for (const port of ['wx', 'wh', 'b']) {
    const sources = cells.map(
      (cell) => graph.edges.find((edge) => edge.to.node === cell.id && edge.to.port === port)!.from,
    );
    expect(sources).toEqual(Array(6).fill({ node: port, port: 'out' }));
  }
});

it('trains the short RNN delayed-copy task and grades a true held-out batch', async () => {
  expect(temporalRnnDelayLevel.presets).toEqual(['rnn_window']);
  expect(temporalRnnDelayLevel.componentDefaults?.rnn_window).toEqual({
    automatic_parameters: true,
    input_features: 1,
    hidden_features: 8,
  });
  expect(temporalRnnDelayLevel.evaluationDataset).toBeTypeOf('function');
  expect(graphMetrics(temporalRnnDelayLevel.referenceGraph!, registry).parameterCount).toBe(89);

  const training = temporalRnnDelayLevel.dataset(new Random(temporalRnnDelayLevel.seed));
  const evaluation = temporalRnnDelayLevel.evaluationDataset!(
    new Random(temporalRnnDelayLevel.seed + 1),
  );
  try {
    expect(await training.inputs.x.read()).not.toEqual(await evaluation.inputs.x.read());
  } finally {
    for (const value of [
      ...Object.values(training.inputs),
      training.targets,
      ...Object.values(evaluation.inputs),
      evaluation.targets,
    ])
      value?.dispose();
  }

  const result = await evaluateLevel(
    playable(temporalRnnDelayLevel),
    temporalRnnDelayLevel.referenceGraph!,
    { registry },
  );
  expect(result.errorKey, JSON.stringify(result)).toBeUndefined();
  expect(result.passed, JSON.stringify(result.training)).toBe(true);
  expect(result.training!.loss).toBeLessThan(temporalRnnDelayLevel.threshold.loss);
}, 60000);

it('the recall bottleneck fixture pairs different histories with one final state', () => {
  const level = temporalForwardLevels.find(
    (candidate) => candidate.id === 'time-recall-bottleneck',
  )!;
  const fixture = level.fixed.find((candidate) => candidate.id === 'same-final-different-history')!;
  const [batch, length, width] = level.interface.inputs.history;
  expect(batch).toBeGreaterThanOrEqual(2);
  const history = fixture.inputs.history;
  const row = (index: number) =>
    history.slice(index * length * width, (index + 1) * length * width);
  const final = (index: number) => row(index).slice((length - 1) * width);
  expect(row(0).slice(0, (length - 1) * width)).not.toEqual(row(1).slice(0, (length - 1) * width));
  expect(final(0)).toEqual(final(1));
});

it('attention takes an explicit query and forecast uses the final RNN state to look back', () => {
  const attention = temporalForwardLevels.find((level) => level.id === 'time-attention')!;
  expect(attention.interface.inputs).toEqual({ history: [3, 5, 4], query: [3, 4] });
  expect(attention.revalidateOnLoad).toBe(true);
  expect(attention.unlockComponents).toEqual(['temporal_attention']);
  expect(
    attention
      .referenceGraph!.nodes.filter((node) => node.component === 'input')
      .map((node) => node.attrs?.name),
  ).toEqual(['history', 'query']);

  const graph = temporalTrainLevel.referenceGraph!;
  const attentionNode = graph.nodes.find((node) => node.component === 'temporal_attention')!;
  expect(
    Object.fromEntries(
      graph.edges
        .filter((edge) => edge.to.node === attentionNode.id)
        .map((edge) => [edge.to.port, edge.from]),
    ),
  ).toEqual({
    history: { node: 'history', port: 'out' },
    query: { node: 'query', port: 'out' },
  });
  expect(graph.nodes.find((node) => node.id === 'query')?.component).toBe('sum');
  expect(graph.nodes.some((node) => node.id === 'query_parameter')).toBe(false);
});

const recurrentModuleFixtures: {
  component: string;
  inputs: ShapePorts;
  outputShape: number[];
  attrs: Attributes;
  parameters: ShapePorts;
}[] = [
  {
    component: 'rnn_cell',
    inputs: { x: [2, 3], h: [2, 5] },
    outputShape: [2, 5],
    attrs: { input_features: 3, hidden_features: 5 },
    parameters: { wx: [3, 5], wh: [5, 5], b: [5] },
  },
  {
    component: 'rnn_window',
    inputs: { x: [2, 6] },
    outputShape: [2, 6, 5],
    attrs: { input_features: 1, hidden_features: 5 },
    parameters: { wx: [1, 5], wh: [5, 5], b: [5] },
  },
];

it.each(recurrentModuleFixtures)(
  'executes automatic $component parameters at configured widths',
  (fixture) => {
    const inputNames = Object.keys(fixture.inputs);
    const model: GraphJSON = {
      version: 1,
      nodes: [
        ...inputNames.map((name) => ({ id: name, component: 'input', attrs: { name } })),
        {
          id: 'module',
          component: fixture.component,
          attrs: { automatic_parameters: true, ...fixture.attrs },
        },
        { id: 'output', component: 'output', attrs: { name: 'y' } },
      ],
      edges: [
        ...inputNames.map((name) => ({
          from: { node: name, port: 'out' },
          to: { node: 'module', port: name },
        })),
        { from: { node: 'module', port: 'out' }, to: { node: 'output', port: 'x' } },
      ],
    };
    const plan = compileGraph(model, {
      registry,
      inputShapes: fixture.inputs,
      random: new Random(15),
    });
    const inputs = Object.fromEntries(
      Object.entries(fixture.inputs).map(([name, shape]) => [
        name,
        tensor(new Float32Array(shape.reduce((size, dimension) => size * dimension, 1)), shape),
      ]),
    );
    try {
      expect(plan.outputShapes.y).toEqual(fixture.outputShape);
      const outputs = plan.execute(inputs);
      try {
        for (const [name, shape] of Object.entries(fixture.parameters))
          expect(plan.parameters.get(`module/${name}`)?.shape).toEqual(shape);
      } finally {
        plan.releaseExecution(outputs);
      }
    } finally {
      plan.dispose();
      Object.values(inputs).forEach((value) => value.dispose());
    }
  },
);

it('the RNN-attention forecast reaches its held-out threshold with 89 parameters', async () => {
  const level = playable(temporalTrainLevel);
  expect(level.evaluationDataset).toBeTypeOf('function');
  expect(graphMetrics(level.referenceGraph!, registry).parameterCount).toBe(89);
  const result = await evaluateLevel(level, level.referenceGraph!, { registry });
  expect(result.errorKey, JSON.stringify(result)).toBeUndefined();
  expect(result.passed, JSON.stringify(result.training)).toBe(true);
  expect(result.training!.loss).toBeLessThan(level.threshold.loss);
}, 60000);
