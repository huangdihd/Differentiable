import { expect, it } from 'vitest';
import type { GraphJSON } from '../engine/graph';
import {
  applyResult,
  initialProgress,
  evaluateLevel,
  type CircuitLevel,
  type ForwardLevel,
} from '../engine/levels';
import { gameLevels } from './catalog';
import { chapter3ClassificationLevels, chapter3ForwardLevels } from './chapter3';
import { referenceRegistry } from './level-helpers';
import { availableComponents, buildPlayerRegistry } from './registry';

const circuitLevels = gameLevels.filter((level) => level.kind !== 'observation');
it('earns the player cross_entropy subgraph with p/t ports', async () => {
  let progress = initialProgress(gameLevels.map((l) => l.id));
  for (const level of gameLevels.filter(
    (l): l is ForwardLevel => l.kind === 'forward' || l.kind === 'derivative',
  )) {
    if (level.id === '3-7') break;
    const result = await evaluateLevel(
      { ...level, available: availableComponents(level, progress) },
      level.referenceGraph!,
      { registry: buildPlayerRegistry(gameLevels, progress) },
    );
    expect(result.passed, level.id).toBe(true);
    progress = applyResult(progress, level, result, level.referenceGraph!);
  }
  const component = buildPlayerRegistry(gameLevels, progress).get('cross_entropy')!;
  expect(component.inputs).toEqual(['p', 't']);
  expect(component.subgraph).toBeDefined();
  expect(component.native).toBeUndefined();
});
it('classification worker traces retain three steps, accuracy and output loss gradients', async () => {
  const { WorkerRuntime } = await import('../worker/runtime');
  const messages: import('../worker/protocol').WorkerResponse[] = [];
  const progress = initialProgress(
    gameLevels.map((l) => l.id),
    gameLevels.flatMap((l) => l.unlockComponents ?? []),
  );
  progress.passed = gameLevels.map((l) => l.id);
  progress.passedSolutions = Object.fromEntries(
    circuitLevels.map((l) => [l.id, l.referenceGraph!]),
  );
  const runtime = new WorkerRuntime((m) => messages.push(m), gameLevels, progress);
  let requestId = 0;
  try {
    await runtime.handle({ type: 'init', backend: 'cpu', revision: 0, requestId: ++requestId });
    for (const level of [
      ...chapter3ClassificationLevels,
      circuitLevels.find((level) => level.id === 'xor')!,
    ]) {
      let last = Infinity;
      for (let i = 1; i <= 3; i++) {
        await runtime.handle({
          type: 'trainStep',
          levelId: level.id,
          graph: level.referenceGraph!,
          learningRate: 0.001,
          optimizer: level.id === '3-8' ? 'sgd' : 'adam',
          revision: 0,
          requestId: ++requestId,
        });
        const reply = messages.at(-1)!;
        expect(reply.type).toBe('reply');
        if (reply.type !== 'reply') throw new Error(JSON.stringify(reply));
        const trace = reply.data as import('../worker/protocol').ReplyData['trainStep'];
        expect(trace.step).toBe(i);
        expect(trace.accuracy).toBeTypeOf('number');
        expect(trace.gradient.output.values.some((v) => v !== 0)).toBe(true);
        // Adam need not decrease monotonically on every early step (notably fixed-rate XOR).
        // The simple two-cluster lesson is the three-step descent regression fixture.
        if (level.id === '3-6') expect(trace.loss).toBeLessThan(last);
        expect(Number.isFinite(trace.loss)).toBe(true);
        last = trace.loss;
      }
    }
  } finally {
    await runtime.dispose();
  }
});
const forwards = gameLevels.filter(
  (l): l is ForwardLevel => l.kind === 'forward' || l.kind === 'derivative',
);
const registry = referenceRegistry(forwards);
const playable = (level: CircuitLevel) => ({
  ...level,
  available: [...level.available, ...forwards.flatMap((l) => l.unlockComponents ?? [])],
});
it.each([
  {
    id: '3-6',
    hidden: 0,
    learningRate: 0.1,
    parameterCount: 6,
  },
  {
    id: '3-9',
    hidden: 16,
    learningRate: 0.05,
    parameterCount: 82,
  },
])(
  '$id trains with configured Linear nodes that own their parameters',
  async ({ id, hidden, learningRate, parameterCount }) => {
    const level = chapter3ClassificationLevels.find((candidate) => candidate.id === id)!;
    const nodes: GraphJSON['nodes'] = [{ id: 'x', component: 'input', attrs: { name: 'x' } }];
    const edges: GraphJSON['edges'] = [];
    let source = 'x';
    if (hidden) {
      nodes.push(
        {
          id: 'hidden',
          component: 'linear',
          attrs: {
            automatic_parameters: true,
            in_features: 2,
            out_features: hidden,
            bias: true,
          },
        },
        { id: 'activation', component: 'sigmoid' },
      );
      edges.push(
        { from: { node: source, port: 'out' }, to: { node: 'hidden', port: 'x' } },
        { from: { node: 'hidden', port: 'out' }, to: { node: 'activation', port: 'x' } },
      );
      source = 'activation';
    }
    nodes.push(
      {
        id: 'classifier',
        component: 'linear',
        attrs: {
          automatic_parameters: true,
          in_features: hidden || 2,
          out_features: 2,
          bias: true,
        },
      },
      { id: 'probability', component: 'softmax' },
      { id: 'output', component: 'output', attrs: { name: 'probabilities' } },
    );
    edges.push(
      { from: { node: source, port: 'out' }, to: { node: 'classifier', port: 'x' } },
      { from: { node: 'classifier', port: 'out' }, to: { node: 'probability', port: 'x' } },
      { from: { node: 'probability', port: 'out' }, to: { node: 'output', port: 'x' } },
    );
    const result = await evaluateLevel(
      playable(level),
      { version: 1, nodes, edges },
      {
        registry,
        learningRate,
      },
    );
    expect(result.passed).toBe(true);
    expect(result.metrics.parameterCount).toBe(parameterCount);
  },
  60000,
);

/** Lookup table over the train batch; ignores x. Must fail hold-out. */
function memorizerGraph(rows: number): GraphJSON {
  return {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'logits', component: 'param', attrs: { shape: [rows, 2], init: 'zeros' } },
      { id: 'probability', component: 'sigmoid' },
      { id: 'output', component: 'output', attrs: { name: 'probabilities' } },
    ],
    edges: [
      { from: { node: 'logits', port: 'out' }, to: { node: 'probability', port: 'x' } },
      { from: { node: 'probability', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  };
}

it.each(['3-6', '3-8', '3-9'] as const)(
  '%s rejects param×N memorizer on held-out evaluation',
  async (id) => {
    const level = chapter3ClassificationLevels.find((candidate) => candidate.id === id)!;
    expect(level.evaluationDataset).toBeTypeOf('function');
    const rows = level.interface.outputShape[0]!;
    const result = await evaluateLevel(playable(level), memorizerGraph(rows), {
      registry,
      learningRate: 0.1,
      optimizer: 'adam',
    });
    expect(result.errorKey).not.toBe('levels.output_shape');
    expect(result.passed).toBe(false);
    // Train-set memorizer collapses on a fresh seed+1 batch.
    expect(result.training?.accuracy ?? 0).toBeLessThan(level.threshold.accuracy!);
  },
  60000,
);

for (const level of chapter3ClassificationLevels) {
  it(`${level.id} measured training settings`, async () => {
    const settings =
      level.id === '3-6'
        ? [{ learningRate: undefined }, { learningRate: 0.1 }]
        : level.id === '3-8'
          ? [0.001, 0.005, 0.01, 0.05, 0.1].map((learningRate) => ({
              learningRate,
              optimizer: 'sgd' as const,
            }))
          : [{ learningRate: undefined }];
    for (const setting of [
      ...settings,
      ...(level.id === '3-8' ? [{ optimizer: 'adam' as const }] : []),
    ]) {
      const result = await evaluateLevel(playable(level), level.referenceGraph!, {
        registry,
        ...setting,
      });
      console.info(level.id, 'par', level.par, setting, result.training, result.errorKey);
      const shouldPass =
        level.id === '3-9' ||
        ('optimizer' in setting && setting.optimizer === 'adam') ||
        (level.id === '3-6' && 'learningRate' in setting && setting.learningRate === 0.1);
      expect(result.passed).toBe(shouldPass);
    }
  }, 60000);
}
for (const level of chapter3ForwardLevels.filter((level) => ['3-5', '3-7'].includes(level.id))) {
  it(`${level.id} reference and mistakes`, async () => {
    const result = await evaluateLevel(playable(level), level.referenceGraph!, { registry });
    expect(result.passed).toBe(true);
    if (level.id === '3-5') {
      expect(result.tests).toHaveLength(15);
      for (const mistake of ['sign', 'sum']) {
        const wrong = structuredClone(level.referenceGraph!);
        if (mistake === 'sign') wrong.nodes.find((n) => n.id === 'minus')!.attrs = { value: 1 };
        else
          Object.assign(
            wrong.nodes.find((n) => n.id === 'average')!,
            { component: 'sum', attrs: { axes: [-1], keepdims: true } },
          );
        expect((await evaluateLevel(playable(level), wrong, { registry })).passed).toBe(false);
      }
    } else {
      const wrong = structuredClone(level.referenceGraph!);
      for (const edge of wrong.edges) {
        if (edge.from.node === 'm') edge.from.node = 'g';
        else if (edge.from.node === 'g') edge.from.node = 'm';
      }
      expect((await evaluateLevel(playable(level), wrong, { registry })).passed).toBe(false);
    }
    console.info('par', level.id, level.par);
  });
}
