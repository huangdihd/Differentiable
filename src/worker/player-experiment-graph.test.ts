import { expect, it } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { evaluateLevel } from '../engine/levels';
import { tensor } from '../engine/tensor';
import { gameLevels } from '../levels/catalog';
import { availableComponents, buildPlayerRegistry } from '../levels/registry';
import { playerExperimentGraph } from './player-experiment-graph';
import { recurrentCharacterTestProgress } from './recurrent-character-test-fixture';

it('returns an independent deep copy of the actual passed graph, never the reference root', () => {
  const progress = recurrentCharacterTestProgress();
  const saved = structuredClone(progress.passedSolutions);
  const { graph } = playerExperimentGraph(progress, 'time-rnn');
  expect(graph).toEqual(saved['time-rnn']);
  expect(graph).not.toBe(progress.passedSolutions['time-rnn']);
  expect(graph.nodes).toContainEqual({
    id: 'player_one',
    component: 'const',
    attrs: { value: 1 },
  });
  graph.nodes.find(({ id }) => id === 'player_one')!.attrs!.value = 0;
  graph.edges[0].to.port = 'changed';
  graph.nodes.pop();
  expect(progress.passedSolutions).toEqual(saved);
});

it.each(['not-passed', 'missing-saved'])(
  'refuses a %s root even when its reference and dependencies are available',
  (missing) => {
    const progress = recurrentCharacterTestProgress();
    if (missing === 'not-passed')
      progress.passed = progress.passed.filter((id) => id !== 'time-rnn');
    else delete progress.passedSolutions['time-rnn'];
    expect(() => playerExperimentGraph(progress, 'time-rnn')).toThrow(
      expect.objectContaining({
        name: 'GraphError',
        key: 'worker.player_experiment_graph_required',
        params: { level: 'time-rnn' },
      }),
    );
  },
);

it.each([
  { missingComponents: ['tanh'] },
  { missingComponents: ['matmul'] },
  { missingComponents: ['tanh', 'matmul'] },
])(
  'keeps the editor-passing RNN usable with missing unlocked saved dependencies: $missingComponents',
  async ({ missingComponents }) => {
    const progress = recurrentCharacterTestProgress();
    for (const component of missingComponents) {
      const provider = gameLevels.find((level) => level.unlockComponents?.includes(component))!;
      expect(progress.unlockedComponents).toContain(component);
      expect(progress.passed).toContain(provider.id);
      delete progress.passedSolutions[provider.id];
    }
    const level = gameLevels.find((candidate) => candidate.id === 'time-rnn')!;
    if (level.kind !== 'forward') throw new Error('expected forward RNN level');
    const saved = structuredClone(progress.passedSolutions);
    const editorRegistry = buildPlayerRegistry(gameLevels, progress);
    const result = await evaluateLevel(
      { ...level, available: availableComponents(level, progress) },
      progress.passedSolutions[level.id],
      { registry: editorRegistry },
    );
    expect(result.passed, JSON.stringify(result)).toBe(true);
    expect(result.tests).toHaveLength(16);
    expect(result.tests.every((test) => test.passed)).toBe(true);

    const { graph, registry } = playerExperimentGraph(progress, level.id);
    expect(graph).toEqual(saved[level.id]);
    for (const component of missingComponents)
      expect(registry.get(component)).toEqual(editorRegistry.get(component));
    const plan = compileGraph(graph, {
      registry,
      inputShapes: level.interface.inputs,
      preferNative: false,
    });
    try {
      expect(plan.outputShapes.hiddenState).toEqual(level.interface.outputShape);
    } finally {
      plan.dispose();
    }
    expect(progress.passedSolutions).toEqual(saved);
  },
);

it('preserves and executes the saved nested graph with native shortcuts disabled', async () => {
  const progress = recurrentCharacterTestProgress();
  const provider = gameLevels.find((level) => level.unlockComponents?.includes('tanh'))!;
  if (provider.kind === 'observation') throw new Error('expected circuit provider');
  // Deliberately differs from native tanh so execution detects a substituted dependency.
  const nested: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'player_zero', component: 'const', attrs: { value: 0 } },
      { id: 'player_product', component: 'mul' },
      { id: 'output', component: 'output', attrs: { name: provider.interface.output } },
    ],
    edges: [
      { from: { node: 'x', port: 'out' }, to: { node: 'player_product', port: 'a' } },
      { from: { node: 'player_zero', port: 'out' }, to: { node: 'player_product', port: 'b' } },
      { from: { node: 'player_product', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  };
  progress.passedSolutions[provider.id] = nested;
  const saved = structuredClone(progress.passedSolutions);
  const { graph, registry } = playerExperimentGraph(progress, 'time-rnn');
  const registered = registry.get('tanh')!.subgraph as GraphJSON;
  expect(registered).not.toBe(nested);
  expect(registered.nodes).toContainEqual(nested.nodes[1]);
  const inputs = {
    x: tensor([1], [1, 1]),
    h: tensor([0], [1, 1]),
    wx: tensor([1], [1, 1]),
    wh: tensor([1], [1, 1]),
    b: tensor([0], [1]),
  };
  const plan = compileGraph(graph, {
    registry,
    inputShapes: Object.fromEntries(
      Object.entries(inputs).map(([name, value]) => [name, value.shape]),
    ),
    preferNative: false,
  });
  try {
    const outputs = plan.execute(inputs);
    try {
      expect(Array.from(await outputs.hiddenState.read())).toEqual([0]);
    } finally {
      plan.releaseExecution(outputs, Object.values(inputs));
    }
  } finally {
    plan.dispose();
    Object.values(inputs).forEach((input) => input.dispose());
  }
  registered.nodes[1].attrs!.value = 1;
  expect(progress.passedSolutions).toEqual(saved);
});
