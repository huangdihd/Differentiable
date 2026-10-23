import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createInitialProgress,
  createThrottledSave,
  isGraphJSON,
  isProgress,
  isExpressionUnlocked,
  LANGUAGE_KEY,
  loadLanguage,
  loadProgress,
  loadProgressWithMigration,
  saveLanguage,
  saveProgress,
  saveSave,
  STORAGE_KEY,
  STORAGE_VERSION,
} from './storage';
import type { GraphJSON } from './engine/graph';
import type { ForwardLevel } from './engine/levels';
import { chapter0Levels } from './levels/chapter0';
import { chapter1Levels } from './levels/chapter1';
import { derivativeSolution } from './levels/examples';

function memory() {
  const data = new Map<string, string>();
  const source = () => ({
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
  });
  return { data, source };
}
const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x', shape: [2] }, ui: { x: 1, y: 2 } },
    { id: 'y', component: 'output' },
  ],
  edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'y', port: 'x' } }],
};
function genericOutputLevel(id: string, unlock: string[]): ForwardLevel {
  const referenceGraph = structuredClone(chapter1Levels[0].referenceGraph!);
  referenceGraph.nodes.find((node) => node.component === 'output')!.attrs = { name: 'y' };
  return {
    ...chapter1Levels[0],
    id,
    interface: { ...chapter1Levels[0].interface, output: 'y' },
    unlock,
    referenceGraph,
  };
}
function staleLossGradientGraph(output: string, vector: boolean): GraphJSON {
  const nodes: GraphJSON['nodes'] = [
    { id: 'w', component: 'input', attrs: { name: 'w' } },
    { id: 'x', component: 'input', attrs: { name: 'x' } },
    { id: 't', component: 'input', attrs: { name: 't' } },
    { id: 'weighted', component: 'mul' },
    ...(vector ? [{ id: 'prediction', component: 'sum', attrs: { axes: [-1] } }] : []),
    { id: 'negativeTarget', component: 'mul' },
    { id: 'minus', component: 'const', attrs: { value: -1 } },
    { id: 'residual', component: 'add' },
    { id: 'twiceResidual', component: 'mul' },
    { id: 'two', component: 'const', attrs: { value: 2 } },
    { id: 'gradient', component: 'mul' },
    ...(!vector
      ? [
          { id: 'step', component: 'mul' },
          { id: 'negativeRate', component: 'const', attrs: { value: -0.1 } },
          { id: 'updated', component: 'add' },
        ]
      : []),
    { id: 'output', component: 'output', attrs: { name: output } },
  ];
  const edge = (from: string, to: string, port = 'a') => ({
    from: { node: from, port: 'out' },
    to: { node: to, port },
  });
  const prediction = vector ? 'prediction' : 'weighted';
  return {
    version: 1,
    nodes,
    edges: [
      edge('w', 'weighted'),
      edge('x', 'weighted', 'b'),
      ...(vector ? [edge('weighted', 'prediction', 'x')] : []),
      edge('t', 'negativeTarget'),
      edge('minus', 'negativeTarget', 'b'),
      edge(prediction, 'residual'),
      edge('negativeTarget', 'residual', 'b'),
      edge('residual', 'twiceResidual'),
      edge('two', 'twiceResidual', 'b'),
      edge('twiceResidual', 'gradient'),
      edge('x', 'gradient', 'b'),
      ...(!vector
        ? [
            edge('gradient', 'step'),
            edge('negativeRate', 'step', 'b'),
            edge('w', 'updated'),
            edge('step', 'updated', 'b'),
          ]
        : []),
      edge(vector ? 'gradient' : 'updated', 'output', 'x'),
    ],
  };
}
afterEach(() => vi.useRealTimers());
describe('storage', () => {
  it('validates and roundtrips optional per-level training settings', () => {
    const { source } = memory();
    const progress = {
      ...createInitialProgress(),
      trainingSettings: { '3-4': { learningRate: 0.2 }, xor: { learningRate: 0.01 } },
    };
    expect(isProgress(createInitialProgress())).toBe(true);
    expect(saveProgress(progress, source)).toBe(true);
    expect(loadProgress(source)).toEqual(progress);
    for (const learningRate of [0, -1, NaN, Infinity, '0.1', null]) {
      expect(isProgress({ ...progress, trainingSettings: { '3-4': { learningRate } } })).toBe(
        false,
      );
    }
    for (const trainingSettings of [null, [], { '': { learningRate: 0.1 } }, { '3-4': {} }]) {
      expect(isProgress({ ...progress, trainingSettings })).toBe(false);
    }
  });
  it('unlocks chapter three for old graduates without removing the XOR unlock', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const progress = {
      ...createInitialProgress(),
      unlocked: ['2-8', 'xor'],
      passed: ['2-8'],
    };
    expect(saveProgress(progress, source)).toBe(true);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toContain('loss-intro');
    expect(restored.unlocked).not.toContain('3-1');
    expect(restored.unlocked).toContain('xor');
  });
  it('unlocks chapter four for saved XOR graduates without changing their graphs', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const solution = structuredClone(graph);
    solution.nodes.find((node) => node.component === 'output')!.attrs = {
      name: 'probabilities',
    };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['xor'],
      passed: ['xor'],
      passedSolutions: { xor: solution },
      savedGraphs: { xor: solution },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toEqual(['xor', '4-1', 'loss-intro']);
    expect(restored.passedSolutions).toEqual(progress.passedSolutions);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
    expect(saveProgress(restored, source)).toBe(true);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
  });
  it('unlocks chapter six for saved chapter-five graduates without changing their graphs', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const solution = structuredClone(graph);
    solution.nodes.find((node) => node.component === 'output')!.attrs = {
      name: 'probabilities',
    };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['5-5'],
      passed: ['5-5'],
      passedSolutions: { '5-5': solution },
      savedGraphs: { '5-5': solution },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toEqual(['5-5', '6-1', 'loss-intro']);
    expect(restored.passed).toEqual(progress.passed);
    expect(restored.passedSolutions).toEqual(progress.passedSolutions);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
    expect(saveProgress(restored, source)).toBe(true);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(saveProgress(createInitialProgress(), source)).toBe(true);
    expect((await loadProgressWithMigration(gameLevels, source)).unlocked).not.toContain('6-1');
  });
  it.each([STORAGE_VERSION])(
    'restores axes access without changing other data in version %s saves',
    async (version) => {
      const levels = [
        genericOutputLevel('1-11', ['axes']),
        genericOutputLevel('axes', ['2-1']),
        genericOutputLevel('2-1', []),
      ];
      for (const [id, passed, expected] of [
        ['1-11', false, ['1-11']],
        ['1-11', true, ['1-11', 'axes']],
        ['2-1', false, ['2-1', 'axes']],
        ['2-1', true, ['2-1', 'axes']],
      ] as const) {
        const { data, source } = memory();
        const progress = {
          ...createInitialProgress(),
          unlocked: [id],
          passed: passed ? [id] : [],
          unlockedComponents: ['identity'],
          savedGraphs: { [id]: graph, axes: graph },
          passedSolutions: passed ? { [id]: graph } : {},
          bestScores: { [id]: { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 } },
          trainingSettings: { [id]: { learningRate: 0.01 } },
        };
        data.set(STORAGE_KEY, JSON.stringify({ version, progress }));
        const restored = await loadProgressWithMigration(levels, source);
        expect(restored).toEqual({ ...progress, unlocked: expected });
        expect(saveProgress(restored, source)).toBe(true);
        expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
      }
      const { data, source } = memory();
      expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
      data.set(STORAGE_KEY, JSON.stringify({ version, progress: createInitialProgress() }));
      expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
    },
  );
  it('restores axes access in v1 without automatically crediting the new prerequisite', async () => {
    const reference = structuredClone(graph);
    reference.nodes[1].attrs = { name: 'y' };
    const lesson: ForwardLevel = {
      id: '1-11',
      titleKey: 'test.title',
      descriptionKey: 'test.description',
      kind: 'forward',
      available: [],
      interface: { inputs: { x: [2] }, output: 'y', outputShape: [2] },
      unlock: ['axes'],
      fixed: [{ id: 'case', inputs: { x: [1, 2] } }],
      reference: (inputs) => inputs.x,
      referenceGraph: reference,
    };
    const levels = [
      lesson,
      { ...lesson, id: 'axes', unlock: ['2-1'] },
      { ...lesson, id: '2-1', unlock: [] },
    ];
    for (const [id, passed] of [
      ['1-10', false],
      ['1-10', true],
      ['2-1', false],
      ['2-1', true],
    ] as const) {
      const { data, source } = memory();
      const legacy = {
        ...createInitialProgress(),
        unlocked: [id],
        passed: passed ? [id] : [],
        savedGraphs: { [id]: reference, axes: graph },
        trainingSettings: { [id]: { learningRate: 0.01 } },
      };
      data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress: legacy }));
      const restored = await loadProgressWithMigration(levels, source);
      expect(restored.unlocked.includes('axes')).toBe(id === '2-1' || passed);
      expect(restored.passed).toEqual(passed ? [id === '1-10' ? '1-11' : id] : []);
      expect(restored.passedSolutions.axes).toBeUndefined();
      expect(restored.bestScores.axes).toBeUndefined();
      const migratedId = id === '1-10' ? '1-11' : id;
      expect(restored.savedGraphs).toEqual({ [migratedId]: reference, axes: graph });
      expect(restored.trainingSettings).toEqual({
        [migratedId]: { learningRate: 0.01 },
      });
      expect(loadProgress(source)).toEqual(restored);
      expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
    }
    const { data, source } = memory();
    data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress: createInitialProgress() }));
    expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
  });
  it.each([2, STORAGE_VERSION])(
    'restores reshape access without changing other data in version %s saves',
    async (version) => {
      const levels = [
        genericOutputLevel('4-4', ['reshape']),
        genericOutputLevel('reshape', ['4-5']),
        genericOutputLevel('4-5', []),
      ];
      for (const [id, passed, expected] of [
        ['4-4', false, ['4-4']],
        ['4-4', true, ['4-4', 'reshape']],
        ['4-5', false, ['4-5', 'reshape']],
        ['4-5', true, ['4-5', 'reshape']],
      ] as const) {
        const { data, source } = memory();
        const progress = {
          ...createInitialProgress(),
          unlocked: [id],
          passed: passed ? [id] : [],
          unlockedComponents: ['identity'],
          savedGraphs: { [id]: graph, reshape: graph },
          passedSolutions: passed ? { [id]: graph } : {},
          bestScores: { [id]: { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 } },
          trainingSettings: { [id]: { learningRate: 0.01 } },
        };
        data.set(STORAGE_KEY, JSON.stringify({ version, progress }));
        const restored = await loadProgressWithMigration(levels, source);
        expect(restored).toEqual({ ...progress, unlocked: expected });
        expect(saveProgress(restored, source)).toBe(true);
        expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
      }
      const { data, source } = memory();
      expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
      data.set(STORAGE_KEY, JSON.stringify({ version, progress: createInitialProgress() }));
      expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
    },
  );
  it('restores reshape access in v1 without automatically crediting the new prerequisite', async () => {
    const reference = structuredClone(graph);
    reference.nodes[1].attrs = { name: 'y' };
    const lesson: ForwardLevel = {
      id: '4-4',
      titleKey: 'test.title',
      descriptionKey: 'test.description',
      kind: 'forward',
      available: [],
      interface: { inputs: { x: [2] }, output: 'y', outputShape: [2] },
      unlock: ['reshape'],
      fixed: [{ id: 'case', inputs: { x: [1, 2] } }],
      reference: (inputs) => inputs.x,
      referenceGraph: reference,
    };
    const levels = [
      lesson,
      { ...lesson, id: 'reshape', unlock: ['4-5'] },
      { ...lesson, id: '4-5', unlock: [] },
    ];
    for (const [id, passed] of [
      ['4-4', false],
      ['4-4', true],
      ['4-5', false],
      ['4-5', true],
    ] as const) {
      const { data, source } = memory();
      const legacy = {
        ...createInitialProgress(),
        unlocked: [id],
        passed: passed ? [id] : [],
        savedGraphs: { [id]: reference, reshape: graph },
        trainingSettings: { [id]: { learningRate: 0.01 } },
      };
      data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress: legacy }));
      const restored = await loadProgressWithMigration(levels, source);
      expect(restored.unlocked.includes('reshape')).toBe(id === '4-5' || passed);
      expect(restored.passed).toEqual(passed ? (id === '4-5' ? ['4-4', '4-5'] : [id]) : []);
      expect(restored.passedSolutions.reshape).toBeUndefined();
      expect(restored.bestScores.reshape).toBeUndefined();
      expect(restored.savedGraphs).toEqual(legacy.savedGraphs);
      expect(restored.trainingSettings).toEqual(legacy.trainingSettings);
      expect(loadProgress(source)).toEqual(restored);
      expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
    }
    const { data, source } = memory();
    data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress: createInitialProgress() }));
    expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
  });
  it.each([2, STORAGE_VERSION])(
    'restores neuron/layer unlocks without crediting new lessons in version %s saves',
    async (version) => {
      const levels = [
        genericOutputLevel('neuron', ['3-2']),
        genericOutputLevel('3-2', ['layer']),
        genericOutputLevel('layer', ['3-3']),
        genericOutputLevel('3-3', ['3-4']),
        genericOutputLevel('3-4', ['3-5']),
        genericOutputLevel('3-5', ['3-6']),
      ];
      for (const [unlocked, passed, additions] of [
        [['3-2'], [], ['neuron']],
        [['3-2'], ['3-2'], ['layer', 'neuron']],
        [['3-3'], [], ['neuron', 'layer']],
        [['3-3'], ['3-3'], ['3-4', 'neuron', 'layer']],
        [
          ['3-2', '3-3', '3-4', '3-5'],
          ['3-2', '3-3', '3-4', '3-5'],
          ['layer', '3-6', 'neuron'],
        ],
        [['neuron'], ['neuron'], ['3-2']],
        [['neuron', 'layer', '3-3'], ['neuron', 'layer'], ['3-2']],
      ]) {
        const { data, source } = memory();
        const progress = {
          ...createInitialProgress(),
          unlocked,
          passed,
          unlockedComponents: ['identity'],
          savedGraphs: { neuron: graph, layer: graph },
          passedSolutions: Object.fromEntries(passed.map((id) => [id, graph])),
          bestScores: Object.fromEntries(
            passed.map((id) => [id, { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 }]),
          ),
          trainingSettings: { neuron: { learningRate: 0.01 }, layer: { learningRate: 0.02 } },
        };
        data.set(STORAGE_KEY, JSON.stringify({ version, progress }));
        const restored = await loadProgressWithMigration(levels, source);
        expect(restored).toEqual({ ...progress, unlocked: [...unlocked, ...additions] });
        expect(saveProgress(restored, source)).toBe(true);
        expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
      }
      const { data, source } = memory();
      data.set(STORAGE_KEY, JSON.stringify({ version, progress: createInitialProgress() }));
      expect(await loadProgressWithMigration(levels, source)).toEqual(createInitialProgress());
    },
  );
  it('restores v1 construction access without automatically crediting neuron or layer', async () => {
    const reference = structuredClone(graph);
    reference.nodes[1].attrs = { name: 'y' };
    const lesson: ForwardLevel = {
      id: '3-2',
      titleKey: 'test.title',
      descriptionKey: 'test.description',
      kind: 'forward',
      available: [],
      interface: { inputs: { x: [2] }, output: 'y', outputShape: [2] },
      unlock: ['layer'],
      fixed: [{ id: 'case', inputs: { x: [1, 2] } }],
      reference: (inputs) => inputs.x,
      referenceGraph: reference,
    };
    const levels = [
      { ...lesson, id: 'neuron', unlock: ['3-2'] },
      lesson,
      { ...lesson, id: 'layer', unlock: ['3-3'] },
      { ...lesson, id: '3-3', unlock: [] },
    ];
    for (const passed of [[], ['3-3']]) {
      const { data, source } = memory();
      const legacy = {
        ...createInitialProgress(),
        unlocked: ['3-3'],
        passed,
        savedGraphs: { neuron: graph, layer: graph },
        trainingSettings: { neuron: { learningRate: 0.01 } },
      };
      data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress: legacy }));
      const restored = await loadProgressWithMigration(levels, source);
      expect(restored.unlocked).toEqual(expect.arrayContaining(['neuron', 'layer']));
      expect(restored.passed).toEqual(passed.length ? ['3-2', '3-3'] : []);
      for (const id of ['neuron', 'layer']) {
        expect(restored.passedSolutions[id]).toBeUndefined();
        expect(restored.bestScores[id]).toBeUndefined();
      }
      expect(restored.savedGraphs).toEqual(legacy.savedGraphs);
      expect(restored.trainingSettings).toEqual(legacy.trainingSettings);
      expect(loadProgress(source)).toEqual(restored);
      expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
    }
  });
  it('does not restore neuron/layer IDs absent from the supplied catalog', async () => {
    const { source } = memory();
    const progress = { ...createInitialProgress(), unlocked: ['3-3'] };
    expect(saveProgress(progress, source)).toBe(true);
    expect(await loadProgressWithMigration(chapter1Levels, source)).toEqual(progress);
  });
  it('gates expressions for new players, graduates and grandfathered players', () => {
    const fresh = createInitialProgress();
    expect(isExpressionUnlocked(fresh, chapter1Levels)).toBe(false);
    expect(
      isExpressionUnlocked({ ...fresh, passed: ['1-7'], unlocked: ['1-7', '1-8'] }, chapter1Levels),
    ).toBe(false);
    expect(isExpressionUnlocked({ ...fresh, passed: ['1-11'] }, chapter1Levels)).toBe(true);
    expect(
      isExpressionUnlocked({ ...fresh, passed: ['1-7'], unlocked: ['2-1'] }, chapter1Levels),
    ).toBe(true);
    expect(
      isExpressionUnlocked({ ...fresh, passed: ['custom'] }, [
        {
          ...genericOutputLevel('custom', []),
          unlockFeatures: ['levels.features.expression'],
        },
      ]),
    ).toBe(true);
  });
  it('drops an unfinished v3 backprop draft that cannot satisfy the v7 contract', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const score = { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['1-10'],
      savedGraphs: { '1-10': graph },
      bestScores: { '1-10': score },
      trainingSettings: { '1-10': { learningRate: 0.01 } },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 3, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);

    expect(restored.unlocked).toContain('1-10');
    expect(restored.unlocked).not.toContain('1-11');
    expect(restored.unlocked).not.toContain('1-12');
    expect(restored.passed).toEqual([]);
    expect(restored.savedGraphs).toEqual({});
    expect(restored.bestScores).toEqual({});
    expect(restored.trainingSettings).toEqual({});
    const persisted = data.get(STORAGE_KEY)!;
    expect(JSON.parse(persisted)).toEqual({ version: STORAGE_VERSION, progress: restored });
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });

  it('revokes a passed v3 backprop under v7 while retaining gradient, axes, and chapter 2 access', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const score = { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['1-10', 'axes', '2-1'],
      passed: ['1-10'],
      savedGraphs: { '1-10': graph },
      passedSolutions: { '1-10': graph },
      bestScores: { '1-10': score },
      trainingSettings: { '1-10': { learningRate: 0.01 } },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 3, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);

    expect(restored.unlocked).toEqual(expect.arrayContaining(['1-10', '1-11', 'axes', '2-1']));
    expect(restored.unlocked).not.toContain('1-12');
    expect(restored.passed).toEqual([]);
    expect(restored.savedGraphs).toEqual({});
    expect(restored.passedSolutions).toEqual({});
    expect(restored.bestScores).toEqual({});
    expect(restored.trainingSettings).toEqual({});
    const persisted = data.get(STORAGE_KEY)!;
    expect(JSON.parse(persisted)).toEqual({ version: STORAGE_VERSION, progress: restored });
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });

  it('drops a v5 1-11 solution under the new v7 interface but preserves axes access', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const score = { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['1-11'],
      passed: ['1-11'],
      savedGraphs: { '1-11': graph },
      passedSolutions: { '1-11': graph },
      bestScores: { '1-11': score },
      trainingSettings: { '1-11': { learningRate: 0.1 } },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 5, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);

    expect(restored.unlocked).toEqual(expect.arrayContaining(['1-11', 'axes']));
    expect(restored.passed).not.toContain('1-11');
    expect(restored.savedGraphs['1-11']).toBeUndefined();
    expect(restored.passedSolutions['1-11']).toBeUndefined();
    expect(restored.bestScores['1-11']).toBeUndefined();
    expect(restored.trainingSettings?.['1-11']).toBeUndefined();
    expect(JSON.parse(data.get(STORAGE_KEY)!)).toEqual({
      version: STORAGE_VERSION,
      progress: restored,
    });
  });

  it('drops a v6 1-11 draft whose obsolete protected b input cannot be removed in the editor', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const stale = structuredClone(graph);
    stale.nodes.unshift(
      { id: 'w', component: 'input', attrs: { name: 'w' } },
      { id: 'b', component: 'input', attrs: { name: 'b' } },
    );
    const progress = {
      ...createInitialProgress(),
      unlocked: ['1-11'],
      savedGraphs: { '1-11': stale },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 6, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);

    expect(restored.unlocked).toContain('1-11');
    expect(restored.savedGraphs['1-11']).toBeUndefined();
    expect(JSON.parse(data.get(STORAGE_KEY)!)).toEqual({
      version: STORAGE_VERSION,
      progress: restored,
    });
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
  });

  it('drops v4 redundant 1-11 completion data instead of applying it to final backprop', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const score = { topNodes: 1, expandedPrimitives: 1, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['1-11'],
      passed: ['1-11'],
      savedGraphs: { '1-11': graph },
      passedSolutions: { '1-11': graph },
      bestScores: { '1-11': score },
      trainingSettings: { '1-11': { learningRate: 0.1 } },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 4, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);

    expect(restored.unlocked).toContain('1-11');
    expect(restored.passed).not.toContain('1-11');
    expect(restored.savedGraphs['1-11']).toBeUndefined();
    expect(restored.passedSolutions['1-11']).toBeUndefined();
    expect(restored.bestScores['1-11']).toBeUndefined();
    expect(restored.trainingSettings?.['1-11']).toBeUndefined();
    expect(Object.keys(restored.savedGraphs)).not.toContain('1-12');
    expect(JSON.parse(data.get(STORAGE_KEY)!)).toEqual({
      version: STORAGE_VERSION,
      progress: restored,
    });
  });

  it('discards both v4 backprop contracts under v7 while preserving later access', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const obsolete = structuredClone(graph);
    obsolete.nodes[0].id = 'obsolete';
    obsolete.edges[0].from.node = 'obsolete';
    const backprop = structuredClone(graph);
    backprop.nodes[0].id = 'backprop';
    backprop.edges[0].from.node = 'backprop';
    const obsoleteScore = { topNodes: 1, expandedPrimitives: 1, parameterCount: 0 };
    const backpropScore = { topNodes: 2, expandedPrimitives: 2, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['1-11', '1-12'],
      passed: ['1-11', '1-12'],
      savedGraphs: { '1-11': obsolete, '1-12': backprop },
      passedSolutions: { '1-11': obsolete, '1-12': backprop },
      bestScores: { '1-11': obsoleteScore, '1-12': backpropScore },
      trainingSettings: {
        '1-11': { learningRate: 0.1 },
        '1-12': { learningRate: 0.2, optimizer: 'adam' as const },
      },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 4, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);

    expect(restored.unlocked).toContain('1-11');
    expect(restored.unlocked).not.toContain('1-12');
    expect(restored.unlocked).toContain('axes');
    expect(restored.passed).toEqual([]);
    expect(restored.savedGraphs).toEqual({});
    expect(restored.passedSolutions).toEqual({});
    expect(restored.bestScores).toEqual({});
    expect(restored.trainingSettings).toEqual({});
    const persisted = data.get(STORAGE_KEY)!;
    expect(JSON.parse(persisted)).toEqual({ version: STORAGE_VERSION, progress: restored });
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });

  it('roundtrips versioned progress with a fresh chapter-zero initial state', () => {
    const { source } = memory();
    const progress = loadProgress(source);
    expect(progress.unlocked).toEqual(['0-1']);
    progress.savedGraphs.sigmoid = graph;
    progress.bestScores.sigmoid = { topNodes: 2, expandedPrimitives: 0, parameterCount: 0 };
    expect(saveProgress(progress, source)).toBe(true);
    expect(loadProgress(source)).toEqual(progress);
    expect(createInitialProgress().savedGraphs).toEqual({});
  });
  it('refreshes catalog unlocks in current saves without changing passed circuits or edits', async () => {
    const { source } = memory();
    const lesson = chapter1Levels.find((level) => level.id === '1-7')!;
    const semanticGraph = structuredClone(graph);
    semanticGraph.nodes.find((node) => node.component === 'output')!.attrs = { name: 'f′' };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['0-1', '1-7', '2-1', '2-2', 'xor'],
      passed: ['1-7'],
      passedSolutions: { '1-7': semanticGraph },
      savedGraphs: { '1-7': semanticGraph },
    };
    saveProgress(progress, source);
    const restored = await loadProgressWithMigration([lesson], source);
    expect(restored.unlocked).toEqual([...progress.unlocked, '1-8']);
    expect(restored.unlocked).toContain('xor');
    expect(restored.passed).toEqual(progress.passed);
    expect(restored.passedSolutions).toEqual(progress.passedSolutions);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
    saveProgress(restored, source);
    expect(await loadProgressWithMigration([lesson], source)).toEqual(restored);
    saveProgress(createInitialProgress(), source);
    expect((await loadProgressWithMigration([lesson], source)).unlocked).not.toContain('2-1');
  });
  it('remaps all chapter-two progress fields once before refreshing unlocks', async () => {
    const { source, data } = memory();
    const oldIds = ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', 'custom'];
    const newIds = ['2-1', '2-3', '2-4', '2-5', '2-7', '2-8', 'custom'];
    const graphs = oldIds.map((id) => ({
      ...graph,
      nodes: [{ id, component: '2-2' }],
      edges: [],
    }));
    const scores = oldIds.map((_, index) => ({
      topNodes: index,
      expandedPrimitives: index,
      parameterCount: index,
    }));
    const progress = {
      ...createInitialProgress(),
      unlocked: oldIds,
      passed: oldIds,
      trainingSettings: { '2-6': { learningRate: 0.2 } },
      unlockedComponents: ['2-2', 'sigmoid'],
      savedGraphs: Object.fromEntries(oldIds.map((id, i) => [id, graphs[i]])),
      passedSolutions: Object.fromEntries(oldIds.map((id, i) => [id, graphs[i]])),
      bestScores: Object.fromEntries(oldIds.map((id, i) => [id, scores[i]])),
    };
    const lesson = {
      ...genericOutputLevel('2-3', ['2-4', '2-6']),
      unlockComponents: ['tanh'],
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 2, progress }));
    const restored = await loadProgressWithMigration([lesson], source);
    expect(restored).toEqual({
      ...progress,
      unlocked: [...newIds, '2-6'],
      passed: newIds,
      trainingSettings: { '2-8': { learningRate: 0.2 } },
      unlockedComponents: ['2-2', 'sigmoid', 'tanh'],
      savedGraphs: Object.fromEntries(newIds.map((id, i) => [id, graphs[i]])),
      passedSolutions: Object.fromEntries(newIds.map((id, i) => [id, graphs[i]])),
      bestScores: Object.fromEntries(newIds.map((id, i) => [id, scores[i]])),
    });
    expect(JSON.parse(data.get(STORAGE_KEY)!).version).toBe(STORAGE_VERSION);
    expect(loadProgress(source)).toEqual(restored);
    expect(await loadProgressWithMigration([lesson], source)).toEqual(restored);
  });
  it.each([
    ['3-1', 'updatedWeight', false],
    ['3-2', 'gradient', true],
  ] as const)(
    'revokes a current-version %s solution with the obsolete target input and preserves its draft',
    async (id, output, vector) => {
      const { gameLevels } = await import('./levels/catalog');
      const { source } = memory();
      const solution = staleLossGradientGraph(output, vector);
      const draft = structuredClone(solution);
      draft.nodes.find((node) => node.id === 't')!.ui = { x: 41, y: 73 };
      const next = gameLevels.find((level) => level.id === id)!.unlock[0];
      const progress = {
        ...createInitialProgress(),
        unlocked: [id, next],
        passed: [id],
        savedGraphs: { [id]: draft },
        passedSolutions: { [id]: solution },
        bestScores: {
          [id]: { topNodes: solution.nodes.length, expandedPrimitives: 7, parameterCount: 0 },
        },
      };
      expect(saveProgress(progress, source)).toBe(true);
      const onRevoked = vi.fn();

      const restored = await loadProgressWithMigration(gameLevels, source, onRevoked);

      const editableDraft = structuredClone(draft);
      editableDraft.nodes.find((node) => node.id === 't')!.attrs = { name: '∂L/∂p' };
      if (id === '3-2') {
        editableDraft.nodes = editableDraft.nodes.filter((node) => node.id !== 'w');
        editableDraft.edges = editableDraft.edges.filter(
          (edge) => edge.from.node !== 'w' && edge.to.node !== 'w',
        );
      }
      expect(restored.unlocked).toEqual(expect.arrayContaining([id, next]));
      expect(restored.passed).not.toContain(id);
      expect(restored.savedGraphs[id]).toEqual(editableDraft);
      expect(restored.passedSolutions).not.toHaveProperty(id);
      expect(restored.bestScores).not.toHaveProperty(id);
      expect(onRevoked).toHaveBeenCalledWith([id]);
      expect(loadProgress(source)).toEqual(restored);
    },
  );
  it('adds the required momentum output to old drafts and revokes the old single-output pass', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const oldMomentum: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'w', component: 'input', attrs: { name: 'w' } },
        { id: 'g', component: 'input', attrs: { name: 'g' } },
        { id: 'm', component: 'input', attrs: { name: 'm' } },
        { id: 'decay', component: 'const', attrs: { value: 0.9 } },
        { id: 'decayed', component: 'mul' },
        { id: 'nextMomentum', component: 'add' },
        { id: 'rate', component: 'const', attrs: { value: -0.1 } },
        { id: 'step', component: 'mul' },
        { id: 'nextWeight', component: 'add' },
        {
          id: 'output',
          component: 'output',
          attrs: { name: 'y', note: 'preserve' },
          ui: { x: 700, y: 120 },
        },
      ],
      edges: [
        { from: { node: 'm', port: 'out' }, to: { node: 'decayed', port: 'a' } },
        { from: { node: 'decay', port: 'out' }, to: { node: 'decayed', port: 'b' } },
        { from: { node: 'decayed', port: 'out' }, to: { node: 'nextMomentum', port: 'a' } },
        { from: { node: 'g', port: 'out' }, to: { node: 'nextMomentum', port: 'b' } },
        { from: { node: 'nextMomentum', port: 'out' }, to: { node: 'step', port: 'a' } },
        { from: { node: 'rate', port: 'out' }, to: { node: 'step', port: 'b' } },
        { from: { node: 'w', port: 'out' }, to: { node: 'nextWeight', port: 'a' } },
        { from: { node: 'step', port: 'out' }, to: { node: 'nextWeight', port: 'b' } },
        { from: { node: 'nextWeight', port: 'out' }, to: { node: 'output', port: 'x' } },
      ],
    };
    const score = { topNodes: 10, expandedPrimitives: 6, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['3-7', '3-8'],
      passed: ['3-7'],
      savedGraphs: { '3-7': oldMomentum },
      passedSolutions: { '3-7': oldMomentum },
      bestScores: { '3-7': score },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const onRevoked = vi.fn();

    const restored = await loadProgressWithMigration(gameLevels, source, onRevoked);

    expect(restored.passed).not.toContain('3-7');
    expect(restored.passedSolutions).not.toHaveProperty('3-7');
    expect(restored.bestScores).not.toHaveProperty('3-7');
    expect(restored.unlocked).toEqual(expect.arrayContaining(['3-7', '3-8']));
    const draft = restored.savedGraphs['3-7'];
    expect(draft.edges).toEqual(oldMomentum.edges);
    expect(draft.nodes.find((node) => node.id === 'output')!.attrs).toEqual({
      name: 'updatedWeight',
      note: 'preserve',
    });
    expect(draft.nodes.find((node) => node.attrs?.name === 'updatedMomentum')).toEqual({
      id: 'output_momentum',
      component: 'output',
      attrs: { name: 'updatedMomentum' },
      ui: { x: 700, y: 320 },
    });
    expect(onRevoked).toHaveBeenCalledWith(['3-7']);
    expect(loadProgress(source)).toEqual(restored);

    const persisted = data.get(STORAGE_KEY);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });
  it('removes the obsolete vector-gradient weight input without revoking a compatible solution', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const level = gameLevels.find((candidate) => candidate.id === '3-2')!;
    const solution = JSON.parse(JSON.stringify(level.referenceGraph!)) as GraphJSON;
    solution.nodes.unshift({
      id: 'legacy-w',
      component: 'input',
      attrs: { name: 'w' },
      ui: { x: 41, y: 73 },
    });
    const score = { topNodes: solution.nodes.length, expandedPrimitives: 1, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['3-2', 'layer'],
      passed: ['3-2'],
      savedGraphs: { '3-2': solution },
      passedSolutions: { '3-2': solution },
      bestScores: { '3-2': score },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const onRevoked = vi.fn();

    const restored = await loadProgressWithMigration(gameLevels, source, onRevoked);

    expect(restored.passed).toContain('3-2');
    expect(restored.savedGraphs['3-2'].nodes.some((node) => node.id === 'legacy-w')).toBe(false);
    expect(restored.passedSolutions['3-2'].nodes.some((node) => node.id === 'legacy-w')).toBe(
      false,
    );
    expect(restored.bestScores['3-2']).toEqual(score);
    expect(onRevoked).not.toHaveBeenCalled();
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
  });
  it('repairs the old regression Linear defaults only in the matching saved draft', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const draft: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'x', component: 'input', attrs: { name: 'x', shape: [16] }, ui: { x: 20, y: 80 } },
        { id: 't', component: 'input', attrs: { name: 't', shape: [16] }, ui: { x: 20, y: 260 } },
        {
          id: 'prediction',
          component: 'linear',
          attrs: {
            automatic_parameters: true,
            in_features: 1,
            out_features: 2,
            bias: false,
            note: 'preserve',
          },
          ui: { x: 250, y: 80, collapsed: true },
        },
        {
          id: 'unrelated',
          component: 'linear',
          attrs: { automatic_parameters: true, in_features: 2, out_features: 2, bias: true },
          ui: { x: 250, y: 420 },
        },
        { id: 'mse', component: 'mse', ui: { x: 500, y: 170 } },
        { id: 'loss', component: 'output', attrs: { name: 'loss' }, ui: { x: 740, y: 170 } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 'prediction', port: 'x' } },
        { from: { node: 'prediction', port: 'out' }, to: { node: 'mse', port: 'p' } },
        { from: { node: 't', port: 'out' }, to: { node: 'mse', port: 't' } },
        { from: { node: 'mse', port: 'out' }, to: { node: 'loss', port: 'x' } },
        { from: { node: 'x', port: 'out' }, to: { node: 'unrelated', port: 'x' } },
      ],
    };
    const submitted = structuredClone(draft);
    const score = { topNodes: 6, expandedPrimitives: 12, parameterCount: 3 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['3-4', '3-5', 'loss-intro'],
      unlockedComponents: ['linear', 'mse'],
      passed: ['3-4'],
      savedGraphs: { '3-4': draft },
      passedSolutions: { '3-4': submitted },
      bestScores: { '3-4': score },
      trainingSettings: { '3-4': { learningRate: 0.3 } },
    };
    expect(saveProgress(progress, source)).toBe(true);

    const restored = await loadProgressWithMigration(gameLevels, source);

    const migrated = structuredClone(draft);
    const prediction = migrated.nodes.find((node) => node.id === 'prediction')!;
    prediction.attrs = { ...prediction.attrs, in_features: 1, out_features: 1 };
    expect(restored).toEqual({
      ...progress,
      savedGraphs: { '3-4': migrated },
    });
    expect(
      restored.savedGraphs['3-4'].nodes.find((node) => node.id === 'unrelated')!.attrs,
    ).toEqual(draft.nodes.find((node) => node.id === 'unrelated')!.attrs);
    expect(restored.passedSolutions['3-4']).toEqual(submitted);
    expect(restored.bestScores['3-4']).toEqual(score);

    const persisted = data.get(STORAGE_KEY);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });
  it('revokes changed-contract pass credit when a current save has no submitted solution', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const draft = staleLossGradientGraph('gradient', true);
    const progress = {
      ...createInitialProgress(),
      unlocked: ['3-2', 'layer'],
      passed: ['3-2'],
      savedGraphs: { '3-2': draft },
      bestScores: {
        '3-2': { topNodes: draft.nodes.length, expandedPrimitives: 7, parameterCount: 0 },
      },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const onRevoked = vi.fn();

    const restored = await loadProgressWithMigration(gameLevels, source, onRevoked);

    expect(restored.passed).not.toContain('3-2');
    expect(restored.savedGraphs['3-2'].nodes.find((node) => node.id === 't')!.attrs).toEqual({
      name: '∂L/∂p',
    });
    expect(restored.savedGraphs['3-2'].nodes.some((node) => node.id === 'w')).toBe(false);
    expect(restored.savedGraphs['3-2'].edges.some((edge) => edge.from.node === 'w')).toBe(false);
    expect(restored.bestScores).not.toHaveProperty('3-2');
    expect(onRevoked).toHaveBeenCalledWith(['3-2']);
  });
  it('leaves current chapter-two numbering unchanged', async () => {
    const { source } = memory();
    const progress = {
      ...createInitialProgress(),
      unlocked: ['2-2', '2-3', '2-4', '2-5', '2-6'],
      passed: ['2-2'],
      savedGraphs: { '2-2': graph },
      passedSolutions: { '2-2': graph },
      bestScores: { '2-2': { topNodes: 2, expandedPrimitives: 0, parameterCount: 0 } },
    };
    saveProgress(progress, source);
    expect(await loadProgressWithMigration([], source)).toEqual(progress);
  });
  it('preserves partial old 2-2 work under 2-3 without crediting a pass', async () => {
    const { source, data } = memory();
    const progress = {
      ...createInitialProgress(),
      unlocked: ['2-2'],
      savedGraphs: { '2-2': graph },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 2, progress }));
    expect(await loadProgressWithMigration([], source)).toEqual({
      ...progress,
      unlocked: ['2-3'],
      savedGraphs: { '2-3': graph },
    });
  });
  it('rejects corrupt, wrong-version and deeply invalid stored values', () => {
    const { data, source } = memory();
    for (const raw of [
      '{',
      'null',
      '{}',
      JSON.stringify({ version: 99, progress: createInitialProgress() }),
      JSON.stringify({
        version: 1,
        progress: {
          ...createInitialProgress(),
          savedGraphs: { sigmoid: { ...graph, edges: [{ from: null, to: {} }] } },
        },
      }),
    ]) {
      data.set(STORAGE_KEY, raw);
      expect(loadProgress(source)).toEqual(createInitialProgress());
    }
  });
  it('migrates legacy milestones with validated graphs and reference fallback', async () => {
    const { source, data } = memory();
    const reference = structuredClone(graph);
    reference.nodes[1].attrs = { name: 'y' };
    const level: ForwardLevel = {
      id: '0-4',
      titleKey: 'levels.sigmoid.title',
      descriptionKey: 'test.description',
      kind: 'forward',
      available: [],
      interface: { inputs: { x: [2] }, output: 'y', outputShape: [2] },
      unlock: ['1-1'],
      unlockComponents: ['identity'],
      fixed: [{ id: 'case', inputs: { x: [1, 2] } }],
      reference: (inputs) => inputs.x,
      referenceGraph: reference,
    };
    for (const saved of [reference, { version: 1, nodes: [], edges: [] }]) {
      const legacy = {
        ...createInitialProgress(),
        unlocked: ['sigmoid'],
        passed: ['sigmoid'],
        savedGraphs: { sigmoid: saved },
      };
      data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress: legacy }));
      const result = await loadProgressWithMigration([level], source);
      expect(result.passed).toEqual(['0-4']);
      expect(result.passedSolutions['0-4']).toEqual(reference);
      expect(result.unlockedComponents).toEqual(['identity']);
      expect(result.unlocked).toContain('1-1');
      expect(JSON.parse(data.get(STORAGE_KEY)!).version).toBe(STORAGE_VERSION);
      expect(loadProgress(source)).toEqual(result);
    }
    data.set(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        progress: { ...createInitialProgress(), unlocked: ['sigmoid'], passed: ['sigmoid'] },
      }),
    );
    expect(
      (await loadProgressWithMigration([{ ...level, referenceGraph: undefined }], source)).passed,
    ).toEqual([]);
  });
  it('migrates v8 semantic output names without changing circuit structure or progress', async () => {
    const { data, source } = memory();
    const explicit = structuredClone(graph);
    explicit.nodes[1] = {
      ...explicit.nodes[1],
      attrs: { name: 'y', note: 'keep' },
      ui: { x: 320, y: 180 },
    };
    const implicit = structuredClone(graph);
    const score = { topNodes: 3, expandedPrimitives: 2, parameterCount: 0 };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['0-1', '1-1'],
      passed: ['0-1', '1-1'],
      unlockedComponents: ['add'],
      savedGraphs: { '0-1': graph, '1-1': explicit },
      passedSolutions: { '0-1': graph, '1-1': implicit },
      bestScores: { '1-1': score },
      trainingSettings: { '1-1': { learningRate: 0.02 } },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 8, progress }));
    const levels = [
      { ...chapter0Levels[0], unlock: [], unlockComponents: [] },
      { ...chapter1Levels[0], unlock: [], unlockComponents: [] },
    ];

    const restored = await loadProgressWithMigration(levels, source);
    const migratedExplicit = structuredClone(explicit);
    migratedExplicit.nodes[1].attrs = { name: 'f′', note: 'keep' };
    const migratedImplicit = structuredClone(implicit);
    migratedImplicit.nodes[1].attrs = { name: 'f′' };
    expect(restored).toEqual({
      ...progress,
      savedGraphs: { '0-1': graph, '1-1': migratedExplicit },
      passedSolutions: { '0-1': graph, '1-1': migratedImplicit },
    });
    expect(restored.savedGraphs['1-1'].nodes.map((node) => node.id)).toEqual(
      explicit.nodes.map((node) => node.id),
    );
    expect(restored.savedGraphs['1-1'].edges).toEqual(explicit.edges);
    expect(restored.savedGraphs['1-1'].nodes[1].ui).toEqual(explicit.nodes[1].ui);
    expect(restored.savedGraphs['0-1']).toEqual(graph);
    expect(JSON.parse(data.get(STORAGE_KEY)!).version).toBe(STORAGE_VERSION);

    const persisted = data.get(STORAGE_KEY);
    expect(await loadProgressWithMigration(levels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });
  it('repairs a v8 descent output and unlocks the inserted loss game without granting completion', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { data, source } = memory();
    const descent = structuredClone(
      gameLevels.find((level) => level.id === '3-1')!.referenceGraph!,
    );
    const output = descent.nodes.find((node) => node.component === 'output')!;
    output.attrs = { name: 'y', note: 'preserve' };
    output.ui = { x: 730, y: 210 };
    const solution = structuredClone(descent);
    const progress = {
      ...createInitialProgress(),
      unlocked: ['3-1', '3-2'],
      passed: ['3-1'],
      savedGraphs: { '3-1': descent },
      passedSolutions: { '3-1': solution },
      bestScores: { '3-1': { topNodes: 7, expandedPrimitives: 7, parameterCount: 0 } },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: 8, progress }));

    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toEqual(['3-1', '3-2', 'neuron', 'loss-intro']);
    expect(restored.passed).toEqual(['3-1']);
    for (const migrated of [restored.savedGraphs['3-1'], restored.passedSolutions['3-1']]) {
      expect(migrated.nodes.find((node) => node.component === 'output')!.attrs).toEqual({
        name: 'updatedWeight',
        note: 'preserve',
      });
      expect(migrated.nodes.find((node) => node.component === 'output')!.ui).toEqual(output.ui);
      expect(migrated.nodes.map((node) => node.id)).toEqual(descent.nodes.map((node) => node.id));
      expect(migrated.edges).toEqual(descent.edges);
    }
    expect(restored.passed).not.toContain('loss-intro');
    expect(restored.passedSolutions).not.toHaveProperty('loss-intro');
    expect(JSON.parse(data.get(STORAGE_KEY)!).version).toBe(STORAGE_VERSION);

    const persisted = data.get(STORAGE_KEY);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(data.get(STORAGE_KEY)).toBe(persisted);
  });
  it('maps the old derivative to the matching product-rule lesson and restores prerequisites', async () => {
    const { data, source } = memory();
    data.set(
      STORAGE_KEY,
      JSON.stringify({
        version: 1,
        progress: {
          ...createInitialProgress(),
          unlocked: ['sigmoid', 'derivative'],
          passed: ['sigmoid', 'derivative'],
          savedGraphs: { derivative: derivativeSolution },
        },
      }),
    );
    const progress = await loadProgressWithMigration(
      [...chapter0Levels, ...chapter1Levels],
      source,
    );
    // The square lesson was added later: it opens up but is never marked complete for the player.
    expect(progress.passed).toEqual(
      [...chapter0Levels, ...chapter1Levels.slice(0, 2)]
        .map((level) => level.id)
        .filter((id) => id !== 'square'),
    );
    expect(progress.unlocked).toContain('square');
    const migratedDerivative = structuredClone(derivativeSolution);
    migratedDerivative.nodes.find((node) => node.component === 'output')!.attrs = { name: 'f′' };
    expect(progress.savedGraphs['1-2']).toEqual(migratedDerivative);
    expect(progress.passedSolutions['1-2']).toEqual(migratedDerivative);
    expect(derivativeSolution.nodes.find((node) => node.component === 'output')?.attrs).toEqual({
      name: 'y',
    });
    expect(progress.unlocked).toContain('1-3');
    expect(progress.unlockedComponents).toContain('sigmoid');
  });
  it('keeps legacy language preferences during the version upgrade', () => {
    const { data, source } = memory();
    for (const version of [1, 2, 7, 8]) {
      data.set(LANGUAGE_KEY, JSON.stringify({ version, language: 'en' }));
      expect(loadLanguage(source, 'zh-CN')).toBe('en');
    }
  });
  it('validates graph structure and recursive JSON without requiring a complete circuit', () => {
    expect(isGraphJSON(graph)).toBe(true);
    expect(isGraphJSON({ version: 1, nodes: [], edges: [] })).toBe(true);
    expect(isGraphJSON({ ...graph, nodes: [graph.nodes[0], graph.nodes[0]] })).toBe(false);
    expect(isGraphJSON({ ...graph, nodes: [] })).toBe(false);
    expect(
      isGraphJSON({ ...graph, nodes: [{ id: 'a', component: 'const', attrs: { value: NaN } }] }),
    ).toBe(false);
    expect(isGraphJSON(JSON.parse('{"version":1,"nodes":[],"edges":[],"__proto__":{}}'))).toBe(
      false,
    );
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    expect(isGraphJSON({ ...graph, extra: cyclic })).toBe(false);
    expect(
      isProgress({
        ...createInitialProgress(),
        bestScores: { sigmoid: { topNodes: -1, parameterCount: 0, expandedPrimitives: 0 } },
      }),
    ).toBe(false);
    expect(isProgress({ ...createInitialProgress(), passed: ['locked'] })).toBe(false);
  });
  it('catches storage acquisition, get/set method getters, and quota errors', () => {
    const denied = () => {
      throw new Error('denied');
    };
    const getter = () => ({
      get getItem(): (key: string) => string | null {
        throw new Error('getter');
      },
      get setItem(): (key: string, value: string) => void {
        throw new Error('getter');
      },
    });
    const quota = () => ({
      getItem: () => null,
      setItem: () => {
        throw new Error('quota');
      },
    });
    for (const source of [denied, getter, quota, () => undefined]) {
      expect(loadProgress(source)).toEqual(createInitialProgress());
      expect(saveProgress(createInitialProgress(), source)).toBe(false);
      expect(loadLanguage(source, 'en')).toBe('en');
      expect(saveLanguage('zh-CN', source)).toBe(false);
    }
  });
  it('still saves language when the progress write fails', () => {
    const { source, data } = memory();
    const partial = () => ({
      ...source(),
      setItem(key: string, value: string) {
        if (key === STORAGE_KEY) throw new Error('quota');
        source().setItem(key, value);
      },
    });
    expect(
      saveSave(
        {
          version: STORAGE_VERSION,
          progress: createInitialProgress(),
          settings: { language: 'en' },
        },
        partial,
      ),
    ).toBe(false);
    expect(JSON.parse(data.get(LANGUAGE_KEY)!).language).toBe('en');
  });
  it('roundtrips registered languages in a versioned envelope', () => {
    const { source, data } = memory();
    for (const language of ['zh-CN', 'en'] as const) {
      expect(saveLanguage(language, source)).toBe(true);
      expect(JSON.parse(data.get(LANGUAGE_KEY)!)).toEqual({
        version: STORAGE_VERSION,
        language,
      });
      expect(loadLanguage(source, language === 'en' ? 'zh-CN' : 'en')).toBe(language);
    }
  });
  it('falls back for missing, corrupt, unversioned, unsupported, or invalid language data', () => {
    const { source, data } = memory();
    expect(loadLanguage(source, 'zh-CN')).toBe('zh-CN');
    for (const raw of [
      '{',
      'null',
      '[]',
      '{}',
      'en',
      JSON.stringify('en'),
      JSON.stringify({ language: 'en' }),
      JSON.stringify({ version: STORAGE_VERSION + 1, language: 'en' }),
      JSON.stringify({ version: String(STORAGE_VERSION), language: 'en' }),
      JSON.stringify({ version: STORAGE_VERSION }),
      JSON.stringify({ version: STORAGE_VERSION, language: null }),
      JSON.stringify({ version: STORAGE_VERSION, language: 'fr' }),
      JSON.stringify({ version: STORAGE_VERSION, language: 'toString' }),
    ]) {
      data.set(LANGUAGE_KEY, raw);
      expect(loadLanguage(source, 'zh-CN')).toBe('zh-CN');
    }
    expect(saveLanguage('fr' as Parameters<typeof saveLanguage>[0], source)).toBe(false);
  });
  it('throttles continuous edits, flushes the newest value, and cancels pending saves', () => {
    vi.useFakeTimers();
    const save = vi.fn();
    const saver = createThrottledSave(save, 100);
    saver.schedule(1);
    vi.advanceTimersByTime(50);
    saver.schedule(2);
    vi.advanceTimersByTime(50);
    expect(save.mock.calls).toEqual([[2]]);
    saver.schedule(3);
    expect(saver.flush()).toBe(true);
    vi.advanceTimersByTime(100);
    expect(save.mock.calls).toEqual([[2], [3]]);
    saver.schedule(4);
    saver.cancel();
    vi.advanceTimersByTime(100);
    expect(save).toHaveBeenCalledTimes(2);
    const fails = createThrottledSave(() => {
      throw new Error('denied');
    });
    fails.schedule(undefined);
    expect(fails.flush()).toBe(false);
  });
});

it('the gradient view unlocks with the autodiff level, not with chapter 2', async () => {
  const { isAutodiffUnlocked } = await import('./storage');
  const { gameLevels } = await import('./levels/catalog');
  const progress = createInitialProgress();
  progress.unlocked.push('2-1');
  expect(isAutodiffUnlocked(progress, gameLevels)).toBe(false);
  progress.passed.push('1-11');
  expect(isAutodiffUnlocked(progress, gameLevels)).toBe(true);
});
