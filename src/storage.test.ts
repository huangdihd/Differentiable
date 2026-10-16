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
    expect(restored.unlocked).toContain('3-1');
    expect(restored.unlocked).toContain('xor');
  });
  it('unlocks chapter four for saved XOR graduates without changing their graphs', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const progress = {
      ...createInitialProgress(),
      unlocked: ['xor'],
      passed: ['xor'],
      passedSolutions: { xor: graph },
      savedGraphs: { xor: graph },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toEqual(['xor', '4-1']);
    expect(restored.passedSolutions).toEqual(progress.passedSolutions);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
    expect(saveProgress(restored, source)).toBe(true);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
  });
  it('unlocks chapter six for saved chapter-five graduates without changing their graphs', async () => {
    const { gameLevels } = await import('./levels/catalog');
    const { source } = memory();
    const progress = {
      ...createInitialProgress(),
      unlocked: ['5-5'],
      passed: ['5-5'],
      passedSolutions: { '5-5': graph },
      savedGraphs: { '5-5': graph },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toEqual(['5-5', '6-1']);
    expect(restored.passed).toEqual(progress.passed);
    expect(restored.passedSolutions).toEqual(progress.passedSolutions);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
    expect(saveProgress(restored, source)).toBe(true);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
    expect(saveProgress(createInitialProgress(), source)).toBe(true);
    expect((await loadProgressWithMigration(gameLevels, source)).unlocked).not.toContain('6-1');
  });
  it.each([2, STORAGE_VERSION])(
    'restores axes access without changing other data in version %s saves',
    async (version) => {
      const levels = [
        { ...chapter1Levels[0], id: '1-10', unlock: ['axes'] },
        { ...chapter1Levels[0], id: 'axes', unlock: ['2-1'] },
        { ...chapter1Levels[0], id: '2-1', unlock: [] },
      ];
      for (const [id, passed, expected] of [
        ['1-10', false, ['1-10']],
        ['1-10', true, ['1-10', 'axes']],
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
      id: '1-10',
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
      expect(restored.passed).toEqual(passed ? (id === '2-1' ? ['1-10', '2-1'] : [id]) : []);
      expect(restored.passedSolutions.axes).toBeUndefined();
      expect(restored.bestScores.axes).toBeUndefined();
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
    'restores reshape access without changing other data in version %s saves',
    async (version) => {
      const levels = [
        { ...chapter1Levels[0], id: '4-4', unlock: ['reshape'] },
        { ...chapter1Levels[0], id: 'reshape', unlock: ['4-5'] },
        { ...chapter1Levels[0], id: '4-5', unlock: [] },
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
        { ...chapter1Levels[0], id: '3-2', unlock: ['neuron'] },
        { ...chapter1Levels[0], id: 'neuron', unlock: ['layer'] },
        { ...chapter1Levels[0], id: 'layer', unlock: ['3-3'] },
        { ...chapter1Levels[0], id: '3-3', unlock: ['3-4'] },
        { ...chapter1Levels[0], id: '3-4', unlock: ['3-5'] },
        { ...chapter1Levels[0], id: '3-5', unlock: ['3-6'] },
      ];
      for (const [unlocked, passed, additions] of [
        [['3-2'], [], []],
        [['3-2'], ['3-2'], ['neuron']],
        [['3-3'], [], ['neuron', 'layer']],
        [['3-3'], ['3-3'], ['3-4', 'neuron', 'layer']],
        [
          ['3-2', '3-3', '3-4', '3-5'],
          ['3-2', '3-3', '3-4', '3-5'],
          ['neuron', '3-6', 'layer'],
        ],
        [['neuron'], ['neuron'], ['layer']],
        [['neuron', 'layer', '3-3'], ['neuron', 'layer'], []],
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
      unlock: ['neuron'],
      fixed: [{ id: 'case', inputs: { x: [1, 2] } }],
      reference: (inputs) => inputs.x,
      referenceGraph: reference,
    };
    const levels = [
      lesson,
      { ...lesson, id: 'neuron', unlock: ['layer'] },
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
    expect(isExpressionUnlocked({ ...fresh, passed: ['1-10'] }, chapter1Levels)).toBe(true);
    expect(
      isExpressionUnlocked({ ...fresh, passed: ['1-7'], unlocked: ['2-1'] }, chapter1Levels),
    ).toBe(true);
    expect(
      isExpressionUnlocked({ ...fresh, passed: ['custom'] }, [
        { ...chapter1Levels[0], id: 'custom', unlockFeatures: ['levels.features.expression'] },
      ]),
    ).toBe(true);
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
    const progress = {
      ...createInitialProgress(),
      unlocked: ['0-1', '1-7', '2-1', '2-2', 'xor'],
      passed: ['1-7'],
      passedSolutions: { '1-7': graph },
      savedGraphs: { '1-7': graph },
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
      ...chapter1Levels[0],
      id: '2-3',
      unlock: ['2-4', '2-6'],
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
    expect(progress.passedSolutions['1-2']).toEqual(derivativeSolution);
    expect(progress.unlocked).toContain('1-3');
    expect(progress.unlockedComponents).toContain('sigmoid');
  });
  it('keeps legacy language preferences during the version upgrade', () => {
    const { data, source } = memory();
    for (const version of [1, 2]) {
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
  progress.passed.push('1-10');
  expect(isAutodiffUnlocked(progress, gameLevels)).toBe(true);
});
