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
  it('refreshes catalog unlocks in old v2 saves without changing passed circuits or edits', async () => {
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
      expect(JSON.parse(data.get(STORAGE_KEY)!).version).toBe(2);
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
    expect(progress.passed).toEqual(
      [...chapter0Levels, ...chapter1Levels.slice(0, 2)].map((level) => level.id),
    );
    expect(progress.passedSolutions['1-2']).toEqual(derivativeSolution);
    expect(progress.unlocked).toContain('1-3');
    expect(progress.unlockedComponents).toContain('sigmoid');
  });
  it('keeps legacy language preferences during the version upgrade', () => {
    const { data, source } = memory();
    data.set(LANGUAGE_KEY, JSON.stringify({ version: 1, language: 'en' }));
    expect(loadLanguage(source, 'zh-CN')).toBe('en');
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
