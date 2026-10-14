import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createInitialProgress,
  createThrottledSave,
  isGraphJSON,
  isProgress,
  LANGUAGE_KEY,
  loadLanguage,
  loadProgress,
  saveLanguage,
  saveProgress,
  saveSave,
  STORAGE_KEY,
  STORAGE_VERSION,
} from './storage';
import type { GraphJSON } from './engine/graph';

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
  it('roundtrips versioned progress with a fresh sigmoid initial state', () => {
    const { source } = memory();
    const progress = loadProgress(source);
    expect(progress.unlocked).toEqual(['sigmoid']);
    progress.savedGraphs.sigmoid = graph;
    progress.bestScores.sigmoid = { topNodes: 2, expandedPrimitives: 0, parameterCount: 0 };
    expect(saveProgress(progress, source)).toBe(true);
    expect(loadProgress(source)).toEqual(progress);
    expect(createInitialProgress().savedGraphs).toEqual({});
  });
  it('rejects corrupt, wrong-version and deeply invalid stored values', () => {
    const { data, source } = memory();
    for (const raw of [
      '{',
      'null',
      '{}',
      JSON.stringify({ version: 2, progress: createInitialProgress() }),
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
        { version: 1, progress: createInitialProgress(), settings: { language: 'en' } },
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
