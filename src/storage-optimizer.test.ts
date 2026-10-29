import { describe, expect, it } from 'vitest';
import {
  createInitialProgress,
  isProgress,
  loadProgress,
  loadProgressWithMigration,
  saveProgress,
  STORAGE_KEY,
  STORAGE_VERSION,
} from './storage';
import { gameLevels } from './levels/catalog';
import { SYMBOLS } from './ui/graph-geometry';
import { locales, translate } from './i18n';

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

describe('optimizer save compatibility', () => {
  it('loads old saves without requiring an optimizer or training settings', () => {
    const { data, source } = memory();
    for (const progress of [
      createInitialProgress(),
      { ...createInitialProgress(), trainingSettings: { '3-4': { learningRate: 0.2 } } },
    ]) {
      data.set(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, progress }));
      expect(loadProgress(source)).toEqual(progress);
    }
  });

  it.each(['sgd', 'adam'] as const)(
    'roundtrips %s without changing the learning rate',
    (optimizer) => {
      const { source } = memory();
      const progress = {
        ...createInitialProgress(),
        trainingSettings: { '3-4': { learningRate: 0.2, optimizer } },
      };
      expect(isProgress(progress)).toBe(true);
      expect(saveProgress(progress, source)).toBe(true);
      expect(loadProgress(source)).toEqual(progress);
    },
  );

  it.each(['Adam', 'rmsprop', '', null, 0, false, {}, []])(
    'rejects an invalid optimizer %j without replacing the stored progress',
    async (optimizer) => {
      const { data, source } = memory();
      const progress = {
        ...createInitialProgress(),
        trainingSettings: { '3-4': { learningRate: 0.2, optimizer } },
      };
      expect(isProgress(progress)).toBe(false);
      const raw = JSON.stringify({ version: STORAGE_VERSION, progress });
      data.set(STORAGE_KEY, raw);
      expect(() => loadProgress(source)).toThrow('storage.progress_load_failed');
      await expect(loadProgressWithMigration(gameLevels, source)).rejects.toThrow(
        'storage.progress_load_failed',
      );
      expect(data.get(STORAGE_KEY)).toBe(raw);
    },
  );

  it('refreshes the 3-4 successor without losing legacy XOR progress or circuits', async () => {
    const { source } = memory();
    const graph = { version: 1 as const, nodes: [], edges: [] };
    const progress = {
      ...createInitialProgress(),
      unlocked: ['3-4', 'xor'],
      passed: ['3-4', 'xor'],
      savedGraphs: { '3-4': graph, xor: graph },
      passedSolutions: { '3-4': graph, xor: graph },
      bestScores: { xor: { topNodes: 0, expandedPrimitives: 0, parameterCount: 0 } },
      trainingSettings: { '3-4': { learningRate: 0.2 }, xor: { learningRate: 0.01 } },
    };
    expect(saveProgress(progress, source)).toBe(true);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).toContain('3-5');
    expect(restored.unlocked).toContain('xor');
    expect(restored.passed).toEqual(progress.passed);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
    expect(restored.passedSolutions).toEqual(progress.passedSolutions);
    expect(restored.bestScores).toEqual(progress.bestScores);
    expect(restored.trainingSettings).toEqual(progress.trainingSettings);
    expect(saveProgress(restored, source)).toBe(true);
    expect(await loadProgressWithMigration(gameLevels, source)).toEqual(restored);
  });

  it('does not unlock 3-5 before 3-4 is passed', async () => {
    const { source } = memory();
    saveProgress({ ...createInitialProgress(), unlocked: ['3-4', 'xor'] }, source);
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.unlocked).not.toContain('3-5');
    expect(restored.unlocked).toContain('xor');
  });

  it('localizes all new lessons and optimizer controls in both languages', () => {
    const keys = [
      ...[5, 6, 7, 8, 9].flatMap((index) =>
        ['title', 'description', 'hint1', 'hint2'].map((field) => `levels.3-${index}.${field}`),
      ),
      'components.cross_entropy',
      'levels.train_accuracy',
      'ui.optimizer',
      'ui.optimizer.sgd',
      'ui.optimizer.adam',
    ];
    for (const locale of ['en', 'zh-CN'] as const) {
      for (const key of keys) {
        expect(Object.hasOwn(locales[locale], key)).toBe(true);
        expect(translate(locale, key)).not.toBe(key);
      }
      for (const id of ['3-6', '3-8', '3-9']) {
        const level = gameLevels.find((entry) => entry.id === id)!;
        if (level.kind !== 'train') throw new Error(id);
        expect(translate(locale, `levels.${id}.description`)).toContain(String(level.steps));
      }
      expect(translate(locale, 'levels.train_accuracy', { accuracy: 0.9 })).not.toContain(
        '{accuracy}',
      );
    }
  });

  it('uses a compact cross entropy symbol', () => {
    expect(SYMBOLS.cross_entropy).toBe('CE');
  });
});
