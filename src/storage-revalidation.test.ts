import { describe, expect, it, vi } from 'vitest';
import { evaluateLevel } from './engine/levels';
import { chapter0Levels } from './levels/chapter0';
import { formulaGraph } from './levels/level-helpers';
import { availableComponents, buildPlayerRegistry } from './levels/registry';
import {
  createInitialProgress,
  loadProgress,
  loadProgressWithMigration,
  STORAGE_KEY,
  STORAGE_VERSION,
} from './storage';

const recip = chapter0Levels.find((level) => level.id === '0-3')!;
const div = chapter0Levels.find((level) => level.id === '0-4')!;
const oldRecip = formulaGraph('exp(neg(log(x)))');
const oldDiv = formulaGraph('a * exp(neg(log(b)))', ['a', 'b']);
function fixture(version: number, modern = false) {
  const progress = {
    ...createInitialProgress(),
    unlocked: ['0-1', '0-2', '0-3', '0-4', '0-5'],
    unlockedComponents: ['neg', 'sub', 'recip', 'div'],
    passed: ['0-3', '0-4'],
    passedSolutions: {
      '0-3': modern ? recip.referenceGraph! : oldRecip,
      '0-4': div.referenceGraph!,
    },
    savedGraphs: {},
    bestScores: { '0-3': { topNodes: 1, expandedPrimitives: 1, parameterCount: 0 } },
  };
  const data = new Map([[STORAGE_KEY, JSON.stringify({ version, progress })]]);
  const source = () => ({
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
  });
  return { progress, data, source };
}

describe('changed-contract storage revalidation', () => {
  it.each([2, STORAGE_VERSION])(
    'revokes old reciprocal before checking dependent division in v%s',
    async (version) => {
      const { source, progress } = fixture(version);
      const onRevoked = vi.fn();
      const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
      expect(onRevoked).toHaveBeenCalledExactlyOnceWith(['0-3']);
      expect(restored.passed).toEqual(['0-4']);
      expect(restored.passedSolutions['0-3']).toBeUndefined();
      expect(restored.bestScores['0-3']).toBeUndefined();
      expect(restored.savedGraphs['0-3']).toEqual(oldRecip);
      expect(restored.unlocked).toEqual(progress.unlocked);
      expect(restored.unlockedComponents).toEqual(progress.unlockedComponents);
      expect(loadProgress(source)).toEqual(restored);
      const negative = {
        ...div,
        fixed: [{ id: 'negative-denominator', inputs: { a: [6], b: [-2] } }],
        generated: undefined,
        shapeVariants: undefined,
      };
      // The complete level includes negative denominators; its implementation uses the fallback recip.
      expect(
        (
          await evaluateLevel(
            { ...negative, available: availableComponents(div, restored) },
            div.referenceGraph!,
            { registry: buildPlayerRegistry(chapter0Levels, restored) },
          )
        ).passed,
      ).toBe(true);
      onRevoked.mockClear();
      expect(await loadProgressWithMigration(chapter0Levels, source, onRevoked)).toEqual(restored);
      expect(onRevoked).not.toHaveBeenCalled();
    },
  );

  it.each([2, STORAGE_VERSION])('keeps new sign-safe solutions in v%s', async (version) => {
    const { source, progress } = fixture(version, true);
    const onRevoked = vi.fn();
    expect(await loadProgressWithMigration(chapter0Levels, source, onRevoked)).toEqual(progress);
    expect(onRevoked).not.toHaveBeenCalled();
  });

  it('revokes directly embedded positive-only division and preserves an existing editor draft', async () => {
    const { source, data, progress } = fixture(STORAGE_VERSION);
    const draft = formulaGraph('x');
    progress.passedSolutions['0-4'] = oldDiv;
    progress.savedGraphs = { '0-3': draft };
    data.set(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, progress }));
    const onRevoked = vi.fn();
    const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
    expect(onRevoked).toHaveBeenCalledExactlyOnceWith(['0-3', '0-4']);
    expect(restored.passed).toEqual([]);
    expect(restored.savedGraphs).toEqual({ '0-3': draft, '0-4': oldDiv });
    expect(restored.unlockedComponents).toEqual(progress.unlockedComponents);
  });

  it('revokes legacy positive-only division while retaining earned access', async () => {
    const { source, data, progress } = fixture(1);
    progress.savedGraphs = { '0-3': oldRecip, '0-4': oldDiv };
    data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress }));
    const onRevoked = vi.fn();
    const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
    expect(onRevoked).toHaveBeenCalledExactlyOnceWith(['0-3', '0-4']);
    expect(restored.passed).not.toContain('0-3');
    expect(restored.passed).not.toContain('0-4');
    expect(restored.unlocked).toEqual(progress.unlocked);
    expect(restored.unlockedComponents).toEqual(progress.unlockedComponents);
    expect(restored.savedGraphs).toEqual(progress.savedGraphs);
  });

  it.each([false, true])(
    'validates legacy reciprocal without auto-crediting a replacement (modern=%s)',
    async (modern) => {
      const { source, data, progress } = fixture(1, modern);
      progress.savedGraphs = { ...progress.passedSolutions };
      data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress }));
      const onRevoked = vi.fn();
      const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
      expect(restored.passed.includes('0-3')).toBe(modern);
      expect(restored.passed).toContain('0-4');
      expect(restored.savedGraphs).toEqual(progress.savedGraphs);
      expect(restored.unlockedComponents).toEqual(progress.unlockedComponents);
      if (modern) expect(onRevoked).not.toHaveBeenCalled();
      else expect(onRevoked).toHaveBeenCalledExactlyOnceWith(['0-3']);
      expect(loadProgress(source)).toEqual(restored);
    },
  );
});
