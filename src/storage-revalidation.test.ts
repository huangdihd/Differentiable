// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
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

const div = chapter0Levels.find((level) => level.id === '0-4')!;
// Saves from before reciprocal became a basic primitive still carry the retired 0-3 lesson.
const oldRecip = formulaGraph('exp(neg(log(x)))');
const oldDiv = formulaGraph('a * exp(neg(log(b)))', ['a', 'b']);
function fixture(version: number) {
  const progress = {
    ...createInitialProgress(),
    unlocked: ['0-1', '0-2', '0-3', '0-4', '0-5'],
    unlockedComponents: ['neg', 'sub', 'recip', 'div'],
    passed: ['0-3', '0-4'],
    passedSolutions: { '0-3': oldRecip, '0-4': div.referenceGraph! },
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
    'keeps sign-safe division in v%s even when the old reciprocal was positive-only',
    async (version) => {
      const { source } = fixture(version);
      const onRevoked = vi.fn();
      const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
      expect(onRevoked).not.toHaveBeenCalled();
      expect(restored.passed).toContain('0-4');
      expect(restored.passedSolutions['0-4']).toEqual(div.referenceGraph);
      const registry = buildPlayerRegistry(chapter0Levels, restored);
      // The retired lesson's saved solution no longer replaces the native primitive.
      expect(registry.get('recip')?.subgraph).toBeUndefined();
      const negative = {
        ...div,
        fixed: [{ id: 'negative-denominator', inputs: { a: [6], b: [-2] } }],
        generated: undefined,
        shapeVariants: undefined,
      };
      expect(
        (
          await evaluateLevel(
            { ...negative, available: availableComponents(div, restored) },
            div.referenceGraph!,
            { registry },
          )
        ).passed,
      ).toBe(true);
      expect(loadProgress(source)).toEqual(restored);
    },
  );

  it('revokes directly embedded positive-only division and preserves an existing editor draft', async () => {
    const { source, data, progress } = fixture(STORAGE_VERSION);
    const draft = formulaGraph('x');
    progress.passedSolutions['0-4'] = oldDiv;
    progress.savedGraphs = { '0-3': draft };
    data.set(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, progress }));
    const onRevoked = vi.fn();
    const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
    expect(onRevoked).toHaveBeenCalledExactlyOnceWith(['0-4']);
    expect(restored.passed).not.toContain('0-4');
    expect(restored.savedGraphs).toEqual({ '0-3': draft, '0-4': oldDiv });
    expect(restored.unlockedComponents).toEqual(progress.unlockedComponents);
  });

  it('revokes legacy positive-only division while retaining earned access', async () => {
    const { source, data, progress } = fixture(1);
    progress.savedGraphs = { '0-3': oldRecip, '0-4': oldDiv };
    data.set(STORAGE_KEY, JSON.stringify({ version: 1, progress }));
    const onRevoked = vi.fn();
    const restored = await loadProgressWithMigration(chapter0Levels, source, onRevoked);
    expect(onRevoked).toHaveBeenCalledExactlyOnceWith(['0-4']);
    expect(restored.passed).not.toContain('0-4');
    // The retired lesson drops out; passing 0-2 also opens the square lesson inserted after it.
    expect(restored.unlocked).toEqual(['0-1', '0-2', '0-4', '0-5', 'square']);
    expect(restored.unlockedComponents).toEqual(progress.unlockedComponents);
    // Drafts for the retired lesson are dropped with it.
    expect(restored.savedGraphs).toEqual({ '0-4': oldDiv });
  });
});
