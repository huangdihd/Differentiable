// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  STORAGE_KEY,
  STORAGE_VERSION,
  createInitialProgress,
  isProgress,
  loadProgress,
  loadProgressWithMigration,
} from '../storage';
import { gameLevels } from '../levels/catalog';
import { createSlot } from './slots';

function memory() {
  const data = new Map<string, string>();
  const source = () => ({
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
  });
  return { data, source };
}

describe('sandbox storage', () => {
  it('upgrades a v10 save to the current version without changing it', async () => {
    expect(STORAGE_VERSION).toBe(11);
    const { data, source } = memory();
    const progress = createInitialProgress();
    data.set(STORAGE_KEY, JSON.stringify({ version: 10, progress }));
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.sandbox).toBeUndefined();
    expect(restored.passed).toEqual(progress.passed);
    expect(JSON.parse(data.get(STORAGE_KEY)!).version).toBe(STORAGE_VERSION);
  });
  it('keeps slots with imported data and weights across reloads', async () => {
    const { data, source } = memory();
    let progress = createSlot(createInitialProgress(), {
      id: 'a',
      name: 'A',
      now: 1,
      dataset: {
        kind: 'imported',
        name: 'n',
        modality: 'line',
        task: 'regression',
        x: [[1], [2]],
        t: [[2], [4]],
      },
    });
    progress = {
      ...progress,
      sandbox: {
        slots: progress.sandbox!.slots.map((s) => ({
          ...s,
          weights: {
            format: 'differentiable-weights' as const,
            version: 1 as const,
            parameters: { w: { shape: [1], values: [3] } },
          },
        })),
      },
    };
    data.set(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, progress }));
    const restored = await loadProgressWithMigration(gameLevels, source);
    expect(restored.sandbox).toEqual(progress.sandbox);
    expect(loadProgress(source).sandbox).toEqual(progress.sandbox);
  });
  it('rejects malformed slots', () => {
    const base = createInitialProgress();
    expect(isProgress({ ...base, sandbox: { slots: [{ id: 'a' }] } })).toBe(false);
    expect(isProgress({ ...base, sandbox: { slots: 'x' } })).toBe(false);
    expect(isProgress({ ...base, sandbox: { slots: [] } })).toBe(true);
  });
});
