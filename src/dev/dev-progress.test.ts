// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { gameLevels } from '../levels/catalog';
import { isSandboxUnlocked } from '../sandbox/slots';
import {
  createInitialProgress,
  isProgress,
  loadProgressWithMigration,
  saveProgress,
} from '../storage';
import { devPassAll, devPassUntil, devReset, devUnlockAll } from './dev-progress';

function memoryStorage() {
  const items = new Map<string, string>();
  return {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => void items.set(key, value),
    removeItem: (key: string) => void items.delete(key),
  };
}

describe('developer progress shortcuts', () => {
  it('passes everything before the target and opens the target', () => {
    const progress = devPassUntil(createInitialProgress(), gameLevels, 'axes');
    const index = gameLevels.findIndex((level) => level.id === 'axes');
    expect(isProgress(progress)).toBe(true);
    for (const level of gameLevels.slice(0, index)) expect(progress.passed).toContain(level.id);
    expect(progress.passed).not.toContain('axes');
    expect(progress.unlocked).toContain('axes');
    expect(progress.passedSolutions['calc-numeric']).toBeDefined();
    expect(progress.passedSolutions['calc-slope']).toBeUndefined();
  });

  it('unlocks without passing, and passing everything opens the sandbox', () => {
    const unlocked = devUnlockAll(createInitialProgress(), gameLevels);
    expect(unlocked.passed).toEqual([]);
    expect(gameLevels.every((level) => unlocked.unlocked.includes(level.id))).toBe(true);
    expect(isSandboxUnlocked(devPassAll(createInitialProgress(), gameLevels), gameLevels)).toBe(
      true,
    );
    expect(devReset()).toEqual(createInitialProgress());
  });

  it('survives a reload: migration keeps every shortcut pass', async () => {
    const storage = memoryStorage();
    const progress = devPassAll(createInitialProgress(), gameLevels);
    expect(saveProgress(progress, () => storage)).toBe(true);
    const revoked: string[] = [];
    const loaded = await loadProgressWithMigration(
      gameLevels,
      () => storage,
      (ids) => revoked.push(...ids),
    );
    expect(revoked).toEqual([]);
    expect([...loaded.passed].sort()).toEqual([...progress.passed].sort());
  }, 120_000);
});
