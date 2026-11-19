// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Developer-menu progress shortcuts. Dev server only: nothing here is reachable from a
 * production build. Passes are recorded with each level's reference solution, so later
 * levels that compile earlier circuits (player components, 5-x experiments) still work.
 */
import { createInitialProgress } from '../storage';
import type { Level, Progress } from '../engine/levels';

const union = (a: readonly string[], b: readonly string[]) => [...new Set([...a, ...b])];
// JSON round-trip, not structuredClone: some reference graphs carry `attrs: undefined`,
// which the save validator rightly rejects and which an editor-built graph never has.
const plain = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/** Records one level as passed, as a real pass would: successors, components, solution. */
export function devPass(progress: Progress, level: Level): Progress {
  const next: Progress = {
    ...progress,
    unlocked: union(progress.unlocked, [level.id, ...level.unlock]),
    passed: union(progress.passed, [level.id]),
    unlockedComponents: union(progress.unlockedComponents, level.unlockComponents ?? []),
  };
  if (level.kind === 'observation' || !level.referenceGraph) return next;
  return {
    ...next,
    passedSolutions: { ...next.passedSolutions, [level.id]: plain(level.referenceGraph) },
    // Keep a draft the developer already wired; otherwise open on the reference.
    savedGraphs: next.savedGraphs[level.id]
      ? next.savedGraphs
      : { ...next.savedGraphs, [level.id]: plain(level.referenceGraph) },
  };
}

/** Passes every level before `targetId` in road order and unlocks the target itself. */
export function devPassUntil(
  progress: Progress,
  levels: readonly Level[],
  targetId: string,
): Progress {
  const index = levels.findIndex((level) => level.id === targetId);
  if (index < 0) throw new RangeError(`Unknown level: ${targetId}`);
  const passed = levels.slice(0, index).reduce(devPass, progress);
  return { ...passed, unlocked: union(passed.unlocked, [targetId]) };
}

export function devPassAll(progress: Progress, levels: readonly Level[]): Progress {
  return levels.reduce(devPass, progress);
}

/** Opens every level without passing any (components stay as earned). */
export function devUnlockAll(progress: Progress, levels: readonly Level[]): Progress {
  return {
    ...progress,
    unlocked: union(
      progress.unlocked,
      levels.map((level) => level.id),
    ),
  };
}

export function devReset(): Progress {
  return createInitialProgress();
}
