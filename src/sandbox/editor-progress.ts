// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * The Editor saves graphs and training settings per level id. A sandbox slot looks like a level
 * to it; these two functions move that state into and back out of the slot.
 */
import type { Progress } from '../engine/levels';
import { sandboxLevelId, updateSlot, type SandboxSlot } from './slots';

/** Graph and training settings live in the slot; the Editor sees them as a level's own. */
export function editorProgress(progress: Progress, slot: SandboxSlot, levelId: string): Progress {
  return {
    ...progress,
    savedGraphs: slot.graph
      ? { ...progress.savedGraphs, [levelId]: slot.graph }
      : progress.savedGraphs,
    ...(slot.training || progress.trainingSettings
      ? {
          trainingSettings: {
            ...progress.trainingSettings,
            ...(slot.training ? { [levelId]: slot.training } : {}),
          },
        }
      : {}),
  };
}

/** Move the Editor's per-level writes back into the slot and out of curriculum progress. */
export function fromEditorProgress(next: Progress, slotId: string, now: number): Progress {
  const levelId = sandboxLevelId(slotId);
  const graph = next.savedGraphs[levelId];
  const training = next.trainingSettings?.[levelId];
  const savedGraphs = { ...next.savedGraphs };
  delete savedGraphs[levelId];
  const stripped: Progress = { ...next, savedGraphs };
  if (next.trainingSettings) {
    const trainingSettings = { ...next.trainingSettings };
    delete trainingSettings[levelId];
    stripped.trainingSettings = trainingSettings;
  }
  return updateSlot(
    stripped,
    slotId,
    { ...(graph ? { graph } : {}), ...(training ? { training } : {}) },
    now,
  );
}
