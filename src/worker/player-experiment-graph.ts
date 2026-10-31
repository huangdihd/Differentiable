// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { GraphError } from '../engine/graph';
import type { Progress } from '../engine/levels';
import { gameLevels } from '../levels/catalog';
import { buildPlayerRegistry } from '../levels/registry';

/**
 * Experiments require the actual saved, passed root graph; never substitute a reference root.
 * Dependencies follow the same availability and saved-solution rules as the editor registry.
 */
export function playerExperimentGraph(progress: Progress, levelId: string) {
  const graph = progress.passedSolutions[levelId];
  if (!progress.passed.includes(levelId) || !graph)
    throw new GraphError('worker.player_experiment_graph_required', undefined, undefined, {
      level: levelId,
    });
  return { graph: structuredClone(graph), registry: buildPlayerRegistry(gameLevels, progress) };
}
