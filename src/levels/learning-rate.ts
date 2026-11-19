// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ObservationLevel } from '../engine/levels';
import { curves, type CurveId } from './primer-calculus';

export const learningRateLevel: ObservationLevel = {
  id: 'learning-rate',
  chapter: '3',
  kind: 'observation',
  titleKey: 'levels.learning-rate.title',
  descriptionKey: 'levels.learning-rate.description',
  unlock: ['neuron'],
};

/**
 * Gradient descent with a learning rate. Each step is x ← x − rate · f′(x); the
 * player only chooses the rate. Two fixed demos (too small, too large), then two tuning goals
 * on bowls of different steepness, so one rate that worked can fail on the next.
 */
export interface LearningRateStage {
  curve: Extract<CurveId, 'bowl' | 'steep'>;
  start: number;
  /** Fixed demo rate; absent when the player tunes it. */
  rate?: number;
  /** Steps per run (a demo plays them all; a goal must arrive within them). */
  steps: number;
  /** Goal stages: arrive within this distance of the minimum at x = 0. */
  tolerance?: number;
}
export const LEARNING_RATE_RANGE = { min: 0.01, max: 1.2, step: 0.01 } as const;
export const learningRateStages: readonly LearningRateStage[] = [
  { curve: 'bowl', start: 2.5, rate: 0.02, steps: 12 },
  { curve: 'bowl', start: 2.5, rate: 1.05, steps: 8 },
  { curve: 'bowl', start: 2.5, steps: 6, tolerance: 0.05 },
  { curve: 'steep', start: 2.5, steps: 6, tolerance: 0.05 },
];
export function gradientStep(x: number, rate: number, curve: CurveId): number {
  return x - rate * curves[curve].df(x);
}
/** The whole run, start included; stops early once a step leaves any sensible range. */
export function descend(stage: LearningRateStage, rate: number): number[] {
  const path = [stage.start];
  for (let i = 0; i < stage.steps; i += 1) {
    const next = gradientStep(path.at(-1)!, rate, stage.curve);
    path.push(next);
    if (!Number.isFinite(next) || Math.abs(next) > 1e6) break;
  }
  return path;
}
export function reachedMinimum(path: readonly number[], stage: LearningRateStage): boolean {
  if (stage.tolerance === undefined) return false;
  return path.some((x) => Math.abs(x) <= stage.tolerance!);
}
