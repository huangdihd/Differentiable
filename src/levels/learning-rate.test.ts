// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  LEARNING_RATE_RANGE,
  descend,
  gradientStep,
  learningRateStages,
  reachedMinimum,
  learningRateLevel,
} from './learning-rate';

describe('learning rate', () => {
  it('leads from the chapter 3 observation to the neuron lesson', () => {
    expect(learningRateLevel).toMatchObject({
      id: 'learning-rate',
      chapter: '3',
      kind: 'observation',
      unlock: ['neuron'],
    });
  });
  it('steps against the gradient, scaled by the rate', () => {
    expect(gradientStep(2, 0.25, 'bowl')).toBe(1);
    expect(gradientStep(2, 0.25, 'steep')).toBe(0);
    expect(gradientStep(-1, 0.5, 'bowl')).toBe(0);
  });
  it('demos crawl when too small and diverge when too large', () => {
    const [small, large] = learningRateStages;
    const crawl = descend(small, small.rate!);
    expect(crawl).toHaveLength(small.steps + 1);
    // Twelve tiny steps still leave the walker more than halfway up the hill.
    expect(Math.abs(crawl.at(-1)!)).toBeGreaterThan(small.start / 2);
    const bounce = descend(large, large.rate!);
    // Every step overshoots to the other side, and each swing is wider than the last.
    for (let i = 1; i < bounce.length; i++) {
      expect(Math.sign(bounce[i])).toBe(-Math.sign(bounce[i - 1]));
      expect(Math.abs(bounce[i])).toBeGreaterThan(Math.abs(bounce[i - 1]));
    }
  });
  it('each goal has a reachable rate window, and the steep bowl needs a different one', () => {
    const goals = learningRateStages.filter((stage) => stage.tolerance !== undefined);
    expect(goals.map((stage) => stage.curve)).toEqual(['bowl', 'steep']);
    const windows = goals.map((stage) => {
      const ok: number[] = [];
      for (
        let r = LEARNING_RATE_RANGE.min;
        r <= LEARNING_RATE_RANGE.max + 1e-9;
        r += LEARNING_RATE_RANGE.step
      ) {
        const rate = Math.round(r * 100) / 100;
        if (reachedMinimum(descend(stage, rate), stage)) ok.push(rate);
      }
      return ok;
    });
    for (const ok of windows) expect(ok.length).toBeGreaterThanOrEqual(10);
    // A rate that is comfortable on the first bowl overshoots on the steep one.
    expect(windows[0]).toContain(0.6);
    expect(windows[1]).not.toContain(0.6);
    for (const stage of goals) expect(reachedMinimum([stage.start], stage)).toBe(false);
    expect(reachedMinimum([0], learningRateStages[0])).toBe(false);
  });
});
