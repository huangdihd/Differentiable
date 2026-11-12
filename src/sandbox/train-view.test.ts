// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { lossSeries, plotAccuracy } from './train-view';

describe('lossSeries', () => {
  it('uses a log scale for positive losses and keeps every point when short', () => {
    const s = lossSeries([1, 0.5, 0.2, 0.05]);
    expect(s.log).toBe(true);
    expect(s.points).toEqual([
      [0, 1],
      [1, 0.5],
      [2, 0.2],
      [3, 0.05],
    ]);
    expect([s.lo, s.hi]).toEqual([0.01, 1]);
    expect(s.diverged).toBe(false);
  });
  it('thins long runs but keeps spikes', () => {
    const losses = Array.from({ length: 100_000 }, (_, i) => (i === 54_321 ? 50 : 1 / (1 + i)));
    const s = lossSeries(losses, 500);
    expect(s.points.length).toBeLessThanOrEqual(1000);
    expect(s.points.some(([i]) => i === 54_321)).toBe(true);
    expect(s.count).toBe(100_000);
  });
  it('marks NaN and ∞ steps and flags divergence instead of breaking', () => {
    const s = lossSeries([1, 3, 400, 1e9, Infinity, NaN]);
    expect(s.bad).toEqual([4, 5]);
    expect(s.points.map(([i]) => i)).toEqual([0, 1, 2, 3]);
    expect(s.diverged).toBe(true);
    expect(lossSeries([2, 2500]).diverged).toBe(true);
  });
  it('switches to a linear scale when a hand-made loss goes negative', () => {
    const s = lossSeries([1, 0, -1]);
    expect(s.log).toBe(false);
    expect(s.lo).toBeLessThan(-1);
    expect(s.hi).toBeGreaterThan(1);
    expect(lossSeries([]).points).toEqual([]);
  });
});

describe('plotAccuracy', () => {
  it('counts predicted classes that match labels', () => {
    expect(
      plotAccuracy({
        kind: 'classes',
        points: [
          [0, 0],
          [1, 1],
          [2, 2],
          [3, 3],
        ],
        label: [0, 1, 1, 0],
        predicted: [0, 1, 0, 0],
      }),
    ).toBe(0.75);
    expect(plotAccuracy({ kind: 'regression', x: [1], t: [1] })).toBeUndefined();
    expect(plotAccuracy(undefined)).toBeUndefined();
  });
});
