// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { gameLevels } from '../levels/catalog';
import type { ForwardLevel } from '../engine/levels';
import { expectedOutput, expectedOutputs, matchesExpected } from './expected-output';

const level = (id: string) => gameLevels.find((l) => l.id === id) as ForwardLevel;

describe('expected output in the live panel', () => {
  it('shows f for forward levels', async () => {
    const r = await expectedOutput(level('2-4'), { a: [1, 2, 3, 4], b: [1, 0, -1] });
    expect(r.shape).toEqual([4, 3]);
    expect(r.values.slice(0, 6)).toEqual([1, 0, -1, 2, 0, -2]);
  });
  it("shows f' for derivative levels", async () => {
    const r = await expectedOutput(level('1-1'), { x: [3] });
    expect(r).toEqual({ shape: [], values: [6] });
  });
  it('returns derivative answers under their declared output names', async () => {
    const surface = await expectedOutputs(level('1-10'), { a: [2], b: [3] });
    expect(Object.keys(surface)).toEqual(['∂a', '∂b']);
    expect(surface['∂a'].shape).toEqual([]);
    expect(surface['∂a'].values[0]).toBeCloseTo(3 + Math.exp(2), 6);
    expect(surface['∂b']).toEqual({ shape: [], values: [2] });
    expect(await expectedOutput(level('1-10'), { a: [2], b: [3] })).toEqual(surface['∂a']);

    const backprop = await expectedOutputs(level('1-11'), {
      w: [3],
      x: [2],
      '∂f/∂z': [0.5],
    });
    expect(Object.keys(backprop)).toEqual(['∂f/∂w', '∂f/∂x', '∂f/∂b']);
    expect(backprop['∂f/∂w'].values[0]).toBeCloseTo(1, 6);
    expect(backprop['∂f/∂x'].values[0]).toBeCloseTo(1.5, 6);
    expect(backprop['∂f/∂b'].values[0]).toBeCloseTo(0.5, 6);
  });
  it('uses the judge tolerance and shape check', () => {
    const l = level('1-1');
    expect(matchesExpected(l, { shape: [], values: [6.00001] }, { shape: [], values: [6] })).toBe(
      true,
    );
    expect(matchesExpected(l, { shape: [1], values: [6] }, { shape: [], values: [6] })).toBe(false);
    expect(matchesExpected(l, { shape: [], values: [NaN] }, { shape: [], values: [NaN] })).toBe(
      false,
    );
  });
});
