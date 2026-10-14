import { describe, expect, it } from 'vitest';
import { gameLevels } from '../levels/catalog';
import type { ForwardLevel } from '../engine/levels';
import { expectedOutput, matchesExpected } from './expected-output';

const level = (id: string) => gameLevels.find((l) => l.id === id) as ForwardLevel;

describe('expected output in the live panel', () => {
  it('shows f for forward levels', async () => {
    const r = await expectedOutput(level('2-3'), { a: [1, 2, 3, 4], b: [1, 0, -1] });
    expect(r.shape).toEqual([4, 3]);
    expect(r.values.slice(0, 6)).toEqual([1, 0, -1, 2, 0, -2]);
  });
  it("shows f' for derivative levels", async () => {
    const r = await expectedOutput(level('1-1'), { x: [3] });
    expect(r).toEqual({ shape: [], values: [6] });
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
