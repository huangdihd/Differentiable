import { describe, expect, it } from 'vitest';
import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import { plottedInput, sampleXs, segments, yRange } from './function-plot';

const level = (id: string) => [...chapter0Levels, ...chapter1Levels].find((l) => l.id === id)!;

describe('function plot', () => {
  it('applies to one-input scalar lessons only', () => {
    expect(plottedInput(level('0-5'))).toBe('x');
    expect(plottedInput(level('0-2'))).toBeUndefined();
  });

  it('samples around the level cases, never at 0, and stays positive for √x', () => {
    const xs = sampleXs(level('0-5'), 'x');
    expect(xs.includes(0)).toBe(false);
    expect(Math.min(...xs)).toBeLessThanOrEqual(-5);
    expect(Math.min(...sampleXs(level('0-8'), 'x'))).toBeGreaterThan(0);
  });

  it('keeps the shape of 1/x visible and breaks it at the pole', () => {
    const xs = sampleXs(level('0-5'), 'x');
    const ys = xs.map((x) => 1 / x);
    const range = yRange(ys);
    expect(range[1] - range[0]).toBeLessThan(10);
    expect(segments(xs, ys, range).length).toBe(2);
  });

  it('skips undefined points', () => {
    expect(segments([0, 1, 2, 3, 4], [0, 1, NaN, 3, 4], [-5, 5])).toEqual([
      [
        [0, 0],
        [1, 1],
      ],
      [
        [3, 3],
        [4, 4],
      ],
    ]);
  });
});
