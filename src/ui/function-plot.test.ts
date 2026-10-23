import { describe, expect, it } from 'vitest';
import { chapter0Levels } from '../levels/chapter0';
import { chapter1Levels } from '../levels/chapter1';
import {
  functionAsymptotes,
  functionPoint,
  gradientSurfaceData,
  gradientSurfaceInputs,
  plottedInput,
  projectSurfacePoint,
  sampleValue,
  sampleXs,
  segments,
  surfaceValue,
  unprojectSurfaceFloor,
  yRange,
} from './function-plot';

const level = (id: string) => [...chapter0Levels, ...chapter1Levels].find((l) => l.id === id)!;

describe('function plot', () => {
  it('plots one-input functions and the differentiated input of partial-derivative lessons', () => {
    expect(plottedInput(level('0-5'))).toBe('x');
    expect(plottedInput(level('0-2'))).toBeUndefined();
    expect(plottedInput(level('1-8'))).toBe('a');
    expect(plottedInput(level('1-9'))).toBe('b');
    expect(plottedInput(level('1-10'))).toBeUndefined();
    expect(plottedInput(level('1-11'))).toBeUndefined();
  });

  it('selects both scalar differentiated inputs only for gradient surfaces', () => {
    expect(gradientSurfaceInputs(level('1-10'))).toEqual(['a', 'b']);
    expect(gradientSurfaceInputs(level('1-8'))).toBeUndefined();
    expect(gradientSurfaceInputs(level('1-11'))).toBeUndefined();
  });

  it('evaluates an off-grid probe exactly, including the original function behind a derivative', async () => {
    const x = 0.037;
    const forward = await functionPoint(level('0-5'), 'x', x);
    expect(forward.target).toBeCloseTo(1 / (1 + Math.exp(-x)), 6);
    expect(forward.original).toBeUndefined();
    const derivative = await functionPoint(level('1-1'), 'x', x);
    expect(derivative.target).toBeCloseTo(2 * x, 6);
    expect(derivative.original).toBeCloseTo(x * x, 6);
  });

  it('evaluates a partial-derivative slice while holding the other inputs fixed', async () => {
    const a = 0.5,
      b = 2;
    const point = await functionPoint(level('1-8'), 'a', a, { a: [a], b: [b] });
    expect(point.target).toBeCloseTo(b + Math.exp(a), 6);
    expect(point.original).toBeCloseTo(a * b + Math.exp(a), 6);
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

  it('interpolates a live probe without crossing an explicit discontinuity', () => {
    expect(sampleValue([0, 1, 2], [0, 2, 4], 1.5)).toBe(3);
    expect(sampleValue([-1, 1], [-1, 1], 0, [0])).toBeUndefined();
    expect(sampleValue([0, 1], [0, NaN], 0.5)).toBeUndefined();
  });

  it('samples the original surface and both named partial derivatives', async () => {
    const data = await gradientSurfaceData(level('1-10'), ['a', 'b'], 3);
    expect(data.axes[0]).toEqual([-3, 0, 3]);
    expect(data.axes[1]).toEqual([-3, 0, 3]);
    expect(data.surface).toHaveLength(9);
    expect(data.target[0]).toHaveLength(9);
    expect(data.target[1]).toHaveLength(9);
    const center = 4;
    expect(data.surface[center]).toBeCloseTo(1, 6);
    expect(data.target[0][center]).toBeCloseTo(1, 6);
    expect(data.target[1][center]).toBeCloseTo(0, 6);
  });

  it('bilinearly interpolates the surface including edges and rejects invalid probes', () => {
    const axes = [
      [0, 1],
      [0, 1],
    ] as const;
    const values = [0, 2, 3, 5];
    expect(surfaceValue(axes, values, 0.25, 0.5)).toBeCloseTo(2, 6);
    expect(surfaceValue(axes, values, 0, 0)).toBe(0);
    expect(surfaceValue(axes, values, 1, 1)).toBe(5);
    expect(surfaceValue(axes, values, -0.001, 0.5)).toBeUndefined();
    expect(surfaceValue(axes, [0, 2, 3, NaN], 0.5, 0.5)).toBeUndefined();
  });

  it('round-trips floor projection for multiple camera views', () => {
    for (const view of [
      { yaw: 0, pitch: 0.5 },
      { yaw: Math.PI / 3, pitch: 0.8 },
      { yaw: -Math.PI / 2, pitch: 1.2 },
    ]) {
      const point = [0.37, -0.62] as const;
      const projected = projectSurfacePoint(point[0], point[1], -1, view);
      const restored = unprojectSurfaceFloor(projected[0], projected[1], -1, view);
      expect(restored[0]).toBeCloseTo(point[0], 10);
      expect(restored[1]).toBeCloseTo(point[1], 10);
    }
  });

  it('declares asymptotes separately for target and original curves', () => {
    expect(functionAsymptotes(level('0-5'))).toEqual([
      { axis: 'y', value: 0 },
      { axis: 'y', value: 1 },
    ]);
    expect(functionAsymptotes(level('0-6'))).toEqual([
      { axis: 'y', value: -1 },
      { axis: 'y', value: 1 },
    ]);
    expect(functionAsymptotes(level('1-4'), 'target')).toEqual([
      { axis: 'x', value: 0 },
      { axis: 'y', value: 0 },
    ]);
    expect(functionAsymptotes(level('1-4'), 'original')).toEqual([{ axis: 'x', value: 0 }]);
    expect(functionAsymptotes(level('1-5'), 'target')).toEqual([{ axis: 'y', value: 0 }]);
    expect(functionAsymptotes(level('1-5'), 'original')).toEqual([
      { axis: 'y', value: 0 },
      { axis: 'y', value: 1 },
    ]);
    expect(functionAsymptotes(level('1-6'), 'original')).toEqual([
      { axis: 'y', value: -1 },
      { axis: 'y', value: 1 },
    ]);
    expect(functionAsymptotes(level('1-1'))).toEqual([]);
    expect(functionAsymptotes(level('1-1'), 'original')).toEqual([]);
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
