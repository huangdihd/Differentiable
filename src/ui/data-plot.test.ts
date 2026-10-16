import { expect, it } from 'vitest';
import { boundaryPath, type ClassField } from './DataPlot';

const field = (nx: number, ny: number, f: (i: number, j: number) => number): ClassField => {
  const margin = Array.from({ length: nx * ny }, (_, k) => f(k % nx, Math.floor(k / nx)));
  return {
    nx,
    ny,
    x0: 0,
    x1: nx,
    y0: 0,
    y1: ny,
    margin,
    classes: margin.map((m) => (m > 0 ? 1 : 0)),
  };
};
const segments = (d: string) =>
  [...d.matchAll(/M([\d.-]+),([\d.-]+)L([\d.-]+),([\d.-]+)/g)].map((m) => m.slice(1).map(Number));

it('draws a two-class boundary through the interpolated zero crossing', () => {
  // Margin i - 2.25 is zero at i = 2.25, i.e. x = 2.75 in cell-edge coordinates.
  const d = boundaryPath(
    field(5, 4, (i) => i - 2.25),
    (i) => i,
    (j) => j,
  );
  const lines = segments(d);
  expect(lines).toHaveLength(3);
  for (const [ax, , bx] of lines) {
    expect(ax).toBeCloseTo(2.8, 1);
    expect(bx).toBeCloseTo(2.8, 1);
  }
});

it('outlines disagreeing cells when there are more than two classes', () => {
  const f = field(3, 1, () => 0);
  delete f.margin;
  f.classes = [0, 2, 2];
  expect(
    segments(
      boundaryPath(
        f,
        (i) => i,
        (j) => j,
      ),
    ),
  ).toEqual([[1, 0, 1, 1]]);
});

it('draws nothing when the model predicts one class everywhere', () => {
  expect(
    boundaryPath(
      field(4, 3, () => 1),
      (i) => i,
      (j) => j,
    ),
  ).toBe('');
});
