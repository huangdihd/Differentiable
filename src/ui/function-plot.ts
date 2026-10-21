import type { ForwardLevel } from '../engine/levels';
import { expectedOutput } from './expected-output';

export interface Asymptote {
  axis: 'x' | 'y';
  value: number;
}
export type PlotCurve = 'target' | 'original';

/**
 * Analytic asymptotes for each displayed curve. Keeping these explicit avoids pretending that a
 * finite sample can prove limiting behaviour, and lets derivative lessons label only the current
 * view rather than combining f and f′ into one chart.
 */
const ASYMPTOTES: Readonly<Record<string, Partial<Record<PlotCurve, readonly Asymptote[]>>>> = {
  '0-5': {
    target: [
      { axis: 'y', value: 0 },
      { axis: 'y', value: 1 },
    ],
  },
  '0-6': {
    target: [
      { axis: 'y', value: -1 },
      { axis: 'y', value: 1 },
    ],
  },
  '1-2': { target: [{ axis: 'y', value: 0 }], original: [{ axis: 'y', value: 0 }] },
  '1-3': { target: [{ axis: 'y', value: 0 }], original: [{ axis: 'y', value: 0 }] },
  '1-4': {
    target: [
      { axis: 'x', value: 0 },
      { axis: 'y', value: 0 },
    ],
    original: [{ axis: 'x', value: 0 }],
  },
  '1-5': {
    target: [{ axis: 'y', value: 0 }],
    original: [
      { axis: 'y', value: 0 },
      { axis: 'y', value: 1 },
    ],
  },
  '1-6': {
    target: [{ axis: 'y', value: 0 }],
    original: [
      { axis: 'y', value: -1 },
      { axis: 'y', value: 1 },
    ],
  },
  '1-7': { target: [{ axis: 'y', value: 0 }] },
};

export function functionAsymptotes(
  level: ForwardLevel,
  curve: PlotCurve = 'target',
): readonly Asymptote[] {
  return ASYMPTOTES[level.id]?.[curve] ?? [];
}

/** The scalar input shown on a function plot; partial derivatives use their differentiated input. */
export function plottedInput(level: ForwardLevel): string | undefined {
  const names = Object.keys(level.interface.inputs);
  if (!names.length || names.some((name) => level.interface.inputs[name].length)) return undefined;
  if (level.interface.outputShape?.length) return undefined;
  if (names.length === 1) return names[0];
  const derivative = level.derivativeInput ?? 'x';
  return level.kind === 'derivative' && names.includes(derivative) ? derivative : undefined;
}

/** Exact target (and original function for derivative lessons) at one scalar probe input. */
export async function functionPoint(
  level: ForwardLevel,
  name: string,
  x: number,
  fixed: Readonly<Record<string, readonly number[]>> = {},
): Promise<{ target: number; original?: number }> {
  const names = Object.keys(level.interface.inputs);
  const inputs = Object.fromEntries(
    names.map((input) => [input, [input === name ? x : fixed[input]?.[0]]]),
  ) as Record<string, number[]>;
  if (Object.values(inputs).some(([value]) => !Number.isFinite(value)))
    throw new Error('missing plot input');
  const shapes = Object.fromEntries(names.map((input) => [input, []]));
  const original =
    level.kind === 'derivative'
      ? expectedOutput({ ...level, kind: 'forward' }, inputs, shapes)
      : Promise.resolve(undefined);
  const [target, base] = await Promise.all([expectedOutput(level, inputs, shapes), original]);
  return { target: target.values[0], ...(base ? { original: base.values[0] } : {}) };
}

/** x values to sample: wide enough for the level's own cases, never through x = 0 exactly. */
export function sampleXs(level: ForwardLevel, name: string, n = 161): number[] {
  const seen = level.fixed.map((c) => c.inputs[name]?.[0]).filter(Number.isFinite);
  const positive = seen.length > 0 && seen.every((x) => x > 0);
  const lo = positive ? 0.02 : Math.min(-5, ...seen);
  const hi = Math.max(5, ...seen);
  // An even count over a symmetric range straddles 0; a nudge keeps 1/x and ln x defined.
  return Array.from({ length: n }, (_, i) => {
    const x = lo + ((hi - lo) * i) / (n - 1);
    return Math.abs(x) < 1e-9 ? (hi - lo) / (n - 1) / 2 : x;
  });
}

/** Linear value between sampled points, unless the interval crosses an explicit discontinuity. */
export function sampleValue(
  xs: readonly number[],
  ys: readonly number[],
  x: number,
  breaks: readonly number[] = [],
): number | undefined {
  const right = xs.findIndex((sample) => sample >= x);
  if (right < 0) return Number.isFinite(ys.at(-1)) ? ys.at(-1) : undefined;
  if (right === 0) return Number.isFinite(ys[0]) && !breaks.includes(x) ? ys[0] : undefined;
  const left = right - 1,
    a = ys[left],
    b = ys[right],
    leftX = xs[left],
    rightX = xs[right];
  if (
    !Number.isFinite(a) ||
    !Number.isFinite(b) ||
    breaks.some((point) => x === point || (leftX < point && point < rightX))
  )
    return undefined;
  const mix = (x - leftX) / (rightX - leftX);
  return a + (b - a) * mix;
}

/** A y range that shows the curve's shape: robust to the spikes of 1/x or eˣ. */
export function yRange(ys: readonly number[]): [number, number] {
  const finite = ys.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return [-1, 1];
  const at = (q: number) => finite[Math.min(finite.length - 1, Math.floor(q * finite.length))];
  let lo = at(0.04),
    hi = at(0.96);
  if (hi - lo < 1e-6) [lo, hi] = [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.15;
  return [lo - pad, hi + pad];
}

/** Polyline runs, split where the curve is undefined or jumps across the whole plot (1/x at 0). */
export function segments(
  xs: readonly number[],
  ys: readonly number[],
  [y0, y1]: readonly [number, number],
): [number, number][][] {
  const runs: [number, number][][] = [];
  let run: [number, number][] = [];
  ys.forEach((y, i) => {
    const prev = run.at(-1);
    const jump = prev && Math.abs(y - prev[1]) > (y1 - y0) * 1.5;
    if (!Number.isFinite(y) || jump) {
      if (run.length > 1) runs.push(run);
      run = [];
      if (!Number.isFinite(y)) return;
    }
    run.push([xs[i], y]);
  });
  if (run.length > 1) runs.push(run);
  return runs;
}
