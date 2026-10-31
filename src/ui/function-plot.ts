// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ForwardLevel } from '../engine/levels';
import { expectedOutput, expectedOutputs } from './expected-output';

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

/** Two scalar inputs whose two derivative outputs form a gradient vector on a surface. */
export interface GradientSurfaceData {
  axes: readonly [number[], number[]];
  surface: number[];
  target: readonly [number[], number[]];
}

export interface SurfaceView {
  yaw: number;
  pitch: number;
}

/** Project normalized input-plane coordinates and normalized height into the SVG view. */
export function projectSurfacePoint(
  x: number,
  y: number,
  height: number,
  view: SurfaceView,
  width = 260,
  centerY = 103,
  xyScale = 72,
  zScale = 42,
): readonly [number, number] {
  const across = x * Math.cos(view.yaw) - y * Math.sin(view.yaw);
  const depth = x * Math.sin(view.yaw) + y * Math.cos(view.yaw);
  return [
    width / 2 + across * xyScale,
    centerY + depth * Math.sin(view.pitch) * zScale - height * Math.cos(view.pitch) * zScale,
  ];
}

/** Invert the projected input plane so picking remains accurate after the view rotates. */
export function unprojectSurfaceFloor(
  screenX: number,
  screenY: number,
  floorHeight: number,
  view: SurfaceView,
  width = 260,
  centerY = 103,
  xyScale = 72,
  zScale = 42,
): readonly [number, number] {
  const across = (screenX - width / 2) / xyScale;
  const depth =
    (screenY - centerY + floorHeight * Math.cos(view.pitch) * zScale) /
    (Math.sin(view.pitch) * zScale);
  return [
    across * Math.cos(view.yaw) + depth * Math.sin(view.yaw),
    -across * Math.sin(view.yaw) + depth * Math.cos(view.yaw),
  ];
}

/** A bounded, deterministic grid for the two-input surface and its two partial derivatives. */
export async function gradientSurfaceData(
  level: ForwardLevel,
  names: readonly [string, string],
  n = 13,
): Promise<GradientSurfaceData> {
  const axis = (name: string) => {
    const seen = level.fixed.map((test) => test.inputs[name]?.[0]).filter(Number.isFinite);
    const lo = Math.max(-5, Math.min(-3, ...seen));
    const hi = Math.min(5, Math.max(3, ...seen));
    return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
  };
  const axes = [axis(names[0]), axis(names[1])] as const;
  const values = Object.fromEntries(
    names.map((name, dimension) => [
      name,
      axes[1].flatMap((b) => axes[0].map((a) => (dimension === 0 ? a : b))),
    ]),
  );
  const shapes = Object.fromEntries(names.map((name) => [name, [n * n]]));
  const forward = { ...level, kind: 'forward' as const, derivativeInputs: undefined };
  const [surface, gradients] = await Promise.all([
    expectedOutput(
      {
        ...forward,
        interface: {
          inputs: level.interface.inputs,
          output: level.interface.output,
          outputShape: [n * n],
        },
      },
      values,
      shapes,
    ),
    expectedOutputs(level, values, shapes),
  ]);
  const outputs = Object.keys(level.interface.outputs ?? {});
  return {
    axes,
    surface: surface.values,
    target: [gradients[outputs[0]].values, gradients[outputs[1]].values],
  };
}

/** Bilinear interpolation inside a sampled surface grid. */
export function surfaceValue(
  axes: readonly [readonly number[], readonly number[]],
  values: readonly number[],
  a: number,
  b: number,
): number | undefined {
  const cell = (samples: readonly number[], value: number) => {
    if (value < samples[0] || value > samples.at(-1)!) return undefined;
    const right = samples.findIndex((sample) => sample >= value);
    const left = Math.max(0, right - 1);
    const hi = Math.min(samples.length - 1, Math.max(0, right));
    const span = samples[hi] - samples[left];
    return [left, hi, span ? (value - samples[left]) / span : 0] as const;
  };
  const x = cell(axes[0], a),
    y = cell(axes[1], b);
  if (!x || !y) return undefined;
  const width = axes[0].length;
  const at = (i: number, j: number) => values[j * width + i];
  const top = at(x[0], y[0]) * (1 - x[2]) + at(x[1], y[0]) * x[2];
  const bottom = at(x[0], y[1]) * (1 - x[2]) + at(x[1], y[1]) * x[2];
  const value = top * (1 - y[2]) + bottom * y[2];
  return Number.isFinite(value) ? value : undefined;
}

export function gradientSurfaceInputs(level: ForwardLevel): readonly [string, string] | undefined {
  const inputs = Object.keys(level.interface.inputs);
  const outputs = Object.keys(level.interface.outputs ?? {});
  const derivatives = level.derivativeInputs ?? [];
  return level.kind === 'derivative' &&
    inputs.length === 2 &&
    inputs.every((name) => level.interface.inputs[name].length === 0) &&
    outputs.length === 2 &&
    derivatives.length === 2 &&
    derivatives.every((name) => inputs.includes(name))
    ? [derivatives[0], derivatives[1]]
    : undefined;
}

/** The scalar input shown on a function plot; partial derivatives use their differentiated input. */
export function plottedInput(level: ForwardLevel): string | undefined {
  const names = Object.keys(level.interface.inputs);
  if (!names.length || names.some((name) => level.interface.inputs[name].length)) return undefined;
  if (Object.keys(level.interface.outputs ?? {}).length > 1) return undefined;
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
