import type { ForwardLevel } from '../engine/levels';

/** The one scalar input of a y = f(x) lesson, or undefined when the level is not one. */
export function plottedInput(level: ForwardLevel): string | undefined {
  const names = Object.keys(level.interface.inputs);
  if (names.length !== 1 || level.interface.inputs[names[0]].length) return undefined;
  return level.interface.outputShape?.length ? undefined : names[0];
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
