// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** Pure helpers behind the sandbox training tab, kept out of React so they can be tested. */
import type { PlotData } from '../ui/DataPlot';

export interface LossSeries {
  /** [step index, loss] for finite losses, thinned to at most `max` points. */
  points: [number, number][];
  /** Step indices whose loss was NaN or ±∞ (thinned the same way). */
  bad: number[];
  lo: number;
  hi: number;
  /** Log scale when every finite loss is positive. */
  log: boolean;
  count: number;
  /** The run blew up: training keeps going in the sandbox, the chart just says so. */
  diverged: boolean;
}

export function lossSeries(losses: readonly number[], max = 600): LossSeries {
  const count = losses.length;
  const bucket = Math.max(1, Math.ceil(count / max));
  const points: [number, number][] = [];
  const bad: number[] = [];
  for (let start = 0; start < count; start += bucket) {
    // Keep each bucket's extremes so spikes survive thinning.
    let lo = -1,
      hi = -1,
      broken = -1;
    for (let i = start; i < Math.min(count, start + bucket); i++) {
      const v = losses[i];
      if (!Number.isFinite(v)) {
        if (broken < 0) broken = i;
        continue;
      }
      if (lo < 0 || v < losses[lo]) lo = i;
      if (hi < 0 || v > losses[hi]) hi = i;
    }
    if (broken >= 0) bad.push(broken);
    for (const i of lo < 0 ? [] : lo === hi ? [lo] : [Math.min(lo, hi), Math.max(lo, hi)])
      points.push([i, losses[i]]);
  }
  const finite = points.map(([, v]) => v);
  const log = finite.length > 0 && finite.every((v) => v > 0);
  let lo = finite.length ? Math.min(...finite) : 0.1,
    hi = finite.length ? Math.max(...finite) : 1;
  if (log) {
    lo = 10 ** Math.floor(Math.log10(lo));
    hi = 10 ** Math.ceil(Math.log10(hi));
    if (hi <= lo) hi = lo * 10;
  } else {
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
    lo -= pad;
    hi += pad;
  }
  const first = losses.find(Number.isFinite);
  const last = losses.at(-1);
  const diverged =
    count > 0 &&
    (!Number.isFinite(last!) ||
      Math.abs(last!) > 1e12 ||
      (first !== undefined && first > 0 && last! > first * 1e3));
  return { points, bad, lo, hi, log, count, diverged };
}

/** Share of points whose predicted class matches the label; undefined until there is a model. */
export function plotAccuracy(plot: PlotData | undefined): number | undefined {
  if (!plot || plot.kind !== 'classes' || !plot.predicted || !plot.label.length) return undefined;
  const right = plot.label.filter((label, i) => plot.predicted![i] === label).length;
  return right / plot.label.length;
}
