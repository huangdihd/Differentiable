// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { GraphJSON } from '../engine/graph';
import { Random } from '../engine/random';

/**
 * How a sequence training level shows the model beyond its few training rows: a fixed set of
 * probe windows swept over the whole input range (or a long time axis), and a "try it" input.
 */
export type SequenceProbeKind = 'forecast' | 'delay';

export interface SequenceProbe {
  /** One input window per row, each `length` long. */
  rows: number[][];
  /** Horizontal coordinate of each row's prediction on the plot. */
  at: number[];
  /** The curve the model should match: points (x, t). */
  curve: { x: number[]; t: number[] };
  /** Draw the target as a line (an identity line) instead of points. */
  targetLine: boolean;
}

export const PROBE_FORECAST = { amplitude: 0.8, frequency: 0.45, phase: 0.3, rows: 32 };

export function sequenceProbe(kind: SequenceProbeKind, length: number): SequenceProbe {
  if (kind === 'forecast') {
    const { amplitude, frequency, phase, rows } = PROBE_FORECAST;
    const series = Array.from(
      { length: length + rows },
      (_, k) => amplitude * Math.sin(phase + k * frequency),
    );
    return {
      rows: Array.from({ length: rows }, (_, k) => series.slice(k, k + length)),
      at: Array.from({ length: rows }, (_, k) => k + length),
      curve: { x: series.map((_, k) => k), t: series },
      targetLine: false,
    };
  }
  const count = 32;
  const remembered = Array.from({ length: count }, (_, i) => -0.8 + (1.6 * i) / (count - 1));
  return {
    // Small fixed distractors after the remembered first value, like the training data.
    rows: remembered.map((r, i) => [
      r,
      ...Array.from({ length: length - 1 }, (_, s) => 0.15 * Math.sin(i * 1.7 + s * 2.3)),
    ]),
    at: remembered,
    curve: { x: remembered, t: remembered },
    targetLine: true,
  };
}

/** A fresh example for the "try it" panel, with the value the model should output. */
export function sequenceExample(
  kind: SequenceProbeKind,
  length: number,
  seed: number,
): { row: number[]; expected: number } {
  const random = new Random(seed);
  if (kind === 'forecast') {
    const phase = random.uniform() * Math.PI * 2;
    const frequency = 0.35 + random.uniform() * 0.25;
    const amplitude = 0.6 + random.uniform() * 0.4;
    const value = (k: number) => amplitude * Math.sin(phase + k * frequency);
    return {
      row: Array.from({ length }, (_, k) => round(value(k))),
      expected: value(length),
    };
  }
  const remembered = round(random.uniform() * 1.6 - 0.8);
  return {
    row: [
      remembered,
      ...Array.from({ length: length - 1 }, () => round(random.uniform() * 0.4 - 0.2)),
    ],
    expected: remembered,
  };
}

/** The delay task has a rule for any input; a forecast only knows the answer for a real wave. */
export function sequenceRule(kind: SequenceProbeKind, row: number[]): number | undefined {
  return kind === 'delay' ? row[0] : undefined;
}

const round = (v: number) => Math.round(v * 100) / 100;

/** The node whose output feeds the locked MSE loss's prediction port. */
export function predictionSource(graph: GraphJSON): string | undefined {
  return graph.edges.find(
    (e) => e.to.port === 'p' && graph.nodes.find((n) => n.id === e.to.node)?.component === 'mse',
  )?.from.node;
}
