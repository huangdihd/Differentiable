// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ForwardLevel, ObservationLevel } from '../engine/levels';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { formulaGraph, scalarPrimitives, withPar } from './level-helpers';

export type CurveId = 'square' | 'cubic' | 'wave' | 'bowl' | 'steep';
export interface Curve {
  id: CurveId;
  f(x: number): number;
  df(x: number): number;
  domain: readonly [number, number];
}
export const curves: Readonly<Record<CurveId, Curve>> = {
  square: { id: 'square', domain: [-3, 3], f: (x) => x * x, df: (x) => 2 * x },
  cubic: { id: 'cubic', domain: [-2.5, 2.5], f: (x) => x ** 3 / 3 - x, df: (x) => x * x - 1 },
  // Loss bowls for the learning-rate lesson: same shape, the second twice as steep.
  bowl: { id: 'bowl', domain: [-4, 4], f: (x) => x * x, df: (x) => 2 * x },
  steep: { id: 'steep', domain: [-4, 4], f: (x) => 2 * x * x, df: (x) => 4 * x },
  wave: {
    id: 'wave',
    domain: [-4, 4],
    f: (x) => Math.sin(x) + 0.3 * x,
    df: (x) => Math.cos(x) + 0.3,
  },
};
export interface SlopeRound {
  curve: CurveId;
  x: number;
}
export const slopeRounds: readonly SlopeRound[] = [
  { curve: 'square', x: 1 },
  { curve: 'wave', x: Math.PI },
  { curve: 'cubic', x: 1 },
];
export const SLOPE_PASS_SCORE = 0.8;
export function slopeScore(guessSlope: number, round: SlopeRound): number {
  if (Number.isNaN(guessSlope)) return 0;
  const truth = curves[round.curve].df(round.x);
  return Math.max(0, 1 - Math.abs(Math.atan(guessSlope) - Math.atan(truth)) / (Math.PI / 12));
}
export interface NudgeRound extends SlopeRound {
  dx: number;
}
export const nudgeRounds: readonly NudgeRound[] = [
  { curve: 'square', x: 1, dx: 0.1 },
  { curve: 'cubic', x: 0, dx: 0.01 },
  { curve: 'wave', x: 0, dx: 0.01 },
];
export function nudgeActual({ curve, x, dx }: NudgeRound): number {
  return curves[curve].f(x + dx) - curves[curve].f(x);
}
export function nudgeCorrect(answer: number, round: NudgeRound): boolean {
  const actual = nudgeActual(round);
  const tolerance = Math.max(0.15 * Math.abs(actual), 0.002);
  return answer >= actual - tolerance && answer <= actual + tolerance;
}
export function nudgePassed(results: readonly boolean[]): boolean {
  return results.length === nudgeRounds.length && results.filter(Boolean).length >= 2;
}
export interface GearStage {
  rate: number;
  /** A function gearbox: its label names the function and the point; the rate stays hidden. */
  labelKey?: string;
}
/**
 * calc-gears: a whole gear train is shown, the input gear turning once per second. The player
 * fills in each following gear's speed, left to right (previous speed × that stage's rate), so
 * the overall rate is built up stage by stage. Function gearboxes hide their rate: it is the
 * function's derivative at the given point.
 */
export interface GearTrain {
  stages: readonly GearStage[];
}
export const gearTrains: readonly GearTrain[] = [
  { stages: [{ rate: 2 }, { rate: 3 }] },
  { stages: [{ rate: 0.5 }, { rate: 4 }, { rate: -1 }] },
  {
    stages: [
      { rate: 3, labelKey: 'primer.gear.tripleAtOne' },
      { rate: 6, labelKey: 'primer.gear.squareAtThree' },
    ],
  },
];
export function chainRate(stages: readonly GearStage[]): number {
  return stages.reduce((product, stage) => product * stage.rate, 1);
}
/** Speed of every gear, input first, when the input turns `inputSpeed` times per second. */
export function gearSpeeds(stages: readonly GearStage[], inputSpeed = 1): number[] {
  return stages.reduce((speeds, stage) => [...speeds, speeds.at(-1)! * stage.rate], [inputSpeed]);
}
/** A filled-in speed is right within 1% (or 1e-6 for a stopped gear). */
export function gearSpeedCorrect(guess: number, expected: number): boolean {
  return (
    Number.isFinite(guess) &&
    Math.abs(guess - expected) <= Math.max(0.01 * Math.abs(expected), 1e-6)
  );
}
function observation(id: string, next: string): ObservationLevel {
  return {
    id,
    chapter: '1',
    kind: 'observation',
    titleKey: `levels.${id}.title`,
    descriptionKey: `levels.${id}.description`,
    unlock: [next],
  };
}
export const numericLevel = withPar<ForwardLevel>(
  {
    id: 'calc-numeric',
    chapter: '1',
    kind: 'forward',
    titleKey: 'levels.calc-numeric.title',
    descriptionKey: 'levels.calc-numeric.description',
    hintsKeys: ['levels.calc-numeric.hint1', 'levels.calc-numeric.hint2'],
    available: [...scalarPrimitives, 'max'],
    interface: { inputs: { x: [], h: [] }, output: 'slope', outputShape: [] },
    unlock: ['1-1'],
    referenceGraph: formulaGraph('div(sub(square(x + h), square(x)), h)', ['x', 'h'], 'slope'),
    // (x+h)² − x² cancels catastrophically in float32: the reference wiring loses 4.7e-5 at
    // x=1, h=0.001, and equivalent wirings (e.g. dividing each square by h first) lose a few
    // ulps of ~1000 more. Leave room for any honest wiring; 2x vs 2x+h still differs by ≥ h.
    atol: 3e-4,
    rtol: 0,
    fixed: [0.1, 0.01, 0.001].map((h, i) => ({ id: `fixed-${i}`, inputs: { x: [1], h: [h] } })),
    generated: {
      seed: 3401,
      count: 8,
      generate: (random, i) => ({
        id: `seeded-${i}`,
        inputs: {
          x: [Math.floor(random.uniform() * 33) / 4 - 4],
          h: [2 ** (-1 - (i % 4))],
        },
      }),
    },
    // For x² the forward difference quotient simplifies exactly to 2x + h, not 2x.
    reference: ({ x, h }) => T.add(T.add(x, x), h),
  },
  chapter0Levels,
);
export const calculusPrimerLevels = [
  observation('calc-slope', 'calc-nudge'),
  observation('calc-nudge', 'calc-numeric'),
  numericLevel,
  observation('calc-gears', '1-3'),
] as const;
