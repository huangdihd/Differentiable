// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { evaluateLevel, initialProgress } from '../engine/levels';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { formulaGraph, scalarPrimitives } from './level-helpers';
import { availableComponents, buildPlayerRegistry } from './registry';
import {
  curves,
  SLOPE_PASS_SCORE,
  slopeRounds,
  slopeScore,
  nudgeActual,
  nudgeCorrect,
  nudgeRounds,
  nudgePassed,
  gearSpeedCorrect,
  gearSpeeds,
  gearTrains,
  chainRate,
  numericLevel,
} from './primer-calculus';

describe('calculus primer', () => {
  it('exposes each curve, domain, and analytic derivative', () => {
    expect(Object.keys(curves).sort()).toEqual(['bowl', 'cubic', 'square', 'steep', 'wave']);
    expect(curves.square.domain).toEqual([-3, 3]);
    expect(curves.cubic.domain).toEqual([-2.5, 2.5]);
    expect(curves.wave.domain).toEqual([-4, 4]);
    for (const x of [-2, -0.25, 0, 1.25, 2]) {
      expect(curves.square.f(x)).toBe(x * x);
      expect(curves.cubic.f(x)).toBeCloseTo((x * x * x) / 3 - x, 12);
      expect(curves.wave.f(x)).toBeCloseTo(Math.sin(x) + 0.3 * x, 12);
      for (const [id, curve] of Object.entries(curves)) {
        expect(curve.id).toBe(id);
        expect(curve.df(x)).toBeCloseTo((curve.f(x + 1e-5) - curve.f(x - 1e-5)) / 2e-5, 8);
      }
    }
  });
  it('grades signed tangent angles on a fifteen-degree scale', () => {
    expect(SLOPE_PASS_SCORE).toBe(0.8);
    expect(slopeRounds.map(({ curve, x }) => Math.sign(curves[curve].df(x))).sort()).toEqual([
      -1, 0, 1,
    ]);
    for (const round of slopeRounds) {
      const slope = curves[round.curve].df(round.x);
      expect(slopeScore(slope, round)).toBe(1);
      for (const degrees of [-20, -15, -7.5, -3, 3, 7.5, 15, 20]) {
        const guess = Math.tan(Math.atan(slope) + (degrees * Math.PI) / 180);
        expect(slopeScore(guess, round)).toBeCloseTo(Math.max(0, 1 - Math.abs(degrees) / 15), 12);
      }
      expect(slopeScore(NaN, round)).toBe(0);
    }
  });
  it('compares nudges to the finite change with inclusive relative and absolute bands', () => {
    expect(nudgeRounds.map((round) => round.dx)).toEqual([0.1, 0.01, 0.01]);
    expect(nudgeRounds.some(({ curve, x }) => curves[curve].df(x) < 0)).toBe(true);
    for (const round of nudgeRounds) {
      const curve = curves[round.curve];
      const actual = curve.f(round.x + round.dx) - curve.f(round.x);
      expect(nudgeActual(round)).toBe(actual);
      expect(nudgeCorrect(curve.df(round.x) * round.dx, round)).toBe(true);
      expect(nudgeCorrect(-actual, round)).toBe(false);
      expect(nudgeCorrect(NaN, round)).toBe(false);
      const tolerance = Math.max(0.002, 0.15 * Math.abs(actual));
      for (const sign of [-1, 1]) {
        expect(nudgeCorrect(actual + sign * tolerance, round)).toBe(true);
        expect(nudgeCorrect(actual + sign * (tolerance + 1e-8), round)).toBe(false);
      }
    }
  });
  it('requires a completed three-answer ballot with a majority correct', () => {
    for (const a of [false, true])
      for (const b of [false, true])
        for (const c of [false, true]) {
          expect(nudgePassed([a, b, c])).toBe(Number(a) + Number(b) + Number(c) >= 2);
        }
    expect(nudgePassed([true, true])).toBe(false);
    expect(nudgePassed([true, true, true, true])).toBe(false);
  });
  it('fills gear speeds left to right as running products', () => {
    expect(gearTrains.map((train) => gearSpeeds(train.stages))).toEqual([
      [1, 2, 6],
      [1, 0.5, 2, -2],
      [1, 3, 18],
    ]);
    for (const train of gearTrains)
      expect(gearSpeeds(train.stages).at(-1)).toBe(chainRate(train.stages));
    // Only the function train hides its rates behind labels.
    expect(gearTrains.map((train) => train.stages.some((stage) => stage.labelKey))).toEqual([
      false,
      false,
      true,
    ]);
  });
  it('accepts a filled speed within 1%', () => {
    expect(gearSpeedCorrect(6, 6)).toBe(true);
    expect(gearSpeedCorrect(6.05, 6)).toBe(true);
    expect(gearSpeedCorrect(6.1, 6)).toBe(false);
    expect(gearSpeedCorrect(-2, -2)).toBe(true);
    expect(gearSpeedCorrect(2, -2)).toBe(false);
    expect(gearSpeedCorrect(0, 0)).toBe(true);
    expect(gearSpeedCorrect(NaN, 1)).toBe(false);
  });
  it('executes the earned circuit on positive steps and visibly converges at fixed x', async () => {
    expect(numericLevel.available).toEqual([...scalarPrimitives, 'max']);
    expect(numericLevel.fixed.map((c) => c.inputs.h[0])).toEqual([0.1, 0.01, 0.001]);
    expect(new Set(numericLevel.fixed.map((c) => c.inputs.x[0])).size).toBe(1);
    const generated = numericLevel.generated!;
    const random = new Random(generated.seed);
    for (let i = 0; i < generated.count; i++)
      expect(generated.generate(random, i).inputs.h[0]).toBeGreaterThan(0);
    const progress = initialProgress(
      [],
      chapter0Levels.flatMap((level) => level.unlockComponents ?? []),
    );
    const registry = buildPlayerRegistry(chapter0Levels, progress);
    const level = { ...numericLevel, available: availableComponents(numericLevel, progress) };
    const result = await evaluateLevel(level, level.referenceGraph!, { registry });
    expect(result.tests).toHaveLength(level.fixed.length + generated.count);
    expect(result.passed).toBe(true);
    const derivative = 2 * level.fixed[0].inputs.x[0];
    const errors = result.tests.slice(0, 3).map((test) => Math.abs(test.actual![0] - derivative));
    expect(errors[2]).toBeLessThan(errors[1]);
    expect(errors[1]).toBeLessThan(errors[0]);
    for (const test of result.tests.slice(0, 3))
      expect(Math.abs(test.expected![0] - test.actual![0])).toBeLessThanOrEqual(4.7e-5);
    const wrong = await evaluateLevel(level, formulaGraph('x+x', ['x', 'h'], 'slope'), {
      registry,
      checkDependency: false,
    });
    expect(wrong.tests).toHaveLength(result.tests.length);
    expect(wrong.tests.every((test) => !test.passed)).toBe(true);
  });
});
