// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { initialProgress } from '../engine/levels';
import { chapter4TrainLevels } from '../levels/chapter4';
import { vocabulary } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowSnapshot } from '../worker/fixed-window-protocol';
import {
  FIXED_WINDOW_CACHE_KEY,
  fixedWindowCircuitHash,
  loadFixedWindowResult,
  saveFixedWindowResult,
} from './fixed-window-cache';
import { STORAGE_KEY } from '../storage';

const level = chapter4TrainLevels[1];
const progress = () => {
  const p = initialProgress([]);
  return {
    ...p,
    passed: [level.id],
    passedSolutions: { [level.id]: structuredClone(level.referenceGraph!) },
  };
};
const run = (step: number): FixedWindowSnapshot => ({
  model: 'fixed-window',
  parameterCount: 10,
  sampleCount: 1495,
  step,
  stepsPerRun: 320,
  seed: 1,
  learningRate: 0.02,
  loss: 1,
  predictions: fixedWindowExamples.map(({ history }) => ({
    history,
    input: 'th',
    prediction: 'e',
    probabilities: vocabulary.map((c) => (c === 'e' ? 1 : 0)),
  })),
});
const memory = () => {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
  };
};

describe('fixed-window experiment cache', () => {
  it('round-trips under its own key, never the progress key', () => {
    const store = memory();
    expect(saveFixedWindowResult(progress(), [run(320), run(640)], store)).toBe(true);
    expect([...store.data.keys()]).toEqual([FIXED_WINDOW_CACHE_KEY]);
    expect(store.data.has(STORAGE_KEY)).toBe(false);
    expect(loadFixedWindowResult(progress(), store)?.map((s) => s.step)).toEqual([320, 640]);
  });
  it('ignores layout but misses when the circuit changes', () => {
    const store = memory();
    saveFixedWindowResult(progress(), [run(320), run(640)], store);
    const moved = progress();
    moved.passedSolutions[level.id].nodes[0].ui = { x: 999, y: 1 };
    expect(fixedWindowCircuitHash(moved)).toBe(fixedWindowCircuitHash(progress()));
    const changed = progress();
    changed.passedSolutions[level.id].nodes.push({ id: 'extra', component: 'tanh' });
    expect(loadFixedWindowResult(changed, store)).toBeUndefined();
  });
  it('treats malformed, partial or missing entries as a miss', () => {
    const store = memory();
    for (const raw of [
      '{',
      'null',
      '{"hash":1}',
      JSON.stringify({ hash: fixedWindowCircuitHash(progress()), snapshots: [run(320)] }),
    ]) {
      store.setItem(FIXED_WINDOW_CACHE_KEY, raw);
      expect(loadFixedWindowResult(progress(), store)).toBeUndefined();
    }
    expect(saveFixedWindowResult(progress(), [run(320)], store)).toBe(false);
    expect(loadFixedWindowResult(initialProgress([]), store)).toBeUndefined();
    const throwing = {
      getItem: () => {
        throw new Error('denied');
      },
      setItem: () => {
        throw new Error('full');
      },
    };
    expect(loadFixedWindowResult(progress(), throwing)).toBeUndefined();
    expect(saveFixedWindowResult(progress(), [run(320), run(640)], throwing)).toBe(false);
  });
});
