// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { createInitialProgress, isGraphJSON } from '../storage';
import { catalog } from '../levels/catalog';
import {
  MAX_SLOTS,
  createSlot,
  exportWeights,
  isSandboxSave,
  isSandboxUnlocked,
  parseWeights,
  removeSlot,
  sandboxSlots,
  updateSlot,
} from './slots';
import { buildSandboxLevel, SANDBOX_STEPS, sandboxComponents } from './level';
import { trainingSteps } from '../engine/levels';
import { fromEditorProgress } from '../ui/Sandbox';

describe('sandbox unlock', () => {
  const levels = catalog.map((e) => e.level);
  it('opens only when every catalog level is passed', () => {
    const progress = createInitialProgress();
    expect(isSandboxUnlocked(progress, levels)).toBe(false);
    const ids = levels.map((l) => l.id);
    expect(isSandboxUnlocked({ passed: ids.slice(1) }, levels)).toBe(false);
    expect(isSandboxUnlocked({ passed: ids }, levels)).toBe(true);
    expect(isSandboxUnlocked({ passed: ids }, [])).toBe(false);
  });
  it('offers every component from the curriculum', () => {
    const all = sandboxComponents();
    for (const level of levels)
      for (const c of 'available' in level ? level.available : []) expect(all).toContain(c);
  });
});

describe('sandbox slots', () => {
  it('create, update, remove and cap', () => {
    let p = createInitialProgress();
    p = createSlot(p, { id: 'a', name: 'A', now: 1 });
    expect(sandboxSlots(p)).toMatchObject([
      { id: 'a', name: 'A', createdAt: 1, dataset: { kind: 'builtin' } },
    ]);
    p = updateSlot(p, 'a', { name: 'B', weights: undefined }, 2);
    expect(sandboxSlots(p)[0]).toMatchObject({ name: 'B', updatedAt: 2 });
    expect('weights' in sandboxSlots(p)[0]).toBe(false);
    expect(isSandboxSave(p.sandbox, isGraphJSON)).toBe(true);
    for (let i = 0; i < MAX_SLOTS + 2; i++) p = createSlot(p, { id: `s${i}`, name: 'x', now: 3 });
    expect(sandboxSlots(p)).toHaveLength(MAX_SLOTS);
    p = removeSlot(p, 'a');
    expect(sandboxSlots(p).some((s) => s.id === 'a')).toBe(false);
  });
  it('keeps the editor graph in the slot, not in curriculum drafts', () => {
    const graph = { version: 1 as const, nodes: [], edges: [] };
    let p = createSlot(createInitialProgress(), { id: 'a', name: 'A', now: 1 });
    p = fromEditorProgress({ ...p, savedGraphs: { ...p.savedGraphs, 'sandbox:a': graph } }, 'a', 5);
    expect(p.savedGraphs['sandbox:a']).toBeUndefined();
    expect(sandboxSlots(p)[0]).toMatchObject({ graph, updatedAt: 5 });
  });
  it('builds a level for built-in and imported data', () => {
    const builtin = buildSandboxLevel('a', { kind: 'builtin', id: 'two-blobs' });
    expect(builtin).toMatchObject({ id: 'sandbox:a', sandbox: true });
    expect(Object.keys(builtin.interface.inputs).sort()).toEqual(['t', 'x']);
    const imported = buildSandboxLevel('b', {
      kind: 'imported',
      name: 'n',
      modality: 'line',
      task: 'regression',
      x: [[1], [2], [3]],
      t: [[2], [4], [6]],
    });
    expect(imported.interface.inputs).toEqual({ x: [3], t: [3] });
  });
  it('lets the player pick a step budget (default 500)', () => {
    const level = buildSandboxLevel('a', { kind: 'builtin', id: 'noisy-line' });
    expect(level.steps).toBe(SANDBOX_STEPS);
    expect(level.tunable?.steps).toEqual({ min: 1, max: 100_000, default: SANDBOX_STEPS });
    expect(trainingSteps(level)).toBe(SANDBOX_STEPS);
    expect(trainingSteps(level, 2000)).toBe(2000);
    expect(trainingSteps(level, 0)).toBe(1);
    expect(trainingSteps(level, 999_999)).toBe(100_000);
  });
});

describe('weights files', () => {
  it('round-trips parameters', () => {
    const file = exportWeights({ w: { shape: [2], values: new Float32Array([1.5, -2]) } });
    expect(parseWeights(JSON.stringify(file))).toEqual({
      format: 'differentiable-weights',
      version: 1,
      parameters: { w: { shape: [2], values: [1.5, -2] } },
    });
  });
  it('rejects broken files', () => {
    expect(() => parseWeights('{')).toThrow('sandbox.weights.error.json');
    expect(() => parseWeights('{"format":"x"}')).toThrow('sandbox.weights.error.format');
    expect(() =>
      parseWeights(
        JSON.stringify({
          format: 'differentiable-weights',
          version: 1,
          parameters: { w: { shape: [2], values: [1] } },
        }),
      ),
    ).toThrow('sandbox.weights.error.format');
  });
});
