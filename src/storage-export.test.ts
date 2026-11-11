// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import type { GraphJSON } from './engine/graph';
import {
  APP_VERSION,
  STORAGE_KEY,
  STORAGE_VERSION,
  collectProgressSaveExport,
  createInitialProgress,
  errorCauseChain,
  explainProgressInvalid,
  explainStoredProgress,
  isProgress,
  loadProgress,
  loadProgressWithMigration,
  progressSaveExportFilename,
  serializeProgressSaveExport,
} from './storage';

function memory(initial?: Iterable<readonly [string, string]>) {
  const data = new Map<string, string>(initial);
  let writes = 0;
  const source = () => ({
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      writes += 1;
      data.set(key, value);
    },
  });
  return { data, source, writes: () => writes };
}

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'x', component: 'input', attrs: { name: 'x', shape: [2] } },
    { id: 'y', component: 'output' },
  ],
  edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'y', port: 'x' } }],
};

function envelope(progress: unknown, version: number = STORAGE_VERSION): string {
  return JSON.stringify({ version, progress });
}

describe('explainProgressInvalid', () => {
  it('returns null for valid progress and agrees with isProgress', () => {
    const progress = createInitialProgress();
    expect(isProgress(progress)).toBe(true);
    expect(explainProgressInvalid(progress)).toBeNull();
    const withSettings = {
      ...progress,
      trainingSettings: { '3-4': { learningRate: 0.2, optimizer: 'adam' as const } },
      savedGraphs: { '0-1': structuredClone(graph) },
      unlocked: ['0-1'],
      passed: ['0-1'],
      passedSolutions: { '0-1': structuredClone(graph) },
      bestScores: { '0-1': { topNodes: 1, expandedPrimitives: 2, parameterCount: 0 } },
    };
    expect(isProgress(withSettings)).toBe(true);
    expect(explainProgressInvalid(withSettings)).toBeNull();
  });

  it('reports the first failing field path and reason', () => {
    expect(explainProgressInvalid(null)).toEqual({ path: '', reason: 'not a plain object' });
    const base = createInitialProgress();
    expect(explainProgressInvalid({ ...base, unlocked: '0-1' })?.path).toBe('unlocked');
    expect(explainProgressInvalid({ ...base, unlocked: ['0-1', '0-1'] })).toEqual({
      path: 'unlocked[1]',
      reason: 'duplicate value',
    });
    expect(explainProgressInvalid({ ...base, passed: ['missing'] })).toEqual({
      path: 'passed[0]',
      reason: '"missing" is not in unlocked',
    });
    expect(
      explainProgressInvalid({
        ...base,
        trainingSettings: { '3-4': { learningRate: 0.2, optimizer: 'Adam' } },
      }),
    ).toEqual({
      path: 'trainingSettings["3-4"].optimizer',
      reason: 'must be "sgd" or "adam" when present',
    });
    expect(
      explainProgressInvalid({
        ...base,
        savedGraphs: { '0-1': { ...graph, version: 2 } },
      }),
    ).toEqual({
      path: 'savedGraphs["0-1"].version',
      reason: 'must be 1',
    });
    expect(
      explainProgressInvalid({
        ...base,
        passed: ['0-1'],
        unlocked: ['0-1'],
        passedSolutions: {
          '0-1': { version: 1, nodes: [], edges: [] },
          extra: structuredClone(graph),
        },
      }),
    ).toEqual({
      path: 'passedSolutions.extra',
      reason: 'level is not in passed',
    });
    expect(
      explainProgressInvalid({
        ...base,
        bestScores: { '0-1': { topNodes: 1, expandedPrimitives: 2, parameterCount: -1 } },
      }),
    ).toEqual({
      path: 'bestScores["0-1"].parameterCount',
      reason: 'must be a safe integer >= 0',
    });
    const cyclic: Record<string, unknown> = { ...base };
    cyclic.self = cyclic;
    expect(explainProgressInvalid(cyclic)).toEqual({ path: 'self', reason: 'cyclic reference' });
    expect(isProgress(cyclic)).toBe(false);
  });

  it('stays in agreement with isProgress for crafted invalid saves', () => {
    const base = createInitialProgress();
    const cases: unknown[] = [
      undefined,
      1,
      [],
      { ...base, unlocked: [1] },
      { ...base, unlockedComponents: null },
      { ...base, passed: ['0-1', ''] },
      { ...base, savedGraphs: [] },
      { ...base, passedSolutions: null },
      {
        ...base,
        bestScores: { '0-1': { topNodes: 1.5, expandedPrimitives: 0, parameterCount: 0 } },
      },
      { ...base, trainingSettings: null },
      { ...base, trainingSettings: { '3-4': { learningRate: 0 } } },
      { ...base, trainingSettings: { '': { learningRate: 0.1 } } },
      {
        ...base,
        savedGraphs: { '0-1': { version: 1, nodes: [{ id: '', component: 'add' }], edges: [] } },
      },
      {
        ...base,
        savedGraphs: {
          '0-1': {
            version: 1,
            nodes: [
              { id: 'a', component: 'add' },
              { id: 'a', component: 'mul' },
            ],
            edges: [],
          },
        },
      },
      {
        ...base,
        savedGraphs: {
          '0-1': {
            ...graph,
            edges: [{ from: { node: 'missing', port: 'out' }, to: { node: 'y', port: 'x' } }],
          },
        },
      },
    ];
    for (const value of cases) {
      expect(explainProgressInvalid(value) === null, JSON.stringify(value)).toBe(isProgress(value));
      if (!isProgress(value)) expect(explainProgressInvalid(value)?.reason).toBeTruthy();
    }
  });
});

describe('collectProgressSaveExport', () => {
  it('preserves the raw storage string byte-for-byte and does not write storage', () => {
    const raw = '  {"version":10,"progress":{"unlocked":["0-1"]}}\n';
    const { data, source, writes } = memory([[STORAGE_KEY, raw]]);
    const snapshot = new Map(data);
    const loadError = new Error('storage.progress_load_failed', {
      cause: new Error('Unsupported or invalid progress'),
    });
    const now = new Date('2026-11-11T12:34:56.000Z');
    const payload = collectProgressSaveExport(source, loadError, {
      now,
      userAgent: 'TestAgent/1.0',
      commit: null,
    });
    expect(payload.raw).toBe(raw);
    expect(payload.raw === raw).toBe(true);
    expect(JSON.parse(serializeProgressSaveExport(payload)).raw).toBe(raw);
    expect(payload.storageKey).toBe('differentiable.progress');
    expect(payload.appVersion).toBe(APP_VERSION);
    expect(payload.commit).toBeNull();
    expect(payload.userAgent).toBe('TestAgent/1.0');
    expect(payload.exportedAt).toBe(now.toISOString());
    expect(payload.legacy).toEqual({});
    expect(payload.error.message).toBe('storage.progress_load_failed');
    expect(payload.error.causeChain).toEqual([
      'storage.progress_load_failed',
      'Unsupported or invalid progress',
    ]);
    expect(payload.error.validation?.path).toBe('progress.unlockedComponents');
    expect(data).toEqual(snapshot);
    expect(writes()).toBe(0);
    expect(() => loadProgress(source)).toThrow('storage.progress_load_failed');
    expect(data.get(STORAGE_KEY)).toBe(raw);
  });

  it('includes the failing field for a crafted invalid save', async () => {
    const progress = {
      ...createInitialProgress(),
      trainingSettings: { '3-4': { learningRate: 0.2, optimizer: 'Adam' } },
    };
    expect(isProgress(progress)).toBe(false);
    expect(explainProgressInvalid(progress)).toEqual({
      path: 'trainingSettings["3-4"].optimizer',
      reason: 'must be "sgd" or "adam" when present',
    });
    const raw = envelope(progress);
    const { data, source, writes } = memory([[STORAGE_KEY, raw]]);
    let loadError: unknown;
    try {
      await loadProgressWithMigration([], source);
    } catch (error) {
      loadError = error;
    }
    const payload = collectProgressSaveExport(source, loadError);
    expect(payload.raw).toBe(raw);
    expect(payload.error.validation).toEqual({
      path: 'progress.trainingSettings["3-4"].optimizer',
      reason: 'must be "sgd" or "adam" when present',
    });
    expect(payload.error.causeChain[0]).toBe('storage.progress_load_failed');
    expect(data.get(STORAGE_KEY)).toBe(raw);
    expect(writes()).toBe(0);
  });

  it('records a localStorage access failure and still returns an export without writing', () => {
    const source = () => ({
      getItem: () => {
        throw new Error('quota');
      },
      setItem: () => {
        throw new Error('should not write');
      },
    });
    const loadError = new Error('storage.progress_load_failed', { cause: new Error('quota') });
    const payload = collectProgressSaveExport(source, loadError, { userAgent: null });
    expect(payload.raw).toBeNull();
    expect(payload.error.message).toBe('storage.progress_load_failed');
    expect(payload.error.causeChain).toEqual(['storage.progress_load_failed', 'quota']);
    expect(payload.error.validation).toEqual({
      path: STORAGE_KEY,
      reason: 'localStorage access failed: quota',
    });
  });

  it('records a thrown storage source without writing', () => {
    const source = () => {
      throw new Error('denied');
    };
    const payload = collectProgressSaveExport(source);
    expect(payload.raw).toBeNull();
    expect(payload.error.validation).toEqual({
      path: STORAGE_KEY,
      reason: 'localStorage access failed: denied',
    });
    expect(payload.error.causeChain).toEqual(['denied']);
  });

  it('explains invalid JSON, missing keys, and unsupported versions', () => {
    expect(explainStoredProgress(null)).toEqual({
      path: '',
      reason: 'no value stored for this key',
    });
    expect(explainStoredProgress('{')?.reason).toMatch(/^JSON.parse failed:/);
    expect(explainStoredProgress(envelope(createInitialProgress(), 99))).toEqual({
      path: 'version',
      reason: 'unsupported envelope version 99',
    });
    const { source, writes } = memory();
    const payload = collectProgressSaveExport(source);
    expect(payload.raw).toBeNull();
    expect(payload.error.validation).toEqual({
      path: '',
      reason: 'no value stored for this key',
    });
    expect(writes()).toBe(0);
  });
});

describe('progressSaveExportFilename', () => {
  it('formats the local timestamp as differentiable-save-YYYYMMDD-HHMM.json', () => {
    expect(progressSaveExportFilename(new Date(2026, 10, 11, 20, 45, 59))).toBe(
      'differentiable-save-20261111-2045.json',
    );
  });
});

describe('errorCauseChain', () => {
  it('walks nested Error.cause values', () => {
    const error = new Error('outer', { cause: new Error('inner', { cause: 'root' }) });
    expect(errorCauseChain(error)).toEqual(['outer', 'inner', 'root']);
  });
});
