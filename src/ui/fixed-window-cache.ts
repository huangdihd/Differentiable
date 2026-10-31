// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { computationGraph, type GraphJSON } from '../engine/graph';
import type { Progress } from '../engine/levels';
import { gameLevels } from '../levels/catalog';
import { chapter4TrainLevels } from '../levels/chapter4';
import { VOCAB } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import type { FixedWindowSnapshot } from '../worker/fixed-window-protocol';

/**
 * 5-1 trains the player's fixed-window circuit; 5-2 reuses that measured result.
 * Deliberately separate from `differentiable.progress`: only predictions are kept here,
 * experiment weights never enter player storage.
 */
export const FIXED_WINDOW_CACHE_KEY = 'differentiable.experiments.fixed-window';

/** Anything that changes what training would produce. Bump `version` when the experiment does. */
export const FIXED_WINDOW_TRAINING_CONFIG = {
  version: 1,
  optimizer: 'adam',
  learningRate: 0.02,
  stepsPerRun: 320,
  runs: 2,
  clipNorm: 1,
  examples: fixedWindowExamples.map(({ history, target }) => history + target),
} as const;

const stripUi = (graph: GraphJSON) => ({
  ...computationGraph(graph),
  nodes: computationGraph(graph).nodes.map(({ ui: _ui, ...node }) => node),
});

/** Key-sorted JSON so attribute order never changes the hash. */
function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  if (value && typeof value === 'object')
    return `{${Object.keys(value)
      .sort()
      .filter((key) => (value as Record<string, unknown>)[key] !== undefined)
      .map(
        (key) =>
          `${JSON.stringify(key)}:${stableStringify((value as Record<string, unknown>)[key])}`,
      )
      .join(',')}}`;
  return JSON.stringify(value);
}

/** FNV-1a 64-bit (two 32-bit halves), hex. Stable across sessions and browsers. */
function hash(text: string): string {
  let a = 0x811c9dc5;
  let b = 0x01000193 ^ 0x5bd1e995;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    a = Math.imul(a ^ c, 0x01000193) >>> 0;
    b = Math.imul(b ^ c, 0x5bd1e995) >>> 0;
  }
  return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0');
}

/**
 * Hash of the player's fixed-window circuit, every saved solution it can call as a component,
 * and the training settings. Backend is deliberately excluded. Undefined if there is no circuit.
 */
export function fixedWindowCircuitHash(progress: Progress | undefined): string | undefined {
  const root = chapter4TrainLevels[1].id;
  const graph = progress?.passedSolutions[root];
  if (!progress || !graph || !progress.passed.includes(root)) return undefined;
  const end = gameLevels.findIndex((level) => level.id === root);
  const dependencies = Object.fromEntries(
    gameLevels
      .slice(0, end + 1)
      .filter((level) => progress.passedSolutions[level.id])
      .map((level) => [level.id, stripUi(progress.passedSolutions[level.id])]),
  );
  return hash(
    stableStringify({
      root,
      dependencies,
      unlocked: progress.unlockedComponents ?? [],
      config: FIXED_WINDOW_TRAINING_CONFIG,
    }),
  );
}

const isPrediction = (value: unknown, history: string) => {
  const p = value as Record<string, unknown> | null;
  return (
    !!p &&
    p.history === history &&
    typeof p.input === 'string' &&
    typeof p.prediction === 'string' &&
    Array.isArray(p.probabilities) &&
    p.probabilities.length === VOCAB.length &&
    p.probabilities.every((x) => typeof x === 'number' && Number.isFinite(x) && x >= 0)
  );
};

export function isCachedFixedWindowSnapshots(value: unknown): value is FixedWindowSnapshot[] {
  const { stepsPerRun, runs } = FIXED_WINDOW_TRAINING_CONFIG;
  return (
    Array.isArray(value) &&
    value.length === runs &&
    value.every((raw, index) => {
      const s = raw as Record<string, unknown> | null;
      return (
        !!s &&
        s.model === 'fixed-window' &&
        s.step === stepsPerRun * (index + 1) &&
        s.stepsPerRun === stepsPerRun &&
        [s.parameterCount, s.sampleCount, s.loss, s.seed, s.learningRate].every(
          (x) => typeof x === 'number' && Number.isFinite(x),
        ) &&
        Array.isArray(s.predictions) &&
        s.predictions.length === fixedWindowExamples.length &&
        fixedWindowExamples.every(({ history }, i) =>
          isPrediction((s.predictions as unknown[])[i], history),
        )
      );
    })
  );
}

type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem'>;
const storage = (): Storage | undefined => {
  try {
    return globalThis.localStorage ?? undefined;
  } catch {
    return undefined;
  }
};

/** Malformed, old, or mismatched entries read as a miss; never throws. */
export function loadFixedWindowResult(
  progress: Progress | undefined,
  source: Storage | undefined = storage(),
): FixedWindowSnapshot[] | undefined {
  const key = fixedWindowCircuitHash(progress);
  if (!key || !source) return undefined;
  try {
    const raw = source.getItem(FIXED_WINDOW_CACHE_KEY);
    if (!raw) return undefined;
    const entry = JSON.parse(raw) as { hash?: unknown; snapshots?: unknown } | null;
    if (!entry || entry.hash !== key || !isCachedFixedWindowSnapshots(entry.snapshots))
      return undefined;
    return entry.snapshots;
  } catch {
    return undefined;
  }
}

/** Saves only complete two-run results. Storage failures are ignored (results stay on screen). */
export function saveFixedWindowResult(
  progress: Progress | undefined,
  snapshots: readonly FixedWindowSnapshot[],
  source: Storage | undefined = storage(),
): boolean {
  const key = fixedWindowCircuitHash(progress);
  if (!key || !source || !isCachedFixedWindowSnapshots(snapshots)) return false;
  try {
    // resetPredictions belongs to the RNN; keep the entry to what 5-2's fixed column shows.
    const slim = snapshots.map(({ resetPredictions: _reset, ...rest }) => rest);
    source.setItem(FIXED_WINDOW_CACHE_KEY, JSON.stringify({ hash: key, snapshots: slim }));
    return true;
  } catch {
    return false;
  }
}
