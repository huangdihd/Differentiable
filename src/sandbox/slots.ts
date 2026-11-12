// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** Sandbox saves: local slots of free-play graphs, their data and imported weights. */
import type { GraphJSON } from '../engine/graph';
import type { Level, Progress } from '../engine/levels';
import { isComponentLibrary, type CustomComponent } from './components';
import { validateRows, type ImportedDataset } from './dataset-import';

import { SANDBOX_BUILTINS, type SandboxBuiltin } from './datasets';

export { SANDBOX_BUILTINS, type SandboxBuiltin };

export type SandboxDataset =
  { kind: 'builtin'; id: SandboxBuiltin } | ({ kind: 'imported' } & ImportedDataset);

export interface WeightsFile {
  format: 'differentiable-weights';
  version: 1;
  /** Keyed by parameter storage path (a Parameter node id, or `module/param` inside a module). */
  parameters: Record<string, { shape: number[]; values: number[] }>;
}

export interface SandboxSlot {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  dataset: SandboxDataset;
  graph?: GraphJSON;
  training?: { learningRate: number; optimizer?: 'sgd' | 'adam'; steps?: number };
  /** Imported weights; training starts from them where paths and shapes match. */
  weights?: WeightsFile;
}

export interface SandboxSave {
  slots: SandboxSlot[];
  /** "My components": parts packaged from sandbox graphs; absent until the first one. */
  components?: CustomComponent[];
}

export const MAX_SLOTS = 12;
export const SANDBOX_PREFIX = 'sandbox:';
export const sandboxLevelId = (slotId: string) => `${SANDBOX_PREFIX}${slotId}`;
export const isSandboxLevelId = (id: string) => id.startsWith(SANDBOX_PREFIX);

/** The sandbox opens once every level in the catalog is passed; then every component is in. */
export function isSandboxUnlocked(
  progress: Pick<Progress, 'passed'>,
  levels: readonly Pick<Level, 'id'>[],
): boolean {
  return levels.length > 0 && levels.every((level) => progress.passed.includes(level.id));
}

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const time = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const names = (value: unknown) =>
  value === undefined || (Array.isArray(value) && value.every((v) => typeof v === 'string'));

export function isSandboxDataset(value: unknown): value is SandboxDataset {
  if (!record(value)) return false;
  if (value.kind === 'builtin') return (SANDBOX_BUILTINS as readonly unknown[]).includes(value.id);
  if (value.kind !== 'imported') return false;
  if (
    typeof value.name !== 'string' ||
    !['line', 'points', 'table'].includes(value.modality as string) ||
    !['regression', 'classification'].includes(value.task as string) ||
    !names(value.xColumns) ||
    !names(value.tColumns)
  )
    return false;
  try {
    validateRows(value.x, value.t);
    return true;
  } catch {
    return false;
  }
}

export function isWeightsFile(value: unknown): value is WeightsFile {
  if (
    !record(value) ||
    value.format !== 'differentiable-weights' ||
    value.version !== 1 ||
    !record(value.parameters)
  )
    return false;
  return Object.entries(value.parameters).every(
    ([id, p]) =>
      text(id) &&
      !['__proto__', 'prototype', 'constructor'].includes(id) &&
      record(p) &&
      Array.isArray(p.shape) &&
      p.shape.every((d) => Number.isSafeInteger(d) && d >= 0) &&
      Array.isArray(p.values) &&
      p.values.every((v) => typeof v === 'number' && Number.isFinite(v)) &&
      p.values.length === (p.shape as number[]).reduce((a, b) => a * b, 1),
  );
}

export function isSandboxSlot(
  value: unknown,
  isGraph: (graph: unknown) => boolean,
): value is SandboxSlot {
  return (
    record(value) &&
    text(value.id) &&
    typeof value.name === 'string' &&
    time(value.createdAt) &&
    time(value.updatedAt) &&
    isSandboxDataset(value.dataset) &&
    (value.graph === undefined || isGraph(value.graph)) &&
    (value.weights === undefined || isWeightsFile(value.weights)) &&
    (value.training === undefined ||
      (record(value.training) &&
        typeof value.training.learningRate === 'number' &&
        Number.isFinite(value.training.learningRate) &&
        value.training.learningRate > 0 &&
        (value.training.optimizer === undefined ||
          value.training.optimizer === 'sgd' ||
          value.training.optimizer === 'adam') &&
        (value.training.steps === undefined ||
          (typeof value.training.steps === 'number' &&
            Number.isSafeInteger(value.training.steps) &&
            value.training.steps >= 1))))
  );
}

export function isSandboxSave(
  value: unknown,
  isGraph: (graph: unknown) => boolean,
): value is SandboxSave {
  if (!record(value) || !Array.isArray(value.slots) || value.slots.length > MAX_SLOTS) return false;
  if (value.components !== undefined && !isComponentLibrary(value.components, isGraph))
    return false;
  const ids = new Set<string>();
  return value.slots.every((slot) => {
    if (!isSandboxSlot(slot, isGraph) || ids.has(slot.id)) return false;
    ids.add(slot.id);
    return true;
  });
}

export const sandboxSlots = (progress: Progress): SandboxSlot[] => progress.sandbox?.slots ?? [];

function withSlots(progress: Progress, slots: SandboxSlot[]): Progress {
  // The component library rides along untouched.
  return { ...progress, sandbox: { ...progress.sandbox, slots } };
}

export function createSlot(
  progress: Progress,
  options: { id: string; name: string; now: number; dataset?: SandboxDataset },
): Progress {
  const slots = sandboxSlots(progress);
  if (slots.length >= MAX_SLOTS || slots.some((s) => s.id === options.id)) return progress;
  return withSlots(progress, [
    ...slots,
    {
      id: options.id,
      name: options.name,
      createdAt: options.now,
      updatedAt: options.now,
      dataset: options.dataset ?? { kind: 'builtin', id: 'noisy-line' },
    },
  ]);
}

export function updateSlot(
  progress: Progress,
  id: string,
  change: Partial<Omit<SandboxSlot, 'id' | 'createdAt'>>,
  now: number,
): Progress {
  const slots = sandboxSlots(progress);
  if (!slots.some((s) => s.id === id)) return progress;
  return withSlots(
    progress,
    slots.map((slot) => {
      if (slot.id !== id) return slot;
      const next: SandboxSlot = { ...slot, ...change, id, updatedAt: now };
      // `undefined` clears a field, so stored slots never carry empty keys.
      for (const key of Object.keys(next) as (keyof SandboxSlot)[])
        if (next[key] === undefined) delete next[key];
      return next;
    }),
  );
}

/** A copy placed right after the original: same data, graph, settings and weights. */
export function duplicateSlot(
  progress: Progress,
  id: string,
  options: { id: string; name: string; now: number },
): Progress {
  const slots = sandboxSlots(progress);
  const index = slots.findIndex((s) => s.id === id);
  if (index < 0 || slots.length >= MAX_SLOTS || slots.some((s) => s.id === options.id))
    return progress;
  const copy: SandboxSlot = {
    ...structuredClone(slots[index]),
    id: options.id,
    name: options.name,
    createdAt: options.now,
    updatedAt: options.now,
  };
  return withSlots(progress, [...slots.slice(0, index + 1), copy, ...slots.slice(index + 1)]);
}

/** A name not yet taken: "base", then "base 2", "base 3", … */
export function uniqueName(base: string, taken: readonly string[]): string {
  if (!taken.includes(base)) return base;
  for (let n = 2; ; n++) if (!taken.includes(`${base} ${n}`)) return `${base} ${n}`;
}

export function removeSlot(progress: Progress, id: string): Progress {
  return withSlots(
    progress,
    sandboxSlots(progress).filter((slot) => slot.id !== id),
  );
}

/** Serialize trained parameters. Values come from the worker as plain arrays. */
export function exportWeights(
  parameters: Record<string, { shape: readonly number[]; values: ArrayLike<number> }>,
): WeightsFile {
  return {
    format: 'differentiable-weights',
    version: 1,
    parameters: Object.fromEntries(
      Object.entries(parameters).map(([id, p]) => [
        id,
        { shape: [...p.shape], values: Array.from(p.values) },
      ]),
    ),
  };
}

export function parseWeights(source: string): WeightsFile {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    throw new Error('sandbox.weights.error.json');
  }
  if (!isWeightsFile(value)) throw new Error('sandbox.weights.error.format');
  return value;
}
