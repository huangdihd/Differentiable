// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** What a slot's data looks like before any model: summary numbers and a planar preview. */
import type { PlotData } from '../ui/DataPlot';
import {
  builtinRows,
  builtinStarter,
  builtinTask,
  type SandboxRows,
  type SandboxTask,
} from './datasets';
import type { SandboxDataset } from './slots';
import type { StarterKind } from './starter';

export const datasetTask = (dataset: SandboxDataset): SandboxTask =>
  dataset.kind === 'builtin' ? builtinTask(dataset.id) : dataset.task;

export const datasetStarter = (dataset: SandboxDataset): StarterKind =>
  dataset.kind === 'builtin' ? builtinStarter(dataset.id) : 'mlp';

export const datasetRows = (dataset: SandboxDataset): Promise<SandboxRows> =>
  dataset.kind === 'builtin'
    ? builtinRows(dataset.id)
    : Promise.resolve({ x: dataset.x, t: dataset.t });

/** The picture the training tab can draw: a curve for 1→1 regression, a plane for 2-feature classes. */
export function plotView(rows: SandboxRows, task: SandboxTask): 'fit' | 'boundary' | 'none' {
  const xw = rows.x[0]?.length ?? 0,
    tw = rows.t[0]?.length ?? 0;
  if (task === 'regression' && xw === 1 && tw === 1) return 'fit';
  if (task === 'classification' && xw === 2) return 'boundary';
  return 'none';
}

const argmax = (row: readonly number[]) =>
  row.reduce((best, v, i) => (v > row[best] ? i : best), 0);

export function rowsPlot(rows: SandboxRows, task: SandboxTask): PlotData | undefined {
  const view = plotView(rows, task);
  if (view === 'fit')
    return { kind: 'regression', x: rows.x.map((r) => r[0]), t: rows.t.map((r) => r[0]) };
  if (view === 'boundary')
    return {
      kind: 'classes',
      points: rows.x.map((r) => [r[0], r[1]] as [number, number]),
      label: rows.t.map(argmax),
    };
  return undefined;
}

export interface DatasetSummary {
  rows: number;
  features: number;
  /** Classification: number of classes; regression: number of target columns. */
  outputs: number;
  task: SandboxTask;
}

export function summarize(rows: SandboxRows, task: SandboxTask): DatasetSummary {
  return {
    rows: rows.x.length,
    features: rows.x[0]?.length ?? 0,
    outputs: rows.t[0]?.length ?? 0,
    task,
  };
}
