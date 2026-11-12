// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Example datasets for the sandbox: the curriculum's planar sets plus a few shapes that only
 * make sense once every component is available (a sine wave, XOR as a point cloud, two moons).
 */
import { datasets, shuffledOrder, type Dataset } from '../datasets';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import type { TrainingBatch } from '../engine/training';

export type SandboxTask = 'regression' | 'classification';

/** Plain rows of one sample each; the shape every preview and starter model works from. */
export interface SandboxRows {
  x: number[][];
  t: number[][];
}

/** A labelled point cloud, shuffled with the dataset RNG so row index never gives the class. */
function classes(rows: { x: number[]; label: number }[], random: Random, count = 2): TrainingBatch {
  const order = shuffledOrder(random, rows.length);
  const shuffled = order.map((i) => rows[i]);
  return {
    inputs: {
      x: T.tensor(
        shuffled.flatMap((r) => r.x),
        [rows.length, 2],
      ),
    },
    targets: T.tensor(
      shuffled.flatMap((r) => Array.from({ length: count }, (_, c) => (c === r.label ? 1 : 0))),
      [rows.length, count],
    ),
  };
}

export const sine = {
  id: 'sine',
  modality: 'line',
  build: (random: Random): TrainingBatch => {
    const n = 40;
    const x = Array.from({ length: n }, (_, i) => -3 + (6 * i) / (n - 1));
    const t = x.map((v) => Math.sin(v) + (random.uniform() - 0.5) * 0.16);
    return { inputs: { x: T.tensor(x, [n]), t: T.tensor(t, [n]) } };
  },
} satisfies Dataset;

export const xorPoints = {
  id: 'xor-points',
  modality: 'points',
  build: (random: Random): TrainingBatch => {
    const rows: { x: number[]; label: number }[] = [];
    for (let i = 0; i < 80; i++) {
      // Keep a small gap around the axes so every point clearly sits in one quadrant.
      const coord = () => (random.uniform() < 0.5 ? -1 : 1) * (0.15 + random.uniform() * 1.35);
      const a = coord(),
        b = coord();
      rows.push({ x: [a, b], label: a * b > 0 ? 0 : 1 });
    }
    return classes(rows, random);
  },
} satisfies Dataset;

export const moons = {
  id: 'moons',
  modality: 'points',
  build: (random: Random): TrainingBatch => {
    const rows: { x: number[]; label: number }[] = [];
    const n = 80;
    for (let i = 0; i < n; i++) {
      const label = i % 2;
      const angle = Math.PI * ((Math.floor(i / 2) + random.uniform() * 0.5) / (n / 2));
      const noise = () => (random.uniform() - 0.5) * 0.24;
      const x = label ? 1 - Math.cos(angle) : Math.cos(angle);
      const y = label ? 0.5 - Math.sin(angle) : Math.sin(angle);
      rows.push({ x: [x - 0.5 + noise(), y - 0.25 + noise()], label });
    }
    return classes(rows, random);
  },
} satisfies Dataset;

/** Every built-in a slot may name. Older slots can still point at the last two. */
export const SANDBOX_BUILTINS = [
  'noisy-line',
  'sine',
  'two-blobs',
  'xor-points',
  'rings',
  'moons',
  'small-and-large',
  'xor-table',
] as const;
export type SandboxBuiltin = (typeof SANDBOX_BUILTINS)[number];

/** The one-click examples, in picker order. */
export const SANDBOX_EXAMPLES: readonly SandboxBuiltin[] = [
  'noisy-line',
  'sine',
  'two-blobs',
  'xor-points',
  'rings',
  'moons',
];

const sandboxOnly: readonly Dataset[] = [sine, xorPoints, moons];

export function builtinSource(id: SandboxBuiltin): Dataset & Required<Pick<Dataset, 'build'>> {
  const source = [...sandboxOnly, ...datasets].find((d) => d.id === id);
  if (!source?.build) throw new Error('sandbox.dataset_unknown');
  return source as Dataset & Required<Pick<Dataset, 'build'>>;
}

export const builtinTask = (id: SandboxBuiltin): SandboxTask =>
  builtinSource(id).modality === 'line' ? 'regression' : 'classification';

/** The straight line wants a straight model; everything else needs a hidden layer. */
export const builtinStarter = (id: SandboxBuiltin): 'linear' | 'mlp' =>
  id === 'noisy-line' ? 'linear' : 'mlp';

/** Seed shared by the sandbox level, the worker and every preview of the same data. */
export const SANDBOX_SEED = 7001;

const rowsCache = new Map<SandboxBuiltin, Promise<SandboxRows>>();

/** Rows of a built-in, read once and cached: previews must match what training sees. */
export function builtinRows(id: SandboxBuiltin): Promise<SandboxRows> {
  let rows = rowsCache.get(id);
  if (!rows) {
    rows = (async () => {
      const batch = builtinSource(id).build(new Random(SANDBOX_SEED));
      const x = batch.inputs.x,
        t = batch.inputs.t ?? batch.targets!;
      try {
        const xs = Array.from(await x.read()),
          ts = Array.from(await t.read());
        const n = x.shape[0];
        const width = (shape: readonly number[]) => (shape.length > 1 ? shape[1] : 1);
        const xw = width(x.shape),
          tw = width(t.shape);
        return {
          x: Array.from({ length: n }, (_, i) => xs.slice(i * xw, (i + 1) * xw)),
          t: Array.from({ length: n }, (_, i) => ts.slice(i * tw, (i + 1) * tw)),
        };
      } finally {
        Object.values(batch.inputs).forEach((v) => v.dispose());
        batch.targets?.dispose();
      }
    })();
    rowsCache.set(id, rows);
  }
  return rows;
}
