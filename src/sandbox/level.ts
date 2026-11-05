// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/** The sandbox as a training "level" with no pass condition, built the same way in UI and worker. */
import { datasets, type Dataset } from '../datasets';
import type { TrainLevel } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import type { TrainingBatch } from '../engine/training';
import { gameLevels } from '../levels/catalog';
import { importedShapes } from './dataset-import';
import { sandboxLevelId, type SandboxDataset } from './slots';

export const SANDBOX_SEED = 7001;
export const SANDBOX_STEPS = 500;

/** Every component any level offers or earns: the sandbox opens only after all of them. */
export function sandboxComponents(): string[] {
  return [
    ...new Set(
      gameLevels.flatMap((level) =>
        level.kind === 'observation'
          ? [...(level.unlockComponents ?? [])]
          : [...level.available, ...(level.unlockComponents ?? []), ...(level.presets ?? [])],
      ),
    ),
  ];
}

/** Classification datasets keep their targets in `inputs.t` so the player wires their own loss. */
function asInputs(batch: TrainingBatch): TrainingBatch {
  if (batch.inputs.t || !batch.targets) return batch;
  return { inputs: { ...batch.inputs, t: batch.targets } };
}

/** Stable key for "same data": the editor and worker restart when it changes. */
export function datasetKey(dataset: SandboxDataset): string {
  if (dataset.kind === 'builtin') return `builtin:${dataset.id}`;
  let hash = 0;
  for (const c of JSON.stringify([dataset.x, dataset.t]))
    hash = (Math.imul(hash, 31) + c.charCodeAt(0)) | 0;
  return `imported:${dataset.x.length}:${(hash >>> 0).toString(36)}`;
}

function sandboxData(dataset: SandboxDataset): {
  build: (random: Random) => TrainingBatch;
  card: Dataset;
} {
  if (dataset.kind === 'builtin') {
    const source = datasets.find((d) => d.id === dataset.id);
    if (!source?.build) throw new Error('sandbox.dataset_unknown');
    const build = (random: Random) => asInputs(source.build!(random));
    return { build, card: { id: source.id, modality: source.modality, build } };
  }
  const shapes = importedShapes(dataset);
  const build = () => ({
    inputs: {
      x: T.tensor(dataset.x.flat(), shapes.x),
      t: T.tensor(dataset.t.flat(), shapes.t),
    },
  });
  return { build, card: { id: 'sandbox-import', modality: dataset.modality, build } };
}

export function buildSandboxLevel(slotId: string, dataset: SandboxDataset): TrainLevel {
  const { build, card } = sandboxData(dataset);
  const probe = build(new Random(SANDBOX_SEED));
  const inputs = Object.fromEntries(
    Object.entries(probe.inputs).map(([name, tensor]) => [name, [...tensor.shape]]),
  );
  Object.values(probe.inputs).forEach((tensor) => tensor.dispose());
  return {
    id: sandboxLevelId(slotId),
    kind: 'train',
    sandbox: true,
    titleKey: 'sandbox.title',
    descriptionKey: 'sandbox.levelDescription',
    available: sandboxComponents(),
    interface: { inputs, output: 'loss', outputShape: [1] },
    unlock: [],
    seed: SANDBOX_SEED,
    steps: SANDBOX_STEPS,
    optimizer: { kind: 'adam', learningRate: 0.01 },
    tunable: {
      learningRate: { min: 0.0001, max: 1, default: 0.01 },
      optimizer: ['sgd', 'adam'],
    },
    lossOutput: 'loss',
    dataset: build,
    data: { dataset: card },
    // No pass condition: a run ends after its steps (or when stopped) and never "wins".
    threshold: { loss: -Infinity },
  };
}
