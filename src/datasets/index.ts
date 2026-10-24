import type { Random } from '../engine/random';
import * as T from '../engine/tensor';
import type { TrainingBatch } from '../engine/training';

/**
 * What a training level learns from, as the player sees it. Name and source are i18n keys
 * `datasets.<id>.name` / `datasets.<id>.source`; the modality picks the preview.
 */
export interface Dataset {
  id: string;
  modality: 'line' | 'points' | 'table' | 'text';
  /** Seeded batch; omitted where each level cuts the data its own way (the rhyme). */
  build?: (random: Random) => TrainingBatch;
}
/** How a level uses a dataset; `context` is how many characters a text model sees. */
export interface DatasetUse {
  dataset: Dataset;
  context?: number;
}

/** A level whose batches come straight from the dataset. */
export function trainOn(dataset: Dataset & Required<Pick<Dataset, 'build'>>) {
  return { dataset: dataset.build, data: { dataset } };
}

export const noisyLine = {
  id: 'noisy-line',
  modality: 'line',
  build: (random: Random) => {
    const x = Array.from({ length: 16 }, (_, i) => -2 + (4 * i) / 15);
    const t = x.map((value) => 1.5 * value - 0.5 + (random.uniform() - 0.5) * 0.1);
    return { inputs: { x: T.tensor(x, [16]), t: T.tensor(t, [16]) } };
  },
} satisfies Dataset;

/** Two-class points in the plane; balanced labels, shuffled so row index ≠ class. */
function pointClasses(count: number, point: (random: Random, i: number, sign: number) => number[]) {
  return (random: Random): TrainingBatch => {
    const rows: { x: number[]; target: number[] }[] = [];
    for (let i = 0; i < count; i++) {
      const label = i % 2;
      rows.push({
        x: point(random, i, label ? 1 : -1),
        target: [1 - label, label],
      });
    }
    // Fisher–Yates with the dataset RNG so seed and seed+1 disagree on index→label
    // (otherwise a param[N,C] memorizer that ignores x still hits hold-out accuracy).
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random.uniform() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    return {
      inputs: {
        x: T.tensor(
          rows.flatMap((row) => row.x),
          [count, 2],
        ),
      },
      targets: T.tensor(
        rows.flatMap((row) => row.target),
        [count, 2],
      ),
    };
  };
}

export const twoBlobs = {
  id: 'two-blobs',
  modality: 'points',
  build: pointClasses(40, (random, _, sign) => [
    sign * (1 + random.uniform() * 0.5),
    sign * (1 + random.uniform() * 0.5),
  ]),
} satisfies Dataset;

/** The small feature determines the label; the large nuisance feature is balanced. */
export const smallAndLarge = {
  id: 'small-and-large',
  modality: 'points',
  build: pointClasses(40, (random, i, sign) => [
    sign * (0.8 + random.uniform() * 0.4) * 0.05,
    (Math.floor(i / 2) % 2 ? 1 : -1) * (0.8 + random.uniform() * 0.4) * 20,
  ]),
} satisfies Dataset;

export const rings = {
  id: 'rings',
  modality: 'points',
  build: pointClasses(64, (random, i, sign) => {
    const angle = (2 * Math.PI * (Math.floor(i / 2) + random.uniform() * 0.2)) / 32;
    const radius = (sign > 0 ? 2 : 0.6) + (random.uniform() - 0.5) * 0.1;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }),
} satisfies Dataset;

export const xorTable = {
  id: 'xor-table',
  modality: 'table',
  build: (random: Random) => {
    const rows = [
      [0, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 0, 0, 1],
      [1, 1, 1, 0],
    ];
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random.uniform() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    return {
      inputs: {
        x: T.tensor(
          rows.flatMap((r) => r.slice(0, 2)),
          [4, 2],
        ),
      },
      targets: T.tensor(
        rows.flatMap((r) => r.slice(2)),
        [4, 2],
      ),
    };
  },
} satisfies Dataset;

/** The nursery rhyme in corpus.ts; chapters 4–6 cut it into pairs, contexts and windows. */
export const gardenRhyme = { id: 'garden-rhyme', modality: 'text' } satisfies Dataset;

export const datasets: readonly Dataset[] = [
  noisyLine,
  twoBlobs,
  smallAndLarge,
  rings,
  xorTable,
  gardenRhyme,
];
