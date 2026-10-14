import type { Level } from '../engine/levels';
import { graphMetrics } from '../engine/levels';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3Levels, chapter3ForwardLevels } from './chapter3';
import { xorLevel, xorSolution } from './examples';
import { referenceRegistry } from './level-helpers';

/** The road from a + b to GPT. Chapters without levels yet are shown as upcoming. */
export const CHAPTERS = [0, 1, 2, 3, 4, 5, 6] as const;

export interface CatalogEntry {
  level: Level;
  chapter: number;
  /** Displayed level number, e.g. "0-4". */
  code: string;
}

export const gameLevels: Level[] = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3Levels,
  {
    ...xorLevel,
    available: [
      'param',
      'const',
      'add',
      'mul',
      'exp',
      'log',
      'max',
      'sum',
      'reduce_max',
      'reshape',
      'transpose',
    ],
    presets: ['matmul', 'softmax'],
    hintsKeys: ['levels.xor.hint1', 'levels.xor.hint2'],
    referenceGraph: xorSolution,
    par: graphMetrics(
      xorSolution,
      referenceRegistry([
        ...chapter0Levels,
        ...chapter1Levels,
        ...chapter2Levels,
        ...chapter3ForwardLevels,
      ]),
    ).expandedPrimitives,
  },
];

export const catalog: CatalogEntry[] = gameLevels.map((level) => ({
  level,
  chapter: level.id === 'xor' ? 3 : Number(level.id.split('-')[0]),
  code: level.id === 'xor' ? '3-5' : level.id,
}));

export function entryFor(level: Level): CatalogEntry {
  return catalog.find((e) => e.level.id === level.id) ?? { level, chapter: 0, code: level.id };
}
