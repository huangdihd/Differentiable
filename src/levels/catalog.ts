import type { Level } from '../engine/levels';
import { graphMetrics } from '../engine/levels';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3Levels, chapter3ForwardLevels } from './chapter3';
import { chapter4Levels } from './chapter4';
import { temporalLevels } from './chapter-temporal';
import { chapter5Levels } from './chapter5';
import { chapter6Levels } from './chapter6';
import { xorLevel, xorSolution } from './examples';
import { referenceRegistry } from './level-helpers';

/** The road from a + b through time-series prediction and Attention to GPT. */
export const CHAPTERS = [0, 1, 2, 3, 4, 5, 6, 7] as const;

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
    chapter: '3',
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
      'size',
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
  ...chapter4Levels,
  ...temporalLevels,
  ...chapter5Levels,
  ...chapter6Levels,
];

const chapterCounts = new Map<number, number>();
export const catalog: CatalogEntry[] = gameLevels.map((level) => {
  const chapter = Number(level.chapter);
  const ordinal = (chapterCounts.get(chapter) ?? 0) + 1;
  chapterCounts.set(chapter, ordinal);
  return { level, chapter, code: `${chapter}-${ordinal}` };
});

export function entryFor(level: Level): CatalogEntry {
  return catalog.find((e) => e.level.id === level.id) ?? { level, chapter: 0, code: level.id };
}
