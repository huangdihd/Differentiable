import type { Level } from '../engine/levels';
import { exampleLevels } from './examples';

/** The road from a + b to GPT. Chapters without levels yet are shown as upcoming. */
export const CHAPTERS = [0, 1, 2, 3, 4, 5, 6] as const;

export interface CatalogEntry {
  level: Level;
  chapter: number;
  /** Displayed level number, e.g. "0-4". */
  code: string;
}

const placement: Record<string, { chapter: number; code: string }> = {
  sigmoid: { chapter: 0, code: '0-4' },
  derivative: { chapter: 1, code: '1-2' },
  xor: { chapter: 3, code: '3-4' },
};

export const catalog: CatalogEntry[] = exampleLevels.map((level) => ({
  level,
  ...(placement[level.id] ?? { chapter: 0, code: level.id }),
}));

export function entryFor(level: Level): CatalogEntry {
  return catalog.find((e) => e.level.id === level.id) ?? { level, chapter: 0, code: level.id };
}
