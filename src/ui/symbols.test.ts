import { expect, it } from 'vitest';
import { gameLevels } from '../levels/catalog';
import { SYMBOLS } from './graph-geometry';

it('every part a player can place has a node symbol', () => {
  const names = new Set(
    gameLevels.flatMap((level) => [...(level.unlockComponents ?? []), ...level.available]),
  );
  expect([...names].filter((name) => !SYMBOLS[name])).toEqual([]);
});
