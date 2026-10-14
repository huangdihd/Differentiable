import { expect, it } from 'vitest';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';

it('preserves every original chapter 0/1/2 par and fixed/seeded case', () => {
  expect(chapter1Levels.slice(0, 7).map((level) => level.par)).toEqual([1, 3, 3, 3, 9, 11, 7]);
  const levels = [...chapter0Levels, ...chapter1Levels.slice(0, 7), ...chapter2Levels];
  expect(
    levels.map((level) => {
      const random = new Random(level.generated!.seed);
      return {
        id: level.id,
        par: level.par,
        fixed: level.fixed,
        seed: level.generated!.seed,
        generated: Array.from({ length: level.generated!.count }, (_, i) =>
          level.generated!.generate(random, i),
        ),
      };
    }),
  ).toMatchSnapshot();
});
