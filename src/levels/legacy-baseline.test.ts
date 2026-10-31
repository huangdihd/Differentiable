// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { Random } from '../engine/random';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';

it('preserves every original chapter 0/1/2 par and fixed/seeded case', () => {
  // Reciprocal is a single native primitive, including in lessons that expand it transitively.
  expect(chapter1Levels.slice(0, 7).map((level) => level.par)).toEqual([1, 3, 3, 1, 6, 7, 4]);
  const levels = [
    // Lessons inserted later have named IDs; only the originals are pinned here.
    ...chapter0Levels.filter((level) => level.id !== 'square'),
    ...chapter1Levels.slice(0, 7),
    ...chapter2Levels.filter((level) => level.unlockComponents),
  ];
  expect(
    levels.map((level) => {
      const random = new Random(level.generated!.seed);
      return {
        // Compare old lessons against their original IDs without rewriting the snapshot.
        id:
          (
            { '2-3': '2-2', '2-4': '2-3', '2-5': '2-4', '2-7': '2-5', '2-8': '2-6' } as Record<
              string,
              string
            >
          )[level.id] ?? level.id,
        par: level.par,
        // The display order may change (the varied case leads); the cases themselves may not.
        fixed: [
          ...level.fixed
            .filter((c) => c.id.startsWith('tensor-'))
            .sort((a, b) => a.id.localeCompare(b.id)),
          ...level.fixed.filter((c) => !c.id.startsWith('tensor-')),
        ],
        seed: level.generated!.seed,
        generated: Array.from({ length: level.generated!.count }, (_, i) =>
          level.generated!.generate(random, i),
        ),
      };
    }),
  ).toMatchSnapshot();
});
