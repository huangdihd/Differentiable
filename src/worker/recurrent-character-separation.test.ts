// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { RecurrentCharacterExperiment } from './recurrent-character-experiment';
import { recurrentCharacterTestProgress } from './recurrent-character-test-fixture';
import { VOCAB } from '../levels/corpus';

// Regression: at the old lr 0.01 the reference RNN still answered "e" for
// "little moth" after 320 steps. With the default plan it must separate both
// sentences at 320 and 640 steps, with a clear margin.
it('reference RNN separates "little moth" (space) from "tap a tune along th" (e)', async () => {
  const experiment = new RecurrentCharacterExperiment(recurrentCharacterTestProgress());
  try {
    for (let run = 0; run < 2; run++) {
      const result = await experiment.run();
      const p = (input: string, c: string) =>
        result.predictions.find((x) => x.input.endsWith(input))!.probabilities[VOCAB.indexOf(c)];
      expect(p('little moth', ' ')).toBeGreaterThan(0.5);
      expect(p('tap a tune along th', 'e')).toBeGreaterThan(0.5);
    }
  } finally {
    experiment.dispose();
  }
}, 3_000_000);
