// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { VOCAB } from '../levels/corpus';
import { FixedWindowExperiment } from './fixed-window-experiment';
import { recurrentCharacterTestProgress } from './recurrent-character-test-fixture';

// Regression: predictions once came from a separately compiled batch-1 plan, which could
// re-initialise parameters and show a trained model as near-uniform (≈4% on one letter).
it('a trained player fixed window puts most probability on space/e after "th"', async () => {
  const experiment = new FixedWindowExperiment(recurrentCharacterTestProgress());
  try {
    const snapshot = await experiment.run();
    expect(snapshot.predictions.length).toBeGreaterThan(0);
    for (const prediction of snapshot.predictions) {
      expect(prediction.input).toBe('th');
      const p = (c: string) => prediction.probabilities[VOCAB.indexOf(c)];
      expect(p(' ') + p('e')).toBeGreaterThan(0.6);
      expect(['e', ' ']).toContain(prediction.prediction);
    }
  } finally {
    experiment.dispose();
  }
}, 600_000);
