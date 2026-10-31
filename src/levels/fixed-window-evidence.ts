// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { corpus } from './corpus';

/** Real continuations in the previous chapter's corpus, not predictions from a saved model. */
export const fixedWindowExamples = [
  { history: 'little moth', target: ' ' },
  { history: 'tap a tune along th', target: 'e' },
] as const;

export const fixedWindowContext = 'th';

/** Validate the evidence against its source rather than silently presenting invented samples. */
export function hasFixedWindowEvidence(): boolean {
  return fixedWindowExamples.every(
    ({ history, target }) =>
      history.endsWith(fixedWindowContext) && corpus.includes(history + target),
  );
}
