// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { gameLevels } from './catalog';
import { levelPapers } from './papers';

it('lists papers only for real levels, with https links', () => {
  const ids = new Set(gameLevels.map((level) => level.id));
  for (const [id, papers] of Object.entries(levelPapers)) {
    expect(ids.has(id), id).toBe(true);
    expect(papers.length).toBeGreaterThan(0);
    for (const paper of papers) expect(paper.url).toMatch(/^https:\/\//);
  }
});
