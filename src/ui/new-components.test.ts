// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { gameLevels } from '../levels/catalog';
import {
  componentSeenKey,
  initializeSeenComponents,
  isNewComponent,
  newlyIntroducedComponents,
} from './new-components';

describe('new component badges', () => {
  it('uses the current library as a baseline for an existing browser', () => {
    const seen = initializeSeenComponents(['expression'], ['add', 'mul']);
    expect(seen).toEqual(['expression', componentSeenKey('add'), componentSeenKey('mul')]);
    expect(isNewComponent(seen, 'add')).toBe(false);
    expect(isNewComponent(seen, 'size')).toBe(true);
  });

  it('does not absorb later components after the baseline has been initialized', () => {
    const baseline = initializeSeenComponents([], ['add']);
    expect(initializeSeenComponents(baseline, ['add', 'size'])).toEqual(baseline);
    expect(isNewComponent(baseline, 'size')).toBe(true);
    expect(isNewComponent([...baseline, componentSeenKey('size')], 'size')).toBe(false);
  });

  it.each([
    ['1-11', ['sum', 'unsqueeze', 'reduce_max']],
    ['2-2', ['size']],
    ['2-5', ['transpose']],
    ['3-3', ['param']],
    ['3-9', ['reshape']],
  ])('announces basic parts first exposed after %s', (id, expected) => {
    const level = gameLevels.find((candidate) => candidate.id === id)!;
    expect(newlyIntroducedComponents(gameLevels, level)).toEqual(expected);
  });
});
