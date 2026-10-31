// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { testCaseLocation } from './test-case-location';

describe('testCaseLocation', () => {
  const groups = [
    { cases: [{ id: 'fixed-0' }, { id: 'generated-0' }] },
    { cases: [{ id: 'fixed-0' }, { id: 'generated-0' }] },
    { cases: [{ id: 'fixed-0' }] },
  ];
  const tests = [
    { id: 'fixed-0' },
    { id: 'generated-0' },
    { id: 'shape-1/fixed-0' },
    { id: 'shape-1/generated-0' },
    { id: 'shape-2/fixed-0' },
  ];

  it('maps flattened variant results back to their shape group and local case index', () => {
    expect(tests.map((_, index) => testCaseLocation(tests, index, groups))).toEqual([
      { group: 0, index: 0 },
      { group: 0, index: 1 },
      { group: 1, index: 0 },
      { group: 1, index: 1 },
      { group: 2, index: 0 },
    ]);
  });

  it('uses occurrence order when fixture ids repeat and ignores synthetic compile failures', () => {
    const duplicateGroups = [{ cases: [{ id: 'same' }, { id: 'same' }] }, { cases: [] }];
    const duplicateTests = [{ id: 'same' }, { id: 'same' }, { id: 'shape-1/compile' }];
    expect(testCaseLocation(duplicateTests, 1, duplicateGroups)).toEqual({ group: 0, index: 1 });
    expect(testCaseLocation(duplicateTests, 2, duplicateGroups)).toBeUndefined();
  });
});
