// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { formatValue } from './format-value';

describe('compact tensor values', () => {
  it.each([
    [2.38e-7, '0'],
    [-9.999e-5, '0'],
    [0, '0'],
    [-0, '0'],
    [1e-4, '1e-4'],
    [-1e-4, '-1e-4'],
    [0.000123456, '1.23e-4'],
    [0.00123456, '0.00123'],
    [1.23456, '1.23'],
    [12, '12'],
    [99999, '1e5'],
    [120000, '1.2e5'],
    [-123456, '-1.23e5'],
    [Number.MAX_VALUE, '1.8e308'],
    [NaN, 'NaN'],
    [Infinity, '∞'],
    [-Infinity, '−∞'],
  ])('formats %s as %s', (value, expected) => {
    expect(formatValue('en', value)).toBe(expected);
    expect(formatValue('zh-CN', value)).toBe(expected);
  });
});
