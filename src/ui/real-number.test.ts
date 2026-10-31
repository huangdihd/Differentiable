// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { parseRealNumber, parseRealValue } from './real-number';
import { validateProperties } from './Properties';

describe('real number parsing', () => {
  it('accepts signed decimals, scientific notation and fractions', () => {
    for (const [text, value] of [
      ['1/2', 0.5],
      [' -3 / 4 ', -0.75],
      ['1/-2', -0.5],
      ['+.5', 0.5],
      ['1.', 1],
      ['2e-3/.5', 0.004],
      ['-2.5E+2', -250],
    ] as const)
      expect(parseRealNumber(text)).toBe(value);
  });
  it('rejects incomplete, nonfinite and expression-like input', () => {
    for (const text of [
      '',
      '-',
      '.',
      '1e',
      '1/',
      '/2',
      '1/2/3',
      '1/0',
      '1/-0',
      '0/0',
      '1e999',
      '1/1e999',
      '1e308/1e-308',
      'NaN',
      'Infinity',
      'null',
      'true',
      '0x10',
      '1+2',
      '(1/2)',
      '1 2',
      '1/**/2',
    ])
      expect(() => parseRealNumber(text), text).toThrow();
  });
  it('accepts flat lists without trailing commas or missing elements', () => {
    expect(parseRealValue('[1/2, -2e-1, .25]')).toEqual([0.5, -0.2, 0.25]);
    expect(parseRealValue('[]')).toEqual([]);
    for (const text of ['[1/2,]', '[,1]', '[[1]]', '[1/0]', '[1', '1,2'])
      expect(() => parseRealValue(text), text).toThrow();
  });
});

describe('fraction property integration', () => {
  it('parses real fields without changing draft text', () => {
    const draft = { value: '[1/2, -3/4]', shape: '2' };
    expect(validateProperties('const', draft)).toEqual({
      attrs: { value: [0.5, -0.75], shape: [2] },
      errors: {},
    });
    expect(draft.value).toBe('[1/2, -3/4]');
    expect(
      validateProperties('param', {
        shape: '[]',
        init: 'constant',
        value: '-1/2',
        scale: '1/4',
      }),
    ).toEqual({
      attrs: { shape: [], init: 'constant', value: -0.5, scale: 0.25 },
      errors: {},
    });
  });
  it('preserves constraints and prior values on invalid input', () => {
    expect(validateProperties('const', { value: '1/' }, { value: 2 })).toEqual({
      attrs: { value: 2 },
      errors: { value: 'ui.invalidNumber' },
    });
    expect(validateProperties('param', { shape: '[]', scale: '-1/2' }).errors.scale).toBe(
      'ui.invalidNumber',
    );
    expect(validateProperties('param', { shape: '[]', value: '[1/2]' }).errors.value).toBe(
      'ui.invalidNumber',
    );
    expect(validateProperties('const', { value: '[1/2]', shape: '2' }).errors.shape).toBe(
      'ui.invalidShape',
    );
  });
  it('does not extend integer field syntax even for integral fractions', () => {
    for (const [component, key] of [
      ['arange', 'n'],
      ['size', 'axis'],
      ['unsqueeze', 'axis'],
      ['reshape', 'shape'],
      ['sum', 'axes'],
      ['transpose', 'perm'],
    ] as const)
      expect(validateProperties(component, { [key]: '2/1' }).errors[key]).toBeTruthy();
    expect(validateProperties('arange', { n: '+2' }).errors.n).toBeTruthy();
  });
});
