import { describe, expect, it } from 'vitest';
import { defaults, validateProperties } from './Properties';

describe('property field validation', () => {
  it('preserves component defaults', () => {
    expect(defaults('sum')).toEqual({ axes: [-1], keepdims: true });
    expect(defaults('reduce_max')).toEqual({ axes: [-1], keepdims: true });
    expect(defaults('unsqueeze')).toEqual({ axis: -1 });
    expect(defaults('param')).toEqual({ shape: [2, 2], init: 'normal', scale: 1 });
  });
  it('accepts negative integer unsqueeze axes but rejects incomplete or fractional axes', () => {
    for (const axis of ['-1', '-3', '0', '2']) {
      const result = validateProperties('unsqueeze', { axis });
      expect(result.errors).toEqual({});
      expect(result.attrs.axis).toBe(Number(axis));
    }
    for (const axis of ['', '-', '1.5', 'null', '1e999', '9007199254740992'])
      expect(validateProperties('unsqueeze', { axis }).errors.axis).toBe('ui.invalidNumber');
  });
  it('supports scalar, vector and broadcast constants', () => {
    expect(validateProperties('const', { value: '2' }).errors).toEqual({});
    expect(validateProperties('const', { value: '[1,2]', shape: '2' }).errors).toEqual({});
    expect(validateProperties('const', { value: '2', shape: '2,3' }).errors).toEqual({});
    expect(validateProperties('const', { value: '[1,2]', shape: '3' }).errors.shape).toBe(
      'ui.invalidShape',
    );
  });
  it('requires finite numbers and complete input', () => {
    for (const value of ['', '-', '1e', '1e999', 'null'])
      expect(validateProperties('const', { value }).errors.value).toBe('ui.invalidNumber');
    for (const n of ['-1', '1.5', ''])
      expect(validateProperties('arange', { n }).errors.n).toBe('ui.invalidNumber');
  });
  it('matches initializer and scale constraints', () => {
    for (const init of ['zeros', 'ones', 'normal', 'uniform', 'constant'])
      expect(
        validateProperties('param', { shape: '2,2', init, scale: '0', value: '3' }).errors,
      ).toEqual({});
    expect(validateProperties('param', { shape: '[]', init: 'constant' }).errors.value).toBe(
      'ui.invalidNumber',
    );
    expect(validateProperties('param', { shape: '[]', scale: '-1' }).errors.scale).toBe(
      'ui.invalidNumber',
    );
    expect(validateProperties('param', { shape: '[]', init: 'unknown' }).errors.init).toBe(
      'ui.invalidInit',
    );
  });
  it('validates reshape, axes and permutations without knowing input rank', () => {
    expect(validateProperties('reshape', { shape: '-1,2' }).errors).toEqual({});
    for (const shape of ['-1,-1', '-2', '-1,0', '2,', '9007199254740991,2'])
      expect(validateProperties('reshape', { shape }).errors.shape).toBe('ui.invalidShape');
    expect(validateProperties('sum', { axes: '-1', keepdims: 'true' }).errors).toEqual({});
    expect(validateProperties('sum', { axes: '1,1' }).errors.axes).toBe('ui.invalidShape');
    expect(validateProperties('transpose', { perm: '1,0' }).errors).toEqual({});
    expect(validateProperties('transpose', { perm: '2,0' }).errors.perm).toBe('ui.invalidShape');
  });
  it('removes blank optional attributes and preserves unrelated attributes', () => {
    expect(
      validateProperties('const', { value: '1', shape: '' }, { shape: [2], custom: true }).attrs,
    ).toEqual({ value: 1, custom: true });
    expect(validateProperties('sum', {}).attrs).toEqual({ keepdims: false });
  });
});
