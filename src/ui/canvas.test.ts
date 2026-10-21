import { describe, expect, it } from 'vitest';
import { nonfiniteKind, signal } from './Canvas';

describe('canvas value semantics', () => {
  it('distinguishes positive, negative and indeterminate nonfinite values', () => {
    expect(nonfiniteKind([1, Infinity])).toBe('positive');
    expect(nonfiniteKind([-1, -Infinity])).toBe('negative');
    expect(nonfiniteKind([NaN])).toBe('invalid');
    expect(nonfiniteKind([Infinity, -Infinity])).toBe('invalid');
    expect(nonfiniteKind([1, -2, 3])).toBeUndefined();
  });

  it('keeps finite values available for wire magnitude when a tensor also overflows', () => {
    expect(signal([1, Infinity, 3])).toBe(2);
    expect(signal([Infinity, -Infinity, NaN])).toBeUndefined();
  });
});
