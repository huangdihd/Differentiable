import { describe, expect, it } from 'vitest';
import { broadcastDisplayShapes, nonfiniteKind, signal } from './Canvas';

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

  it.each(['add', 'sub', 'mul', 'div', 'max', 'ge'])(
    'shows mismatched operand shapes for broadcasting %s',
    (component) => {
      expect(broadcastDisplayShapes(component, [4, 1], [3], [4, 3])).toEqual({
        a: [4, 1],
        b: [3],
        output: [4, 3],
      });
    },
  );

  it('hides the broadcast note for equal shapes, unary ops, and incomplete probes', () => {
    expect(broadcastDisplayShapes('div', [4, 3], [4, 3], [4, 3])).toBeUndefined();
    expect(broadcastDisplayShapes('recip', [4, 1], [3], [4, 3])).toBeUndefined();
    expect(broadcastDisplayShapes('div', [4, 1], undefined, [4, 3])).toBeUndefined();
  });
});
