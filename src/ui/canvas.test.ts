import { describe, expect, it } from 'vitest';
import {
  broadcastDisplayShapes,
  nonfiniteKind,
  signal,
  wireNeedsArc,
  wirePath,
  wireRupture,
} from './Canvas';

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

describe('textbook wire geometry', () => {
  it('draws a straight segment for ordinary forward wires', () => {
    const a = { x: 0, y: 0 };
    const b = { x: 200, y: 20 };
    expect(wireNeedsArc(a, b)).toBe(false);
    expect(wirePath(a, b)).toBe('M 0 0 L 200 20');
    const mid = wireRupture(a, b);
    expect(mid.x).toBe(100);
    expect(mid.y).toBe(10);
    expect(mid.angle).toBeCloseTo((Math.atan2(20, 200) * 180) / Math.PI, 5);
  });

  it('uses a light arc for steep forward diagonals that would tangle', () => {
    const a = { x: 0, y: 0 };
    const b = { x: 80, y: 120 };
    expect(wireNeedsArc(a, b)).toBe(true);
    expect(wirePath(a, b)).toMatch(/^M 0 0 C /);
    expect(wirePath(a, b)).not.toMatch(/ L /);
  });

  it('arcs backward wires so they wrap around neighbouring nodes', () => {
    const a = { x: 200, y: 40 };
    const b = { x: 40, y: 40 };
    expect(wireNeedsArc(a, b)).toBe(true);
    expect(wirePath(a, b)).toMatch(/^M 200 40 C /);
  });
});
