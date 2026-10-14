import { describe, it, expect } from 'vitest';
import { CpuBackend } from './cpu';
import { tensor, add, mul, sum, backward, disposeGraph, gather, reshape, operation } from './tensor';
import { Random } from './random';

describe('backend ownership and reproducibility', () => {
  it('isolates uploads/readbacks and rejects disposed tensors', async () => {
    const backend = new CpuBackend(), data = new Float32Array([1, 2]);
    const x = tensor(data, [2], backend); data[0] = 99;
    const read = await x.read(); read[0] = 88;
    expect(Array.from(await x.read())).toEqual([1, 2]);
    x.dispose(); x.dispose(); expect(backend.liveBuffers).toBe(0);
    expect(() => add(x, x)).toThrow('disposed');
  });
  it('releases all intermediates and adjoints over repeated training-shaped runs', async () => {
    const backend = new CpuBackend(), p = tensor([1, 2], [2], backend);
    for (let i = 0; i < 100; i++) {
      const y = sum(mul(p, p), [0]);
      const [g] = await backward(y, [p]);
      expect(Array.from(await g.read())).toEqual([2, 4]);
      g.dispose(); disposeGraph([y], [p]);
      expect(backend.liveBuffers).toBe(1);
    }
    p.dispose(); expect(backend.liveBuffers).toBe(0);
  });
  it('cleans failed kernel outputs and rejects malformed shapes/indices', () => {
    const b = new CpuBackend(), table = tensor([1, 2], [2, 1], b), idx = tensor([0.5], [1], b);
    expect(() => gather(table, idx)).toThrow('index');
    expect(b.liveBuffers).toBe(2);
    expect(() => reshape(table, [3])).toThrow('reshape');
    expect(() => tensor([], [-1], b)).toThrow('shape');
    table.dispose(); idx.dispose();
  });
  it('registers extensible native kernels with their VJP', async () => {
    const b = new CpuBackend();
    b.register('square', ([x], _s, _a, out) => out.set(x.map(v => v * v)),
      ([x, _y, g], _s, _a, out) => out.set(x.map((v, i) => 2 * v * g[i])));
    const x = tensor([2, 3], [2], b), y = operation('square', [x], {}, x.shape);
    const [g] = await backward(y, [x]);
    expect(Array.from(await y.read())).toEqual([4, 9]);
    expect(Array.from(await g.read())).toEqual([4, 6]);
    g.dispose(); disposeGraph([y]); expect(b.liveBuffers).toBe(0);
  });
  it('has seeded repeatable independent random streams', () => {
    const a = new Random(18), b = new Random(18), c = new Random(19);
    const values = Array.from({ length: 100 }, () => a.normal());
    expect(values).toEqual(Array.from({ length: 100 }, () => b.normal()));
    expect(values).not.toEqual(Array.from({ length: 100 }, () => c.normal()));
  });
});
