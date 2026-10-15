import { expect, it, vi } from 'vitest';
import { BufferPool, FREE_BUFFER_BUDGET } from './buffer-pool';
const buffer = (size: number) => ({ size, destroy: vi.fn() });
it('bounds all size buckets by bytes and evicts least recently used free storage', () => {
  const destroyed = vi.fn();
  const pool = new BufferPool(12, destroyed);
  const a = buffer(4),
    b = buffer(8),
    c = buffer(8);
  pool.put(a);
  pool.put(b);
  expect(pool.take(4)).toBe(a);
  pool.put(a); // used again, now newest
  pool.put(c);
  expect(b.destroy).toHaveBeenCalledOnce();
  expect(a.destroy).not.toHaveBeenCalled();
  expect(pool.bytes).toBe(12);
  expect(pool.count).toBe(2);
  expect(destroyed).toHaveBeenCalledWith(8);
  expect(pool.take(16)).toBeUndefined();
  pool.clear();
  expect(pool.bytes).toBe(0);
  expect(pool.count).toBe(0);
  expect(a.destroy).toHaveBeenCalledOnce();
  expect(c.destroy).toHaveBeenCalledOnce();
});
it('destroys oversize buffers without evicting useful cached storage', () => {
  expect(FREE_BUFFER_BUDGET).toBe(64 * 1024 * 1024);
  const pool = new BufferPool(8);
  const a = buffer(4),
    huge = buffer(12);
  pool.put(a);
  pool.put(a);
  pool.put(huge);
  expect(huge.destroy).toHaveBeenCalledOnce();
  expect(pool.count).toBe(1);
  expect(pool.take(4)).toBe(a);
  expect(pool.bytes).toBe(0);
  expect(a.destroy).not.toHaveBeenCalled();
});
