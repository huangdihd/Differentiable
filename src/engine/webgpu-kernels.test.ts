import { expect, it } from 'vitest';
import { kernel } from './webgpu-kernels';
import type { Command } from './backend';

function reduction(op: string, width: number, vjp = false) {
  const x = { handle: { id: 0 }, shape: [3, width] };
  const y = { handle: { id: 1 }, shape: ['sum', 'reduce_max'].includes(op) ? [3] : x.shape };
  return kernel({
    op: vjp ? 'vjp' : op,
    inputs: vjp ? [x, y, y] : [x],
    output: vjp ? x : y,
    attrs: { axes: [-1], op, input: 0, count: 1 },
  } satisfies Command);
}

it('scatter_add reuses gather accumulation and validates indices even for empty outputs', () => {
  for (const width of [0, 2]) {
    const table = { handle: { id: 0 }, shape: [3, width] };
    const idx = { handle: { id: 1 }, shape: [130] };
    const updates = { handle: { id: 2 }, shape: [130, width] };
    const spec = kernel({
      op: 'scatter_add',
      inputs: [table, idx, updates],
      output: table,
      attrs: {},
    });
    expect(spec.workgroups).toEqual([3]);
    expect(spec.code).toContain('total += x2[');
    expect(spec.code).toContain('atomicStore(&error[0], 1u)');
    expect(spec.code.indexOf('atomicStore')).toBeLessThan(spec.code.indexOf('return;'));
    const adjoint = kernel({
      op: 'vjp',
      inputs: [table, idx, updates, table, table],
      output: updates,
      attrs: { op: 'scatter_add', input: 2, count: 3 },
    });
    expect(adjoint.code).toContain('out[i] = x4[u32(row)');
  }
});

it('uses one 64-lane group per normalization row in both directions', () => {
  for (const op of ['softmax', 'layernorm'])
    for (const width of [1, 1024, 1031])
      for (const vjp of [false, true]) {
        const spec = reduction(op, width, vjp);
        expect(spec.workgroups).toEqual([3]);
        expect(spec.code).toContain('@workgroup_size(64)');
        expect(spec.code).toContain('workgroupBarrier();');
        expect(spec.code).toContain('c += 64u');
        expect(spec.code).not.toContain('bitcast<f32>(0xff800000u)');
      }
});

it('switches long arbitrary-axis reductions to fixed tree order and integer tie winners', () => {
  for (const op of ['sum', 'reduce_max']) {
    expect(reduction(op, 255).workgroups).toEqual([1]);
    for (const width of [256, 1024, 1031]) {
      expect(reduction(op, width).workgroups).toEqual([3]);
      expect(reduction(op, width).code).toContain('stride /= 2u');
      const adjoint = reduction(op, width, true);
      if (op === 'reduce_max') {
        expect(adjoint.workgroups).toEqual([3]);
        expect(adjoint.code).toContain('min(winners[lane], winners[lane + stride])');
        expect(adjoint.code).toContain('r == winner');
      } else {
        expect(adjoint.workgroups).toEqual([Math.ceil((3 * width) / 64)]);
        expect(adjoint.code).not.toContain('workgroupBarrier');
      }
    }
  }
});
