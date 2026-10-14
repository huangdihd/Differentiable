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
