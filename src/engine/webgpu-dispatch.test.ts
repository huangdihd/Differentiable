// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import type { Command, Shape } from './backend';
import { dispatchGrid, limitDispatch, type DispatchKernel } from './webgpu-dispatch';
import { kernel } from './webgpu-kernels';
import { optimizerKernel } from './webgpu-optimizer';

const command = (op: string, inputs: Shape[], output: Shape): Command => ({
  op,
  inputs: inputs.map((shape, id) => ({ shape, handle: { id } })),
  output: { shape: output, handle: { id: inputs.length } },
  attrs: { learningRate: 0.1, beta1: 0.9, beta2: 0.99, epsilon: 1e-8 },
});

describe('WebGPU dispatch remapping', () => {
  it('keeps in-limit shader code and dispatch unchanged', () => {
    const spec = kernel(command('exp', [[64]], [64]));
    expect(limitDispatch(spec, 65535)).toBe(spec);
    expect(dispatchGrid([2, 3], 4)).toEqual([2, 3, 1]);
  });

  it.each<[number, number, number]>([
    [17, 1, 1],
    [2, 17, 1],
    [2, 1, 17],
    [5, 3, 4],
  ])('maps logical grid (%i, %i, %i) exactly once, discarding only padding', (lx, ly, lz) => {
    const [px, py, pz] = dispatchGrid([lx, ly, lz], 4);
    expect(Math.max(px, py, pz)).toBeLessThanOrEqual(4);
    const seen = new Set<string>();
    for (let z = 0; z < pz; z++)
      for (let y = 0; y < py; y++)
        for (let x = 0; x < px; x++) {
          const index = x + px * (y + py * z);
          if (index >= lx * ly * lz) continue;
          const logical = [index % lx, Math.floor(index / lx) % ly, Math.floor(index / (lx * ly))];
          expect(logical[2]).toBeLessThan(lz);
          seen.add(logical.join(','));
        }
    expect(seen.size).toBe(lx * ly * lz);
  });

  it('handles the real device boundary and third physical dimension', () => {
    expect(dispatchGrid([65536], 65535)).toEqual([65535, 2, 1]);
    expect(dispatchGrid([65], 8)).toEqual([8, 8, 2]);
  });

  it.each([0, -1, 1.5, NaN, Infinity, 0x100000000])(
    'rejects invalid/address-overflow counts: %s',
    (n) => {
      expect(() => dispatchGrid([n], 65535)).toThrow(RangeError);
    },
  );
  it('rejects unrepresentable grids instead of silently losing groups', () => {
    expect(() => dispatchGrid([65], 4)).toThrow(RangeError);
    expect(() => dispatchGrid([1], 0)).toThrow(RangeError);
  });

  it('remaps elementwise global IDs including all lanes', () => {
    const mapped = limitDispatch(kernel(command('exp', [[1088]], [1088])), 4);
    expect(mapped.workgroups).toEqual([4, 4, 2]);
    expect(mapped.code).toContain(
      'let gid = dispatchLogical * vec3<u32>(64u, 1u, 1u) + dispatchLocal;',
    );
    expect(mapped.code).toContain('if (dispatchIndex >= 17u) { return; }');
    expect(mapped.code).toContain('let i = gid.x;');
  });

  it.each(['optimizer_sgd', 'optimizer_m', 'optimizer_v', 'optimizer_adam'])(
    'automatically remaps %s through the same backend adapter',
    (op) => {
      const spec = optimizerKernel(command(op, [[1088], [1088], [1088], [2]], [1088]))!;
      const mapped = limitDispatch(spec, 4);
      expect(mapped.workgroups).toEqual([4, 4, 2]);
      expect(mapped.code).toContain(
        'let gid = dispatchLogical * vec3<u32>(64u, 1u, 1u) + dispatchLocal;',
      );
      expect(mapped.code).toContain('let i = gid.x;');
    },
  );

  it('remaps tiled matmul workgroups while retaining local IDs and barriers', () => {
    const spec = kernel(
      command(
        'matmul',
        [
          [80, 16],
          [16, 32],
        ],
        [80, 32],
      ),
    );
    const mapped = limitDispatch(spec, 4);
    expect(mapped.workgroups.every((n) => n! <= 4)).toBe(true);
    expect(mapped.code).toContain('let group = dispatchLogical;');
    expect(mapped.code).toContain('@builtin(local_invocation_id) local: vec3<u32>');
    expect(mapped.code.indexOf('if (dispatchIndex')).toBeLessThan(
      mapped.code.indexOf('workgroupBarrier()'),
    );
    expect(mapped.code.match(/workgroupBarrier\(\)/g)).toEqual(
      spec.code.match(/workgroupBarrier\(\)/g),
    );
  });

  it('preserves local lane IDs and logical num_workgroups in cooperative kernels', () => {
    const spec: DispatchKernel = {
      workgroups: [5, 2, 3],
      code: `@compute @workgroup_size(8, 2, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) index: u32) {
workgroupBarrier();
}`,
    };
    const mapped = limitDispatch(spec, 4);
    expect(mapped.code).toContain('let group = dispatchLogical;');
    expect(mapped.code).toContain('let gid = dispatchLogical * vec3<u32>(8u, 2u, 1u) + lane;');
    expect(mapped.code).toContain('let groups = vec3<u32>(5u, 2u, 3u);');
    expect(mapped.code).toContain('@builtin(local_invocation_index) index: u32');
    expect(mapped.code).not.toContain('dispatchLocal');
  });
});
