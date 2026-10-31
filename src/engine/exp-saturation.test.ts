// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import { add, backward, exp, mul, recip, sum, tensor } from './tensor';
import { kernel } from './webgpu-kernels';

// The 0-6 tanh formula div(sub(exp(2x), 1), exp(2x) + 1) used to give inf/inf = NaN once a
// pre-activation passed ~44, which a fixed-window run at lr 0.02 can reach (seen on WebGPU).
it('player-style exp tanh stays finite with finite gradients for large inputs', async () => {
  const backend = new CpuBackend();
  const x = tensor([60, -60, 0.5, 200], [4], backend);
  const e = exp(mul(tensor([2], [], backend), x));
  const y = mul(add(e, tensor([-1], [], backend)), recip(add(e, tensor([1], [], backend))));
  const values = [...(await y.read())];
  expect(values.every(Number.isFinite)).toBe(true);
  expect(values[0]).toBeCloseTo(1, 5);
  expect(values[1]).toBeCloseTo(-1, 5);
  expect(values[2]).toBeCloseTo(Math.tanh(0.5), 5);
  const [grad] = await backward(sum(y, [0]), [x]);
  expect([...(await grad.read())].every(Number.isFinite)).toBe(true);
});

it('WebGPU exp kernel saturates like the CPU', () => {
  const spec = kernel({
    op: 'exp',
    inputs: [{ shape: [4], handle: { id: 0 } }],
    output: { shape: [4], handle: { id: 1 } },
    attrs: {},
  } as never)!;
  expect(spec.code).toContain('exp(min(x0[i], 88.0))');
});
