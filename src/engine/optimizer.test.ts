// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it, vi } from 'vitest';
import { CpuBackend } from './cpu';
import { mul, sum, tensor } from './tensor';
import { Optimizer, TrainingSession } from './training';

it.each(['sgd', 'adam'] as const)('%s stays resident and releases state', async (kind) => {
  const backend = new CpuBackend();
  const read = vi.spyOn(backend, 'read');
  const parameters = new Map([['w', tensor([1], [1], backend)]]);
  const optimizer = new Optimizer({ kind, learningRate: 0.1 });
  for (let step = 1; step <= 5; step++) {
    const gradient = tensor([2], [1], backend);
    const reads = read.mock.calls.length;
    await optimizer.update(parameters, [gradient]);
    expect(read.mock.calls.length).toBe(reads);
    gradient.dispose();
    expect((await parameters.get('w')!.read())[0]).toBeCloseTo(
      1 - step * (kind === 'sgd' ? 0.2 : 0.1),
      6,
    );
    expect(parameters.get('w')!.tape).toBeUndefined();
    expect(backend.liveBuffers).toBe(kind === 'adam' ? 3 : 1);
  }
  optimizer.dispose();
  optimizer.dispose();
  expect(backend.liveBuffers).toBe(1);
  parameters.forEach((t) => t.dispose());
  expect(backend.liveBuffers).toBe(0);
});

it('reads training loss only at reporting intervals and final update', async () => {
  const backend = new CpuBackend();
  const parameters = new Map([['w', tensor([1], [1], backend)]]);
  const x = tensor([1], [1], backend);
  const session = new TrainingSession(
    { parameters, execute: () => ({ y: mul(parameters.get('w')!, x) }) },
    { inputs: { x }, targets: x },
    (y) => sum(mul(y, y), [0]),
    { kind: 'adam', learningRate: 0.01 },
    'y',
    { accuracy: false },
  );
  const read = vi.spyOn(backend, 'read');
  const reports: number[] = [];
  await session.run(7, { reportEvery: 3, onReport: (r) => reports.push(r.step) });
  expect(reports).toEqual([3, 6, 7]);
  expect(read).toHaveBeenCalledTimes(5);
  read.mockClear();
  reports.length = 0;
  await session.run(5, { onReport: (r) => reports.push(r.step) });
  expect(reports).toEqual([10, 12]);
  expect(read).toHaveBeenCalledTimes(4);
  await expect(session.run(1, { reportEvery: 0 })).rejects.toThrow('training.report_every');
  session.dispose();
  expect(backend.liveBuffers).toBe(2);
  x.dispose();
  parameters.forEach((t) => t.dispose());
  expect(backend.liveBuffers).toBe(0);
});

it('clips gradients to a global norm and releases temporaries', async () => {
  const backend = new CpuBackend();
  const parameters = new Map([
    ['a', tensor([0], [1], backend)],
    ['b', tensor([0], [1], backend)],
  ]);
  const optimizer = new Optimizer({ kind: 'sgd', learningRate: 1, clipNorm: 1 });
  const gradients = [tensor([3], [1], backend), tensor([4], [1], backend)];
  await optimizer.update(parameters, gradients);
  expect((await parameters.get('a')!.read())[0]).toBeCloseTo(-0.6, 6);
  expect((await parameters.get('b')!.read())[0]).toBeCloseTo(-0.8, 6);
  gradients.forEach((g) => g.dispose());
  parameters.forEach((t) => t.dispose());
  expect(backend.liveBuffers).toBe(0);
});

it('Adam stays finite when the second moment is denormal or zero', async () => {
  const backend = new CpuBackend();
  const parameters = new Map([['w', tensor([1, 1, 1], [3], backend)]]);
  const optimizer = new Optimizer({ kind: 'adam', learningRate: 0.02 });
  for (const g of [[1e-30, 0, 1e-20], [0, 0, 0], [1e-38, 1e-45, 0]]) {
    const gradient = tensor(g, [3], backend);
    await optimizer.update(parameters, [gradient]);
    gradient.dispose();
  }
  expect([...(await parameters.get('w')!.read())].every(Number.isFinite)).toBe(true);
  optimizer.dispose();
  parameters.forEach((t) => t.dispose());
});

it('WebGPU Adam kernel guards sqrt of a tiny second moment', async () => {
  const { optimizerKernel } = await import('./webgpu-optimizer');
  const input = (shape: number[]) => ({ shape }) as never;
  const spec = optimizerKernel({
    op: 'optimizer_adam',
    inputs: [input([4]), input([4]), input([4]), input([2])],
    output: input([4]),
    attrs: { learningRate: 0.02, epsilon: 1e-8 },
  } as never)!;
  expect(spec.code).toContain('select(sqrt(x2[i] / x3[1]), 0.0, !(x2[i] / x3[1] > 1.0e-30))');
});
