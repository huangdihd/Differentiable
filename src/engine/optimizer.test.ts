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
