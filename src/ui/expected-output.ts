import { CpuBackend } from '../engine/cpu';
import type { ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';

const backend = new CpuBackend();
type Tolerance = { kind: string; atol?: number; rtol?: number };

/** The answer for one probe case, computed exactly as the judge does (f, or f' by autodiff). */
export async function expectedOutput(
  level: ForwardLevel,
  inputs: Record<string, number[]>,
): Promise<{ shape: number[]; values: number[] }> {
  const tensors = Object.fromEntries(
    Object.entries(level.interface.inputs).map(([name, shape]) => [
      name,
      T.tensor(inputs[name] ?? [], shape, backend),
    ]),
  );
  const reference = level.reference(tensors);
  try {
    const [result] =
      level.kind === 'derivative'
        ? await T.backward(reference, [tensors[level.derivativeInput ?? 'x']])
        : [reference];
    const values = Array.from(await result.read());
    if (result !== reference) result.dispose();
    return { shape: [...result.shape], values };
  } finally {
    T.disposeGraph([reference]);
  }
}

/** Element indices outside the judge tolerance (empty when shapes differ: nothing lines up). */
export function offIndices(
  level: Tolerance,
  actual: { shape: readonly number[]; values: readonly number[] },
  expected: { shape: readonly number[]; values: readonly number[] },
): Set<number> {
  if (actual.shape.join() !== expected.shape.join()) return new Set();
  const atol = level.atol ?? 1e-5,
    rtol = level.rtol ?? 1e-4;
  return new Set(
    actual.values.flatMap((x, i) =>
      Number.isFinite(x) &&
      Math.abs(x - expected.values[i]) <= atol + rtol * Math.abs(expected.values[i])
        ? []
        : [i],
    ),
  );
}

/** Same tolerance as the judge, so the panel's ✓ never disagrees with "Run tests". */
export function matchesExpected(
  level: Tolerance,
  actual: { shape: readonly number[]; values: readonly number[] },
  expected: { shape: readonly number[]; values: readonly number[] },
): boolean {
  const atol = level.atol ?? 1e-5,
    rtol = level.rtol ?? 1e-4;
  return (
    actual.shape.join() === expected.shape.join() &&
    actual.values.length === expected.values.length &&
    actual.values.every(
      (x, i) =>
        Number.isFinite(x) &&
        Number.isFinite(expected.values[i]) &&
        Math.abs(x - expected.values[i]) <= atol + rtol * Math.abs(expected.values[i]),
    )
  );
}
