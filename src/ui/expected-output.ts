// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { CpuBackend } from '../engine/cpu';
import { levelDerivativeInputs, levelOutputShapes, type ForwardLevel } from '../engine/levels';
import * as T from '../engine/tensor';

const backend = new CpuBackend();
type Tolerance = { kind: string; atol?: number; rtol?: number };

export interface ExpectedValue {
  shape: number[];
  values: number[];
}

/** All answers for one probe case, computed exactly as the judge does. */
export async function expectedOutputs(
  level: ForwardLevel,
  inputs: Record<string, number[]>,
  shapes: Readonly<Record<string, readonly number[]>> = level.interface.inputs,
): Promise<Record<string, ExpectedValue>> {
  const tensors = Object.fromEntries(
    Object.entries(shapes).map(([name, shape]) => [
      name,
      T.tensor(inputs[name] ?? [], shape, backend),
    ]),
  );
  let reference: T.Tensor | undefined;
  let results: T.Tensor[] = [];
  try {
    const names = Object.keys(levelOutputShapes(level));
    if (level.referenceOutputs) {
      const named = level.referenceOutputs(tensors);
      if (
        Object.keys(named).length !== names.length ||
        names.some((name) => !Object.hasOwn(named, name))
      )
        throw new Error('levels.output_shape');
      results = names.map((name) => named[name]);
    } else {
      reference = level.reference(tensors);
      results =
        level.kind === 'derivative'
          ? await T.backward(
              reference,
              levelDerivativeInputs(level).map((name) => tensors[name]),
            )
          : [reference];
    }
    if (names.length !== results.length) throw new Error('levels.output_shape');
    return Object.fromEntries(
      await Promise.all(
        names.map(async (name, index) => [
          name,
          { shape: [...results[index].shape], values: Array.from(await results[index].read()) },
        ]),
      ),
    );
  } finally {
    if (reference) T.disposeGraph([reference], Object.values(tensors));
    if (results.length) T.disposeGraph(results, Object.values(tensors));
    Object.values(tensors).forEach((tensor) => tensor.dispose());
  }
}

/** The primary answer retained for existing single-output plots and callers. */
export async function expectedOutput(
  level: ForwardLevel,
  inputs: Record<string, number[]>,
  shapes: Readonly<Record<string, readonly number[]>> = level.interface.inputs,
): Promise<ExpectedValue> {
  return (await expectedOutputs(level, inputs, shapes))[level.interface.output];
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
