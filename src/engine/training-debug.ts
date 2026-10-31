// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { Tensor } from './tensor';

let enabled = false;
let reported = false;
/** Dev-only: when on, the optimizer checks every gradient/parameter and logs the first nonfinite one. */
export function setTrainingDebug(on: boolean): void {
  enabled = on;
  reported = false;
}
export function trainingDebugEnabled(): boolean {
  return enabled && !reported;
}
function summary(values: Float32Array) {
  let nan = 0,
    inf = 0,
    max = 0;
  for (const v of values) {
    if (Number.isNaN(v)) nan++;
    else if (!Number.isFinite(v)) inf++;
    else max = Math.max(max, Math.abs(v));
  }
  return { nan, inf, maxAbsFinite: max, first: values.findIndex((v) => !Number.isFinite(v)) };
}
/** Returns true (and logs once) if the tensor holds a NaN/inf. */
export async function debugCheck(
  step: number,
  stage: string,
  key: string,
  tensor: Tensor | number,
  extra: Record<string, unknown> = {},
): Promise<boolean> {
  if (!trainingDebugEnabled()) return false;
  const values = typeof tensor === 'number' ? new Float32Array([tensor]) : await tensor.read();
  const s = summary(values);
  if (s.nan + s.inf === 0) return false;
  reported = true;
  console.error(
    '[devDebug] first nonfinite',
    JSON.stringify({
      step,
      stage,
      key,
      shape: typeof tensor === 'number' ? [] : tensor.shape,
      ...s,
      ...extra,
    }),
  );
  return true;
}
