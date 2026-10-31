// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { ShapePorts } from '../engine/graph';
import type { LevelMetadata } from '../engine/levels';

/** Flat probe inputs carry no shape; omit labels if the declared shapes are ambiguous. */
export function matrixAxes(
  variants: readonly ShapePorts[],
  labels: (
    inputs: Record<string, number[]>,
    shapes: ShapePorts,
  ) => ReturnType<NonNullable<LevelMetadata['outputAxes']>>,
): NonNullable<LevelMetadata['outputAxes']> {
  return (inputs) => {
    const matches = variants.filter((shapes) =>
      Object.entries(shapes).every(
        ([name, shape]) => inputs[name]?.length === shape.reduce((n, d) => n * d, 1),
      ),
    );
    return matches.length === 1 ? labels(inputs, matches[0]) : {};
  };
}

export const axisLabels = (prefix: string, length: number) =>
  Array.from({ length }, (_, i) => `${prefix}${i}`);
