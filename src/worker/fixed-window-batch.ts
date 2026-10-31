// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import {
  ComponentRegistry,
  compileGraph,
  type ComponentImplementation,
  type GraphJSON,
  type Shape,
  type ShapePorts,
} from '../engine/graph';

function withReshapeInference(
  registry: ComponentRegistry,
  wrap: (implementation: ComponentImplementation) => ComponentImplementation['infer'],
): ComponentRegistry {
  const result = new ComponentRegistry();
  for (const definition of registry.list()) {
    if (definition.name !== 'reshape' || !definition.primitive) {
      result.register(definition);
      continue;
    }
    result.register({
      ...definition,
      primitive: { ...definition.primitive, infer: wrap(definition.primitive) },
    });
  }
  return result;
}

const rowSize = (shape: Shape): number => shape.slice(1).reduce((size, width) => size * width, 1);
const signature = (target: readonly unknown[], input: Shape): string =>
  JSON.stringify([target, input.slice(1)]);

/**
 * Adapt only validated, row-preserving reshapes from the fixed-window circuit's x batch.
 * This is a per-experiment registry, not a general graph or parameter-shape migration.
 */
export function fixedWindowBatchRegistry(
  graph: GraphJSON,
  registry: ComponentRegistry,
  inputShapes: ShapePorts,
  availableComponents?: readonly string[],
): ComponentRegistry {
  const originalRows = inputShapes.x?.[0];
  const signatures = new Set<string>();
  const observingRegistry = withReshapeInference(registry, (implementation) => (inputs, attrs) => {
    const outputs = implementation.infer(inputs, attrs);
    if (
      originalRows !== undefined &&
      originalRows > 0 &&
      Array.isArray(attrs.shape) &&
      attrs.shape[0] === originalRows &&
      inputs.x?.[0] === originalRows &&
      outputs.out?.[0] === originalRows &&
      rowSize(inputs.x) === rowSize(outputs.out)
    )
      signatures.add(signature(attrs.shape, inputs.x));
    return outputs;
  });

  // Validate the whole original circuit before permitting any adaptation. Compilation is
  // allocation-free: do not execute it or initialize player parameters just to inspect shapes.
  const originalPlan = compileGraph(graph, {
    registry: observingRegistry,
    inputShapes,
    availableComponents,
    preferNative: false,
  });
  try {
    return withReshapeInference(registry, (implementation) => (inputs, attrs) => {
      if (
        inputs.x?.length &&
        inputs.x[0] !== originalRows &&
        Array.isArray(attrs.shape) &&
        signatures.has(signature(attrs.shape, inputs.x))
      ) {
        // The compiler owns this attrs object and retains it for execution. Replace only
        // its shape array, including after dimension-port resolution; never mutate the save.
        attrs.shape = [inputs.x[0], ...attrs.shape.slice(1)];
      }
      return implementation.infer(inputs, attrs);
    });
  } finally {
    originalPlan.dispose();
  }
}
