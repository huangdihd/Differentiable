// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  compileGraph,
  createDefaultRegistry,
  expandComponentSubgraph,
  GraphError,
  type GraphJSON,
  type ShapePorts,
} from './graph';
import { standardComponents } from './standard-components';
import { backward, tensor, type Tensor } from './tensor';

function inputShapes(steps: number, batch = 2, width = 3): ShapePorts {
  return { x: [batch, steps], wx: [1, width], wh: [width, width], b: [width] };
}

function model(component: string): GraphJSON {
  return {
    version: 1,
    nodes: [
      ...['x', 'wx', 'wh', 'b'].map((name) => ({
        id: name,
        component: 'input',
        attrs: { name },
      })),
      { id: 'window', component },
      { id: 'output', component: 'output', attrs: { name: 'history' } },
    ],
    edges: [
      ...['x', 'wx', 'wh', 'b'].map((port) => ({
        from: { node: port, port: 'out' },
        to: { node: 'window', port },
      })),
      { from: { node: 'window', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  };
}

function close(actual: Float32Array, expected: readonly number[], tolerance = 1e-5): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < expected.length; index++)
    expect(actual[index]).toBeCloseTo(expected[index], -Math.log10(tolerance));
}

const dotColumn = (
  row: readonly number[],
  matrix: readonly number[],
  columns: number,
  column: number,
) => row.reduce((total, value, index) => total + value * matrix[index * columns + column], 0);

function independentFixture(
  x: readonly number[],
  wx: readonly number[],
  wh: readonly number[],
  bias: readonly number[],
  width: number,
): number[] {
  let hidden = Array(width).fill(0) as number[];
  const history: number[] = [];
  for (const value of x) {
    hidden = hidden.map((_, column) =>
      Math.tanh(value * wx[column] + bias[column] + dotColumn(hidden, wh, width, column)),
    );
    history.push(...hidden);
  }
  return history;
}

function tensors(
  values: Record<string, readonly number[]>,
  shapes: ShapePorts,
): Record<string, Tensor> {
  return Object.fromEntries(
    Object.entries(values).map(([name, data]) => [name, tensor(data, shapes[name])]),
  );
}

describe('standard RNN window', () => {
  it('infers T from x and H from wh for multiple sequence lengths', () => {
    for (const steps of [0, 1, 2, 5]) {
      const plan = compileGraph(model('rnn_window'), { inputShapes: inputShapes(steps) });
      try {
        expect(plan.outputShapes.history).toEqual([2, steps, 3]);
      } finally {
        plan.dispose();
      }
    }
  });

  it('generates exactly T primitive steps sharing wx, wh and b', () => {
    const definition = standardComponents.find((candidate) => candidate.name === 'rnn_window')!;
    const registry = createDefaultRegistry();
    const expanded = expandComponentSubgraph(definition, {}, inputShapes(4))!;
    const primitiveNames = new Set(
      registry
        .list()
        .filter((candidate) => candidate.primitive)
        .map((candidate) => candidate.name),
    );

    expect(expanded.nodes.filter((node) => /^step_\d+_value$/.test(node.id))).toHaveLength(4);
    expect(expanded.nodes.every((node) => primitiveNames.has(node.component))).toBe(true);
    for (const parameter of ['wx', 'wh', 'b']) {
      expect(expanded.nodes.filter((node) => node.id === parameter)).toHaveLength(1);
      expect(expanded.edges.filter((edge) => edge.from.node === parameter)).toHaveLength(4);
    }
  });

  it('matches an independent two-step equation fixture', async () => {
    const width = 2;
    const shapes = inputShapes(2, 1, width);
    const data = {
      x: [0.35, -0.2],
      wx: [-0.22, -0.11],
      wh: [-0.14, -0.07, 0, 0.07],
      b: [0.04, 0],
    };
    const values = tensors(data, shapes);
    const plan = compileGraph(model('rnn_window'), { inputShapes: shapes });
    let output: Record<string, Tensor> | undefined;
    try {
      output = plan.execute(values);
      close(
        await output.history.read(),
        independentFixture(data.x, data.wx, data.wh, data.b, width),
      );
    } finally {
      if (output) plan.releaseExecution(output);
      plan.dispose();
      Object.values(values).forEach((value) => value.dispose());
    }
  });

  it('propagates gradients to shared inputs and parameters', async () => {
    const shapes = inputShapes(3, 1, 2);
    const values = tensors(
      { x: [0.2, -0.1, 0.4], wx: [0.3, -0.2], wh: [0.1, 0.2, -0.3, 0.4], b: [0, 0] },
      shapes,
    );
    const plan = compileGraph(model('rnn_window'), { inputShapes: shapes });
    let output: Record<string, Tensor> | undefined;
    try {
      output = plan.execute(values);
      const gradients = await backward(output.history, Object.values(values));
      try {
        expect(gradients).toHaveLength(4);
        for (const gradient of gradients) {
          expect(gradient.shape.reduce((size, dimension) => size * dimension, 1)).toBeGreaterThan(
            0,
          );
        }
      } finally {
        gradients.forEach((gradient) => gradient.dispose());
      }
    } finally {
      if (output) plan.releaseExecution(output);
      plan.dispose();
      Object.values(values).forEach((value) => value.dispose());
    }
  });

  it('validates sequence rank and parameter shapes with GraphError diagnostics', () => {
    expect(() =>
      compileGraph(model('rnn_window'), {
        inputShapes: { x: [2, 3, 1], wx: [1, 2], wh: [2, 2], b: [2] },
      }),
    ).toThrowError(GraphError);
    expect(() =>
      compileGraph(model('rnn_window'), {
        inputShapes: { x: [2, 3], wx: [1, 2], wh: [2, 3], b: [2] },
      }),
    ).toThrowError(GraphError);
  });

  it('remains subject to top-level component availability filtering', () => {
    expect(() =>
      compileGraph(model('rnn_window'), {
        inputShapes: inputShapes(2),
        availableComponents: [],
      }),
    ).toThrowError(/component_unavailable/);
  });
});
