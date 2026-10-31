// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { safeCompileGraph, type Attributes, type GraphJSON } from './graph';

function errorFor(component: string, attrs: Attributes, shape = [2, 3]) {
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'input', component: 'input', attrs: { name: 'x', shape } },
      { id: 'operation', component, attrs },
    ],
    edges: [{ from: { node: 'input', port: 'out' }, to: { node: 'operation', port: 'x' } }],
  };
  const result = safeCompileGraph(graph);
  expect(result.ok).toBe(false);
  if (result.ok) throw new Error('Expected compilation failure');
  return result.errors[0];
}

describe('graph error parameter contracts', () => {
  it.each([
    ['sum', { axes: [2] }],
    ['sum', { axes: [0, -2] }],
    ['sum', { axes: 'bad' }],
    ['reduce_max', { axes: [0.5] }],
    ['transpose', { axes: [0] }],
  ])('reports original axes and their accepted range for %s', (component, attrs) => {
    expect(errorFor(component, attrs)).toMatchObject({
      key: 'invalid_axes',
      node: 'operation',
      params: { actual: attrs.axes, rank: 2, allowed: [-2, -1, 0, 1] },
    });
  });
  it('includes insertion positions for unsqueeze and an empty range for scalars', () => {
    expect(errorFor('unsqueeze', { axis: 3 })).toMatchObject({
      key: 'invalid_axes',
      params: { actual: 3, rank: 2, allowed: [-3, -2, -1, 0, 1, 2] },
    });
    expect(errorFor('sum', { axes: [0] }, [])).toMatchObject({
      key: 'invalid_axes',
      params: { actual: [0], rank: 0, allowed: [] },
    });
    expect(errorFor('size', { axis: -3 })).toMatchObject({
      key: 'size_axis',
      params: { actual: -3, rank: 2, allowed: [-2, -1, 0, 1] },
    });
  });
  it('reports permutation, invalid shapes and reshape sizes', () => {
    expect(errorFor('transpose', { perm: [0, 0] })).toMatchObject({
      key: 'invalid_permutation',
      params: { actual: [0, 0], rank: 2, allowed: [0, 1] },
    });
    expect(errorFor('reshape', { shape: [-1, -1] })).toMatchObject({
      key: 'invalid_shape',
      params: { shape: [-1, -1] },
    });
    expect(errorFor('reshape', { shape: [4] })).toMatchObject({
      key: 'reshape_mismatch',
      params: { actual: [2, 3], expected: [4], actual_size: 6, expected_size: 4 },
    });
  });
  it.each([
    ['arange', { n: 1 }, 'n'],
    ['reshape', { shape: [1] }, 'd0'],
  ])('preserves node and port for dynamic %s dimensions', (component, attrs, port) => {
    const graph: GraphJSON = {
      version: 1,
      nodes: [
        { id: 'dynamic', component: 'input', attrs: { name: 'n', shape: [] } },
        { id: 'operation', component, attrs },
      ],
      edges: [{ from: { node: 'dynamic', port: 'out' }, to: { node: 'operation', port } }],
    };
    if (component === 'reshape')
      graph.edges.push({
        from: { node: 'dynamic', port: 'out' },
        to: { node: 'operation', port: 'x' },
      });
    expect(safeCompileGraph(graph)).toMatchObject({
      ok: false,
      errors: [{ key: 'dim_not_static', node: 'operation', port, params: { port } }],
    });
  });
  it('reports constant element counts and compile-time input shapes', () => {
    expect(
      safeCompileGraph({
        version: 1,
        nodes: [{ id: 'constant', component: 'const', attrs: { shape: [2], value: [1] } }],
        edges: [],
      }),
    ).toMatchObject({
      ok: false,
      errors: [
        {
          key: 'invalid_constant',
          node: 'constant',
          params: {
            actual: [1],
            shape: [2],
            expected: 2,
            actual_count: 1,
          },
        },
      ],
    });
    expect(
      safeCompileGraph(
        {
          version: 1,
          nodes: [{ id: 'input', component: 'input', attrs: { name: 'x', shape: [2] } }],
          edges: [],
        },
        { inputShapes: { x: [3] } },
      ),
    ).toMatchObject({
      ok: false,
      errors: [
        {
          key: 'input_shape_mismatch',
          node: 'input',
          port: 'x',
          params: { expected: [2], actual: [3] },
        },
      ],
    });
  });
});
