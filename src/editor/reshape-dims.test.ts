// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { createDefaultRegistry } from '../engine/graph';
import {
  concreteReshapeDim,
  reshapeShapeForWiredPorts,
  withWiredReshapeDims,
} from './reshape-dims';
import { createEditorState, editorReducer, type EditorCommand, type EditorState } from './state';
import { reshapeLevel } from '../levels/chapter4';

const registry = createDefaultRegistry();
const reduce = (s: EditorState, c: EditorCommand) => editorReducer(s, c, registry);

describe('reshape wired dimension persistence', () => {
  it('replaces a wired -1 with the concrete wire size', () => {
    expect(concreteReshapeDim(-1, 8)).toBe(8);
    expect(reshapeShapeForWiredPorts([-1, 2], ['d0'], { d0: 8 })).toEqual([8, 2]);
    expect(reshapeShapeForWiredPorts([8, -1], ['d0'])).toEqual([8, -1]);
  });

  it('stamps a concrete dim on connect and keeps it after disconnect', () => {
    let s = createEditorState(reshapeLevel, undefined, [
      ...reshapeLevel.available,
      'reshape',
      'const',
    ]);
    s = reduce(s, {
      type: 'addNode',
      component: 'reshape',
      position: { x: 200, y: 100 },
      attrs: { shape: [-1, 2] },
    });
    s = reduce(s, {
      type: 'addNode',
      component: 'const',
      position: { x: 60, y: 260 },
      attrs: { value: 8, shape: [] },
    });
    const reshape = s.graph.nodes.find((n) => n.component === 'reshape')!;
    const constant = s.graph.nodes.find((n) => n.component === 'const')!;
    s = reduce(s, {
      type: 'connect',
      from: { node: constant.id, port: 'out' },
      to: { node: reshape.id, port: 'd0' },
    });
    expect(s.error).toBeNull();
    expect(s.graph.nodes.find((n) => n.id === reshape.id)?.attrs?.shape).toEqual([8, 2]);

    const edge = s.graph.edges.find(
      (e) => e.to.node === reshape.id && e.to.port === 'd0',
    )!;
    s = reduce(s, { type: 'disconnect', edge });
    expect(s.graph.edges.some((e) => e.to.node === reshape.id && e.to.port === 'd0')).toBe(
      false,
    );
    expect(s.graph.nodes.find((n) => n.id === reshape.id)?.attrs?.shape).toEqual([8, 2]);
  });

  it('falls back to 1 when the wire size is not a static const', () => {
    const graph = {
      version: 1 as const,
      nodes: [
        { id: 'r', component: 'reshape', attrs: { shape: [-1, 3] } },
        { id: 's', component: 'size', attrs: { axis: 0 } },
        { id: 'x', component: 'input', attrs: { name: 'x' } },
      ],
      edges: [
        { from: { node: 'x', port: 'out' }, to: { node: 's', port: 'x' } },
        { from: { node: 's', port: 'out' }, to: { node: 'r', port: 'd0' } },
      ],
    };
    const next = withWiredReshapeDims(graph, graph.edges[1]);
    expect(next.nodes.find((n) => n.id === 'r')?.attrs?.shape).toEqual([1, 3]);
  });
});
