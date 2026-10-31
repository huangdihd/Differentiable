// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import type { ComponentRegistry, GraphJSON } from '../engine/graph';
import {
  NODE_W,
  freeSpot,
  layoutGraph,
  movePositions,
  nearestInput,
  sidePortY,
} from './graph-geometry';

describe('freeSpot', () => {
  it('keeps the wanted spot when it is empty', () => {
    expect(freeSpot({ x: 100, y: 100 }, NODE_W, 90, [])).toEqual({ x: 100, y: 100 });
  });

  it('moves a second node off the first instead of stacking them', () => {
    const first = { x: 100, y: 100, w: NODE_W, h: 90 };
    const spot = freeSpot({ x: 100, y: 100 }, NODE_W, 90, [first]);
    const overlaps =
      spot.x < first.x + first.w &&
      first.x < spot.x + NODE_W &&
      spot.y < first.y + first.h &&
      first.y < spot.y + 90;
    expect(overlaps).toBe(false);
  });

  it('places repeated clicks at distinct, non-overlapping spots', () => {
    const rects: { x: number; y: number; w: number; h: number }[] = [];
    for (let i = 0; i < 6; i++) {
      const p = freeSpot({ x: 300, y: 200 }, NODE_W, 90, rects);
      for (const r of rects)
        expect(p.x >= r.x + r.w || r.x >= p.x + NODE_W || p.y >= r.y + r.h || r.y >= p.y + 90).toBe(
          true,
        );
      rects.push({ ...p, w: NODE_W, h: 90 });
    }
  });
});

describe('nearestInput', () => {
  const registry = new Map([
    ['mul', { inputs: ['a', 'b'], outputs: ['out'] }],
  ]) as unknown as ComponentRegistry;
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'm', component: 'mul' },
    ],
    edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'm', port: 'a' } }],
  };
  const positions = new Map([
    ['x', { x: 0, y: 0 }],
    ['m', { x: 300, y: 0 }],
  ]);
  const portB = sidePortY(graph.nodes[1], registry, 1, 2);

  it('snaps to an input port within the radius', () => {
    const hit = nearestInput(graph, registry, positions, { x: 290, y: portB + 5 }, 30);
    expect(hit).toMatchObject({ node: 'm', port: 'b' });
  });

  it('ignores ports outside the radius and on the dragged node', () => {
    expect(nearestInput(graph, registry, positions, { x: 200, y: portB }, 30)).toBeUndefined();
    expect(nearestInput(graph, registry, positions, { x: 300, y: portB }, 30, 'm')).toBeUndefined();
  });
});

describe('movePositions', () => {
  // Reference solutions ship without `ui`, so every node is auto-laid-out.
  const registry = new Map([
    ['mul', { inputs: ['a', 'b'], outputs: ['out'] }],
  ]) as unknown as ComponentRegistry;
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'y', component: 'input', attrs: { name: 'y' } },
      { id: 'm', component: 'mul' },
      { id: 'o', component: 'output' },
    ],
    edges: [
      { from: { node: 'x', port: 'out' }, to: { node: 'm', port: 'a' } },
      { from: { node: 'y', port: 'out' }, to: { node: 'm', port: 'b' } },
      { from: { node: 'm', port: 'out' }, to: { node: 'o', port: 'x' } },
    ],
  };

  it('dragging one auto-laid-out node leaves the others where they were', () => {
    const before = layoutGraph(graph, registry);
    const moved = { m: { x: 900, y: 900 } };
    const positions = movePositions(graph, before, moved);
    const next: GraphJSON = {
      ...graph,
      nodes: graph.nodes.map((n) => (positions[n.id] ? { ...n, ui: { ...positions[n.id] } } : n)),
    };
    const after = layoutGraph(next, registry);
    expect(after.get('m')).toEqual({ x: 900, y: 900 });
    for (const id of ['x', 'y', 'o']) expect(after.get(id)).toEqual(before.get(id));
  });

  it('does not override nodes the user already placed', () => {
    const placed: GraphJSON = {
      ...graph,
      nodes: graph.nodes.map((n) => ({ ...n, ui: { x: 1, y: 2 } })),
    };
    expect(movePositions(placed, layoutGraph(placed, registry), { m: { x: 5, y: 5 } })).toEqual({
      m: { x: 5, y: 5 },
    });
  });
});
