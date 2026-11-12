// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  isClick,
  marqueeSelection,
  nodesInRect,
  pressNode,
  rectFromPoints,
  type NodeRect,
} from './selection';

const nodes: NodeRect[] = [
  { id: 'a', x: 0, y: 0, w: 150, h: 90 },
  { id: 'b', x: 200, y: 0, w: 150, h: 90 },
  { id: 'c', x: 0, y: 200, w: 150, h: 90 },
];

describe('rectFromPoints', () => {
  it('normalises any drag direction', () => {
    expect(rectFromPoints({ x: 10, y: 50 }, { x: 0, y: 20 })).toEqual({
      x: 0,
      y: 20,
      w: 10,
      h: 30,
    });
  });
});

describe('nodesInRect', () => {
  it('picks nodes the box touches', () => {
    expect(nodesInRect(nodes, { x: 100, y: 50, w: 120, h: 10 })).toEqual(['a', 'b']);
  });
  it('picks nothing in empty space', () => {
    expect(nodesInRect(nodes, { x: 160, y: 100, w: 30, h: 90 })).toEqual([]);
  });
  it('counts a grazed edge', () => {
    expect(nodesInRect(nodes, { x: 150, y: 290, w: 10, h: 10 })).toEqual(['c']);
  });
});

describe('marqueeSelection', () => {
  it('replaces by default', () => {
    expect(marqueeSelection(['a'], ['b', 'c'], false)).toEqual(['b', 'c']);
  });
  it('adds with a modifier, without duplicates', () => {
    expect(marqueeSelection(['a', 'b'], ['b', 'c'], true)).toEqual(['a', 'b', 'c']);
  });
  it('clears when a plain marquee hits nothing', () => {
    expect(marqueeSelection(['a'], [], false)).toEqual([]);
  });
});

describe('pressNode', () => {
  it('selects only an unselected node and drags it alone', () => {
    expect(pressNode(['a', 'b'], 'c', false)).toEqual({ select: ['c'], drag: ['c'] });
  });
  it('keeps a group when pressing one of its nodes, narrowing only on click', () => {
    expect(pressNode(['a', 'b'], 'b', false)).toEqual({ drag: ['a', 'b'], onClick: ['b'] });
  });
  it('does nothing extra when the only selected node is pressed', () => {
    expect(pressNode(['a'], 'a', false)).toEqual({ drag: ['a'], onClick: undefined });
  });
  it('toggles in and out with a modifier and never drags', () => {
    expect(pressNode(['a'], 'b', true)).toEqual({ select: ['a', 'b'], drag: [] });
    expect(pressNode(['a', 'b'], 'a', true)).toEqual({ select: ['b'], drag: [] });
  });
});

describe('isClick', () => {
  it('treats tiny jitter as a click', () => {
    expect(isClick(1, 2)).toBe(true);
    expect(isClick(3, 0)).toBe(false);
  });
});
