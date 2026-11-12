// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_INSPECTOR_POS,
  INSPECTOR_MARGIN as M,
  INSPECTOR_MIN_VISIBLE,
  clampInspectorPos,
  draggedInspectorPos,
  inspectorMaxHeight,
  parseInspectorPos,
} from './inspector-position';

const panel = { w: 320, h: 400 };
const area = { w: 1000, h: 600 };

describe('clampInspectorPos', () => {
  it('leaves a position inside the area alone', () => {
    expect(clampInspectorPos({ right: 200, top: 100 }, panel, area)).toEqual({
      right: 200,
      top: 100,
    });
  });
  it('keeps the panel fully inside left and right', () => {
    expect(clampInspectorPos({ right: -50, top: 100 }, panel, area).right).toBe(M);
    expect(clampInspectorPos({ right: 5000, top: 100 }, panel, area).right).toBe(1000 - 320 - M);
  });
  it('keeps the whole panel on screen at the bottom', () => {
    expect(clampInspectorPos({ right: 20, top: -30 }, panel, area).top).toBe(M);
    expect(clampInspectorPos({ right: 20, top: 9999 }, panel, area).top).toBe(600 - 400 - M);
  });
  it('re-clamps a remembered spot when the area shrinks', () => {
    expect(clampInspectorPos({ right: 600, top: 400 }, panel, { w: 700, h: 450 })).toEqual({
      right: 700 - 320 - M,
      top: 450 - 400 - M,
    });
  });
  it('sends a panel taller than the area to the top', () => {
    expect(clampInspectorPos({ right: 20, top: 300 }, { w: 320, h: 900 }, area).top).toBe(M);
  });
  it('falls back to the margin when the area is smaller than the panel', () => {
    expect(clampInspectorPos({ right: 100, top: 100 }, panel, { w: 200, h: 100 })).toEqual({
      right: M,
      top: M,
    });
  });
});

describe('drag and height', () => {
  it('moves right when the pointer moves right, down when it moves down', () => {
    expect(draggedInspectorPos({ right: 100, top: 50 }, 30, 20)).toEqual({ right: 70, top: 70 });
  });
  it('lets the panel use the height left below it', () => {
    expect(inspectorMaxHeight({ right: 0, top: 100 }, area)).toBe(600 - 100 - M);
    expect(inspectorMaxHeight({ right: 0, top: 590 }, area)).toBe(INSPECTOR_MIN_VISIBLE);
  });
});

describe('parseInspectorPos', () => {
  it('reads a stored position and rejects junk', () => {
    expect(parseInspectorPos('{"right":40,"top":80}')).toEqual({ right: 40, top: 80 });
    expect(parseInspectorPos(null)).toBeUndefined();
    expect(parseInspectorPos('nope')).toBeUndefined();
    expect(parseInspectorPos('null')).toBeUndefined();
    expect(parseInspectorPos('{"right":"1","top":2}')).toBeUndefined();
    expect(DEFAULT_INSPECTOR_POS).toEqual({ right: M, top: M });
  });
});
