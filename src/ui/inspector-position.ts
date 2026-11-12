// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Where the sandbox's floating inspector sits: an offset from the canvas area's top-right corner,
 * so a remembered spot keeps hugging the same edge when the window or the chart strip resizes.
 */

export interface InspectorPos {
  /** Gap between the panel's right edge and the canvas area's right edge, in px. */
  right: number;
  /** Gap between the panel's top edge and the canvas area's top edge, in px. */
  top: number;
}

export interface Size {
  w: number;
  h: number;
}

/** Breathing room kept between the panel and the canvas area's edges. */
export const INSPECTOR_MARGIN = 12;
/** The panel never gets shorter than this (header and a little body); it scrolls inside. */
export const INSPECTOR_MIN_VISIBLE = 120;
export const DEFAULT_INSPECTOR_POS: InspectorPos = {
  right: INSPECTOR_MARGIN,
  top: INSPECTOR_MARGIN,
};
export const INSPECTOR_POS_KEY = 'differentiable.sandbox.inspectorPos';

const clampTo = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), Math.max(lo, hi));

/**
 * Keep the panel wholly inside the area. `panel.h` is its natural (unscrolled) height; one taller
 * than the area goes to the top and scrolls inside.
 */
export function clampInspectorPos(pos: InspectorPos, panel: Size, area: Size): InspectorPos {
  const m = INSPECTOR_MARGIN;
  const h = Math.min(panel.h, area.h - 2 * m);
  return {
    right: clampTo(pos.right, m, area.w - panel.w - m),
    top: clampTo(pos.top, m, area.h - h - m),
  };
}

/** The tallest the panel may be from where it sits; it scrolls inside beyond that. */
export const inspectorMaxHeight = (pos: InspectorPos, area: Size) =>
  Math.max(INSPECTOR_MIN_VISIBLE, area.h - pos.top - INSPECTOR_MARGIN);

/** The position after a drag of (dx, dy) from where the drag started. */
export const draggedInspectorPos = (start: InspectorPos, dx: number, dy: number): InspectorPos => ({
  right: start.right - dx,
  top: start.top + dy,
});

export function parseInspectorPos(text: string | null): InspectorPos | undefined {
  if (!text) return undefined;
  try {
    const value = JSON.parse(text) as Partial<InspectorPos> | null;
    if (
      typeof value?.right === 'number' &&
      typeof value.top === 'number' &&
      Number.isFinite(value.right) &&
      Number.isFinite(value.top)
    )
      return { right: value.right, top: value.top };
  } catch {
    // A broken entry just means the default spot.
  }
  return undefined;
}

export function readInspectorPos(): InspectorPos {
  try {
    return parseInspectorPos(localStorage.getItem(INSPECTOR_POS_KEY)) ?? DEFAULT_INSPECTOR_POS;
  } catch {
    return DEFAULT_INSPECTOR_POS;
  }
}

export function writeInspectorPos(pos: InspectorPos | undefined) {
  try {
    if (pos) localStorage.setItem(INSPECTOR_POS_KEY, JSON.stringify(pos));
    else localStorage.removeItem(INSPECTOR_POS_KEY);
  } catch {
    // Only a view preference.
  }
}
