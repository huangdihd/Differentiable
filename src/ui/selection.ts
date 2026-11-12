// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.

/** Pure canvas selection rules, kept out of Canvas.tsx so they can be unit-tested. */

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface NodeRect extends Rect {
  id: string;
}

/** A press that moved less than this many screen pixels is a click, not a drag. */
export const CLICK_SLOP_PX = 3;

export function isClick(dx: number, dy: number, slop = CLICK_SLOP_PX): boolean {
  return Math.hypot(dx, dy) < slop;
}

/** Normalised rectangle spanned by two corners (any drag direction). */
export function rectFromPoints(a: { x: number; y: number }, b: { x: number; y: number }): Rect {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    w: Math.abs(a.x - b.x),
    h: Math.abs(a.y - b.y),
  };
}

/** Nodes touched by the marquee. Touching is enough: grazing a node's edge picks it up. */
export function nodesInRect(nodes: readonly NodeRect[], box: Rect): string[] {
  return nodes
    .filter(
      (n) =>
        n.x <= box.x + box.w && n.x + n.w >= box.x && n.y <= box.y + box.h && n.y + n.h >= box.y,
    )
    .map((n) => n.id);
}

/**
 * The selection a marquee produces. A plain marquee replaces the selection; with Shift or
 * Ctrl/Cmd held it adds to what was already selected.
 */
export function marqueeSelection(
  base: readonly string[],
  hits: readonly string[],
  additive: boolean,
): string[] {
  return additive ? [...new Set([...base, ...hits])] : [...new Set(hits)];
}

/** What pressing a node does, before we know whether the press becomes a drag. */
export interface NodePress {
  /** Selection to apply right away, or undefined to leave it as is. */
  select?: string[];
  /** Nodes that move if the press turns into a drag; empty when the press cannot drag. */
  drag: string[];
  /** Selection to apply if the press is released without moving (click on a selected node). */
  onClick?: string[];
}

/**
 * - Shift or Ctrl/Cmd toggles the node in or out of the selection and never drags.
 * - Pressing an unselected node selects only it (and drags it alone).
 * - Pressing a selected node keeps the group so dragging moves all of it; releasing without
 *   moving narrows the selection to that node.
 */
export function pressNode(selected: readonly string[], id: string, toggle: boolean): NodePress {
  if (toggle) {
    const select = selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id];
    return { select, drag: [] };
  }
  if (!selected.includes(id)) return { select: [id], drag: [id] };
  return {
    drag: [...selected],
    onClick: selected.length > 1 ? [id] : undefined,
  };
}
