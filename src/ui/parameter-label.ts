// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { GraphJSON, GraphNode } from '../engine/graph';

/** Display metadata only; node IDs remain the parameter identity. */
export function parameterName(node: GraphNode): string | undefined {
  return node.component === 'param' && typeof node.attrs?.label === 'string'
    ? node.attrs.label.trim() || undefined
    : undefined;
}

export function gradientParameterLabel(node: GraphNode, graph: GraphJSON): string {
  const name = parameterName(node);
  if (name) return name;
  const tag = (n: GraphNode) => ((n.attrs?.shape as number[] | undefined) ?? []).join('×');
  const twins = graph.nodes.filter((n) => n.component === 'param' && tag(n) === tag(node));
  const i = twins.findIndex((n) => n.id === node.id);
  const sub = twins.length > 1 ? ('₁₂₃₄₅₆₇₈₉'[i] ?? String(i + 1)) : '';
  return tag(node) ? `θ${sub} ${tag(node)}` : `θ${sub}`;
}
