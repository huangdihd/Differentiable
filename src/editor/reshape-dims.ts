// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { Attributes, GraphEndpoint, GraphJSON, GraphNode } from '../engine/graph';

/** Reshape exposes d0–d3; rank is attrs.shape.length. */
export const RESHAPE_DIM_PORTS = ['d0', 'd1', 'd2', 'd3'] as const;

export function reshapeShapeOf(attrs: Attributes | undefined): number[] {
  const shape = attrs?.shape;
  if (Array.isArray(shape) && shape.length >= 1 && shape.every((n) => Number.isSafeInteger(n)))
    return shape as number[];
  return [-1];
}

/** Prefer a static integer from the wire source; otherwise keep a positive concrete dim (never -1). */
export function concreteReshapeDim(
  stored: number | undefined,
  resolved?: number,
): number {
  if (
    resolved !== undefined &&
    Number.isSafeInteger(resolved) &&
    resolved >= 0
  ) {
    return resolved;
  }
  const dim = stored ?? 1;
  return dim === -1 ? 1 : dim;
}

/** Read a compile-time scalar from a const node when the wire source is that simple. */
export function staticWireDimValue(
  graph: GraphJSON,
  from: GraphEndpoint,
): number | undefined {
  const source = graph.nodes.find((node) => node.id === from.node);
  if (!source || source.component !== 'const') return undefined;
  const value = source.attrs?.value;
  if (typeof value === 'number' && Number.isSafeInteger(value) && value >= 0) return value;
  return undefined;
}

/**
 * When a reshape dimension port is wired, persist a concrete size into attrs.shape so another
 * axis can use -1, the panel can lock a number, and disconnect keeps that number (never -1).
 */
export function withWiredReshapeDims(
  graph: GraphJSON,
  edge: { from: GraphEndpoint; to: GraphEndpoint },
  resolvedByPort: Readonly<Record<string, number>> = {},
): GraphJSON {
  const target = graph.nodes.find((node) => node.id === edge.to.node);
  if (!target || target.component !== 'reshape') return graph;
  const match = /^d(\d+)$/.exec(edge.to.port);
  if (!match) return graph;
  const index = Number(match[1]);
  const shape = reshapeShapeOf(target.attrs);
  if (index < 0 || index >= shape.length) return graph;
  const resolved =
    resolvedByPort[edge.to.port] ?? staticWireDimValue(graph, edge.from);
  const nextDim = concreteReshapeDim(shape[index], resolved);
  if (shape[index] === nextDim) return graph;
  const nextShape = shape.map((dim, i) => (i === index ? nextDim : dim));
  return {
    ...graph,
    nodes: graph.nodes.map((node) =>
      node.id === target.id
        ? { ...node, attrs: { ...node.attrs, shape: nextShape } }
        : node,
    ),
  };
}

/** Replace any wired -1 (or apply known resolutions) without inventing new rank. */
export function reshapeShapeForWiredPorts(
  shape: readonly number[],
  wiredPorts: readonly string[],
  resolvedByPort: Readonly<Record<string, number>> = {},
): number[] {
  const wired = new Set(wiredPorts);
  return shape.map((dim, index) => {
    const port = `d${index}`;
    if (!wired.has(port)) return dim;
    return concreteReshapeDim(dim, resolvedByPort[port]);
  });
}

export function reshapeNodeWithShape(node: GraphNode, shape: number[]): GraphNode {
  return { ...node, attrs: { ...node.attrs, shape } };
}
