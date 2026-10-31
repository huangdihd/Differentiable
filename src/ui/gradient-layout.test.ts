// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { buildBackwardGraph } from '../engine/backward-graph';
import { createTranslator } from '../i18n';
import { formulaGraph } from '../levels/level-helpers';
import {
  backwardTeachingSegments,
  gradientEdgeKey,
  gradientLayout,
  namedMultiplicationRule,
} from './gradient-layout';

const layoutOf = (formula: string, inputs: string[], wrt: string[]) => {
  const graph = formulaGraph(formula, inputs);
  const backward = buildBackwardGraph(graph, {
    inputShapes: Object.fromEntries(inputs.map((n) => [n, []])),
    output: 'y',
    wrt,
  });
  return { graph, layout: gradientLayout(graph, backward, wrt) };
};

describe('gradient graph layout', () => {
  it('mirrors the forward graph column by column', () => {
    const { layout } = layoutOf('exp(x * w)', ['x', 'w'], ['w']);
    const col = (c: string) => layout.nodes.find((n) => n.component === c)!.column;
    expect(col('input')).toBe(0);
    expect(col('mul')).toBeLessThan(col('exp'));
    expect(col('exp')).toBeLessThan(col('output'));
  });
  it('only nodes between a differentiated input and the output carry gradient', () => {
    const { layout } = layoutOf('exp(x * w)', ['x', 'w'], ['w']);
    const carries = (id: string) => layout.nodes.find((n) => n.id === id)!.carries;
    expect(carries('w')).toBe(true);
    expect(carries('x')).toBe(false);
    expect(layout.edges.find((e) => e.from === 'x')!.backward).toBe(false);
  });
  it('labels each backward wire with the rule of the node it leaves', () => {
    const { layout } = layoutOf('exp(x * w)', ['x', 'w'], ['w']);
    const intoExp = layout.edges.find((e) => e.backward && e.port === 'x')!;
    expect(intoExp.rule).toBe('backward.rule.exp');
    expect(layout.edges.find((e) => e.from === 'w')!.rule).toBe('backward.rule.mul');
  });
  it('differentiates parameters by ID without differentiating data or unused parameters', () => {
    const graph = formulaGraph('x * w', ['x', 'w']);
    const parameter = graph.nodes.find((n) => n.id === 'w')!;
    parameter.component = 'param';
    parameter.attrs = { name: 'display-name', shape: [], init: 'ones' };
    graph.nodes.push({ id: 'unused', component: 'param', attrs: { shape: [], init: 'ones' } });
    const backward = buildBackwardGraph(graph, {
      inputShapes: { x: [] },
      output: 'y',
      wrt: ['w', 'unused'],
    });
    const layout = gradientLayout(graph, backward, ['w', 'unused']);
    expect(layout.nodes.find((n) => n.id === 'w')!.carries).toBe(true);
    expect(layout.nodes.find((n) => n.id === 'x')!.carries).toBe(false);
    expect(layout.nodes.find((n) => n.id === 'unused')!.carries).toBe(false);
    expect(layout.edges.find((e) => e.from === 'w')!.backward).toBe(true);
    expect(gradientLayout(graph, backward, ['display-name']).nodes.some((n) => n.carries)).toBe(
      false,
    );
  });
  it('splits an engine reveal at the upstream gradient supplied to the player', () => {
    const { graph, layout } = layoutOf('exp(w * x + b)', ['w', 'x', 'b'], ['w', 'x', 'b']);
    const start = graph.nodes.find((node) => node.component === 'add')!.id;
    const outer = graph.nodes.find((node) => node.component === 'exp')!.id;
    const product = graph.nodes.find((node) => node.component === 'mul')!.id;
    const output = graph.nodes.find((node) => node.component === 'output')!.id;
    const segments = backwardTeachingSegments(layout, start)!;

    expect([...segments.playerNodes].sort()).toEqual(['b', product, start, 'w', 'x'].sort());
    expect([...segments.engineNodes].sort()).toEqual([start, outer, output].sort());
    expect(segments.playerEdges).toContain(gradientEdgeKey(product, start, 'a'));
    expect(segments.playerEdges).toContain(gradientEdgeKey('b', start, 'b'));
    expect(segments.engineEdges).toContain(gradientEdgeKey(start, outer, 'x'));
    expect(segments.playerEdges).not.toContain(gradientEdgeKey(start, outer, 'x'));
    expect(backwardTeachingSegments(layout, 'missing')).toBeUndefined();
  });
  it('names the actual multiplication operands in the hand-built backward rules', () => {
    const { graph, layout } = layoutOf('exp(w * x + b)', ['w', 'x', 'b'], ['w', 'x', 'b']);
    const product = graph.nodes.find((node) => node.component === 'mul')!.id;
    const edges = layout.edges.filter((edge) => edge.to === product);
    const t = createTranslator('en');
    const labels = edges.map((edge) => {
      const rule = namedMultiplicationRule(graph, edge, (node) => node);
      return rule ? t('gradView.ruleMulNamed', rule) : '';
    });

    expect(labels).toEqual(['∂(w·x)/∂w = x', '∂(w·x)/∂x = w']);
    expect(labels.join(' ')).not.toMatch(/∂[ab]|= [ab]$/);
  });
  it('marks fan-out values whose gradients are summed', () => {
    const { layout } = layoutOf('x * x', ['x'], ['x']);
    expect(layout.sums).toEqual({ x: 2 });
  });
});
