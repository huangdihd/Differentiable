import { describe, expect, it } from 'vitest';
import { buildBackwardGraph } from '../engine/backward-graph';
import { formulaGraph } from '../levels/level-helpers';
import { gradientLayout } from './gradient-layout';

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
  it('marks fan-out values whose gradients are summed', () => {
    const { layout } = layoutOf('x * x', ['x'], ['x']);
    expect(layout.sums).toEqual({ x: 2 });
  });
});
