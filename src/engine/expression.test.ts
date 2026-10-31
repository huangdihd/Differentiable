// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { componentOperators, ExpressionError, parseExpression } from './expression';
import type { ExpressionGraph, ExpressionOptions } from './expression';

function error(source: string, options?: ExpressionOptions): ExpressionError {
  try {
    parseExpression(source, options);
  } catch (caught) {
    expect(caught).toBeInstanceOf(ExpressionError);
    return caught as ExpressionError;
  }
  throw new Error('Expected parsing to fail');
}
function input(graph: ExpressionGraph, node: string, port: string) {
  return graph.edges.find((edge) => edge.to.node === node && edge.to.port === port)!.from;
}

describe('parseExpression', () => {
  it('lowers 1 / x to a single recip instead of mul(1, recip(x))', () => {
    const graph = parseExpression('1 / x', { variables: { x: { node: 'in', port: 'out' } } });
    expect(graph.nodes.map((node) => node.component)).toEqual(['recip']);
    expect(input(graph, graph.output.node, 'x')).toEqual({ node: 'in', port: 'out' });
    expect(
      parseExpression('2 / x', { variables: { x: { node: 'in', port: 'out' } } }).nodes.map(
        (node) => node.component,
      ),
    ).toEqual(['const', 'recip', 'mul']);
  });

  it('uses the player-built sub/div components for a - b and a / b once available', () => {
    const variables = { a: { node: 'a', port: 'out' }, b: { node: 'b', port: 'out' } };
    const available = ['const', 'add', 'mul', 'recip', 'neg', 'sub', 'div'];
    const options = {
      variables,
      availableComponents: available,
      operators: componentOperators(available),
    };
    const sub = parseExpression('a - b', options);
    expect(sub.nodes.map((node) => node.component)).toEqual(['sub']);
    expect(input(sub, sub.output.node, 'a')).toEqual(variables.a);
    expect(input(sub, sub.output.node, 'b')).toEqual(variables.b);
    const div = parseExpression('a / b', options);
    expect(div.nodes.map((node) => node.component)).toEqual(['div']);
    expect(input(div, div.output.node, 'a')).toEqual(variables.a);
    expect(input(div, div.output.node, 'b')).toEqual(variables.b);
    // 1 / x stays a lone recip; unary minus stays neg.
    expect(parseExpression('1 / b', options).nodes.map((node) => node.component)).toEqual([
      'recip',
    ]);
    expect(parseExpression('-a - b', options).nodes.map((node) => node.component)).toEqual([
      'neg',
      'sub',
    ]);
    // Without sub/div (e.g. on the lesson that builds them) the old lowering remains.
    const basic = ['const', 'add', 'mul', 'recip', 'neg'];
    expect(
      parseExpression('a - b', {
        variables,
        availableComponents: basic,
        operators: componentOperators(basic),
      }).nodes.map((node) => node.component),
    ).toEqual(['neg', 'add']);
    expect(componentOperators(basic)).toEqual({});
  });

  it('emits scalar constants with exponent and decimal syntax', () => {
    for (const [source, value] of [
      ['1e-3', 0.001],
      ['.5E+2', 50],
      ['2.', 2],
    ] as const) {
      const graph = parseExpression(source);
      expect(graph.nodes).toEqual([
        { id: 'expr_0', component: 'const', attrs: { value, shape: [] } },
      ]);
      expect(graph.edges).toEqual([]);
      expect(graph.output).toEqual({ node: 'expr_0', port: 'out' });
    }
  });

  it('respects precedence, associativity and parentheses', () => {
    const graph = parseExpression('1 + 2 * 3 - 4');
    expect(graph.nodes.map((node) => node.component)).toEqual([
      'const',
      'const',
      'const',
      'mul',
      'add',
      'const',
      'neg',
      'add',
    ]);
    expect(input(graph, 'expr_4', 'b')).toEqual({ node: 'expr_3', port: 'out' });
    expect(input(graph, graph.output.node, 'a').node).toBe('expr_4');
    const grouped = parseExpression('(1 + 2) * 3');
    expect(grouped.nodes.map((node) => node.component)).toEqual([
      'const',
      'const',
      'add',
      'const',
      'mul',
    ]);
    const division = parseExpression('8 / 4 / 2');
    expect(input(division, division.output.node, 'a').node).toBe('expr_3');
  });

  it('lowers unary minus, subtraction and division with named ports', () => {
    const graph = parseExpression('-x / (y - x)', {
      variables: { x: { node: 'x', port: 'value' }, y: { node: 'y', port: 'result' } },
    });
    expect(graph.nodes.map((node) => node.component)).toEqual([
      'neg',
      'neg',
      'add',
      'recip',
      'mul',
    ]);
    expect(input(graph, 'expr_0', 'x')).toEqual({ node: 'x', port: 'value' });
    expect(input(graph, 'expr_2', 'a')).toEqual({ node: 'y', port: 'result' });
    expect(input(graph, 'expr_3', 'x').node).toBe('expr_2');
    expect(input(graph, 'expr_4', 'b').node).toBe('expr_3');
    expect(parseExpression('--2').nodes.map((node) => node.component)).toEqual([
      'const',
      'neg',
      'neg',
    ]);
  });

  it('references external endpoints without emitting or mutating nodes', () => {
    const endpoint = Object.freeze({ node: 'expr_0', port: 'value' });
    expect(parseExpression('x', { variables: { x: endpoint } })).toEqual({
      nodes: [],
      edges: [],
      output: endpoint,
    });
    const graph = parseExpression('x + 1', { variables: { x: endpoint } });
    expect(graph.nodes.map((node) => node.id)).toEqual(['expr_1', 'expr_2']);
  });

  it('supports custom operators, constants, function signatures and output ports', () => {
    const graph = parseExpression('f(3 - 1, z())', {
      constant: { component: 'scalar', output: 'value' },
      idPrefix: 'custom_',
      operators: {
        '-': { component: 'subtract', inputs: { left: 0, right: 1 }, output: 'difference' },
      },
      functions: {
        f: { component: 'combine', inputs: ['first', 'second'], output: 'result' },
        z: { component: 'zero', inputs: [] },
      },
      availableComponents: ['scalar', 'subtract', 'combine', 'zero'],
    });
    expect(graph.output).toEqual({ node: 'custom_4', port: 'result' });
    expect(input(graph, 'custom_2', 'left').port).toBe('value');
    expect(input(graph, 'custom_4', 'first').port).toBe('difference');
  });

  it('checks every component of composite lowerings', () => {
    for (const [source, components, requiredComponent, operator] of [
      ['x - x', ['add'], 'neg', '-'],
      ['x - x', ['neg'], 'add', '-'],
      ['x / x', ['mul'], 'recip', '/'],
      ['x / x', ['recip'], 'mul', '/'],
    ] as const) {
      const caught = error(source, {
        variables: { x: { node: 'x', port: 'out' } },
        availableComponents: components,
      });
      expect(caught.key).toBe('missing_component');
      expect(caught.position).toBe(2);
      expect(caught.params).toEqual({ requiredComponent, operator });
      expect(caught.requiredComponent).toBe(requiredComponent);
      expect(caught.operator).toBe(operator);
    }
    expect(error('1', { availableComponents: [] }).requiredComponent).toBe('const');
    expect(
      error('f()', { functions: { f: { component: 'f', inputs: [] } }, availableComponents: [] })
        .requiredComponent,
    ).toBe('f');
  });

  it('reports structured errors for names, arity and disabled operators', () => {
    expect(error('missing').key).toBe('unknown_variable');
    expect(error('toString').key).toBe('unknown_variable');
    expect(error('toString()').key).toBe('unknown_function');
    expect(
      error('f(1)', { functions: { f: { component: 'f', inputs: ['a', 'b'] } } }).params,
    ).toEqual({ name: 'f', expected: 2, actual: 1 });
    expect(error('1 + 2', { operators: { '+': null } }).key).toBe('unsupported_operator');
    expect(error('-1', { unaryMinus: null }).key).toBe('unsupported_operator');
    expect(error('-1', { unaryMinus: 1 }).key).toBe('invalid_lowering');
  });

  it.each(['', '1 +', '(1', '1)', '1 2', '2(3)', '*2', '+2', 'x.y', '1;2', 'f(1,)'])(
    'rejects invalid expression %j',
    (source) => {
      expect(() =>
        parseExpression(source, { functions: { f: { component: 'f', inputs: ['x'] } } }),
      ).toThrow(ExpressionError);
    },
  );

  it.each(['1e', '1e+', '1e999', '1.2.3'])('rejects invalid number %j', (source) => {
    expect(error(source).key).toBe('invalid_number');
  });

  it('reports source offsets including whitespace', () => {
    expect(error('  1 + )').position).toBe(6);
    expect(error('  1 +').position).toBe(5);
  });
});
