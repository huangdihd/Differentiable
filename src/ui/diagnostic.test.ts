import { describe, expect, it } from 'vitest';
import { ComponentRegistry, type GraphJSON } from '../engine/graph';
import { translate, translateDiagnostic } from '../i18n';
import { diagnosticNode, graphDiagnostic } from './diagnostic';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'outer/name', component: 'log' },
    { id: 'weight', component: 'param', attrs: { label: 'W_1' } },
  ],
  edges: [],
};
describe('player diagnostic context', () => {
  it('renders axis choices, learning-rate interval and malformed values in both languages', () => {
    for (const locale of ['en', 'zh-CN'] as const) {
      const axes = translateDiagnostic(locale, {
        key: 'invalid_axes',
        params: { actual: 3, rank: 2, allowed: [-2, -1, 0, 1] },
      });
      expect(axes).toContain('3');
      expect(axes).toContain('[-2, -1, 0, 1]');
      const rate = translateDiagnostic(locale, {
        key: 'training.learning_rate',
        params: { actual: '-0.5' },
      });
      expect(rate).toContain('-0.5');
      expect(rate).toContain(locale === 'en' ? 'greater than 0' : '大于 0');
      for (const shape of [undefined, null, ['bad', 2]]) {
        expect(
          translateDiagnostic(locale, { key: 'invalid_shape', params: { shape } }),
        ).not.toContain('{shape}');
      }
    }
  });
  const nonfinite = (node?: string) => ({
    key: 'training.nonfinite',
    node,
    params: { actual: 'NaN', index: 2, stageKey: 'training.stage.forward' },
  });
  it.each(['log', 'exp', 'div', 'recip', 'sqrt', 'param', 'add'])(
    'adds advice for the actual %s operation without losing error details',
    (component) => {
      const local: GraphJSON = {
        version: 1,
        nodes: [{ id: 'operation', component, attrs: { label: 'W_1' } }],
        edges: [],
      };
      const error = nonfinite('operation');
      const result = graphDiagnostic('en', local, error);
      expect(result).toContain(translateDiagnostic('en', error));
      expect(result).toContain(
        translate('en', `training.nonfinite.hint.${component === 'add' ? 'default' : component}`),
      );
      if (component === 'param') expect(result.startsWith('W_1：')).toBe(true);
    },
  );
  it('resolves nested escaped paths through registry subgraphs', () => {
    const registry = new ComponentRegistry()
      .register({
        name: 'wrapper',
        inputs: [],
        outputs: [],
        subgraph: {
          version: 1,
          nodes: [{ id: 'nested', component: 'generated' }],
          edges: [],
        },
      })
      .register({
        name: 'generated',
        inputs: [],
        outputs: [],
        subgraph: () => ({
          version: 1,
          nodes: [{ id: 'inner/log', component: 'log' }],
          edges: [],
        }),
      });
    const local: GraphJSON = {
      version: 1,
      nodes: [{ id: 'outer/name', component: 'wrapper' }],
      edges: [],
    };
    const error = nonfinite('outer%2Fname/nested/inner%2Flog');
    expect(diagnosticNode(local, error)).toBe('outer/name');
    expect(graphDiagnostic('en', local, error, registry)).toContain(
      translate('en', 'training.nonfinite.hint.log'),
    );
  });
  it.each([undefined, 'missing', '%invalid', 'outer%2Fname/missing'])(
    'uses generic advice for unresolved path %s instead of guessing from stage or outer node',
    (path) => {
      expect(graphDiagnostic('en', graph, nonfinite(path))).toContain(
        translate('en', 'training.nonfinite.hint.default'),
      );
    },
  );
  it('falls back when a generated subgraph needs unavailable input shapes', () => {
    const registry = new ComponentRegistry().register({
      name: 'log',
      inputs: [],
      outputs: [],
      subgraph: () => {
        throw new Error('shape required');
      },
    });
    expect(graphDiagnostic('en', graph, nonfinite('outer%2Fname/inner'), registry)).toContain(
      translate('en', 'training.nonfinite.hint.default'),
    );
  });
  it('maps expanded paths onto the existing highlighted canvas node', () => {
    const error = { key: 'dim_not_static', node: 'outer%2Fname/inner', port: 'n' };
    expect(diagnosticNode(graph, error)).toBe('outer/name');
    expect(graphDiagnostic('zh-CN', graph, error)).toBe(
      '对数：输入口 n 的长度必须由常量或输入的形状算出来，不能取决于会变的数据。请接「常量」，或从「维度大小」计算长度。',
    );
    expect(graphDiagnostic('en', graph, error)).toContain('Logarithm');
  });
  it('uses a parameter display name, not its internal identifier', () => {
    expect(
      graphDiagnostic('zh-CN', graph, { key: 'missing_port', node: 'weight', port: 'x' }),
    ).toBe('W_1：输入口 x 还没有接线。把所需数值的输出口接到这里。');
  });
});
