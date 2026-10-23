import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { buildBackwardGraph } from '../engine/backward-graph';
import type { GraphJSON } from '../engine/graph';
import { createTranslator } from '../i18n';
import GradientView from './GradientView';

const graph: GraphJSON = {
  version: 1,
  nodes: [
    { id: 'z', component: 'input', attrs: { name: 'z' } },
    { id: 'f', component: 'exp' },
    { id: 'output', component: 'output', attrs: { name: 'probabilities' } },
  ],
  edges: [
    { from: { node: 'z', port: 'out' }, to: { node: 'f', port: 'x' } },
    { from: { node: 'f', port: 'out' }, to: { node: 'output', port: 'x' } },
  ],
};
const backward = buildBackwardGraph(graph, {
  inputShapes: { z: [] },
  output: 'probabilities',
  wrt: ['z'],
});

describe('gradient view', () => {
  it('localizes semantic output names and marks the player handoff as three separate rows', () => {
    const html = renderToStaticMarkup(
      <GradientView
        t={createTranslator('zh-CN')}
        graph={graph}
        backward={backward}
        wrt={['z']}
        output="probabilities"
        playerBackwardStart="z"
        composite={() => false}
        format={String}
        onClose={() => {}}
      />,
    );

    expect(html).toContain('反向：∂概率/∂·');
    expect(html).toContain('g = ∂概率/∂z');
    expect(html).toContain('你电路的输入 / 起点');
    expect(html).toContain('gv-card is-backward is-target is-handoff is-player-step');
    expect(html).toContain('gv-card-label');
    expect(html).toContain('gv-card-value');
    expect(html).toContain('gv-card-note');
  });
});
