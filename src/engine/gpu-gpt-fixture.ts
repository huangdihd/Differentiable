// Browser benchmark fixture mirrors gpt.test.ts; intentionally no Vitest import.
import { createDefaultRegistry, type GraphJSON, type Attributes } from './graph';
/** Small JSON builder; all numerical operations remain registered graph components. */
class Builder {
  readonly graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  node(
    id: string,
    component: string,
    inputs: Record<string, string> = {},
    attrs?: Attributes,
  ): string {
    this.graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, from] of Object.entries(inputs))
      this.graph.edges.push({ from: { node: from, port: 'out' }, to: { node: id, port } });
    return id;
  }
  input(name: string): string {
    return this.node(name, 'input', {}, { name });
  }
  output(x: string, name = 'out'): void {
    this.node(`output_${name}`, 'output', { x }, { name });
  }
  param(id: string, shape: number[], init = 'normal'): string {
    return this.node(id, 'param', {}, { shape, init, scale: 0.08 });
  }
  constant(id: string, value: number | number[], shape: number[] = []): string {
    return this.node(id, 'const', {}, { value, shape });
  }
}

export interface GptConfig {
  B: number;
  C: number;
  D: number;
  H: number;
  V: number;
  layers: number;
}
export function gptFixture({ B, C, D, H, V, layers }: GptConfig) {
  const registry = createDefaultRegistry();
  // Attribute-driven linear components give each instance independent scoped storage.
  registry.register({
    name: 'linear',
    inputs: ['x'],
    outputs: ['out'],
    subgraph: (attrs) => {
      const g = new Builder(),
        x = g.input('x');
      const w = g.param('weight', [attrs.input as number, attrs.output as number]);
      const bias = g.param('bias', [attrs.output as number], 'zeros');
      g.output(
        g.node('affine', 'add', { a: g.node('product', 'matmul', { a: x, b: w }), b: bias }),
      );
      return g.graph;
    },
  });
  const norm = new Builder(),
    nx = norm.input('x');
  const gain = norm.param('gain', [D], 'ones'),
    bias = norm.param('bias', [D], 'zeros');
  norm.output(
    norm.node('affine', 'add', {
      a: norm.node('scaled', 'mul', { a: norm.node('normalize', 'layernorm', { x: nx }), b: gain }),
      b: bias,
    }),
  );
  registry.register({ name: 'affine_norm', inputs: ['x'], outputs: ['out'], subgraph: norm.graph });
  const attention = new Builder(),
    ax = attention.input('x');
  const project = (id: string) =>
    attention.node(
      `${id}_heads`,
      'transpose',
      {
        x: attention.node(
          `${id}_reshape`,
          'reshape',
          { x: attention.node(id, 'linear', { x: ax }, { input: D, output: D }) },
          { shape: [B, C, H, D / H] },
        ),
      },
      { perm: [0, 2, 1, 3] },
    );
  const q = project('query'),
    k = project('key'),
    v = project('value');
  const scores = attention.node('scores', 'matmul', {
    a: q,
    b: attention.node('key_transpose', 'transpose', { x: k }, { perm: [0, 1, 3, 2] }),
  });
  const scaled = attention.node('scaled', 'mul', {
    a: scores,
    b: attention.constant('scale', 1 / Math.sqrt(D / H)),
  });
  const mask = attention.constant(
    'mask',
    Array.from({ length: C * C }, (_, i) => (i % C > Math.floor(i / C) ? -1e4 : 0)),
    [C, C],
  );
  const weights = attention.node('weights', 'softmax', {
    x: attention.node('masked', 'add', { a: scaled, b: mask }),
  });
  const attended = attention.node('attended', 'matmul', { a: weights, b: v });
  const joined = attention.node(
    'joined',
    'reshape',
    { x: attention.node('unhead', 'transpose', { x: attended }, { perm: [0, 2, 1, 3] }) },
    { shape: [B, C, D] },
  );
  attention.output(attention.node('projection', 'linear', { x: joined }, { input: D, output: D }));
  registry.register({
    name: 'attention',
    inputs: ['x'],
    outputs: ['out'],
    subgraph: attention.graph,
  });
  const mlp = new Builder(),
    mx = mlp.input('x');
  const up = mlp.node('up', 'linear', { x: mx }, { input: D, output: 2 * D });
  const relu = mlp.node('relu', 'max', { a: up, b: mlp.constant('zero', 0) });
  mlp.output(mlp.node('down', 'linear', { x: relu }, { input: 2 * D, output: D }));
  registry.register({ name: 'mlp', inputs: ['x'], outputs: ['out'], subgraph: mlp.graph });
  const block = new Builder(),
    bx = block.input('x');
  const residual = block.node('residual', 'add', {
    a: bx,
    b: block.node('attention', 'attention', { x: block.node('norm1', 'affine_norm', { x: bx }) }),
  });
  block.output(
    block.node('result', 'add', {
      a: residual,
      b: block.node('mlp', 'mlp', { x: block.node('norm2', 'affine_norm', { x: residual }) }),
    }),
  );
  registry.register({ name: 'block', inputs: ['x'], outputs: ['out'], subgraph: block.graph });
  const g = new Builder(),
    tokens = g.input('tokens'),
    targets = g.input('targets');
  let x = g.node('embedding', 'add', {
    a: g.node('lookup', 'gather', { table: g.param('token', [V, D]), idx: tokens }),
    b: g.param('position', [C, D]),
  });
  for (let i = 0; i < layers; i++) x = g.node(`block${i}`, 'block', { x });
  const logits = g.node(
    'head',
    'linear',
    { x: g.node('final_norm', 'affine_norm', { x }) },
    { input: D, output: V },
  );
  const probability = g.node('probability', 'softmax', { x: logits });
  const safe = g.node('safe', 'add', { a: probability, b: g.constant('epsilon', 1e-7) });
  const logp = g.node('logp', 'log', { x: safe });
  const total = g.node(
    'total',
    'sum',
    { x: g.node('selected', 'mul', { a: targets, b: logp }) },
    { axes: [0, 1, 2] },
  );
  const loss = g.node('loss', 'mul', { a: total, b: g.constant('mean', -1 / (B * C)) });
  g.output(logits, 'logits');
  g.output(loss, 'loss');
  // Round-trip the top-level model; nested components expand into scoped GraphJSON nodes.
  const graph = JSON.parse(JSON.stringify(g.graph)) as GraphJSON;
  const shapes = { tokens: [B, C], targets: [B, C, V] };
  return { graph, registry, shapes };
}
