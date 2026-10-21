import { createDefaultRegistry, type GraphJSON, type Attributes } from '../engine/graph';
import type { ForwardLevel, TrainLevel, Level } from '../engine/levels';
import * as T from '../engine/tensor';

/** Small graph construction helper; all example solutions serialize as GraphJSON. */
class Builder {
  graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  node(
    id: string,
    component: string,
    attrs: Attributes = {},
    inputs: Record<string, string> = {},
  ): string {
    this.graph.nodes.push({ id, component, attrs });
    for (const [port, from] of Object.entries(inputs))
      this.graph.edges.push({ from: { node: from, port: 'out' }, to: { node: id, port } });
    return id;
  }
  output(from: string) {
    this.node('output', 'output', { name: 'y' }, { x: from });
    return this.graph;
  }
}
function unaryGraph(kind: 'neg' | 'sigmoid'): GraphJSON {
  const b = new Builder(),
    x = b.node('input', 'input', { name: 'x' });
  if (kind === 'neg') {
    const minus = b.node('minus', 'const', { value: -1 });
    return b.output(b.node('product', 'mul', {}, { a: x, b: minus }));
  }
  const neg = b.node('neg', 'neg', {}, { x });
  const exp = b.node('exp', 'exp', {}, { x: neg });
  const one = b.node('one', 'const', { value: 1 });
  const sum = b.node('sum', 'add', {}, { a: one, b: exp });
  return b.output(b.node('recip', 'recip', {}, { x: sum }));
}
export function createExampleRegistry() {
  const registry = createDefaultRegistry();
  for (const name of ['neg', 'sigmoid'] as const)
    registry.register({
      name,
      inputs: ['x'],
      outputs: ['out'],
      subgraph: (() => {
        const graph = unaryGraph(name);
        graph.nodes.find((n) => n.component === 'output')!.attrs = { name: 'out' };
        return graph;
      })(),
    });
  return registry;
}
export const sigmoidSolution = unaryGraph('sigmoid');
const derivativeBuilder = new Builder();
const dx = derivativeBuilder.node('input', 'input', { name: 'x' });
const dexp = derivativeBuilder.node('exp', 'exp', {}, { x: dx });
const done = derivativeBuilder.node('one', 'const', { value: 1 });
const dsum = derivativeBuilder.node('sum', 'add', {}, { a: dx, b: done });
export const derivativeSolution = derivativeBuilder.output(
  derivativeBuilder.node('product', 'mul', {}, { a: dexp, b: dsum }),
);

export const sigmoidLevel: ForwardLevel = {
  id: 'sigmoid',
  titleKey: 'levels.sigmoid.title',
  descriptionKey: 'levels.sigmoid.description',
  kind: 'forward',
  available: ['const', 'add', 'mul', 'exp', 'log', 'neg', 'recip'],
  interface: { inputs: { x: [5] }, output: 'y', outputShape: [5] },
  unlock: ['derivative'],
  unlockComponents: ['sigmoid'],
  fixed: [{ id: 'fixed', inputs: { x: [-4, -1, 0, 1, 4] } }],
  generated: {
    seed: 42,
    count: 3,
    generate: (random, i) => ({
      id: `seeded-${i}`,
      inputs: { x: Array.from({ length: 5 }, () => random.uniform() * 12 - 6) },
    }),
  },
  reference: ({ x }) =>
    T.exp(
      T.mul(
        T.log(T.add(T.tensor([1], [], x.backend), T.exp(T.mul(x, T.tensor([-1], [], x.backend))))),
        T.tensor([-1], [], x.backend),
      ),
    ),
};
export const derivativeLevel: ForwardLevel = {
  id: 'derivative',
  titleKey: 'levels.derivative.title',
  descriptionKey: 'levels.derivative.description',
  kind: 'derivative',
  available: ['const', 'add', 'mul', 'exp', 'log', 'neg', 'recip'],
  interface: { inputs: { x: [5] }, output: 'y', outputShape: [5] },
  unlock: ['xor'],
  fixed: [{ id: 'fixed', inputs: { x: [-2, -1, 0, 1, 2] } }],
  generated: {
    seed: 71,
    count: 3,
    generate: (random, i) => ({
      id: `seeded-${i}`,
      inputs: { x: Array.from({ length: 5 }, () => random.uniform() * 4 - 2) },
    }),
  },
  derivativeInput: 'x',
  reference: ({ x }) => T.mul(x, T.exp(x)),
};
const mlp = new Builder();
const mx = mlp.node('input', 'input', { name: 'x' });
const w1 = mlp.node('w1', 'param', { shape: [2, 8], init: 'normal', scale: 1 });
const b1 = mlp.node('b1', 'param', { shape: [8] });
const h0 = mlp.node('hiddenLinear', 'matmul', {}, { a: mx, b: w1 });
const h1 = mlp.node('hiddenBias', 'add', {}, { a: h0, b: b1 });
const h = mlp.node('hidden', 'sigmoid', {}, { x: h1 });
const w2 = mlp.node('w2', 'param', { shape: [8, 2], init: 'normal', scale: 0.5 });
const b2 = mlp.node('b2', 'param', { shape: [2] });
const y0 = mlp.node('linear', 'matmul', {}, { a: h, b: w2 });
const y1 = mlp.node('bias', 'add', {}, { a: y0, b: b2 });
export const xorSolution = mlp.output(mlp.node('probabilities', 'softmax', {}, { x: y1 }));
export const xorLevel: TrainLevel = {
  id: 'xor',
  titleKey: 'levels.xor.title',
  descriptionKey: 'levels.xor.description',
  kind: 'train',
  available: [
    'param',
    'const',
    'add',
    'mul',
    'exp',
    'log',
    'neg',
    'recip',
    'sigmoid',
    'matmul',
    'softmax',
  ],
  interface: { inputs: { x: [4, 2] }, output: 'y', outputShape: [4, 2] },
  unlock: ['4-1'],
  seed: 17,
  steps: 400,
  optimizer: { kind: 'adam', learningRate: 0.05 },
  threshold: { loss: 0.05, accuracy: 1 },
  dataset: (random) => {
    const rows = [
      [0, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 0, 0, 1],
      [1, 1, 1, 0],
    ];
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random.uniform() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    return {
      inputs: {
        x: T.tensor(
          rows.flatMap((r) => r.slice(0, 2)),
          [4, 2],
        ),
      },
      targets: T.tensor(
        rows.flatMap((r) => r.slice(2)),
        [4, 2],
      ),
    };
  },
};
export const exampleLevels: Level[] = [sigmoidLevel, derivativeLevel, xorLevel];
export const exampleSolutions: Record<string, GraphJSON> = {
  sigmoid: sigmoidSolution,
  derivative: derivativeSolution,
  xor: xorSolution,
};
