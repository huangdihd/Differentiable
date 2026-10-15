import type { Attributes, GraphJSON, ShapePorts } from '../engine/graph';
import type { ForwardLevel } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { chapter0Levels } from './chapter0';
import { chapter1Levels } from './chapter1';
import { chapter2Levels } from './chapter2';
import { chapter3ForwardLevels } from './chapter3';
import { chapter4ForwardLevels } from './chapter4';
import { chapter5ForwardLevels, chapter5TrainLevels } from './chapter5';
import { withPar } from './level-helpers';

export const chapter6Preceding = [
  ...chapter0Levels,
  ...chapter1Levels,
  ...chapter2Levels,
  ...chapter3ForwardLevels,
  ...chapter4ForwardLevels,
  ...chapter5ForwardLevels,
];
const available = chapter5TrainLevels[0].available;
const heads = 2;
const names = ['multi_head', 'feed_forward', 'transformer_block'];

function answer(index: number): GraphJSON {
  const graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  const node = (
    id: string,
    component: string,
    args: Record<string, string> = {},
    attrs?: Attributes,
  ) => {
    graph.nodes.push({ id, component, ...(attrs ? { attrs } : {}) });
    for (const [port, source] of Object.entries(args))
      graph.edges.push({ from: { node: source, port: 'out' }, to: { node: id, port } });
    return id;
  };
  const input = (name: string) => node(name, 'input', {}, { name });
  const x = index === 1 ? '' : input('x');
  let result: string;
  if (index === 2) {
    const w1 = input('w1'),
      b1 = input('b1'),
      w2 = input('w2'),
      b2 = input('b2');
    const hidden = node('hidden', 'relu', {
      x: node('affine', 'add', {
        a: node('project', 'matmul', { a: x, b: w1 }),
        b: b1,
      }),
    });
    result = node('result', 'add', {
      a: node('projectOut', 'matmul', { a: hidden, b: w2 }),
      b: b2,
    });
  } else if (index === 1) {
    const q = input('q'),
      k = input('k'),
      v = input('v');
    const batch = node('batch', 'size', { x: q }, { axis: 0 });
    const length = node('length', 'size', { x: q }, { axis: 1 });
    const width = node('width', 'size', { x: q }, { axis: -1 });
    // Static scalar arithmetic: D / 2, evaluated from the concrete input shape.
    const headWidth = node('headWidth', 'mul', {
      a: width,
      b: node('inverseHeads', 'const', {}, { value: 1 / heads }),
    });
    const split = (name: string, source: string) =>
      node(
        `${name}Heads`,
        'transpose',
        {
          x: node(
            `${name}Split`,
            'reshape',
            {
              x: source,
              d0: batch,
              d1: length,
              d3: headWidth,
            },
            { shape: [1, 1, heads, 1] },
          ),
        },
        { axes: [1, 2] },
      );
    const attended = node('attended', 'attention', {
      q: split('q', q),
      k: split('k', k),
      v: split('v', v),
    });
    result = node(
      'result',
      'reshape',
      {
        x: node('restored', 'transpose', { x: attended }, { axes: [1, 2] }),
        d0: batch,
        d1: length,
        d2: width,
      },
      { shape: [1, 1, 1] },
    );
  } else {
    // Eight parameter inputs: four attention matrices plus w1, b1, w2, b2.
    const weights = Object.fromEntries(
      ['wq', 'wk', 'wv', 'wo', 'w1', 'b1', 'w2', 'b2'].map((name) => [name, input(name)]),
    );
    const normalized = node('norm1', 'layernorm', { x });
    const project = (name: string) =>
      node(name, 'matmul', { a: normalized, b: weights[`w${name}`] });
    const attended = node('attended', 'matmul', {
      a: node('heads', 'multi_head', { q: project('q'), k: project('k'), v: project('v') }),
      b: weights.wo,
    });
    const residual = node('residual', 'add', { a: x, b: attended });
    const feedforward = node('feedforward', 'feed_forward', {
      x: node('norm2', 'layernorm', { x: residual }),
      w1: weights.w1,
      b1: weights.b1,
      w2: weights.w2,
      b2: weights.b2,
    });
    result = node('result', 'add', { a: residual, b: feedforward });
  }
  node('output', 'output', { x: result }, { name: 'y' });
  return graph;
}

function inputsFor(index: number, x: number[], hidden: number): ShapePorts {
  const d = x.at(-1)!;
  if (index === 1) return { q: x, k: x, v: x };
  return {
    x,
    ...(index === 3 ? { wq: [d, d], wk: [d, d], wv: [d, d], wo: [d, d] } : {}),
    w1: [d, hidden],
    b1: [hidden],
    w2: [hidden, d],
    b2: [d],
  };
}
function fixtures(index: number, inputs: ShapePorts) {
  const make = (random: Random, i: number) => ({
    id: `case-${i}`,
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([name, shape]) => [
        name,
        Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, j) =>
          i === 0
            ? 0
            : i === 1 && name === 'x'
              ? 0.5
              : i === 2
                ? (((j * 3 + name.length) % 9) - 4) / 8
                : random.uniform() - 0.5,
        ),
      ]),
    ),
  });
  return {
    fixed: Array.from({ length: 5 }, (_, i) => make(new Random(6100 + i), i)),
    generated: { seed: 6600 + index, count: 5, generate: make },
  };
}
function ffn(x: T.Tensor, a: Record<string, T.Tensor>) {
  const affine = T.add(T.matmul(x, a.w1), a.b1);
  return T.add(T.matmul(T.max(affine, T.tensor([0], [], x.backend)), a.w2), a.b2);
}
function multihead(x: T.Tensor, kInput: T.Tensor, vInput: T.Tensor) {
  const d = x.shape.at(-1)!,
    n = x.shape.at(-2)!;
  const batch = x.shape.slice(0, -2).reduce((v, dim) => v * dim, 1);
  const project = (value: T.Tensor) =>
    T.transpose(T.reshape(value, [batch, n, heads, d / heads]), [0, 2, 1, 3]);
  const q = project(x),
    k = project(kInput),
    v = project(vInput);
  const scores = T.mul(
    T.matmul(q, T.transpose(k, [0, 1, 3, 2])),
    T.tensor([1 / Math.sqrt(d / heads)], [], x.backend),
  );
  const mask = T.tensor(
    Array.from({ length: n * n }, (_, i) => (i % n <= Math.floor(i / n) ? 1 : 0)),
    [n, n],
    x.backend,
  );
  const masked = T.add(
    T.mul(scores, mask),
    T.mul(T.add(mask, T.tensor([-1], [], x.backend)), T.tensor([1e9], [], x.backend)),
  );
  const attended = T.matmul(T.softmax(masked), v);
  return T.reshape(T.transpose(attended, [0, 2, 1, 3]), x.shape);
}

export const chapter6ForwardLevels: ForwardLevel[] = [];
for (let index = 1; index <= 3; index++) {
  const shapes = [
    inputsFor(index, [2, 4, 8], 16),
    inputsFor(index, [3, 5, 12], 20),
    inputsFor(index, [1, 3, 4], 7),
  ];
  const inputs = shapes[0];
  chapter6ForwardLevels.push(
    withPar<ForwardLevel>(
      {
        id: `6-${index}`,
        chapter: '6',
        kind: 'forward',
        titleKey: `levels.6-${index}.title`,
        descriptionKey: `levels.6-${index}.description`,
        hintsKeys: [`levels.6-${index}.hint1`, `levels.6-${index}.hint2`],
        available,
        unlock: [`6-${index + 1}`],
        unlockComponents: [names[index - 1]],
        forbidden: [names[index - 1]],
        referenceGraph: answer(index),
        interface: { inputs, output: 'y', outputShape: inputs.x ?? inputs.q },
        ...fixtures(index, inputs),
        shapeVariants: shapes.slice(1).map((inputs) => ({ inputs, ...fixtures(index, inputs) })),
        reference: (args) => {
          if (index === 1) return multihead(args.q, args.k, args.v);
          if (index === 2) return ffn(args.x, args);
          const normalized = T.layernorm(args.x);
          const attended = multihead(
            T.matmul(normalized, args.wq),
            T.matmul(normalized, args.wk),
            T.matmul(normalized, args.wv),
          );
          const residual = T.add(args.x, T.matmul(attended, args.wo));
          return T.add(residual, ffn(T.layernorm(residual), args));
        },
      },
      [...chapter6Preceding, ...chapter6ForwardLevels],
    ),
  );
}
