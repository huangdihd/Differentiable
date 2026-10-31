// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import {
  compileGraph,
  createDefaultRegistry,
  type GraphJSON,
  type Attributes,
  type GraphPlan,
} from './graph';
import { Random } from './random';
import { tensor, backward, type Tensor } from './tensor';
import { TrainingSession, type TrainingBatch } from './training';

// Lewis Carroll, Alice's Adventures in Wonderland (1865), chapter I.
// One contiguous public-domain excerpt; no repeated/constructed training sentences.
const excerpt = `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
In another moment down went Alice after it, never once considering how in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.
Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody, so managed to put it into one of the cupboards as she fell past it.
"Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!" (Which was very likely true.)`;

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

it('trains a nested GraphJSON GPT with fresh Adam batches and matches primitive forward/backward', async () => {
  const started = performance.now(),
    backend = new CpuBackend();
  const B = 8,
    C = 12,
    D = 24,
    H = 3,
    V = 27;
  const alphabet = ' abcdefghijklmnopqrstuvwxyz';
  const text = excerpt
    .toLowerCase()
    .replace(/[^a-z]+/g, ' ')
    .trim();
  expect(text.length).toBeGreaterThan(2300);
  expect(text.length).toBeLessThan(2800);
  const split = Math.floor(text.length * 0.8);
  const train = text.slice(0, split),
    validation = text.slice(split);
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
  for (let i = 0; i < 2; i++) x = g.node(`block${i}`, 'block', { x });
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
  const native = compileGraph(graph, {
    registry,
    inputShapes: shapes,
    backend,
    random: new Random(123),
    preferNative: true,
  });
  const rng = new Random(456);
  const makeBatch = (source: string, offset?: number): TrainingBatch => {
    const ids: number[] = [],
      labels = new Float32Array(B * C * V);
    for (let b = 0; b < B; b++) {
      const start =
        offset !== undefined
          ? Math.min(offset + b * C, source.length - C - 1)
          : Math.floor(rng.uniform() * (source.length - C));
      for (let t = 0; t < C; t++) {
        ids.push(alphabet.indexOf(source[start + t]));
        labels[(b * C + t) * V + alphabet.indexOf(source[start + t + 1])] = 1;
      }
    }
    const target = tensor(labels, [B, C, V], backend);
    return { inputs: { tokens: tensor(ids, [B, C], backend), targets: target }, targets: target };
  };
  const evaluation = makeBatch(validation, 0);
  const evaluateHoldout = async (session: TrainingSession) => {
    let totalLoss = 0,
      batches = 0;
    // Cover the entire held-out suffix, overlapping only the final partial batch.
    for (let offset = 0; offset < validation.length - 1; offset += B * C) {
      const batch = makeBatch(validation, offset);
      try {
        totalLoss += (await session.evaluate(batch)).loss;
        batches++;
      } finally {
        new Set([...Object.values(batch.inputs), batch.targets]).forEach((t) => t?.dispose());
      }
    }
    return totalLoss / batches;
  };
  const session = new TrainingSession(
    native,
    () => makeBatch(train),
    undefined,
    { kind: 'adam', learningRate: 0.008 },
    'logits',
    { lossOutput: 'loss', accuracy: false, evaluationBatch: evaluation },
  );
  let pure: GraphPlan | undefined;
  const close = (a: Float32Array, b: Float32Array, label: string) => {
    expect(a.length, label).toBe(b.length);
    let worst = 0;
    for (let i = 0; i < a.length; i++) {
      expect(Number.isFinite(a[i]) && Number.isFinite(b[i]), label).toBe(true);
      worst = Math.max(worst, Math.abs(a[i] - b[i]) / (3e-4 + 5e-3 * Math.abs(b[i])));
    }
    expect(worst, label).toBeLessThanOrEqual(1);
  };
  try {
    const initialLoss = await evaluateHoldout(session);
    const block0 = [...native.parameters].filter(([key]) => key.startsWith('block0/'));
    expect(block0.length).toBe(16);
    for (const [key, value] of block0) {
      const other = native.parameters.get(key.replace('block0/', 'block1/'));
      expect(other, key).toBeDefined();
      expect(other, key).not.toBe(value);
      expect(other!.shape).toEqual(value.shape);
      if (key.endsWith('/weight')) expect(await other!.read()).not.toEqual(await value.read());
    }
    const baselineBuffers = backend.liveBuffers;
    await session.run(100, { yieldEvery: 100 });
    const finalLoss = await evaluateHoldout(session);
    session.dispose();
    expect(backend.liveBuffers).toBe(baselineBuffers);
    // Held-out positional suffix is never sampled by the training factory.
    const counts = new Float64Array(V);
    for (const c of validation) counts[alphabet.indexOf(c)]++;
    const entropy = -counts.reduce(
      (s, n) => (n ? s + (n / validation.length) * Math.log(n / validation.length) : s),
      0,
    );
    expect(finalLoss).toBeLessThan(initialLoss);
    expect(finalLoss).toBeLessThan(entropy);
    const corpusCounts = new Float64Array(V);
    for (const c of text) corpusCounts[alphabet.indexOf(c)]++;
    const corpusEntropy = -corpusCounts.reduce(
      (s, n) => (n ? s + (n / text.length) * Math.log(n / text.length) : s),
      0,
    );
    expect(finalLoss).toBeLessThan(corpusEntropy);
    pure = compileGraph(graph, {
      registry,
      inputShapes: shapes,
      backend,
      parameters: native.parameters,
      preferNative: false,
    });
    const a = native.execute(evaluation.inputs),
      b = pure.execute(evaluation.inputs);
    const params = [...native.parameters.values()];
    let ga: Tensor[] = [],
      gb: Tensor[] = [];
    try {
      close(await b.logits.read(), await a.logits.read(), 'logits');
      close(await b.loss.read(), await a.loss.read(), 'loss');
      ga = await backward(a.loss, params);
      gb = await backward(b.loss, params);
      for (let i = 0; i < params.length; i++)
        close(await gb[i].read(), await ga[i].read(), [...native.parameters.keys()][i]);
    } finally {
      ga.forEach((t) => t.dispose());
      gb.forEach((t) => t.dispose());
      native.releaseExecution(a);
      pure.releaseExecution(b);
    }
    const elapsed = (performance.now() - started) / 1000;
    console.info(
      `GraphJSON GPT: chars=${text.length}, train=${train.length}, validation=${validation.length}, B=${B}, C=${C}, D=${D}, H=${H}, blocks=2, Adam=100; validation ${initialLoss.toFixed(4)} -> ${finalLoss.toFixed(4)}, validation unigram entropy=${entropy.toFixed(4)}, corpus unigram entropy=${corpusEntropy.toFixed(4)}, parameters=${params.reduce((n, p) => n + p.shape.reduce((a, b) => a * b, 1), 0)},  elapsed=${elapsed.toFixed(2)}s`,
    );
    expect(elapsed).toBeLessThan(60);
  } finally {
    pure?.dispose();
    native.dispose();
    new Set([...Object.values(evaluation.inputs), evaluation.targets]).forEach((t) => t?.dispose());
  }
  expect(backend.liveBuffers).toBe(0);
}, 60_000);
