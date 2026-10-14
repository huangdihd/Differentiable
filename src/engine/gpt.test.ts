import { expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import {
  Tensor, tensor, add, mul, max, exp, log, sum, reduceMax, reshape, transpose,
  gather, arange, ge, matmul, softmax, layernorm, backward, disposeGraph,
} from './tensor';

it('trains a two-layer character GPT using only primitive graphs, matching native kernels', async () => {
  const started = performance.now();
  const backend = new CpuBackend();
  const B = 8, T = 16, D = 32, H = 4, V = 27;
  let seed = 0x12345678;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; };
  const scalar = (x: number) => tensor([x], [], backend);
  const scale = (x: Tensor, s: number) => mul(x, scalar(s));
  const sub = (a: Tensor, b: Tensor) => add(a, scale(b, -1));
  const reciprocal = (x: Tensor) => exp(scale(log(x), -1));
  // Broadcast [batch,m,k,1] * [batch,1,k,n], then reduce k.
  // These deliberately do not call the native matmul/softmax/layernorm kernels.
  const primitiveMatmul = (a: Tensor, b: Tensor) => {
    const as = a.shape, bs = b.shape;
    return sum(mul(
      reshape(a, [...as, 1]),
      reshape(b, [...bs.slice(0, -2), 1, bs.at(-2)!, bs.at(-1)!]),
    ), [-2]);
  };
  const primitiveSoftmax = (x: Tensor) => {
    const e = exp(sub(x, reduceMax(x, [-1], true)));
    return mul(e, reciprocal(sum(e, [-1], true)));
  };
  const primitiveLayernorm = (x: Tensor) => {
    const centered = sub(x, scale(sum(x, [-1], true), 1 / x.shape.at(-1)!));
    const variance = scale(sum(mul(centered, centered), [-1], true), 1 / x.shape.at(-1)!);
    return mul(centered, exp(scale(log(add(variance, scalar(1e-5))), -0.5)));
  };

  const params: Tensor[] = [];
  const names: string[] = [];
  const parameter = (name: string, shape: number[], constant?: number) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const p = tensor(Float32Array.from({ length: n }, () => constant ?? (random() * 2 - 1) * 0.08), shape, backend);
    names.push(name); params.push(p); return params.length - 1;
  };
  const token = parameter('token embedding', [V, D]);
  const position = parameter('position embedding', [T, D]);
  const normParameters = (prefix: string) => [parameter(`${prefix} gain`, [D], 1), parameter(`${prefix} bias`, [D], 0)];
  const linearParameters = (prefix: string, input: number, output: number) => [
    parameter(`${prefix} weight`, [input, output]), parameter(`${prefix} bias`, [output], 0),
  ];
  const blocks = Array.from({ length: 2 }, (_, i) => ({
    ln1: normParameters(`block ${i} ln1`), ln2: normParameters(`block ${i} ln2`),
    q: linearParameters(`block ${i} query`, D, D),
    k: linearParameters(`block ${i} key`, D, D),
    v: linearParameters(`block ${i} value`, D, D),
    out: linearParameters(`block ${i} attention output`, D, D),
    up: linearParameters(`block ${i} mlp up`, D, 2 * D),
    down: linearParameters(`block ${i} mlp down`, 2 * D, D),
  }));
  const finalNorm = normParameters('final norm');
  const head = linearParameters('language head', D, V);
  const alphabet = ' abcdefghijklmnopqrstuvwxyz';
  const text = 'the cat sat on the mat the dog sat on the log '.repeat(32);
  const inputData: number[] = [], targets = new Float32Array(B * T * V);
  for (let b = 0; b < B; b++) {
    const start = Math.floor(random() * (text.length - T - 1));
    for (let t = 0; t < T; t++) {
      inputData.push(alphabet.indexOf(text[start + t]));
      targets[(b * T + t) * V + alphabet.indexOf(text[start + t + 1])] = 1;
    }
  }
  const input = tensor(inputData, [B, T], backend);
  const target = tensor(targets, [B, T, V], backend);
  const positions = arange(T, backend);
  const allowed = ge(reshape(positions, [T, 1]), reshape(positions, [1, T]));
  const maskGraph = scale(sub(scalar(1), allowed), -1e4);
  const mask = tensor(await maskGraph.read(), [T, T], backend);
  disposeGraph([maskGraph]);
  const persistent = [input, target, mask];
  const forward = (native: boolean) => {
    const mm = native ? matmul : primitiveMatmul;
    const sm = native ? softmax : primitiveSoftmax;
    const ln = native ? layernorm : primitiveLayernorm;
    const linear = (x: Tensor, ids: number[]) => add(mm(x, params[ids[0]]), params[ids[1]]);
    const norm = (x: Tensor, ids: number[]) => add(mul(ln(x), params[ids[0]]), params[ids[1]]);
    const split = (x: Tensor) => transpose(reshape(x, [B, T, H, D / H]), [0, 2, 1, 3]);
    let x = add(gather(params[token], input), params[position]);
    for (const block of blocks) {
      const z = norm(x, block.ln1);
      const q = split(linear(z, block.q)), k = split(linear(z, block.k)), v = split(linear(z, block.v));
      const weights = sm(add(scale(mm(q, transpose(k, [0, 1, 3, 2])), 1 / Math.sqrt(D / H)), mask));
      const attention = reshape(transpose(mm(weights, v), [0, 2, 1, 3]), [B, T, D]);
      x = add(x, linear(attention, block.out));
      // ReLU is an exact primitive activation; this feasibility model needs no GELU kernel.
      x = add(x, linear(max(linear(norm(x, block.ln2), block.up), scalar(0)), block.down));
    }
    const logits = linear(norm(x, finalNorm), head);
    const loss = scale(sum(mul(target, log(sm(logits))), [0, 1, 2]), -1 / (B * T));
    return { logits, loss };
  };
  const close = (actual: Float32Array, expected: Float32Array, label: string) => {
    expect(actual.length, label).toBe(expected.length);
    let worst = 0;
    for (let i = 0; i < actual.length; i++) {
      expect(Number.isFinite(actual[i]) && Number.isFinite(expected[i]), `${label}[${i}] finite`).toBe(true);
      worst = Math.max(worst, Math.abs(actual[i] - expected[i]) / (2e-4 + 3e-3 * Math.abs(expected[i])));
    }
    expect(worst, `${label}: normalized absolute/relative error`).toBeLessThanOrEqual(1);
  };

  try {
    const pure = forward(false), native = forward(true);
    const pureGrad = await backward(pure.loss, params), nativeGrad = await backward(native.loss, params);
    try {
      close(await pure.logits.read(), await native.logits.read(), 'logits');
      close(await pure.loss.read(), await native.loss.read(), 'cross entropy');
      for (let i = 0; i < params.length; i++) close(await pureGrad[i].read(), await nativeGrad[i].read(), names[i]);
    } finally {
      pureGrad.forEach(g => g.dispose()); nativeGrad.forEach(g => g.dispose());
      disposeGraph([pure.loss, native.loss], [...params, ...persistent]);
    }
    expect(backend.liveBuffers).toBe(params.length + persistent.length);

    // Adam is intentionally host-side: parameters remain backend-owned tensors,
    // and every training forward/backward uses the pure primitive implementation.
    const first = params.map(p => new Float32Array(p.shape.reduce((a, b) => a * b, 1)));
    const second = first.map(m => new Float32Array(m.length));
    let initialLoss = NaN;
    for (let step = 1; step <= 16; step++) {
      const { loss } = forward(false);
      const grads = await backward(loss, params);
      try {
        if (step === 1) initialLoss = (await loss.read())[0];
        const next: Tensor[] = [];
        for (let p = 0; p < params.length; p++) {
          const data = await params[p].read(), grad = await grads[p].read();
          for (let j = 0; j < data.length; j++) {
            first[p][j] = 0.9 * first[p][j] + 0.1 * grad[j];
            second[p][j] = 0.999 * second[p][j] + 0.001 * grad[j] ** 2;
            data[j] -= 0.015 * (first[p][j] / (1 - 0.9 ** step)) / (Math.sqrt(second[p][j] / (1 - 0.999 ** step)) + 1e-8);
          }
          next.push(tensor(data, params[p].shape, backend));
        }
        disposeGraph([loss], [...params, ...persistent]);
        params.forEach(p => p.dispose());
        params.splice(0, params.length, ...next);
      } finally {
        grads.forEach(g => g.dispose());
        disposeGraph([loss], [...params, ...persistent]);
      }
      expect(backend.liveBuffers).toBe(params.length + persistent.length);
    }
    const final = forward(false);
    const finalLoss = (await final.loss.read())[0];
    disposeGraph([final.loss], [...params, ...persistent]);
    const elapsed = performance.now() - started;
    console.info(`GPT primitives: batch=${B}, context=${T}, width=${D}, layers=2, vocab=${V}, Adam steps=16; loss ${initialLoss.toFixed(4)} -> ${finalLoss.toFixed(4)}; elapsed ${(elapsed / 1000).toFixed(2)}s`);
    expect(finalLoss).toBeLessThan(initialLoss);
    expect(finalLoss).toBeLessThan(0.8 * Math.log(V));
    expect(elapsed).toBeLessThan(60_000);
  } finally {
    params.forEach(p => p.dispose()); persistent.forEach(p => p.dispose());
  }
  expect(backend.liveBuffers).toBe(0);
}, 60_000);
