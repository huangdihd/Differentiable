import { describe, expect, it } from 'vitest';
import { CpuBackend } from './cpu';
import {
  add,
  arange,
  backward,
  disposeGraph,
  exp,
  gather,
  ge,
  log,
  max,
  mul,
  reduceMax,
  reshape,
  sum,
  tensor,
  transpose,
} from './tensor';
import type { Tensor } from './tensor';
import { compileGraph, type GraphJSON, type GraphPlan } from './graph';

const executionPlans: GraphPlan[] = [];
function registered(name: string, inputs: Tensor[], preferNative: boolean, eps?: number): Tensor {
  const ports = name === 'matmul' ? ['a', 'b'] : ['x'];
  const graph: GraphJSON = {
    version: 1,
    nodes: [
      ...ports.map((port, i) => ({
        id: port,
        component: 'input',
        attrs: { name: port, shape: inputs[i].shape },
      })),
      { id: 'op', component: name, attrs: eps === undefined ? {} : { eps } },
      { id: 'output', component: 'output', attrs: { name: 'out' } },
    ],
    edges: [
      ...ports.map((port) => ({ from: { node: port, port: 'out' }, to: { node: 'op', port } })),
      { from: { node: 'op', port: 'out' }, to: { node: 'output', port: 'x' } },
    ],
  };
  const plan = compileGraph(graph, { backend: inputs[0].backend, preferNative });
  executionPlans.push(plan);
  return plan.execute(Object.fromEntries(ports.map((port, i) => [port, inputs[i]]))).out;
}
const primitiveMatmul = (a: Tensor, b: Tensor) => registered('matmul', [a, b], false);
const primitiveSoftmax = (x: Tensor) => registered('softmax', [x], false);
const primitiveLayernorm = (x: Tensor, eps = 1e-5) => registered('layernorm', [x], false, eps);
const matmul = (a: Tensor, b: Tensor) => registered('matmul', [a, b], true);
const softmax = (x: Tensor) => registered('softmax', [x], true);
const layernorm = (x: Tensor, eps = 1e-5) => registered('layernorm', [x], true, eps);

type Input = { data: number[]; shape: number[] };
type Formula = (...inputs: Tensor[]) => Tensor;
const input = (data: number[], shape = [data.length]): Input => ({ data, shape });
const scalar = (n: number, x: Tensor) => tensor([n], [], x.backend);
const neg = (x: Tensor) => mul(x, scalar(-1, x));
const sigmoid = (x: Tensor) => exp(neg(log(add(scalar(1, x), exp(neg(x))))));
const total = (x: Tensor) =>
  sum(
    x,
    x.shape.map((_, i) => i),
  );

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, atol = 2e-4, rtol = 2e-4) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Number.isFinite(actual[i]), `element ${i} must be finite`).toBe(true);
    expect(
      Math.abs(actual[i] - expected[i]),
      `element ${i}: ${actual[i]} vs ${expected[i]}`,
    ).toBeLessThanOrEqual(atol + rtol * Math.abs(expected[i]));
  }
}

/** Every evaluation has its own backend and tape, including perturbations. */
async function evaluate(specs: Input[], formula: Formula, gradients = false) {
  const backend = new CpuBackend();
  const inputs = specs.map((s) => tensor(s.data, s.shape, backend));
  const roots: Tensor[] = [...inputs];
  try {
    const output = formula(...inputs);
    roots.push(output);
    const value = Array.from(await output.read());
    const shape = [...output.shape];
    if (!gradients) return { value, shape, gradients: [] as number[][] };
    const loss = total(output);
    roots.push(loss);
    const grads = await backward(loss, inputs);
    roots.push(...grads);
    return {
      value,
      shape,
      gradients: await Promise.all(grads.map(async (g) => Array.from(await g.read()))),
    };
  } finally {
    disposeGraph(roots);
    executionPlans.splice(0).forEach((plan) => plan.dispose());
  }
}

async function finiteDifference(specs: Input[], formula: Formula, atol = 3e-3, rtol = 1e-2) {
  const analytical = await evaluate(specs, formula, true);
  const h = 2e-3;
  for (let p = 0; p < specs.length; p++) {
    const numerical: number[] = [];
    for (let j = 0; j < specs[p].data.length; j++) {
      const perturb = (delta: number) =>
        specs.map((s, i) => ({
          shape: s.shape,
          data: s.data.map((v, k) => v + (i === p && k === j ? delta : 0)),
        }));
      const plus = await evaluate(perturb(h), formula);
      const minus = await evaluate(perturb(-h), formula);
      numerical.push(
        (plus.value.reduce((a, b) => a + b, 0) - minus.value.reduce((a, b) => a + b, 0)) / (2 * h),
      );
    }
    close(analytical.gradients[p], numerical, atol, rtol);
  }
}

async function equivalent(specs: Input[], native: Formula, primitive: Formula, atol = 3e-4) {
  const a = await evaluate(specs, native, true),
    b = await evaluate(specs, primitive, true);
  expect(a.shape).toEqual(b.shape);
  close(a.value, b.value, atol);
  a.gradients.forEach((g, i) => close(g, b.gradients[i], atol));
}

// Nonuniform objectives prevent softmax and normalization gradient checks from
// degenerating into the derivative of a constant row sum.
function weighted(x: Tensor): Tensor {
  const count = x.shape.reduce((a, b) => a * b, 1);
  return mul(
    x,
    tensor(
      Array.from({ length: count }, (_, i) => (((i * 7) % 11) - 5) / 7),
      [...x.shape],
      x.backend,
    ),
  );
}

describe('primitive autodiff: central finite differences', () => {
  it('add/mul undo leading and singleton broadcast dimensions', async () => {
    await finiteDifference(
      [
        input([0.2, -0.4, 0.7, 1.1, -0.6, 0.3], [2, 1, 3]),
        input([0.5, -0.2, 0.8, 0.9], [1, 4, 1]),
        input([0.3], []),
      ],
      (a, b, c) => weighted(add(mul(a, b), c)),
    );
  });
  it('exp and log chain rule, including a reused intermediate', async () => {
    await finiteDifference([input([0.2, 0.7, 1.4, 2])], (x) => {
      const y = log(x);
      return add(exp(y), mul(y, y));
    });
  });
  it('sigmoid gradients at negative, zero, and positive inputs', async () => {
    const specs = [input([-4, -1, 0, 0.5, 3])];
    await finiteDifference(specs, sigmoid);
    const result = await evaluate(specs, sigmoid, true);
    close(
      result.value,
      specs[0].data.map((x) => 1 / (1 + Math.exp(-x))),
    );
    close(
      result.gradients[0],
      result.value.map((y) => y * (1 - y)),
    );
  });
  it('reshape, non-self-inverse transpose, and multi-axis reductions', async () => {
    const specs = [
      input(
        Array.from({ length: 24 }, (_, i) => (i - 8) / 13),
        [2, 3, 4],
      ),
    ];
    await finiteDifference(specs, (x) => weighted(sum(transpose(x, [1, 2, 0]), [0, -1], true)));
    await finiteDifference(specs, (x) => weighted(reshape(sum(transpose(x, [1, 2, 0]), [1]), [6])));
  });
  it('sum with empty axes is identity and keepdims preserves reduced axes', async () => {
    const specs = [input([1, 2, 3, 4, 5, 6], [2, 3])];
    expect((await evaluate(specs, (x) => sum(x, [], true))).value).toEqual(specs[0].data);
    const result = await evaluate(specs, (x) => sum(x, [-1], true), true);
    expect(result.shape).toEqual([2, 1]);
    close(result.value, [6, 15]);
    close(result.gradients[0], [1, 1, 1, 1, 1, 1]);
  });
  it('max and reduce_max away from nondifferentiable ties', async () => {
    await finiteDifference([input([-2, 0.4, 3, 1.2], [2, 2]), input([0, 1], [2])], (a, b) =>
      weighted(max(a, b)),
    );
    await finiteDifference([input([1, 4, -1, 2, 0, 5], [2, 3])], (x) =>
      weighted(reduceMax(x, [1], true)),
    );
  });
  it('gather accumulates repeated rows in the table gradient', async () => {
    const specs = [input([0.2, 0.4, 0.6, 0.8, 1, 1.2], [3, 2])];
    const formula = (x: Tensor) => gather(x, tensor([2, 0, 2], [3], x.backend));
    await finiteDifference(specs, (x) => weighted(formula(x)));
    const result = await evaluate(specs, formula, true);
    close(result.value, [1, 1.2, 0.2, 0.4, 1, 1.2]);
    close(result.gradients[0], [1, 1, 0, 0, 2, 2]);
  });
});

describe('nondifferentiable primitive conventions', () => {
  it('arange supplies row indices for gather on the same backend', async () => {
    const specs = [input([1, 2, 3, 4, 5, 6], [3, 2])];
    const result = await evaluate(specs, (x) => gather(x, arange(3, x.backend)), true);
    expect(result.shape).toEqual([3, 2]);
    close(result.value, specs[0].data);
    close(result.gradients[0], [1, 1, 1, 1, 1, 1]);
  });
  it('binary max assigns equality to the first operand, including broadcast', async () => {
    const result = await evaluate([input([2, 1, 2]), input([2], [])], (a, b) => max(a, b), true);
    close(result.gradients[0], [1, 0, 1]);
    close(result.gradients[1], [1]);
  });
  it('reduce_max selects the first tied element in each reduction', async () => {
    const specs = [input([3, 3, 1, 2, 4, 4], [2, 3])];
    const result = await evaluate(specs, (x) => reduceMax(x, [1]), true);
    close(result.value, [3, 4]);
    close(result.gradients[0], [1, 0, 0, 0, 1, 0]);
    const all = await evaluate(specs, (x) => reduceMax(x, [0, 1]), true);
    close(all.gradients[0], [0, 0, 0, 0, 1, 0]);
  });
  it('ge includes equality and has zero gradients', async () => {
    const result = await evaluate([input([-1, 0, 2]), input([0], [])], (a, b) => ge(a, b), true);
    close(result.value, [0, 1, 1]);
    close(result.gradients[0], [0, 0, 0]);
    close(result.gradients[1], [0]);
  });
});

describe('native versus primitive composite operations', () => {
  it('rectangular and broadcast batched matmul values and gradients', async () => {
    const cases: Input[][] = [
      [input([1, 2, 3, 4, 5, 6], [2, 3]), input([0.2, -0.4, 0.5, 0.7, -0.2, 0.9], [3, 2])],
      [
        input(
          Array.from({ length: 12 }, (_, i) => (i - 5) / 9),
          [2, 1, 2, 3],
        ),
        input(
          Array.from({ length: 18 }, (_, i) => (i - 7) / 11),
          [1, 3, 3, 2],
        ),
      ],
    ];
    for (const specs of cases) {
      await equivalent(specs, matmul, primitiveMatmul);
      await equivalent(
        specs,
        (a, b) => weighted(matmul(a, b)),
        (a, b) => weighted(primitiveMatmul(a, b)),
      );
      await finiteDifference(specs, (a, b) => weighted(matmul(a, b)));
    }
  });
  it('stable softmax for large logits and tied maxima', async () => {
    const specs = [input([1000, 1001, 999, -1000, -1000, -1002], [2, 3])];
    await equivalent(specs, softmax, primitiveSoftmax);
    await equivalent(
      specs,
      (x) => weighted(softmax(x)),
      (x) => weighted(primitiveSoftmax(x)),
    );
    const result = await evaluate(specs, softmax);
    close(
      [
        result.value.slice(0, 3).reduce((a, b) => a + b),
        result.value.slice(3).reduce((a, b) => a + b),
      ],
      [1, 1],
    );
    await finiteDifference([input([-0.5, 0.8, 1.2, 0.1, -0.3, 0.7], [2, 3])], (x) =>
      weighted(softmax(x)),
    );
  });
  it('layernorm for rank-three tensors, constant rows, and custom epsilon', async () => {
    for (const eps of [1e-5, 0.03]) {
      const specs = [input([0.2, -0.4, 0.8, 2, 2, 2, -1, 0, 1, 4, 3, 2], [2, 2, 3])];
      await equivalent(
        specs,
        (x) => layernorm(x, eps),
        (x) => primitiveLayernorm(x, eps),
      );
      await equivalent(
        specs,
        (x) => weighted(layernorm(x, eps)),
        (x) => weighted(primitiveLayernorm(x, eps)),
        2e-3,
      );
    }
    await finiteDifference([input([-0.3, 0.7, 1.2, 0.2, -0.5, 0.9], [2, 3])], (x) =>
      weighted(layernorm(x)),
    );
    await equivalent(
      [input([2, -3], [2, 1])],
      (x) => weighted(layernorm(x)),
      (x) => weighted(primitiveLayernorm(x)),
    );
  });
});

describe('end-to-end differentiable models', () => {
  const crossEntropy = (logits: Tensor, primitive = false) => {
    const labels = tensor([0, 1, 0, 0, 0, 1], [2, 3], logits.backend);
    return mul(
      total(mul(labels, log(primitive ? primitiveSoftmax(logits) : softmax(logits)))),
      scalar(-0.5, logits),
    );
  };
  it('softmax cross entropy has (probabilities - labels) / batch gradient', async () => {
    const specs = [input([0.3, -0.5, 1, 0.7, 0.2, -0.1], [2, 3])];
    await finiteDifference(specs, (x) => crossEntropy(x));
    await equivalent(
      specs,
      (x) => crossEntropy(x),
      (x) => crossEntropy(x, true),
    );
    const result = await evaluate(specs, (x) => crossEntropy(x), true);
    const probabilities = (await evaluate(specs, softmax)).value;
    close(
      result.gradients[0],
      probabilities.map((p, i) => (p - (i === 1 || i === 5 ? 1 : 0)) / 2),
    );
  });
  it('small sigmoid MLP differentiates inputs, both weight matrices, and broadcast biases', async () => {
    const specs = [
      input([0.2, -0.6, 0.7, 0.3], [2, 2]),
      input([0.4, -0.5, 0.2, 0.8], [2, 2]),
      input([0.1, -0.2], [2]),
      input([0.3, -0.7, 0.4, 0.6, 0.2, -0.5], [2, 3]),
      input([0.1, 0.2, -0.1], [3]),
    ];
    const mlp =
      (primitive: boolean): Formula =>
      (x, w1, b1, w2, b2) => {
        const mm = primitive ? primitiveMatmul : matmul;
        return crossEntropy(add(mm(sigmoid(add(mm(x, w1), b1)), w2), b2), primitive);
      };
    await finiteDifference(specs, mlp(false));
    await equivalent(specs, mlp(false), mlp(true));
  });
});
