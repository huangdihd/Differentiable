import { add, exp, log, mul, reduceMax, reshape, sum, tensor, transpose } from './tensor';
import type { Tensor } from './tensor';
import { infer } from './backend';

const scalar = (value: number, x: Tensor) => tensor([value], [], x.backend);
const negate = (x: Tensor) => mul(x, scalar(-1, x));
const reciprocal = (x: Tensor) => exp(negate(log(x)));

/** Primitive-only batched matmul. Batch dimensions follow tensor broadcasting.
 * This reference implementation materializes the product before reducing it;
 * prefer native matmul for large tensors. Dispose the returned graph when done.
 */
export function primitiveMatmul(a: Tensor, b: Tensor): Tensor {
  infer('matmul', [a.shape, b.shape]);
  const m = a.shape.at(-2)!, k = a.shape.at(-1)!, n = b.shape.at(-1)!;
  const perm = b.shape.map((_, i) => i);
  [perm[perm.length - 2], perm[perm.length - 1]] = [perm.at(-1)!, perm.at(-2)!];
  const left = reshape(a, [...a.shape.slice(0, -2), m, 1, k]);
  const right = reshape(transpose(b, perm), [...b.shape.slice(0, -2), 1, n, k]);
  return sum(mul(left, right), [-1]);
}

/** Numerically stable softmax along the final axis, using only primitives. */
export function primitiveSoftmax(x: Tensor): Tensor {
  infer('softmax', [x.shape]);
  const shifted = add(x, negate(reduceMax(x, [-1], true)));
  const weights = exp(shifted);
  return mul(weights, reciprocal(sum(weights, [-1], true)));
}

/** Final-axis layer normalization (population variance, no affine parameters). */
export function primitiveLayernorm(x: Tensor, eps = 1e-5): Tensor {
  infer('layernorm', [x.shape], { eps });
  const scale = scalar(1 / x.shape.at(-1)!, x);
  const mean = mul(sum(x, [-1], true), scale);
  const centered = add(x, negate(mean));
  const variance = mul(sum(mul(centered, centered), [-1], true), scale);
  const inverseStd = exp(mul(log(add(variance, scalar(eps, x))), scalar(-0.5, x)));
  return mul(centered, inverseStd);
}
