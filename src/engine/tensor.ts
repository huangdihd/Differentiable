import { type Backend, type Value, type Shape, type Attributes, infer, size } from './backend';
import { CpuBackend } from './cpu';
/** Set once before constructing a game; existing tensors keep their backend. */
export let defaultBackend: Backend = new CpuBackend();
export function setDefaultBackend(backend: Backend): void {
  defaultBackend = backend;
}
interface Tape {
  op: string;
  inputs: Tensor[];
  attrs: Attributes;
}
export class Tensor implements Value {
  readonly handle;
  readonly shape: readonly number[];
  private disposed = false;
  constructor(
    readonly backend: Backend,
    shape: Shape,
    data?: ArrayLike<number>,
    readonly tape?: Tape,
  ) {
    this.shape = Object.freeze([...shape]);
    this.handle = backend.allocate(shape, data);
  }
  read(): Promise<Float32Array> {
    this.assertLive();
    return this.backend.read(this.handle);
  }
  assertLive(): void {
    if (this.disposed) throw new Error('engine.tensor.disposed');
  }
  dispose(): void {
    if (!this.disposed) {
      this.backend.release(this.handle);
      this.disposed = true;
    }
  }
}
export function tensor(
  data: ArrayLike<number>,
  shape: Shape,
  backend: Backend = defaultBackend,
): Tensor {
  return new Tensor(backend, shape, data);
}
/** Backend-independent kernel dispatch. Backend owns execution and all data. */
export function operation(
  op: string,
  inputs: Tensor[],
  attrs: Attributes = {},
  shape?: Shape,
): Tensor {
  const backend = inputs[0]?.backend ?? defaultBackend;
  for (const x of inputs) {
    x.assertLive();
    if (x.backend !== backend) throw new Error('engine.backend.mismatch');
  }
  const result = new Tensor(
    backend,
    shape ??
      infer(
        op,
        inputs.map((x) => x.shape),
        attrs,
      ),
    undefined,
    { op, inputs, attrs },
  );
  try {
    backend.submit([{ op, inputs, attrs, output: result }]);
  } catch (error) {
    result.dispose();
    throw error;
  }
  return result;
}
export const add = (a: Tensor, b: Tensor) => operation('add', [a, b]);
export const mul = (a: Tensor, b: Tensor) => operation('mul', [a, b]);
export const max = (a: Tensor, b: Tensor) => operation('max', [a, b]);
export const ge = (a: Tensor, b: Tensor) => operation('ge', [a, b]);
export const exp = (x: Tensor) => operation('exp', [x]);
export const log = (x: Tensor) => operation('log', [x]);
export const recip = (x: Tensor) => operation('recip', [x]);
export const sum = (x: Tensor, axes: number[], keepdims = false) =>
  operation('sum', [x], { axes, keepdims });
export const reduceMax = (x: Tensor, axes: number[], keepdims = false) =>
  operation('reduce_max', [x], { axes, keepdims });
export const reshape = (x: Tensor, shape: Shape) => operation('reshape', [x], { shape });
export const transpose = (x: Tensor, perm: number[]) => operation('transpose', [x], { perm });
export const gather = (table: Tensor, idx: Tensor) => operation('gather', [table, idx]);
/** Scatter into a zero table; table supplies only the destination shape. */
export const scatterAdd = (table: Tensor, idx: Tensor, updates: Tensor) =>
  operation('scatter_add', [table, idx, updates]);
export const matmul = (a: Tensor, b: Tensor) => operation('matmul', [a, b]);
export const softmax = (x: Tensor) => operation('softmax', [x]);
export const layernorm = (x: Tensor, eps = 1e-5) => operation('layernorm', [x], { eps });
export function arange(n: number, backend: Backend = defaultBackend): Tensor {
  size([n]);
  return tensor(
    Float32Array.from({ length: n }, (_, i) => i),
    [n],
    backend,
  );
}
function order(outputs: Tensor[]): Tensor[] {
  const seen = new Set<Tensor>(),
    nodes: Tensor[] = [];
  const visit = (x: Tensor) => {
    if (seen.has(x)) return;
    seen.add(x);
    x.tape?.inputs.forEach(visit);
    nodes.push(x);
  };
  outputs.forEach(visit);
  return nodes;
}
export function disposeGraph(outputs: Tensor[], keep: Tensor[] = []): void {
  const retained = new Set(keep);
  for (const x of order(outputs)) if (!retained.has(x)) x.dispose();
}
/** First-order reverse AD. Non-scalar outputs are seeded with ones (sum VJP).
 * Returned gradients are caller-owned; temporary adjoints are released here. */
export async function backward(loss: Tensor, inputs: Tensor[]): Promise<Tensor[]> {
  const nodes = order([loss]);
  const needed = new Set<Tensor>(inputs);
  for (const x of nodes) if (x.tape?.inputs.some((p) => needed.has(p))) needed.add(x);
  const grads = new Map<Tensor, Tensor>();
  grads.set(loss, tensor(new Float32Array(size(loss.shape)).fill(1), loss.shape, loss.backend));
  try {
    for (const x of nodes.reverse()) {
      const g = grads.get(x);
      if (!g || !x.tape) continue;
      const { op, inputs: parents, attrs } = x.tape;
      for (let i = 0; i < parents.length; i++) {
        const p = parents[i];
        if (!needed.has(p)) continue;
        const dx = new Tensor(p.backend, p.shape);
        try {
          p.backend.submit([
            {
              op: 'vjp',
              inputs: [...parents, x, g],
              output: dx,
              attrs: { ...attrs, op, input: i, count: parents.length },
            },
          ]);
        } catch (error) {
          dx.dispose();
          throw error;
        }
        const old = grads.get(p);
        if (old) {
          const combined = new Tensor(p.backend, p.shape);
          p.backend.submit([{ op: 'add', inputs: [old, dx], output: combined, attrs: {} }]);
          old.dispose();
          dx.dispose();
          grads.set(p, combined);
        } else grads.set(p, dx);
      }
    }
    return inputs.map((x) => {
      const result = grads.get(x);
      return result ? tensorCopy(result) : new Tensor(x.backend, x.shape);
    });
  } finally {
    for (const g of grads.values()) g.dispose();
  }
}
function tensorCopy(x: Tensor): Tensor {
  const out = new Tensor(x.backend, x.shape);
  x.backend.submit([{ op: 'reshape', inputs: [x], output: out, attrs: { shape: x.shape } }]);
  return out;
}
