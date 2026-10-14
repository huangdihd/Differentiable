import {
  type Backend,
  type Handle,
  type Shape,
  type Command,
  size,
  strides,
  axesFor,
} from './backend';

/** Reference f32 backend. Native kernels are registered, not special tensor types. */
export type CpuKernel = (
  inputs: Float32Array[],
  shapes: Shape[],
  attrs: Record<string, unknown>,
  output: Float32Array,
  outputShape: Shape,
) => void;
export class CpuBackend implements Backend {
  private next = 0;
  private buffers = new Map<number, Float32Array>();
  private kernels = new Map<string, CpuKernel>();
  private adjoints = new Map<string, CpuKernel>();
  get liveBuffers(): number {
    return this.buffers.size;
  }
  /** VJP receives original inputs, forward output, upstream gradient; attrs.input selects the differentiated input. */
  register(name: string, kernel: CpuKernel, vjp?: CpuKernel): void {
    this.kernels.set(name, kernel);
    if (vjp) this.adjoints.set(name, vjp);
  }
  allocate(shape: Shape, data?: ArrayLike<number>): Handle {
    const n = size(shape);
    if (data && data.length !== n) throw new Error('engine.data.length');
    const handle = { id: this.next++ };
    this.buffers.set(handle.id, data ? Float32Array.from(data) : new Float32Array(n));
    return handle;
  }
  private get(h: Handle): Float32Array {
    const data = this.buffers.get(h.id);
    if (!data) throw new Error('engine.tensor.disposed');
    return data;
  }
  async read(h: Handle): Promise<Float32Array> {
    return this.get(h).slice();
  }
  release(h: Handle): void {
    this.buffers.delete(h.id);
  }
  submit(commands: readonly Command[]): void {
    for (const c of commands) {
      const xs = c.inputs.map((v) => this.get(v.handle));
      const shapes = c.inputs.map((v) => v.shape);
      const out = this.get(c.output.handle);
      const custom =
        c.op === 'vjp' ? this.adjoints.get(String(c.attrs.op)) : this.kernels.get(c.op);
      if (custom) custom(xs, shapes, c.attrs, out, c.output.shape);
      else execute(c.op, xs, shapes, c.attrs, out, c.output.shape);
    }
  }
}

function mapOffsets(from: Shape, to: Shape): Int32Array {
  const result = new Int32Array(size(from)),
    fs = strides(from),
    ts = strides(to),
    shift = from.length - to.length;
  for (let i = 0; i < result.length; i++) {
    let j = 0;
    for (let k = 0; k < to.length; k++)
      if (to[k] !== 1) j += (Math.floor(i / fs[k + shift]) % from[k + shift]) * ts[k];
    result[i] = j;
  }
  return result;
}
function reductionMap(shape: Shape, axes: number[]): Int32Array {
  const target = shape.map((d, i) => (axes.includes(i) ? 1 : d));
  return mapOffsets(shape, target);
}
function execute(
  op: string,
  xs: Float32Array[],
  shapes: Shape[],
  attrs: Record<string, unknown>,
  out: Float32Array,
  shape: Shape,
): void {
  const vjp = op === 'vjp';
  if (vjp) op = String(attrs.op);
  const n = Number(attrs.input ?? 0),
    a = xs[0],
    b = xs[1];
  const count = Number(attrs.count ?? xs.length);
  const y = vjp ? xs[count] : out,
    g = vjp ? xs[count + 1] : out;
  const ys = vjp ? shapes[count] : shape;
  if (['add', 'mul', 'max', 'ge'].includes(op)) {
    const ai = mapOffsets(ys, shapes[0]),
      bi = mapOffsets(ys, shapes[1]);
    for (let i = 0; i < y.length; i++) {
      const av = a[ai[i]],
        bv = b[bi[i]];
      if (!vjp)
        out[i] =
          op === 'add'
            ? av + bv
            : op === 'mul'
              ? av * bv
              : op === 'max'
                ? Math.max(av, bv)
                : Number(av >= bv);
      else {
        const d =
          op === 'add'
            ? 1
            : op === 'mul'
              ? n === 0
                ? bv
                : av
              : op === 'ge'
                ? 0
                : Number(n === 0 ? av >= bv : bv > av);
        out[n === 0 ? ai[i] : bi[i]] += g[i] * d;
      }
    }
  } else if (op === 'exp' || op === 'log') {
    for (let i = 0; i < out.length; i++)
      out[i] = vjp
        ? g[i] * (op === 'exp' ? y[i] : 1 / a[i])
        : op === 'exp'
          ? Math.exp(a[i])
          : Math.log(a[i]);
  } else if (op === 'reshape') out.set(vjp ? g : a);
  else if (op === 'transpose') {
    const p = attrs.perm as number[],
      ss = strides(shapes[0]),
      os = strides(ys);
    for (let i = 0; i < y.length; i++) {
      let j = 0;
      for (let k = 0; k < p.length; k++) j += (Math.floor(i / os[k]) % ys[k]) * ss[p[k]];
      if (vjp) out[j] = g[i];
      else out[i] = a[j];
    }
  } else if (op === 'sum' || op === 'reduce_max') {
    const map = reductionMap(shapes[0], axesFor(shapes[0], attrs.axes as number[]));
    if (!vjp) {
      if (op === 'reduce_max') out.fill(-Infinity);
      for (let i = 0; i < a.length; i++)
        if (op === 'sum') out[map[i]] += a[i];
        else out[map[i]] = Math.max(out[map[i]], a[i]);
    } else {
      const seen = new Uint8Array(y.length);
      for (let i = 0; i < a.length; i++) {
        const j = map[i];
        if (op === 'sum') out[i] = g[j];
        else if (!seen[j] && a[i] === y[j]) {
          out[i] = g[j];
          seen[j] = 1;
        }
      }
    }
  } else if (op === 'gather') {
    const [rows, d] = shapes[0];
    for (let i = 0; i < b.length; i++) {
      const row = b[i];
      if (!Number.isInteger(row) || row < 0 || row >= rows) throw new Error('engine.gather.index');
      for (let j = 0; j < d; j++)
        if (!vjp) out[i * d + j] = a[row * d + j];
        else if (n === 0) out[row * d + j] += g[i * d + j];
    }
  } else if (op === 'matmul') {
    const as = shapes[0],
      bs = shapes[1],
      m = as.at(-2)!,
      k = as.at(-1)!,
      d = bs.at(-1)!;
    const batch = ys.slice(0, -2),
      ai = mapOffsets(batch, as.slice(0, -2)),
      bi = mapOffsets(batch, bs.slice(0, -2));
    for (let q = 0; q < size(batch); q++)
      for (let i = 0; i < m; i++)
        for (let j = 0; j < d; j++) {
          const oi = (q * m + i) * d + j;
          let total = 0;
          for (let t = 0; t < k; t++) {
            const ia = (ai[q] * m + i) * k + t,
              ib = (bi[q] * k + t) * d + j;
            if (!vjp) total += a[ia] * b[ib];
            else if (n === 0) out[ia] += g[oi] * b[ib];
            else out[ib] += g[oi] * a[ia];
          }
          if (!vjp) out[oi] = total;
        }
  } else if (op === 'softmax' || op === 'layernorm') {
    const d = shapes[0].at(-1)!;
    for (let start = 0; start < a.length; start += d) {
      let mean = 0,
        variance = 0,
        mx = -Infinity;
      for (let j = 0; j < d; j++) {
        mean += a[start + j];
        mx = Math.max(mx, a[start + j]);
      }
      mean /= d;
      for (let j = 0; j < d; j++) variance += (a[start + j] - mean) ** 2;
      const inv = 1 / Math.sqrt(variance / d + Number(attrs.eps ?? 1e-5));
      let total = 0,
        dot = 0;
      for (let j = 0; j < d; j++) {
        const i = start + j;
        if (vjp) {
          total += g[i];
          dot += g[i] * y[i];
        } else if (op === 'softmax') total += Math.exp(a[i] - mx);
      }
      for (let j = 0; j < d; j++) {
        const i = start + j;
        if (vjp) {
          const centeredGradient = g[i] - total / d - (y[i] * dot) / d;
          out[i] = op === 'softmax' ? y[i] * (g[i] - dot) : inv * centeredGradient;
        } else {
          out[i] = op === 'softmax' ? Math.exp(a[i] - mx) / total : (a[i] - mean) * inv;
        }
      }
    }
  } else throw new Error(`engine.op.unknown:${op}`);
}
