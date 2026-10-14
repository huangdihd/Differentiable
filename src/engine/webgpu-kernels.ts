import { axesFor, type Command, type Shape, size, strides } from './backend';

/** Shape-specialized kernels. Each output has one writer, including adjoints. */
export function kernel(command: Command): {
  code: string;
  workgroups: [number, number?, number?];
} {
  const { inputs, output, attrs } = command;
  const vjp = command.op === 'vjp';
  const op = vjp ? String(attrs.op) : command.op;
  const count = Number(attrs.count ?? inputs.length);
  const input = Number(attrs.input ?? 0);
  const a = inputs[0]?.shape ?? [];
  const b = inputs[1]?.shape ?? [];
  const ys = vjp ? inputs[count].shape : output.shape;
  const y = `x${count}`;
  const g = `x${count + 1}`;
  const declarations = inputs
    .map((_, i) => `@group(0) @binding(${i}) var<storage, read> x${i}: array<f32>;`)
    .join('\n');
  const header = `${declarations}
@group(0) @binding(${inputs.length}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${inputs.length + 1}) var<storage, read_write> error: array<atomic<u32>>;
// WGSL rejects non-finite const-expressions, so special values go through runtime bit patterns.
var<private> POS_INF_BITS: u32 = 0x7f800000u;
var<private> NEG_INF_BITS: u32 = 0xff800000u;
var<private> NAN_BITS: u32 = 0x7fc00000u;
`;
  const finish = (body: string, length = size(output.shape), before = '') => ({
    code: `${header}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  ${before}
  if (i >= ${u(length)}) { return; }
  ${body}
}`,
    workgroups: [Math.max(1, Math.ceil(length / 64))] as [number],
  });

  if (['add', 'mul', 'max', 'ge'].includes(op)) {
    const values = (index: string) => `let av = x0[${mapped(index, ys, a)}];
      let bv = x1[${mapped(index, ys, b)}];`;
    if (!vjp) {
      const expression = {
        add: 'av + bv',
        mul: 'av * bv',
        max: 'max(av, bv)',
        ge: 'select(0.0, 1.0, av >= bv)',
      }[op];
      return finish(`${values('i')} out[i] = ${expression};`);
    }
    const fiber = broadcastFiber('i', ys, input === 0 ? a : b);
    const derivative =
      op === 'add'
        ? '1.0'
        : op === 'mul'
          ? input === 0
            ? 'bv'
            : 'av'
          : op === 'ge'
            ? '0.0'
            : `select(0.0, 1.0, ${input === 0 ? 'av >= bv' : 'bv > av'})`;
    return finish(`var total = 0.0;
      for (var r = 0u; r < ${u(fiber.length)}; r++) {
        let j = ${fiber.index};
        ${values('j')}
        total += ${g}[j] * ${derivative};
      }
      out[i] = total;`);
  }
  if (op === 'exp' || op === 'log') {
    return finish(
      `out[i] = ${
        vjp ? `${g}[i] * (${op === 'exp' ? `${y}[i]` : '1.0 / x0[i]'})` : `${op}(x0[i])`
      };`,
    );
  }
  if (op === 'reshape') return finish(`out[i] = ${vjp ? g : 'x0'}[i];`);
  if (op === 'transpose') {
    const perm = attrs.perm as number[];
    const source = vjp ? a : ys;
    const targetStrides = strides(vjp ? ys : a);
    const terms = perm.map(
      (axis, k) => `${coord('i', source, vjp ? axis : k)} * ${u(targetStrides[vjp ? k : axis])}`,
    );
    return finish(`out[i] = ${vjp ? g : 'x0'}[${sum(terms)}];`);
  }
  if (op === 'sum' || op === 'reduce_max') {
    const axes = axesFor(a, attrs.axes as number[]);
    const reduced = a.map((d, axis) => (axes.includes(axis) ? 1 : d));
    const fiber = broadcastFiber('i', a, reduced);
    if (vjp && op === 'sum') {
      return finish(`out[i] = ${g}[${mapped('i', a, reduced)}];`);
    }
    if (vjp) {
      // One invocation owns an entire reduction fiber: first equal maximum wins.
      return finish(
        `var seen = false;
        for (var r = 0u; r < ${u(fiber.length)}; r++) {
          let j = ${fiber.index};
          var value = 0.0;
          if (!seen && x0[j] == ${y}[i]) { value = ${g}[i]; seen = true; }
          out[j] = value;
        }`,
        size(ys),
      );
    }
    return finish(`var total = ${op === 'sum' ? '0.0' : 'bitcast<f32>(NEG_INF_BITS)'};
      for (var r = 0u; r < ${u(fiber.length)}; r++) {
        let j = ${fiber.index};
        total = ${op === 'sum' ? 'total + x0[j]' : 'max(total, x0[j])'};
      }
      out[i] = total;`);
  }
  if (op === 'gather') {
    const [rows, width] = a;
    const indices = size(b);
    // Validate independently of output size (including a zero-width table).
    const validation = `if (i < ${u(indices)}) {
      let row = x1[i];
      if (!(row >= 0.0 && row < ${f(rows)} && row == floor(row))) {
        atomicStore(&error[0], 1u);
      }
    }`;
    let body: string;
    if (!vjp) {
      body = `let row = x1[i / ${u(Math.max(1, width))}];
        if (row >= 0.0 && row < ${f(rows)} && row == floor(row)) {
          out[i] = x0[u32(row) * ${u(width)} + i % ${u(Math.max(1, width))}];
        } else { out[i] = 0.0; }`;
    } else if (input !== 0) body = 'out[i] = 0.0;';
    else {
      body = `var total = 0.0;
        for (var j = 0u; j < ${u(indices)}; j++) {
          if (x1[j] == f32(i / ${u(Math.max(1, width))})) {
            total += ${g}[j * ${u(width)} + i % ${u(Math.max(1, width))}];
          }
        }
        out[i] = total;`;
    }
    const result = finish(body, size(output.shape), validation);
    result.workgroups = [Math.max(1, Math.ceil(Math.max(indices, size(output.shape)) / 64))];
    return result;
  }
  if (op === 'matmul') {
    const m = a.at(-2)!;
    const k = a.at(-1)!;
    const n = b.at(-1)!;
    const batch = ys.slice(0, -2);
    const ab = a.slice(0, -2);
    const bb = b.slice(0, -2);
    const targetBatch = vjp ? (input === 0 ? ab : bb) : batch;
    const rows = vjp && input !== 0 ? k : m;
    const cols = vjp && input === 0 ? k : n;
    const inner = vjp ? (input === 0 ? n : m) : k;
    const fiber = broadcastFiber('batchIndex', batch, targetBatch);
    const ai = mapped('q', batch, ab);
    const bi = mapped('q', batch, bb);
    const lhs = !vjp
      ? `x0[(${ai} * ${u(m)} + row) * ${u(k)} + lt]`
      : input === 0
        ? `${g}[(q * ${u(m)} + row) * ${u(n)} + lt]`
        : `x0[(${ai} * ${u(m)} + lt) * ${u(k)} + row]`;
    const rhs = !vjp
      ? `x1[(${bi} * ${u(k)} + rt) * ${u(n)} + col]`
      : input === 0
        ? `x1[(${bi} * ${u(k)} + col) * ${u(n)} + rt]`
        : `${g}[(q * ${u(m)} + rt) * ${u(n)} + col]`;
    // All lanes participate in barriers, including edge tiles. Batch adjoints
    // reduce only their broadcast fibers, in ascending forward-batch order.
    return {
      code: `${header}
var<workgroup> leftTile: array<f32, 256>;
var<workgroup> rightTile: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.y * 16u + local.y;
  let col = group.x * 16u + local.x;
  let batchIndex = group.z;
  let lane = local.y * 16u + local.x;
  var total = 0.0;
  for (var r = 0u; r < ${u(fiber.length)}; r++) {
    let q = ${fiber.index};
    for (var base = 0u; base < ${u(inner)}; base += 16u) {
      let lt = base + local.x;
      let rt = base + local.y;
      leftTile[lane] = 0.0;
      rightTile[lane] = 0.0;
      if (batchIndex < ${u(size(targetBatch))} && row < ${u(rows)} && lt < ${u(inner)}) {
        leftTile[lane] = ${lhs};
      }
      if (batchIndex < ${u(size(targetBatch))} && col < ${u(cols)} && rt < ${u(inner)}) {
        rightTile[lane] = ${rhs};
      }
      workgroupBarrier();
      for (var t = 0u; t < min(16u, ${u(inner)} - base); t++) {
        total += leftTile[local.y * 16u + t] * rightTile[t * 16u + local.x];
      }
      workgroupBarrier();
    }
  }
  if (batchIndex < ${u(size(targetBatch))} && row < ${u(rows)} && col < ${u(cols)}) {
    out[(batchIndex * ${u(rows)} + row) * ${u(cols)} + col] = total;
  }
}`,
      workgroups: [
        Math.max(1, Math.ceil(cols / 16)),
        Math.max(1, Math.ceil(rows / 16)),
        Math.max(1, size(targetBatch)),
      ],
    };
  }
  if (op === 'softmax' || op === 'layernorm') {
    const width = a.at(-1)!;
    const stats =
      op === 'softmax'
        ? vjp
          ? ''
          : `var mx = bitcast<f32>(NEG_INF_BITS);
          for (var j = 0u; j < ${u(width)}; j++) { mx = max(mx, x0[start + j]); }`
        : `var mean = 0.0;
        for (var j = 0u; j < ${u(width)}; j++) { mean += x0[start + j]; }
        mean /= ${f(width)};
        var variance = 0.0;
        for (var j = 0u; j < ${u(width)}; j++) {
          let centered = x0[start + j] - mean;
          variance += centered * centered;
        }
        let inv = 1.0 / sqrt(variance / ${f(width)} + ${f(Number(attrs.eps ?? 1e-5))});`;
    const totals = vjp
      ? `var total = 0.0;
        var dot = 0.0;
        for (var j = 0u; j < ${u(width)}; j++) {
          total += ${g}[start + j];
          dot += ${g}[start + j] * ${y}[start + j];
        }`
      : op === 'softmax'
        ? `var total = 0.0;
          for (var j = 0u; j < ${u(width)}; j++) { total += exp(x0[start + j] - mx); }`
        : '';
    const expression = vjp
      ? op === 'softmax'
        ? `${y}[j] * (${g}[j] - dot)`
        : `inv * (${g}[j] - total / ${f(width)} - (${y}[j] * dot) / ${f(width)})`
      : op === 'softmax'
        ? 'exp(x0[j] - mx) / total'
        : '(x0[j] - mean) * inv';
    return finish(
      `let start = i * ${u(width)};
      ${stats}
      ${totals}
      for (var c = 0u; c < ${u(width)}; c++) {
        let j = start + c;
        out[j] = ${expression};
      }`,
      size(a) / width,
    );
  }
  throw new Error(`engine.op.unknown:${op}`);
}

function u(value: number): string {
  return `${value}u`;
}
function f(value: number): string {
  if (value === Infinity) return 'bitcast<f32>(POS_INF_BITS)';
  if (value === -Infinity) return 'bitcast<f32>(NEG_INF_BITS)';
  if (Number.isNaN(value)) return 'bitcast<f32>(NAN_BITS)';
  const text = String(value);
  return /[.eE]/.test(text) ? `${text}f` : `${text}.0`;
}
function sum(terms: string[]): string {
  return terms.length ? `(${terms.join(' + ')})` : '0u';
}
function coord(index: string, shape: Shape, axis: number): string {
  // Zero-volume tensors still produce syntactically valid, unreachable indexing.
  return `(${index} / ${u(Math.max(1, strides(shape)[axis]))} % ${u(Math.max(1, shape[axis]))})`;
}
function mapped(index: string, from: Shape, to: Shape): string {
  const shift = from.length - to.length;
  const ts = strides(to);
  return sum(
    to.flatMap((d, axis) =>
      d === 1 ? [] : [`${coord(index, from, axis + shift)} * ${u(ts[axis])}`],
    ),
  );
}

/** Enumerate only expanded axes, not every element of the broadcast result.
 * r traverses the fiber in ascending flat-index order for deterministic sums. */
function broadcastFiber(index: string, from: Shape, to: Shape): { index: string; length: number } {
  const shift = from.length - to.length;
  const expanded = from.map((d, axis) => ((to[axis - shift] ?? 1) === 1 ? d : 1));
  const fs = strides(from);
  const terms = from.map((_, axis) => {
    const targetAxis = axis - shift;
    const coordinate =
      (to[targetAxis] ?? 1) === 1 ? coord('r', expanded, axis) : coord(index, to, targetAxis);
    return `${coordinate} * ${u(fs[axis])}`;
  });
  return { index: sum(terms), length: size(expanded) };
}
