export type Shape = readonly number[];
export type Attributes = Record<string, unknown>;
/** Opaque backend-owned storage identity, never a host array. */
export interface Handle {
  readonly id: number;
}
export interface Value {
  readonly handle: Handle;
  readonly shape: Shape;
}
export interface Command {
  readonly op: string;
  readonly inputs: readonly Value[];
  readonly output: Value;
  readonly attrs: Attributes;
}
/** Commands are ordered, may be queued by a GPU implementation. read is a fence.
 * release must defer recycling until queued users complete. */
export interface Backend {
  allocate(shape: Shape, data?: ArrayLike<number>): Handle;
  submit(commands: readonly Command[]): void;
  read(handle: Handle): Promise<Float32Array>;
  release(handle: Handle): void;
  dispose?(): void;
}
export function size(shape: Shape): number {
  if (shape.some((d) => !Number.isSafeInteger(d) || d < 0)) throw new Error('engine.shape.invalid');
  return shape.reduce((a, b) => a * b, 1);
}
export function strides(shape: Shape): number[] {
  return shape.map((_, i) => size(shape.slice(i + 1)));
}
export function broadcast(a: Shape, b: Shape): number[] {
  const n = Math.max(a.length, b.length);
  return Array.from({ length: n }, (_, i) => {
    const x = a[i - n + a.length] ?? 1,
      y = b[i - n + b.length] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error('engine.shape.broadcast');
    return x === 1 ? y : x;
  });
}
export function coordinates(i: number, shape: Shape): number[] {
  return strides(shape).map((s, k) => Math.floor(i / s) % shape[k]);
}
export function offset(coords: readonly number[], shape: Shape): number {
  const s = strides(shape),
    shift = coords.length - shape.length;
  return s.reduce((n, v, k) => n + (shape[k] === 1 ? 0 : coords[k + shift]) * v, 0);
}
export function axesFor(shape: Shape, axes: number[]): number[] {
  const result = axes.map((a) => (a < 0 ? a + shape.length : a));
  if (
    new Set(result).size !== result.length ||
    result.some((a) => !Number.isInteger(a) || a < 0 || a >= shape.length)
  )
    throw new Error('engine.axis.invalid');
  return result;
}
export function infer(op: string, shapes: Shape[], attrs: Attributes = {}): number[] {
  const a = shapes[0] ?? [];
  if (['add', 'mul', 'max', 'ge'].includes(op)) return broadcast(a, shapes[1]);
  if (['exp', 'log', 'recip', 'softmax', 'layernorm'].includes(op)) {
    if (['softmax', 'layernorm'].includes(op) && (!a.length || a[a.length - 1] === 0))
      throw new Error('engine.shape.lastAxis');
    if (op === 'layernorm' && !(Number(attrs.eps ?? 1e-5) > 0))
      throw new Error('engine.eps.invalid');
    return [...a];
  }
  if (op === 'reshape') {
    const shape = attrs.shape as number[];
    if (size(shape) !== size(a)) throw new Error('engine.shape.reshape');
    return [...shape];
  }
  if (op === 'transpose') {
    const p = attrs.perm as number[];
    if (
      p.length !== a.length ||
      new Set(p).size !== a.length ||
      p.some((v) => !Number.isInteger(v) || v < 0 || v >= a.length)
    )
      throw new Error('engine.shape.transpose');
    return p.map((v) => a[v]);
  }
  if (op === 'sum' || op === 'reduce_max') {
    const axes = axesFor(a, attrs.axes as number[]);
    if (op === 'reduce_max' && axes.some((i) => a[i] === 0))
      throw new Error('engine.shape.emptyReduction');
    return a.flatMap((d, i) => (axes.includes(i) ? (attrs.keepdims ? [1] : []) : [d]));
  }
  if (op === 'scatter_add') {
    const gathered = infer('gather', shapes);
    if (gathered.length !== shapes[2].length || gathered.some((d, i) => d !== shapes[2][i]))
      throw new Error('engine.shape.gather');
    return [...a];
  }
  if (op === 'gather') {
    if (a.length !== 2) throw new Error('engine.shape.gather');
    return [...shapes[1], a[1]];
  }
  if (op === 'matmul') {
    const b = shapes[1];
    if (a.length < 2 || b.length < 2 || a.at(-1) !== b.at(-2))
      throw new Error('engine.shape.matmul');
    return [...broadcast(a.slice(0, -2), b.slice(0, -2)), a.at(-2)!, b.at(-1)!];
  }
  throw new Error('engine.op.unknown');
}
