import {
  GraphError,
  type Attributes,
  type ComponentAttributeDefinition,
  type ComponentDefinition,
  type GraphJSON,
  type ShapePorts,
} from './graph';
import { clampCount, MAX_PORTS, SPLIT_OUTPUTS, TENSOR_OP_PORTS } from './tensor-op-ports';

const int = (
  name: string,
  value: number,
  min: number,
  max: number,
): ComponentAttributeDefinition => ({ name, kind: 'integer', default: value, min, max });

class Builder {
  readonly graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  private n = 0;
  node(component: string, inputs: Record<string, string> = {}, attrs?: Attributes, id?: string) {
    const name = id ?? `${component}_${this.n++}`;
    this.graph.nodes.push({ id: name, component, ...(attrs ? { attrs } : {}) });
    for (const [port, source] of Object.entries(inputs))
      this.graph.edges.push({ from: { node: source, port: 'out' }, to: { node: name, port } });
    return name;
  }
  input(name: string) {
    return this.node('input', {}, { name }, name);
  }
  output(name: string, x: string) {
    this.node('output', { x }, { name }, `output_${name}`);
  }
}

const normalizeAxis = (axis: unknown, rank: number, port?: string): number => {
  const a = typeof axis === 'number' ? axis : 0;
  if (!Number.isInteger(a) || a < -rank || a >= rank)
    throw new GraphError('tensor_axis', undefined, port, { axis: a, rank });
  return a < 0 ? a + rank : a;
};
const product = (dims: readonly number[]) => dims.reduce((p, d) => p * d, 1);
const inverse = (perm: readonly number[]) => {
  const out: number[] = [];
  perm.forEach((p, i) => (out[p] = i));
  return out;
};
const moveFront = (axis: number, rank: number) => [
  axis,
  ...Array.from({ length: rank }, (_, i) => i).filter((i) => i !== axis),
];
const range = (start: number, length: number) => Array.from({ length }, (_, i) => start + i);

/** x [..n..] → rows [n, R] with the chosen axis first. */
function rows(b: Builder, x: string, shape: readonly number[], axis: number): string {
  const perm = moveFront(axis, shape.length);
  const moved = axis === 0 ? x : b.node('transpose', { x }, { perm });
  const rest = perm.slice(1).map((i) => shape[i]);
  return b.node('reshape', { x: moved }, { shape: [shape[axis], product(rest)] });
}
/** rows [m, R] → original layout with the axis (now length m) back in place. */
function unrows(b: Builder, r: string, shape: readonly number[], axis: number, m: number): string {
  const perm = moveFront(axis, shape.length);
  const rest = perm.slice(1).map((i) => shape[i]);
  const front = b.node('reshape', { x: r }, { shape: [m, ...rest] });
  return axis === 0 ? front : b.node('transpose', { x: front }, { perm: inverse(perm) });
}
function take(
  b: Builder,
  x: string,
  shape: readonly number[],
  axis: number,
  start: number,
  length: number,
  keep: boolean,
): string {
  const idx = b.node('const', {}, { value: range(start, length), shape: [length] });
  const picked = b.node('gather', { table: rows(b, x, shape, axis), idx });
  if (keep) return unrows(b, picked, shape, axis, length);
  return b.node('reshape', { x: picked }, { shape: shape.filter((_, i) => i !== axis) });
}

function slice(attrs: Attributes, inputs?: ShapePorts): GraphJSON {
  const b = new Builder();
  const x = b.input('x');
  if (!inputs?.x) {
    b.output('out', x);
    return b.graph;
  }
  const shape = inputs.x;
  if (!shape.length) throw new GraphError('tensor_axis', undefined, 'x', { axis: 0, rank: 0 });
  const axis = normalizeAxis(attrs.axis, shape.length, 'x');
  const n = shape[axis];
  const length = Number(attrs.length ?? 1);
  const raw = Number(attrs.index ?? 0);
  const start = raw < 0 ? raw + n : raw;
  if (start < 0 || start + length > n)
    throw new GraphError('slice_range', undefined, 'x', {
      shape,
      axis,
      start: raw,
      end: raw + length,
      size: n,
    });
  b.output('out', take(b, x, shape, axis, start, length, length !== 1 || attrs.keepdims === true));
  return b.graph;
}

function connectedParts(attrs: Attributes, inputs?: ShapePorts): string[] {
  const count = clampCount(attrs.count, 2);
  const ports = TENSOR_OP_PORTS.slice(0, count);
  if (inputs)
    for (const port of Object.keys(inputs))
      if (!ports.includes(port as (typeof ports)[number]))
        throw new GraphError('unknown_input_port', undefined, port);
  if (inputs)
    for (const port of ports)
      if (!inputs[port]) throw new GraphError('missing_port', undefined, port);
  return ports;
}

function concatGraph(
  b: Builder,
  parts: readonly { id: string; shape: readonly number[]; port: string }[],
  axis: number,
  key: 'concat_shape' | 'stack_shape',
): string {
  const first = parts[0].shape;
  for (const part of parts) {
    const same =
      part.shape.length === first.length &&
      part.shape.every((d, i) => (key === 'concat_shape' && i === axis) || d === first[i]);
    if (!same) throw new GraphError(key, undefined, part.port, { a: first, b: part.shape, axis });
  }
  if (parts.length === 1) return parts[0].id;
  const total = parts.reduce((t, p) => t + p.shape[axis], 0);
  const rest = product(first.filter((_, i) => i !== axis));
  const table = b.node('const', {}, { value: 0, shape: [total, rest] });
  let offset = 0;
  let sum: string | undefined;
  for (const part of parts) {
    const n = part.shape[axis];
    const idx = b.node('const', {}, { value: range(offset, n), shape: [n] });
    const placed = b.node('scatter_add', {
      table,
      idx,
      updates: rows(b, part.id, part.shape, axis),
    });
    sum = sum ? b.node('add', { a: sum, b: placed }) : placed;
    offset += n;
  }
  const outShape = first.map((d, i) => (i === axis ? total : d));
  return unrows(b, sum!, outShape, axis, total);
}

function joined(kind: 'stack' | 'concat', attrs: Attributes, inputs?: ShapePorts): GraphJSON {
  const b = new Builder();
  const ports = connectedParts(attrs, inputs);
  const ids = ports.map((port) => b.input(port));
  if (!inputs) {
    b.output('out', ids[0]);
    return b.graph;
  }
  const rank = (inputs.a ?? []).length + (kind === 'stack' ? 1 : 0);
  if (kind === 'concat' && rank === 0)
    throw new GraphError('tensor_axis', undefined, 'a', { axis: Number(attrs.axis ?? 0), rank });
  const axis = normalizeAxis(attrs.axis, rank);
  const parts = ports.map((port, i) => {
    const shape = inputs[port] ?? [];
    if (kind === 'concat') return { id: ids[i], shape, port };
    if (shape.length !== (inputs.a ?? []).length)
      throw new GraphError('stack_shape', undefined, port, { a: inputs.a, b: shape, axis });
    const lifted = [...shape];
    lifted.splice(axis, 0, 1);
    return { id: b.node('unsqueeze', { x: ids[i] }, { axis }), shape: lifted, port };
  });
  b.output('out', concatGraph(b, parts, axis, kind === 'stack' ? 'stack_shape' : 'concat_shape'));
  return b.graph;
}

function split(attrs: Attributes, inputs?: ShapePorts): GraphJSON {
  const b = new Builder();
  const x = b.input('x');
  const parts = clampCount(attrs.parts, 2);
  const shape = inputs?.x;
  const outs: string[] = [];
  if (shape) {
    if (!shape.length) throw new GraphError('tensor_axis', undefined, 'x', { axis: 0, rank: 0 });
    const axis = normalizeAxis(attrs.axis, shape.length, 'x');
    const n = shape[axis];
    if (n % parts)
      throw new GraphError('split_uneven', undefined, 'x', { shape, axis, size: n, parts });
    const width = n / parts;
    for (let i = 0; i < parts; i++) outs.push(take(b, x, shape, axis, i * width, width, true));
  }
  // Every declared output must exist; ports beyond `parts` are hidden and carry a scalar zero.
  const unused = b.node('const', {}, { value: 0, shape: [] }, 'unused');
  SPLIT_OUTPUTS.forEach((port, i) => b.output(port, outs[i] ?? (shape ? unused : x)));
  return b.graph;
}

const keepdims: ComponentAttributeDefinition = {
  name: 'keepdims',
  kind: 'boolean',
  default: false,
};

export const tensorOpComponents: ComponentDefinition[] = [
  {
    name: 'slice',
    inputs: ['x'],
    outputs: ['out'],
    attributes: [
      int('axis', 1, -8, 7),
      int('index', 0, -4096, 4096),
      int('length', 1, 1, 4096),
      keepdims,
    ],
    defaultAttributes: { axis: 1, index: 0, length: 1, keepdims: false },
    builtin: true,
    subgraph: (attrs, inputs) => slice(attrs, inputs),
  },
  {
    name: 'stack',
    inputs: [...TENSOR_OP_PORTS],
    optionalInputs: TENSOR_OP_PORTS.slice(2),
    outputs: ['out'],
    attributes: [int('count', 2, 2, MAX_PORTS), int('axis', 1, -9, 8)],
    defaultAttributes: { count: 2, axis: 1 },
    builtin: true,
    subgraph: (attrs, inputs) => joined('stack', attrs, inputs),
  },
  {
    name: 'concat',
    inputs: [...TENSOR_OP_PORTS],
    optionalInputs: TENSOR_OP_PORTS.slice(2),
    outputs: ['out'],
    attributes: [int('count', 2, 2, MAX_PORTS), int('axis', -1, -8, 7)],
    defaultAttributes: { count: 2, axis: -1 },
    builtin: true,
    subgraph: (attrs, inputs) => joined('concat', attrs, inputs),
  },
  {
    name: 'split',
    inputs: ['x'],
    outputs: SPLIT_OUTPUTS,
    attributes: [int('parts', 2, 2, MAX_PORTS), int('axis', -1, -8, 7)],
    defaultAttributes: { parts: 2, axis: -1 },
    builtin: true,
    subgraph: (attrs, inputs) => split(attrs, inputs),
  },
];
