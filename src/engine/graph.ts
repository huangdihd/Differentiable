import * as T from './tensor';
import type { Tensor } from './tensor';
import type { Random } from './random';
import { standardComponents } from './standard-components';

type Backend = Tensor['backend'];
export type Shape = readonly number[];
export type Attributes = Record<string, unknown>;
export interface GraphNode {
  id: string;
  component: string;
  attrs?: Attributes;
  ui?: Record<string, unknown>;
}
export interface GraphEndpoint {
  node: string;
  port: string;
}
export interface GraphEdge {
  from: GraphEndpoint;
  to: GraphEndpoint;
}
export interface GraphJSON {
  version: 1;
  nodes: GraphNode[];
  edges: GraphEdge[];
}
export type TensorPorts = Record<string, Tensor>;
export type ShapePorts = Record<string, Shape>;
export interface ExecutionContext {
  backend?: Backend;
  random?: Pick<Random, 'uniform' | 'normal'>;
}
export interface ComponentImplementation {
  infer(inputs: ShapePorts, attrs: Attributes): ShapePorts;
  execute(inputs: TensorPorts, attrs: Attributes, context: ExecutionContext): TensorPorts;
}
export interface ComponentDefinition {
  name: string;
  inputs: readonly string[];
  outputs: readonly string[];
  primitive?: ComponentImplementation;
  native?: ComponentImplementation;
  subgraph?: GraphJSON | ((attrs: Attributes, inputs?: ShapePorts) => GraphJSON);
}
export class GraphError extends Error {
  readonly name = 'GraphError';
  constructor(
    public readonly key: string,
    public readonly node?: string,
    public readonly port?: string,
    public readonly params: Record<string, unknown> = {},
  ) {
    super(`${key}${node ? ` at ${node}` : ''}${port ? `:${port}` : ''}`);
  }
}
export class ComponentRegistry {
  private readonly components = new Map<string, ComponentDefinition>();
  register(component: ComponentDefinition): this {
    if (this.components.has(component.name))
      throw new GraphError('duplicate_component', undefined, undefined, {
        component: component.name,
      });
    for (const ports of [component.inputs, component.outputs]) {
      if (new Set(ports).size !== ports.length) throw new GraphError('duplicate_port');
    }
    this.components.set(component.name, component);
    return this;
  }
  get(name: string): ComponentDefinition | undefined {
    return this.components.get(name);
  }
  list(): ComponentDefinition[] {
    return [...this.components.values()];
  }
}
export interface CompileOptions extends ExecutionContext {
  registry?: ComponentRegistry;
  inputShapes?: ShapePorts;
  preferNative?: boolean;
  availableComponents?: readonly string[];
  parameters?: Map<string, Tensor>;
}
interface Step {
  id: string;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
  attrs: Attributes;
  implementation: ComponentImplementation;
}
interface StorageSpec {
  shape: Shape;
  attrs: Attributes;
  parameter: boolean;
}
const sameShape = (a: Shape, b: Shape) => a.length === b.length && a.every((d, i) => d === b[i]);
const size = (shape: Shape) => shape.reduce((a, b) => a * b, 1);
function shape(value: unknown): number[] {
  if (!Array.isArray(value) || !value.every((d) => Number.isInteger(d) && d >= 0))
    throw new GraphError('invalid_shape', undefined, undefined, { shape: value });
  return [...value];
}
function requiredName(attrs: Attributes): string {
  if (typeof attrs.name !== 'string' || !attrs.name) throw new GraphError('missing_name');
  return attrs.name;
}
function contextual(error: unknown, node: string): GraphError {
  return error instanceof GraphError
    ? new GraphError(error.key, error.node ?? node, error.port, error.params)
    : new GraphError('component_error', node, undefined, {
        message: error instanceof Error ? error.message : String(error),
      });
}

/** Compiled, allocation-free shape analysis with lazy persistent parameter storage.
 * Input tensors belong to the caller. Execution outputs/intermediates are not disposed
 * automatically because the tensor autodiff tape may still reference them.
 */
export class GraphPlan {
  readonly parameters: Map<string, Tensor>;
  private readonly constants = new Map<string, Tensor>();
  private readonly ownedParameters = new Set<string>();
  private readonly executions = new WeakMap<TensorPorts, Set<Tensor>>();
  private disposed = false;
  constructor(
    readonly inputShapes: ShapePorts,
    readonly outputShapes: ShapePorts,
    private readonly inputs: Record<string, string>,
    private readonly outputs: Record<string, string>,
    private readonly steps: Step[],
    private readonly storage: Map<string, StorageSpec>,
    private readonly options: CompileOptions,
  ) {
    this.parameters = options.parameters ?? new Map();
  }
  execute(inputs: TensorPorts): TensorPorts {
    if (this.disposed) throw new GraphError('disposed_plan');
    const values = new Map<string, Tensor>();
    for (const [name, id] of Object.entries(this.inputs)) {
      const value = inputs[name];
      if (!value) throw new GraphError('missing_input', undefined, name);
      if (!sameShape(value.shape, this.inputShapes[name]))
        throw new GraphError('input_shape_mismatch', undefined, name, {
          expected: this.inputShapes[name],
          actual: value.shape,
        });
      values.set(id, value);
    }
    for (const [id, spec] of this.storage) {
      const store = spec.parameter ? this.parameters : this.constants;
      let value = store.get(id);
      if (value && !sameShape(value.shape, spec.shape))
        throw new GraphError('parameter_shape_mismatch', id);
      if (!value) {
        const n = size(spec.shape);
        const data = new Float32Array(n);
        const init = spec.attrs.init ?? 'zeros';
        if (!spec.parameter) {
          const source = spec.attrs.value;
          if (typeof source === 'number') data.fill(source);
          else data.set(source as number[]);
        } else if (init === 'constant') data.fill(spec.attrs.value as number);
        else if (init === 'ones') data.fill(1);
        else if (init === 'normal' || init === 'uniform') {
          if (!this.options.random) throw new GraphError('random_required', id);
          const scale = typeof spec.attrs.scale === 'number' ? spec.attrs.scale : 1;
          for (let i = 0; i < n; i++)
            data[i] =
              (init === 'normal' ? this.options.random.normal() : this.options.random.uniform()) *
              scale;
        }
        value = T.tensor(data, [...spec.shape], this.options.backend);
        store.set(id, value);
        if (spec.parameter) this.ownedParameters.add(id);
      }
      values.set(id, value);
    }
    const allocated = new Set<Tensor>();
    const protectedTensors = new Set([
      ...Object.values(inputs),
      ...this.parameters.values(),
      ...this.constants.values(),
    ]);
    const protect = (value: Tensor): void => {
      if (protectedTensors.has(value)) return;
      protectedTensors.add(value);
      value.tape?.inputs.forEach(protect);
    };
    // Caller inputs can themselves have tapes: never claim their ancestors.
    for (const value of [...protectedTensors]) value.tape?.inputs.forEach(protect);
    const track = (value: Tensor): void => {
      if (protectedTensors.has(value) || allocated.has(value)) return;
      allocated.add(value);
      value.tape?.inputs.forEach(track);
    };
    try {
      for (const step of this.steps) {
        try {
          const args = Object.fromEntries(
            Object.entries(step.inputs).map(([port, id]) => [port, values.get(id)!]),
          );
          const result = step.implementation.execute(args, step.attrs, this.options);
          Object.values(result).forEach(track);
          for (const [port, id] of Object.entries(step.outputs)) {
            if (!result[port]) throw new GraphError('missing_output', step.id, port);
            values.set(id, result[port]);
          }
        } catch (error) {
          throw contextual(error, step.id);
        }
      }
      const result = Object.fromEntries(
        Object.entries(this.outputs).map(([name, id]) => [name, values.get(id)!]),
      );
      this.executions.set(result, allocated);
      return result;
    } catch (error) {
      for (const value of allocated) value.dispose();
      throw error;
    }
  }
  /** Release one execution after autodiff, including disconnected step results.
   * Pass the exact object returned by execute. Inputs and persistent storage are
   * always protected; keep additionally retains caller-selected tensors.
   * Custom implementations must clean up allocations they never return.
   */
  releaseExecution(outputs: TensorPorts, keep: readonly Tensor[] | TensorPorts = []): void {
    const allocated = this.executions.get(outputs);
    if (!allocated) return;
    const retained = new Set([
      ...Object.values(keep),
      ...this.parameters.values(),
      ...this.constants.values(),
    ]);
    for (const value of allocated) if (!retained.has(value)) value.dispose();
    this.executions.delete(outputs);
  }
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const tensor of this.constants.values()) tensor.dispose();
    this.constants.clear();
    for (const id of this.ownedParameters) {
      this.parameters.get(id)?.dispose();
      this.parameters.delete(id);
    }
    this.ownedParameters.clear();
  }
}

export function compileGraph(graph: GraphJSON, options: CompileOptions = {}): GraphPlan {
  const registry = options.registry ?? createDefaultRegistry();
  const steps: Step[] = [];
  const storage = new Map<string, StorageSpec>();
  const shapes = new Map<string, Shape>();
  const externalInputs: Record<string, string> = Object.create(null);
  const inputShapes: ShapePorts = Object.create(null);
  const active = new Set<string>();
  const available = options.availableComponents && new Set(options.availableComponents);
  function checkAvailable(node: GraphNode, id: string): void {
    if (
      available &&
      !['input', 'output'].includes(node.component) &&
      !available.has(node.component)
    ) {
      throw new GraphError('component_unavailable', id, undefined, { component: node.component });
    }
  }
  // Native implementations have no parameter-binding interface yet.
  // This inspection is independent of the top-level palette policy.
  function inspect(g: GraphJSON, prefix: string, seen: Set<string>): boolean {
    let parameters = false;
    for (const node of g.nodes) {
      const id = `${prefix}/${encodeURIComponent(node.id)}`;
      if (node.component === 'param') parameters = true;
      const def = registry.get(node.component);
      if (def?.subgraph) {
        if (seen.has(def.name)) throw new GraphError('recursive_component', id);
        const next = new Set(seen);
        next.add(def.name);
        const nested =
          typeof def.subgraph === 'function' ? def.subgraph(node.attrs ?? {}) : def.subgraph;
        if (inspect(nested, id, next)) parameters = true;
      }
    }
    return parameters;
  }
  function compile(
    g: GraphJSON,
    prefix: string,
    bindings?: Record<string, string>,
  ): Record<string, string> {
    if (g.version !== 1)
      throw new GraphError('unsupported_version', prefix, undefined, { version: g.version });
    const nodes = new Map<string, GraphNode>();
    const incoming = new Map<string, Map<string, GraphEndpoint>>();
    const resolved = new Map<string, Record<string, string>>();
    const visiting = new Set<string>();
    const names = new Set<string>();
    const result: Record<string, string> = Object.create(null);
    const path = (id: string) =>
      prefix ? `${prefix}/${encodeURIComponent(id)}` : encodeURIComponent(id);
    function ports(n: GraphNode): { inputs: readonly string[]; outputs: readonly string[] } {
      if (['input', 'const', 'param'].includes(n.component))
        return { inputs: [], outputs: ['out'] };
      if (n.component === 'output') return { inputs: ['x'], outputs: [] };
      const def = registry.get(n.component);
      if (!def)
        throw new GraphError('unknown_component', path(n.id), undefined, {
          component: n.component,
        });
      return def;
    }
    for (const node of g.nodes) {
      if (!bindings) checkAvailable(node, path(node.id));
      if (!node.id || nodes.has(node.id)) throw new GraphError('duplicate_node', path(node.id));
      nodes.set(node.id, node);
      incoming.set(node.id, new Map());
      ports(node);
    }
    for (const edge of g.edges) {
      const from = nodes.get(edge.from.node),
        to = nodes.get(edge.to.node);
      if (!from || !to)
        throw new GraphError('missing_node', path(!from ? edge.from.node : edge.to.node));
      if (!ports(from).outputs.includes(edge.from.port))
        throw new GraphError('unknown_output_port', path(from.id), edge.from.port);
      if (!ports(to).inputs.includes(edge.to.port))
        throw new GraphError('unknown_input_port', path(to.id), edge.to.port);
      const links = incoming.get(to.id)!;
      if (links.has(edge.to.port))
        throw new GraphError('duplicate_connection', path(to.id), edge.to.port);
      links.set(edge.to.port, edge.from);
    }
    function visit(node: GraphNode): Record<string, string> {
      if (resolved.has(node.id)) return resolved.get(node.id)!;
      const id = path(node.id),
        attrs = node.attrs ?? {};
      if (visiting.has(node.id)) throw new GraphError('cycle', id);
      visiting.add(node.id);
      try {
        const args: Record<string, string> = Object.create(null);
        for (const port of ports(node).inputs) {
          const link = incoming.get(node.id)!.get(port);
          if (!link) throw new GraphError('missing_port', id, port);
          args[port] = visit(nodes.get(link.node)!)[link.port];
        }
        let out: Record<string, string> = Object.create(null);
        if (node.component === 'input') {
          const name = requiredName(attrs);
          if (names.has(name)) throw new GraphError('duplicate_input', id, name);
          names.add(name);
          if (bindings) {
            if (!bindings[name]) throw new GraphError('missing_input', id, name);
            if (
              attrs.shape !== undefined &&
              !sameShape(shape(attrs.shape), shapes.get(bindings[name])!)
            )
              throw new GraphError('input_shape_mismatch', id, name);
            out.out = bindings[name];
          } else {
            const s = shape(options.inputShapes?.[name] ?? attrs.shape);
            if (attrs.shape !== undefined && !sameShape(s, shape(attrs.shape)))
              throw new GraphError('input_shape_mismatch', id, name);
            out.out = `${id}:out`;
            shapes.set(out.out, s);
            externalInputs[name] = out.out;
            inputShapes[name] = s;
          }
        } else if (node.component === 'output') {
          const name = requiredName(attrs);
          if (Object.hasOwn(result, name)) throw new GraphError('duplicate_output', id, name);
          result[name] = args.x;
        } else if (node.component === 'const' || node.component === 'param') {
          const s = shape(attrs.shape ?? (typeof attrs.value === 'number' ? [] : undefined));
          if (
            node.component === 'const' &&
            !(
              typeof attrs.value === 'number' ||
              (Array.isArray(attrs.value) &&
                attrs.value.every((x) => typeof x === 'number') &&
                attrs.value.length === size(s))
            )
          )
            throw new GraphError('invalid_constant', id);
          if (
            node.component === 'param' &&
            !['zeros', 'ones', 'normal', 'uniform', 'constant'].includes(
              String(attrs.init ?? 'zeros'),
            )
          )
            throw new GraphError('invalid_initializer', id);
          if (
            node.component === 'param' &&
            attrs.init === 'constant' &&
            (typeof attrs.value !== 'number' || !Number.isFinite(attrs.value))
          )
            throw new GraphError('invalid_constant', id);
          if (
            node.component === 'param' &&
            attrs.scale !== undefined &&
            (typeof attrs.scale !== 'number' || !Number.isFinite(attrs.scale) || attrs.scale < 0)
          )
            throw new GraphError('invalid_scale', id);
          // Storage keys are node instance paths, not output port names.
          out.out = id;
          shapes.set(id, s);
          storage.set(id, { shape: s, attrs, parameter: node.component === 'param' });
        } else {
          const def = registry.get(node.component)!;
          const argumentShapes = Object.fromEntries(
            Object.entries(args).map(([port, value]) => [port, shapes.get(value)!]),
          );
          const subgraph =
            typeof def.subgraph === 'function' ? def.subgraph(attrs, argumentShapes) : def.subgraph;
          const hasParameters = subgraph ? inspect(subgraph, id, new Set([def.name])) : false;
          const impl = hasParameters
            ? undefined
            : options.preferNative !== false && def.native
              ? def.native
              : def.primitive;
          if (impl) {
            const inferred = impl.infer(argumentShapes, attrs);
            for (const port of def.outputs) {
              if (!inferred[port]) throw new GraphError('missing_output_shape', id, port);
              out[port] = `${id}:${port}`;
              shapes.set(out[port], shape(inferred[port]));
            }
            steps.push({ id, inputs: args, outputs: out, attrs, implementation: impl });
          } else if (def.subgraph) {
            if (active.has(def.name)) throw new GraphError('recursive_component', id);
            active.add(def.name);
            try {
              out = compile(subgraph!, id, args);
            } finally {
              active.delete(def.name);
            }
            for (const port of def.outputs)
              if (!Object.hasOwn(out, port)) throw new GraphError('missing_output', id, port);
            for (const port of Object.keys(out))
              if (!def.outputs.includes(port))
                throw new GraphError('unknown_output_port', id, port);
          } else
            throw new GraphError('implementation_unavailable', id, undefined, {
              component: def.name,
            });
        }
        visiting.delete(node.id);
        resolved.set(node.id, out);
        return out;
      } catch (error) {
        throw contextual(error, id);
      }
    }
    for (const node of g.nodes) visit(node);
    if (bindings)
      for (const name of Object.keys(bindings))
        if (!names.has(name)) throw new GraphError('unused_component_input', prefix, name);
    return result;
  }
  const outputs = compile(graph, '');
  return new GraphPlan(
    inputShapes,
    Object.fromEntries(Object.entries(outputs).map(([k, v]) => [k, shapes.get(v)!])),
    externalInputs,
    outputs,
    steps,
    storage,
    options,
  );
}

/** Non-throwing compilation; errors currently contain the first diagnostic. */
export function safeCompileGraph(
  graph: GraphJSON,
  options: CompileOptions = {},
): { ok: true; plan: GraphPlan } | { ok: false; errors: GraphError[] } {
  try {
    return { ok: true, plan: compileGraph(graph, options) };
  } catch (error) {
    return { ok: false, errors: [error instanceof GraphError ? error : contextual(error, '')] };
  }
}

function broadcast(a: Shape, b: Shape): number[] {
  const out: number[] = [];
  for (let i = 1; i <= Math.max(a.length, b.length); i++) {
    const x = a[a.length - i] ?? 1,
      y = b[b.length - i] ?? 1;
    if (x !== y && x !== 1 && y !== 1)
      throw new GraphError('broadcast_mismatch', undefined, undefined, { a, b });
    out.unshift(x === 1 ? y : x);
  }
  return out;
}
function axes(a: Attributes, rank: number): number[] {
  const raw = a.axes ?? Array.from({ length: rank }, (_, i) => i);
  if (!Array.isArray(raw)) throw new GraphError('invalid_axes');
  const result = raw.map((x) => (typeof x === 'number' && x < 0 ? x + rank : x));
  if (
    result.some((x) => !Number.isInteger(x) || x < 0 || x >= rank) ||
    new Set(result).size !== result.length
  )
    throw new GraphError('invalid_axes');
  return result;
}
function reshapeShape(s: Shape, attrs: Attributes): number[] {
  if (!Array.isArray(attrs.shape)) throw new GraphError('invalid_shape');
  const target = [...attrs.shape] as number[];
  if (
    target.some((d) => !Number.isInteger(d) || d < -1) ||
    target.filter((d) => d === -1).length > 1
  )
    throw new GraphError('invalid_shape');
  const index = target.indexOf(-1);
  if (index >= 0) {
    const n = target.reduce((a, b) => a * (b === -1 ? 1 : b), 1);
    if (!n || size(s) % n) throw new GraphError('reshape_mismatch');
    target[index] = size(s) / n;
  }
  if (size(target) !== size(s)) throw new GraphError('reshape_mismatch');
  return target;
}
function permutation(a: Attributes, rank: number): number[] {
  const p = a.perm ?? Array.from({ length: rank }, (_, i) => rank - i - 1);
  if (
    !Array.isArray(p) ||
    p.length !== rank ||
    new Set(p).size !== rank ||
    p.some((i) => !Number.isInteger(i) || i < 0 || i >= rank)
  )
    throw new GraphError('invalid_permutation');
  return p;
}
export function createDefaultRegistry(): ComponentRegistry {
  const registry = new ComponentRegistry();
  const register = (
    name: string,
    inputs: string[],
    infer: ComponentImplementation['infer'],
    execute: ComponentImplementation['execute'],
  ) => registry.register({ name, inputs, outputs: ['out'], primitive: { infer, execute } });
  // Structural primitives are handled by the compiler/plan rather than tensor kernels.
  for (const name of ['input', 'output', 'const', 'param']) {
    registry.register({
      name,
      inputs: name === 'output' ? ['x'] : [],
      outputs: name === 'output' ? [] : ['out'],
      primitive: {
        infer: () => {
          throw new GraphError('structural_primitive');
        },
        execute: () => {
          throw new GraphError('structural_primitive');
        },
      },
    });
  }
  for (const name of ['add', 'mul', 'max', 'ge'] as const)
    register(
      name,
      ['a', 'b'],
      (i) => ({ out: broadcast(i.a, i.b) }),
      (i) => ({ out: T[name](i.a, i.b) }),
    );
  for (const name of ['exp', 'log'] as const)
    register(
      name,
      ['x'],
      (i) => ({ out: i.x }),
      (i) => ({ out: T[name](i.x) }),
    );
  for (const name of ['sum', 'reduce_max'] as const)
    register(
      name,
      ['x'],
      (i, a) => {
        const ax = axes(a, i.x.length);
        return { out: i.x.flatMap((d, k) => (ax.includes(k) ? (a.keepdims ? [1] : []) : [d])) };
      },
      (i, a) => ({
        out: T[name === 'sum' ? 'sum' : 'reduceMax'](
          i.x,
          axes(a, i.x.shape.length),
          Boolean(a.keepdims),
        ),
      }),
    );
  register(
    'reshape',
    ['x'],
    (i, a) => ({ out: reshapeShape(i.x, a) }),
    (i, a) => ({ out: T.reshape(i.x, reshapeShape(i.x.shape, a)) }),
  );
  register(
    'transpose',
    ['x'],
    (i, a) => ({ out: permutation(a, i.x.length).map((k) => i.x[k]) }),
    (i, a) => ({ out: T.transpose(i.x, permutation(a, i.x.shape.length)) }),
  );
  register(
    'gather',
    ['table', 'idx'],
    (i) => {
      if (!i.table.length) throw new GraphError('gather_rank');
      return { out: [...i.idx, ...i.table.slice(1)] };
    },
    (i) => ({ out: T.gather(i.table, i.idx) }),
  );
  register(
    'arange',
    [],
    (_i, a) => ({ out: shape([a.n]) }),
    (_i, a, c) => ({ out: T.arange(a.n as number, c.backend) }),
  );
  for (const component of standardComponents) registry.register(component);
  return registry;
}
