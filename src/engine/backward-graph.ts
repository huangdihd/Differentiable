import { size } from './backend';
import {
  compileGraph,
  resolveDimensions,
  staticScalar,
  createDefaultRegistry,
  GraphError,
  type ComponentRegistry,
  type GraphJSON,
  type GraphNode,
  type ShapePorts,
  type Attributes,
} from './graph';
import { CpuBackend } from './cpu';
import { Random } from './random';
import { tensor, type Tensor } from './tensor';

export interface BackwardGraphOptions {
  registry?: ComponentRegistry;
  availableComponents?: readonly string[];
  inputShapes: ShapePorts;
  output: string;
  wrt: readonly string[];
}
export interface BackwardRole {
  role: 'forward' | 'backward' | 'accumulate' | 'seed';
  mirrorOf?: string;
  rule?: string;
  origin?: string[];
}
export interface BackwardGraph {
  graph: GraphJSON;
  roles: Record<string, BackwardRole>;
}
export interface BackwardTraceValue {
  shape: number[];
  values: number[];
}
export interface BackwardTrace {
  forward: Record<string, BackwardTraceValue>;
  gradient: Record<string, BackwardTraceValue>;
}
const supported = new Set([
  'input',
  'param',
  'const',
  'output',
  'add',
  'mul',
  'max',
  'ge',
  'exp',
  'log',
  'sum',
  'reduce_max',
  'reshape',
  'unsqueeze',
  'transpose',
  'size',
  'arange',
  'gather',
  'scatter_add',
]);

/** Pure symbolic lowering. Shape inference is the only implementation hook used here.
 * Composite nodes become reshape aliases, preserving their public IDs, while their
 * canonical subgraphs (never native kernels) supply the actual computation.
 */
function build(source: GraphJSON, options: BackwardGraphOptions) {
  const registry = options.registry ?? createDefaultRegistry();
  const validation = compileGraph(source, { ...options, preferNative: false });
  validation.dispose();
  if (!Object.hasOwn(validation.outputShapes, options.output))
    throw new GraphError('missing_output', undefined, options.output);
  const graph: GraphJSON = { version: 1, nodes: [], edges: [] };
  const roles: Record<string, BackwardRole> = {};
  const shapes: Record<string, number[]> = {};
  const scalars = new Map<string, number>();
  const args: Record<string, Record<string, string>> = {};
  const used = new Set(source.nodes.map((n) => n.id));
  const unique = (base: string) => {
    let id = base;
    for (let i = 1; used.has(id); i++) id = `${base}_${i}`;
    used.add(id);
    return id;
  };
  function append(
    id: string,
    component: string,
    inputs: Record<string, string>,
    attrs: Attributes,
    role: BackwardRole,
    explicitShape?: readonly number[],
  ) {
    attrs = structuredClone(attrs);
    inputs = { ...inputs };
    resolveDimensions(component, attrs, inputs, scalars);
    const scalar =
      component === 'const' && ((attrs.shape ?? []) as number[]).length === 0
        ? Math.fround(typeof attrs.value === 'number' ? attrs.value : (attrs.value as number[])[0])
        : staticScalar(
            component,
            Object.fromEntries(Object.entries(inputs).map(([p, ref]) => [p, shapes[ref]])),
            attrs,
            Object.fromEntries(Object.entries(inputs).map(([p, ref]) => [p, scalars.get(ref)])),
          );
    if (scalar !== undefined) scalars.set(id, scalar);
    if (component === 'size') {
      component = 'const';
      inputs = {};
      attrs = { shape: [], value: scalar };
      explicitShape = [];
    }
    graph.nodes.push({ id, component, attrs: structuredClone(attrs) });
    roles[id] = role;
    args[id] = inputs;
    for (const [port, from] of Object.entries(inputs))
      graph.edges.push({ from: { node: from, port: 'out' }, to: { node: id, port } });
    if (component !== 'output') {
      const inferred =
        explicitShape ??
        registry
          .get(component)!
          .primitive!.infer(
            Object.fromEntries(Object.entries(inputs).map(([p, ref]) => [p, shapes[ref]])),
            attrs,
          ).out;
      shapes[id] = [...inferred];
      if (inferred.length !== 0) scalars.delete(id);
    }
    return id;
  }
  function expand(
    g: GraphJSON,
    origin: string[],
    bindings?: Record<string, string>,
  ): Record<string, string> {
    const resolved = new Map<string, Record<string, string>>();
    const nodes = new Map(g.nodes.map((n) => [n.id, n]));
    const outputs: Record<string, string> = {};
    function visit(n: GraphNode): Record<string, string> {
      const cached = resolved.get(n.id);
      if (cached) return cached;
      const attrs = n.attrs ?? {};
      const inputs = Object.fromEntries(
        g.edges
          .filter((e) => e.to.node === n.id)
          .map((e) => [e.to.port, visit(nodes.get(e.from.node)!)[e.from.port]]),
      );
      let result: Record<string, string>;
      if (bindings && n.component === 'input') result = { out: bindings[String(attrs.name)] };
      else if (bindings && n.component === 'output') {
        outputs[String(attrs.name)] = inputs.x;
        result = {};
      } else {
        const id = origin.length ? unique(`${origin.join('/')}/${n.id}`) : n.id;
        const role: BackwardRole = { role: 'forward', origin: [...origin] };
        const def = registry.get(n.component);
        if (def?.subgraph) {
          const nested =
            typeof def.subgraph === 'function'
              ? def.subgraph(
                  attrs,
                  Object.fromEntries(Object.entries(inputs).map(([p, ref]) => [p, shapes[ref]])),
                )
              : def.subgraph;
          const expanded = expand(nested, [...origin, n.id], inputs);
          result = {};
          for (const [i, port] of def.outputs.entries()) {
            const ref = expanded[port];
            result[port] = append(
              i === 0 ? id : unique(`${id}/${port}`),
              'reshape',
              { x: ref },
              { shape: shapes[ref] },
              role,
            );
          }
        } else {
          if (!supported.has(n.component))
            throw new GraphError('backward_unsupported', undefined, undefined, {
              component: n.component,
            });
          const shape =
            n.component === 'input'
              ? validation.inputShapes[String(attrs.name)]
              : ['const', 'param'].includes(n.component)
                ? ((attrs.shape ?? []) as number[])
                : undefined;
          append(id, n.component, inputs, attrs, role, shape);
          if (n.ui) graph.nodes.at(-1)!.ui = structuredClone(n.ui);
          result = n.component === 'output' ? {} : { out: id };
          if (n.component === 'output') outputs[String(attrs.name)] = inputs.x;
        }
      }
      resolved.set(n.id, result);
      return result;
    }
    for (const n of g.nodes) visit(n);
    return outputs;
  }
  const outputs = expand(source, []);
  const forward = [...graph.nodes].filter((n) => n.component !== 'output');
  const targets = options.wrt.map((name) => {
    const n = forward.find(
      (n) =>
        (n.component === 'input' && n.attrs?.name === name) ||
        (n.component === 'param' && n.id === name),
    );
    if (!n) throw new GraphError('missing_input', undefined, name);
    return n.id;
  });
  // Mark only descendants of requested leaves. Reverse traversal additionally
  // restricts this to ancestors of the selected output.
  const needed = new Set(targets);
  for (const n of forward)
    if (Object.values(args[n.id]).some((id) => needed.has(id))) needed.add(n.id);
  const gradients: Record<string, string> = {};
  let mirror = outputs[options.output];
  let rule = 'seed';
  function emit(
    component: string,
    inputs: Record<string, string> = {},
    attrs: Attributes = {},
    role: BackwardRole['role'] = 'backward',
  ) {
    return append(
      unique(`__backward/${mirror}/${component}`),
      component,
      inputs,
      attrs,
      {
        role,
        mirrorOf: mirror,
        rule: `backward.rule.${rule}`,
        origin: roles[mirror]?.origin ?? [],
      },
      component === 'const' ? (attrs.shape as number[]) : undefined,
    );
  }
  const constant = (
    value: number | number[],
    shape: number[] = [],
    role: BackwardRole['role'] = 'backward',
  ) => emit('const', {}, { shape, value }, role);
  const binary = (op: string, a: string, b: string) => emit(op, { a, b });
  const unary = (op: string, x: string, attrs: Attributes = {}) => emit(op, { x }, attrs);
  const reshape = (x: string, shape: number[]) => unary('reshape', x, { shape });
  const complement = (x: string) => binary('add', constant(1), binary('mul', constant(-1), x));
  function contribute(id: string, grad: string) {
    if (!needed.has(id)) return;
    const target = shapes[id],
      actual = shapes[grad];
    if (target.length !== actual.length || target.some((d, i) => d !== actual[i])) {
      const saved = rule;
      rule = 'broadcast';
      const offset = actual.length - target.length;
      const axes = actual.flatMap((d, i) =>
        i < offset || (target[i - offset] === 1 && d !== 1) ? [i] : [],
      );
      grad = reshape(unary('sum', grad, { axes, keepdims: true }), target);
      rule = saved;
    }
    if (gradients[id]) {
      const saved = mirror;
      mirror = id;
      grad = emit('add', { a: gradients[id], b: grad }, {}, 'accumulate');
      mirror = saved;
    }
    gradients[id] = grad;
  }
  gradients[mirror] = constant(1, shapes[mirror], 'seed');
  for (const n of [...forward].reverse()) {
    const grad = gradients[n.id];
    if (!grad || !needed.has(n.id)) continue;
    mirror = n.id;
    rule = n.component;
    const a = args[n.id];
    const send = (port: string, f: () => string) => {
      if (needed.has(a[port])) contribute(a[port], f());
    };
    switch (n.component) {
      case 'add':
        send('a', () => grad);
        send('b', () => grad);
        break;
      case 'mul':
        send('a', () => binary('mul', grad, a.b));
        send('b', () => binary('mul', grad, a.a));
        break;
      case 'gather':
        send('table', () => emit('scatter_add', { table: a.table, idx: a.idx, updates: grad }));
        break;
      case 'scatter_add':
        send('updates', () => emit('gather', { table: grad, idx: a.idx }));
        break;
      case 'arange':
      case 'size':
      case 'ge':
        break; // Comparison is non-differentiable, with zero VJP.
      case 'max': {
        const mask = binary('ge', a.a, a.b);
        send('a', () => binary('mul', grad, mask));
        send('b', () => binary('mul', grad, complement(mask)));
        break;
      }
      case 'exp':
        send('x', () => binary('mul', grad, n.id));
        break;
      case 'log':
        send('x', () =>
          binary('mul', grad, unary('exp', binary('mul', constant(-1), unary('log', a.x)))),
        );
        break;
      case 'reshape':
      case 'unsqueeze':
        send('x', () => reshape(grad, shapes[a.x]));
        break;
      case 'transpose': {
        if (n.attrs && 'axes' in n.attrs) {
          send('x', () => unary('transpose', grad, { axes: n.attrs!.axes }));
          break;
        }
        const perm =
          (n.attrs?.perm as number[] | undefined) ?? shapes[a.x].map((_, i, s) => s.length - i - 1);
        send('x', () => unary('transpose', grad, { perm: perm.map((_, i) => perm.indexOf(i)) }));
        break;
      }
      case 'sum':
      case 'reduce_max': {
        const shape = shapes[a.x];
        const axes = ((n.attrs?.axes as number[] | undefined) ?? shape.map((_, i) => i)).map((i) =>
          i < 0 ? i + shape.length : i,
        );
        const kept = shape.map((d, i) => (axes.includes(i) ? 1 : d));
        send('x', () => {
          const expanded = reshape(grad, kept);
          if (n.component === 'sum') return binary('mul', expanded, constant(1, shape));
          const mask = binary('ge', a.x, reshape(n.id, kept));
          // Positive descending row-major coordinates: non-winners score zero,
          // and reduce_max selects exactly the first tied winner in each group.
          const coordinates = constant(
            Array.from({ length: size(shape) }, (_, i) => size(shape) - i),
            shape,
          );
          const scores = binary('mul', mask, coordinates);
          const first = unary('reduce_max', scores, { axes, keepdims: true });
          return binary('mul', expanded, binary('mul', mask, binary('ge', scores, first)));
        });
        break;
      }
    }
  }
  // Trace disconnected nodes too, with explicit zero gradients.
  for (const n of forward)
    if (!gradients[n.id]) {
      mirror = n.id;
      rule = n.component;
      gradients[n.id] = constant(0, shapes[n.id]);
    }
  options.wrt.forEach((name, i) => {
    mirror = targets[i];
    rule = 'input';
    emit('output', { x: gradients[targets[i]] }, { name: `grad_${name}` });
  });
  return { graph, roles, gradients, forward, shapes };
}

export function buildBackwardGraph(graph: GraphJSON, options: BackwardGraphOptions): BackwardGraph {
  const { graph: generated, roles } = build(graph, options);
  return { graph: generated, roles };
}

/** Trace by probing the generated ordinary graph, not by a second evaluator.
 * CPU and a fixed PRNG seed make parameter initialization reproducible.
 */
export async function traceBackward(
  source: GraphJSON,
  options: BackwardGraphOptions & { inputs: Record<string, number[]> },
): Promise<BackwardTrace> {
  const built = build(source, options);
  const used = new Set(built.graph.nodes.map((n) => n.id));
  const names = new Set(
    built.graph.nodes.filter((n) => n.component === 'output').map((n) => String(n.attrs?.name)),
  );
  const probes: { id: string; forward: string; gradient: string }[] = [];
  function probe(ref: string) {
    let id = `__probe_${used.size}`;
    while (used.has(id) || names.has(id)) id += '_';
    used.add(id);
    names.add(id);
    built.graph.nodes.push({ id, component: 'output', attrs: { name: id } });
    built.graph.edges.push({ from: { node: ref, port: 'out' }, to: { node: id, port: 'x' } });
    return id;
  }
  for (const n of built.forward)
    probes.push({ id: n.id, forward: probe(n.id), gradient: probe(built.gradients[n.id]) });
  const backend = new CpuBackend();
  const plan = compileGraph(built.graph, {
    registry: options.registry,
    inputShapes: options.inputShapes,
    backend,
    random: new Random(42),
  });
  const inputs: Record<string, Tensor> = {};
  let execution: ReturnType<typeof plan.execute> | undefined;
  try {
    for (const [name, shape] of Object.entries(plan.inputShapes)) {
      if (!options.inputs[name]) throw new GraphError('missing_input', undefined, name);
      inputs[name] = tensor(options.inputs[name], shape, backend);
    }
    execution = plan.execute(inputs);
    const trace: BackwardTrace = { forward: {}, gradient: {} };
    for (const p of probes)
      for (const kind of ['forward', 'gradient'] as const) {
        const value = execution[p[kind]];
        trace[kind][p.id] = { shape: [...value.shape], values: Array.from(await value.read()) };
      }
    return trace;
  } finally {
    if (execution) plan.releaseExecution(execution);
    plan.dispose();
    Object.values(inputs).forEach((t) => t.dispose());
  }
}
