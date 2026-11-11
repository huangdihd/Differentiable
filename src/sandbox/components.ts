// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
/**
 * Sandbox components ("我的组件"): a piece of a sandbox graph saved as a reusable part.
 *
 * A component is an ordinary subgraph component: its inner graph has one `input` node per input
 * port and one `output` node per output port, and the compiler expands every placed instance in
 * place. Parameters are stored under the instance's path (`instance/inner`), so each instance
 * starts from its own initialisation and training differentiates through the component exactly
 * as if its parts were wired inline.
 *
 * The library lives in the local save next to the slots that use it (`progress.sandbox`), so a
 * slot and the parts it references are always written together. Nothing is shared online.
 */
import type {
  ComponentDefinition,
  ComponentRegistry,
  GraphEdge,
  GraphEndpoint,
  GraphJSON,
  GraphNode,
} from '../engine/graph';
import type { Progress } from '../engine/levels';
import { isLockedNode } from '../editor/locked-loss';

export interface CustomComponent {
  /** Stable identity; the component name on the canvas is `my:<id>`. Never reused. */
  id: string;
  /** What the player called it. Renaming only changes this. */
  name: string;
  createdAt: number;
  updatedAt: number;
  /** Port names in display order; each has a matching `input`/`output` node in `graph`. */
  inputs: string[];
  outputs: string[];
  graph: GraphJSON;
  /** Shapes seen on the wires when it was packaged. Informational only: instances adapt. */
  shapes?: Record<string, number[]>;
}

export interface ComponentsFile {
  format: 'differentiable-components';
  version: 1;
  components: CustomComponent[];
}

export const CUSTOM_PREFIX = 'my:';
export const MAX_COMPONENTS = 64;
export const MAX_PORT_NAME = 12;
export const MAX_COMPONENT_NAME = 32;

export const customComponentName = (id: string) => `${CUSTOM_PREFIX}${id}`;
export const isCustomComponent = (component: string) => component.startsWith(CUSTOM_PREFIX);
export const customComponentId = (component: string) =>
  isCustomComponent(component) ? component.slice(CUSTOM_PREFIX.length) : undefined;

export const libraryOf = (progress: Pick<Progress, 'sandbox'>): CustomComponent[] =>
  progress.sandbox?.components ?? [];

export const findComponent = (
  library: readonly CustomComponent[],
  component: string,
): CustomComponent | undefined => {
  const id = customComponentId(component);
  return id === undefined ? undefined : library.find((c) => c.id === id);
};

/* ---------------------------------------------------------------- validation */

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const time = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const RESERVED = new Set(['__proto__', 'prototype', 'constructor']);

/** Port names are short identifiers: letters (any script), digits, `_` and primes (x′). */
export function portNameError(name: string, taken: readonly string[] = []): string | undefined {
  if (!name.trim()) return 'sandbox.components.port.empty';
  if (name.length > MAX_PORT_NAME || !/^[\p{L}\p{N}_′']+$/u.test(name) || RESERVED.has(name))
    return 'sandbox.components.port.invalid';
  if (taken.includes(name)) return 'sandbox.components.port.duplicate';
  return undefined;
}

const portList = (value: unknown): value is string[] =>
  Array.isArray(value) &&
  value.every((p) => typeof p === 'string' && portNameError(p) === undefined) &&
  new Set(value).size === value.length;

/** Inner `input`/`output` nodes must match the declared ports one to one. */
function interfaceMatches(component: CustomComponent): boolean {
  const names = (kind: 'input' | 'output') =>
    component.graph.nodes
      .filter((n) => n.component === kind)
      .map((n) => n.attrs?.name)
      .sort();
  const same = (a: unknown[], b: string[]) =>
    a.length === b.length && [...b].sort().every((name, i) => a[i] === name);
  return same(names('input'), component.inputs) && same(names('output'), component.outputs);
}

export function isCustomComponentRecord(
  value: unknown,
  isGraph: (graph: unknown) => boolean,
): value is CustomComponent {
  if (
    !record(value) ||
    typeof value.id !== 'string' ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(value.id) ||
    typeof value.name !== 'string' ||
    value.name.length > MAX_COMPONENT_NAME * 2 ||
    !time(value.createdAt) ||
    !time(value.updatedAt) ||
    !portList(value.inputs) ||
    !portList(value.outputs) ||
    value.outputs.length === 0 ||
    !isGraph(value.graph) ||
    (value.shapes !== undefined &&
      (!record(value.shapes) ||
        !Object.entries(value.shapes).every(
          ([k, s]) =>
            !RESERVED.has(k) &&
            Array.isArray(s) &&
            s.every((d) => Number.isSafeInteger(d) && (d as number) >= 0),
        )))
  )
    return false;
  return interfaceMatches(value as unknown as CustomComponent);
}

/** Components another one uses directly. */
export const dependenciesOf = (component: Pick<CustomComponent, 'graph'>): string[] => [
  ...new Set(
    component.graph.nodes.flatMap((n) => {
      const id = customComponentId(n.component);
      return id === undefined ? [] : [id];
    }),
  ),
];

/** True when some component (transitively) contains itself. */
export function hasCycle(library: readonly CustomComponent[]): boolean {
  const byId = new Map(library.map((c) => [c.id, c]));
  const state = new Map<string, 'visiting' | 'done'>();
  const visit = (id: string): boolean => {
    const s = state.get(id);
    if (s === 'done') return false;
    if (s === 'visiting') return true;
    state.set(id, 'visiting');
    const c = byId.get(id);
    if (c && dependenciesOf(c).some(visit)) return true;
    state.set(id, 'done');
    return false;
  };
  return library.some((c) => visit(c.id));
}

export function isComponentLibrary(
  value: unknown,
  isGraph: (graph: unknown) => boolean,
): value is CustomComponent[] {
  if (!Array.isArray(value) || value.length > MAX_COMPONENTS) return false;
  const ids = new Set<string>();
  for (const c of value) {
    if (!isCustomComponentRecord(c, isGraph) || ids.has(c.id)) return false;
    ids.add(c.id);
  }
  return !hasCycle(value);
}

/* ---------------------------------------------------------------- registry */

/** Engine view: a plain subgraph component, labelled with the player's name. */
export function customDefinition(component: CustomComponent): ComponentDefinition {
  return {
    name: customComponentName(component.id),
    label: component.name,
    inputs: [...component.inputs],
    outputs: [...component.outputs],
    subgraph: component.graph,
  };
}

/** Add the library to a registry. A malformed entry is skipped rather than breaking the rest. */
export function registerCustomComponents(
  registry: ComponentRegistry,
  library: readonly CustomComponent[],
): ComponentRegistry {
  for (const component of library) {
    try {
      registry.register(customDefinition(component));
    } catch {
      // Duplicate or invalid ports: the instance will report an unknown component instead.
    }
  }
  return registry;
}

/* ---------------------------------------------------------------- packaging */

export interface PackagePort {
  /** Suggested port name, unique on its side. */
  name: string;
  /** Input: the outside endpoint feeding it. Output: the inside endpoint it exposes. */
  source: GraphEndpoint;
  /** Input: inside endpoints it feeds. Output: outside endpoints it feeds. */
  targets: GraphEndpoint[];
}

export interface PackagePlan {
  /** Selected parts that move into the component, in graph order. */
  nodes: string[];
  inputs: PackagePort[];
  outputs: PackagePort[];
}

export type PackageProblem =
  | 'sandbox.components.problem.empty'
  | 'sandbox.components.problem.noOutput'
  | 'sandbox.components.problem.cycle';

const keyOf = (e: GraphEndpoint) => `${e.node}\u0000${e.port}`;
const IO = new Set(['input', 'output']);

/** Pick a name not yet taken: `x`, then `x2`, `x3`, … (kept within the length limit). */
function freeName(base: string, taken: Set<string>): string {
  const clean =
    [...base]
      .filter((ch) => /[\p{L}\p{N}_′']/u.test(ch))
      .join('')
      .slice(0, MAX_PORT_NAME - 2) || 'p';
  if (!taken.has(clean) && !RESERVED.has(clean)) return clean;
  for (let n = 2; ; n++) if (!taken.has(`${clean}${n}`)) return `${clean}${n}`;
}

/**
 * Which wires cross the selection, and what the ports would be. Wires from one outside endpoint
 * share an input port; wires from one inside endpoint share an output port.
 */
export function planPackage(
  graph: GraphJSON,
  selection: readonly string[],
): PackagePlan | PackageProblem {
  const chosen = new Set(
    graph.nodes
      .filter((n) => selection.includes(n.id) && !IO.has(n.component) && !isLockedNode(n))
      .map((n) => n.id),
  );
  if (!chosen.size) return 'sandbox.components.problem.empty';
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const inputs = new Map<string, PackagePort>();
  const outputs = new Map<string, PackagePort>();
  const inputNames = new Set<string>();
  for (const edge of graph.edges) {
    const inFrom = chosen.has(edge.from.node),
      inTo = chosen.has(edge.to.node);
    if (!inFrom && inTo) {
      const port = inputs.get(keyOf(edge.from));
      if (port) port.targets.push({ ...edge.to });
      else {
        const source = nodeById.get(edge.from.node);
        const base = source?.component === 'input' ? String(source.attrs?.name ?? '') : '';
        const name = freeName(base || edge.to.port, inputNames);
        inputNames.add(name);
        inputs.set(keyOf(edge.from), {
          name,
          source: { ...edge.from },
          targets: [{ ...edge.to }],
        });
      }
    } else if (inFrom && !inTo) {
      const port = outputs.get(keyOf(edge.from));
      if (port) port.targets.push({ ...edge.to });
      else
        outputs.set(keyOf(edge.from), {
          name: '',
          source: { ...edge.from },
          targets: [{ ...edge.to }],
        });
    }
  }
  if (!outputs.size) return 'sandbox.components.problem.noOutput';
  // One exposed value reads as the component's `out`, like every built-in part.
  const exposed = [...outputs.values()];
  const outputNames = new Set(inputNames);
  for (const port of exposed) {
    const target = nodeById.get(port.targets[0].node);
    const base =
      exposed.length === 1
        ? 'out'
        : target?.component === 'output'
          ? String(target.attrs?.name ?? '')
          : port.source.port === 'out'
            ? 'y'
            : port.source.port;
    port.name = freeName(base || 'y', outputNames);
    outputNames.add(port.name);
  }
  const plan: PackagePlan = {
    nodes: graph.nodes.filter((n) => chosen.has(n.id)).map((n) => n.id),
    inputs: [...inputs.values()],
    outputs: exposed,
  };
  // An outside path from the selection back into it would turn the new part into a loop.
  const after = new Map<string, string[]>();
  for (const e of graph.edges) {
    const from = chosen.has(e.from.node) ? '\u0000part' : e.from.node;
    const to = chosen.has(e.to.node) ? '\u0000part' : e.to.node;
    if (from === to) continue;
    after.set(from, [...(after.get(from) ?? []), to]);
  }
  const seen = new Set<string>();
  const pending = [...(after.get('\u0000part') ?? [])];
  while (pending.length) {
    const id = pending.pop()!;
    if (id === '\u0000part') return 'sandbox.components.problem.cycle';
    if (seen.has(id)) continue;
    seen.add(id);
    pending.push(...(after.get(id) ?? []));
  }
  return plan;
}

const position = (n: GraphNode) => ({
  x: typeof n.ui?.x === 'number' ? n.ui.x : 0,
  y: typeof n.ui?.y === 'number' ? n.ui.y : 0,
});

function freeId(prefix: string, taken: Set<string>): string {
  let n = 1;
  while (taken.has(`${prefix}${n}`)) n++;
  const id = `${prefix}${n}`;
  taken.add(id);
  return id;
}

/**
 * Turn the plan into a component plus the outer graph with one instance in place of the parts.
 * `inputNames`/`outputNames` override the suggested port names (same order as the plan).
 */
export function applyPackage(
  graph: GraphJSON,
  plan: PackagePlan,
  options: {
    id: string;
    name: string;
    now: number;
    inputNames?: readonly string[];
    outputNames?: readonly string[];
    shapes?: Record<string, number[]>;
  },
): { component: CustomComponent; graph: GraphJSON; instance: string } {
  const inside = new Set(plan.nodes);
  const inputs = plan.inputs.map((p, i) => options.inputNames?.[i] ?? p.name);
  const outputs = plan.outputs.map((p, i) => options.outputNames?.[i] ?? p.name);
  // One namespace for both sides, so a port name never means two things.
  const all = [...inputs, ...outputs];
  all.forEach((name, i) => {
    const problem = portNameError(name, all.slice(0, i));
    if (problem) throw new Error(problem);
  });
  const parts = graph.nodes.filter((n) => inside.has(n.id));
  const xs = parts.map((n) => position(n).x),
    ys = parts.map((n) => position(n).y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  // Inside: parts keep their arrangement, inputs on the left, outputs on the right.
  const innerIds = new Set(parts.map((n) => n.id));
  const inputIds = inputs.map(() => freeId('in_', innerIds));
  const outputIds = outputs.map(() => freeId('out_', innerIds));
  const shift = { x: 280 - x0, y: 40 - y0 };
  const column = (i: number, count: number) =>
    40 + (y1 - y0) / 2 - ((count - 1) * 140) / 2 + i * 140;
  const inner: GraphJSON = {
    version: 1,
    nodes: [
      ...inputs.map((name, i) => ({
        id: inputIds[i],
        component: 'input',
        attrs: { name },
        ui: { x: 40, y: column(i, inputs.length) },
      })),
      ...parts.map((n) => {
        const copy = structuredClone(n);
        const p = position(n);
        copy.ui = { ...copy.ui, x: p.x + shift.x, y: p.y + shift.y };
        return copy;
      }),
      ...outputs.map((name, i) => ({
        id: outputIds[i],
        component: 'output',
        attrs: { name },
        ui: { x: x1 + shift.x + 260, y: column(i, outputs.length) },
      })),
    ],
    edges: [
      ...graph.edges
        .filter((e) => inside.has(e.from.node) && inside.has(e.to.node))
        .map((e) => structuredClone(e)),
      ...plan.inputs.flatMap((p, i) =>
        p.targets.map((to) => ({ from: { node: inputIds[i], port: 'out' }, to: { ...to } })),
      ),
      ...plan.outputs.map((p, i) => ({
        from: { ...p.source },
        to: { node: outputIds[i], port: 'x' },
      })),
    ],
  };
  const component: CustomComponent = {
    id: options.id,
    name: options.name,
    createdAt: options.now,
    updatedAt: options.now,
    inputs,
    outputs,
    graph: inner,
    ...(options.shapes && Object.keys(options.shapes).length
      ? { shapes: structuredClone(options.shapes) }
      : {}),
  };
  // Outside: one instance where the parts were, wired to the same neighbours.
  const outerIds = new Set(graph.nodes.map((n) => n.id));
  const instance = freeId('node_', outerIds);
  const edges: GraphEdge[] = [
    ...graph.edges
      .filter((e) => !inside.has(e.from.node) && !inside.has(e.to.node))
      .map((e) => structuredClone(e)),
    ...plan.inputs.map((p, i) => ({
      from: { ...p.source },
      to: { node: instance, port: inputs[i] },
    })),
    ...plan.outputs.flatMap((p, i) =>
      p.targets.map((to) => ({ from: { node: instance, port: outputs[i] }, to: { ...to } })),
    ),
  ];
  const first = graph.nodes.findIndex((n) => inside.has(n.id));
  const node: GraphNode = {
    id: instance,
    component: customComponentName(options.id),
    attrs: {},
    ui: { x: Math.round((x0 + x1) / 2), y: Math.round((y0 + y1) / 2) },
  };
  const nodes = graph.nodes.filter((n) => !inside.has(n.id));
  nodes.splice(first, 0, node);
  return { component, graph: { ...graph, nodes, edges }, instance };
}

/* ---------------------------------------------------------------- ungrouping */

/**
 * Put an instance's parts back on the canvas, wired to its neighbours. Nested components stay
 * packaged (ungroup again to open them). Returns the ids of the parts that came out.
 */
export function ungroupInstance(
  graph: GraphJSON,
  instanceId: string,
  library: readonly CustomComponent[],
): { graph: GraphJSON; ids: string[] } | undefined {
  const instance = graph.nodes.find((n) => n.id === instanceId);
  const component = instance && findComponent(library, instance.component);
  if (!instance || !component) return undefined;
  const inner = component.graph;
  const innerById = new Map(inner.nodes.map((n) => [n.id, n]));
  const parts = inner.nodes.filter((n) => !IO.has(n.component));
  const taken = new Set(graph.nodes.map((n) => n.id).filter((id) => id !== instanceId));
  const ids = new Map(parts.map((n) => [n.id, freeId('node_', taken)]));
  const xs = parts.map((n) => position(n).x),
    ys = parts.map((n) => position(n).y);
  const at = position(instance);
  const shift = parts.length
    ? {
        x: at.x - (Math.min(...xs) + Math.max(...xs)) / 2,
        y: at.y - (Math.min(...ys) + Math.max(...ys)) / 2,
      }
    : { x: 0, y: 0 };
  const placed = parts.map((n) => {
    const copy = structuredClone(n);
    const p = position(n);
    copy.id = ids.get(n.id)!;
    copy.ui = { ...copy.ui, x: Math.round(p.x + shift.x), y: Math.round(p.y + shift.y) };
    return copy;
  });
  const portOf = (id: string) => String(innerById.get(id)?.attrs?.name ?? '');
  // Outside wires into each input port, and out of each output port.
  const feeding = new Map<string, GraphEndpoint>();
  const fed = new Map<string, GraphEndpoint[]>();
  for (const e of graph.edges) {
    if (e.to.node === instanceId) feeding.set(e.to.port, e.from);
    if (e.from.node === instanceId) fed.set(e.from.port, [...(fed.get(e.from.port) ?? []), e.to]);
  }
  // What each inner output node reads, as an outside-or-inside endpoint.
  const resolve = (from: GraphEndpoint): GraphEndpoint | undefined => {
    const source = innerById.get(from.node);
    if (source?.component === 'input') return feeding.get(portOf(from.node));
    return { node: ids.get(from.node)!, port: from.port };
  };
  const edges: GraphEdge[] = graph.edges
    .filter((e) => e.from.node !== instanceId && e.to.node !== instanceId)
    .map((e) => structuredClone(e));
  for (const e of inner.edges) {
    const target = innerById.get(e.to.node);
    if (target?.component === 'output') {
      const from = resolve(e.from);
      if (from)
        for (const to of fed.get(portOf(e.to.node)) ?? [])
          edges.push({ from: { ...from }, to: { ...to } });
    } else {
      const from = resolve(e.from);
      if (from)
        edges.push({ from: { ...from }, to: { node: ids.get(e.to.node)!, port: e.to.port } });
    }
  }
  const index = graph.nodes.findIndex((n) => n.id === instanceId);
  const nodes = [...graph.nodes];
  nodes.splice(index, 1, ...placed);
  return { graph: { ...graph, nodes, edges }, ids: placed.map((n) => n.id) };
}

/* ---------------------------------------------------------------- library edits */

const withLibrary = (progress: Progress, components: CustomComponent[]): Progress => ({
  ...progress,
  sandbox: { ...progress.sandbox, slots: progress.sandbox?.slots ?? [], components },
});

export function addComponent(progress: Progress, component: CustomComponent): Progress {
  const library = libraryOf(progress);
  if (library.length >= MAX_COMPONENTS || library.some((c) => c.id === component.id))
    return progress;
  return withLibrary(progress, [...library, component]);
}

export function renameComponent(
  progress: Progress,
  id: string,
  name: string,
  now: number,
): Progress {
  const clean = name.trim().slice(0, MAX_COMPONENT_NAME);
  if (!clean) return progress;
  return withLibrary(
    progress,
    libraryOf(progress).map((c) => (c.id === id ? { ...c, name: clean, updatedAt: now } : c)),
  );
}

/** Where a component is placed: slot graphs, and other components that contain it. */
export function componentUsage(
  progress: Progress,
  id: string,
): { slots: string[]; components: string[] } {
  const name = customComponentName(id);
  const uses = (graph?: GraphJSON) => !!graph?.nodes.some((n) => n.component === name);
  return {
    slots: (progress.sandbox?.slots ?? []).filter((s) => uses(s.graph)).map((s) => s.name),
    components: libraryOf(progress)
      .filter((c) => c.id !== id && uses(c.graph))
      .map((c) => c.name),
  };
}

/** Only unused components can go: a placed one would leave a dangling part behind. */
export function removeComponent(progress: Progress, id: string): Progress {
  const usage = componentUsage(progress, id);
  if (usage.slots.length || usage.components.length) return progress;
  return withLibrary(
    progress,
    libraryOf(progress).filter((c) => c.id !== id),
  );
}

/** A name not yet taken in the library: "base", then "base 2", … */
export function uniqueComponentName(base: string, library: readonly CustomComponent[]): string {
  const taken = new Set(library.map((c) => c.name));
  const clean = base.trim().slice(0, MAX_COMPONENT_NAME) || 'C';
  if (!taken.has(clean)) return clean;
  for (let n = 2; ; n++) if (!taken.has(`${clean} ${n}`)) return `${clean} ${n}`;
}

/* ---------------------------------------------------------------- files */

/** The chosen components plus everything they contain, dependencies first. */
export function exportComponents(
  library: readonly CustomComponent[],
  ids: readonly string[],
): ComponentsFile {
  const byId = new Map(library.map((c) => [c.id, c]));
  const ordered: CustomComponent[] = [];
  const seen = new Set<string>();
  const visit = (id: string) => {
    if (seen.has(id)) return;
    seen.add(id);
    const c = byId.get(id);
    if (!c) return;
    dependenciesOf(c).forEach(visit);
    ordered.push(structuredClone(c));
  };
  ids.forEach(visit);
  return { format: 'differentiable-components', version: 1, components: ordered };
}

export function parseComponentsFile(
  source: string,
  isGraph: (graph: unknown) => boolean,
): CustomComponent[] {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    throw new Error('sandbox.components.error.json');
  }
  if (
    !record(value) ||
    value.format !== 'differentiable-components' ||
    value.version !== 1 ||
    !isComponentLibrary(value.components, isGraph)
  )
    throw new Error('sandbox.components.error.format');
  const ids = new Set(value.components.map((c) => c.id));
  // Everything a file's components contain must travel with them.
  if (value.components.some((c) => dependenciesOf(c).some((d) => !ids.has(d))))
    throw new Error('sandbox.components.error.missing');
  return value.components;
}

const sameDefinition = (a: CustomComponent, b: CustomComponent) =>
  JSON.stringify([a.inputs, a.outputs, a.graph]) === JSON.stringify([b.inputs, b.outputs, b.graph]);

/**
 * Bring imported components into the library. An identical one already here is reused; a
 * different one with a clashing id gets a fresh id (and its users are rewired to it).
 */
export function importComponents(
  progress: Progress,
  imported: readonly CustomComponent[],
  options: { newId: () => string; now: number },
): { progress: Progress; added: number; reused: number } {
  const library = [...libraryOf(progress)];
  const rename = new Map<string, string>();
  let added = 0,
    reused = 0;
  const pending = [...imported];
  // Dependencies first, so their final ids are known when their users are rewritten.
  const done = new Set<string>();
  while (pending.length) {
    const index = pending.findIndex((c) =>
      dependenciesOf(c).every((d) => done.has(d) || !imported.some((i) => i.id === d)),
    );
    if (index < 0) break;
    const [source] = pending.splice(index, 1);
    done.add(source.id);
    const graph = structuredClone(source.graph);
    for (const node of graph.nodes) {
      const dep = customComponentId(node.component);
      if (dep !== undefined && rename.has(dep))
        node.component = customComponentName(rename.get(dep)!);
    }
    const candidate: CustomComponent = { ...structuredClone(source), graph };
    const existing = library.find((c) => c.id === candidate.id);
    if (existing && sameDefinition(existing, candidate)) {
      reused++;
      continue;
    }
    const twin = library.find((c) => sameDefinition(c, candidate));
    if (twin) {
      rename.set(source.id, twin.id);
      reused++;
      continue;
    }
    if (library.length >= MAX_COMPONENTS) throw new Error('sandbox.components.error.full');
    if (existing) {
      candidate.id = options.newId();
      rename.set(source.id, candidate.id);
    }
    candidate.name = uniqueComponentName(candidate.name, library);
    candidate.updatedAt = options.now;
    library.push(candidate);
    added++;
  }
  return { progress: withLibrary(progress, library), added, reused };
}
