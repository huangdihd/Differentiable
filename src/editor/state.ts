import { tensorOpPorts } from '../engine/tensor-op-ports';
import {
  createDefaultRegistry,
  type Attributes,
  type ComponentRegistry,
  type GraphEdge,
  type GraphEndpoint,
  type GraphJSON,
  type GraphNode,
} from '../engine/graph';
import type { CircuitLevel } from '../engine/levels';
import type { ExpressionGraph } from '../engine/expression';

export interface Point {
  x: number;
  y: number;
}
export interface Viewport extends Point {
  zoom: number;
}
export interface Rect extends Point {
  width: number;
  height: number;
}
export interface EditorSnapshot {
  graph: GraphJSON;
  selection: string[];
  nextId: number;
}
export interface EditorState extends EditorSnapshot {
  available: readonly string[];
  viewport: Viewport;
  past: EditorSnapshot[];
  future: EditorSnapshot[];
  clipboard: GraphJSON | null;
  /** A drag previews graph changes, then creates exactly one history entry. */
  transaction: EditorSnapshot | null;
  error: string | null;
}
export type EditorCommand =
  | { type: 'undo' | 'redo' | 'beginMove' | 'commitMove' | 'cancelMove' | 'copy' }
  | { type: 'addNode'; component: string; position: Point; attrs?: Attributes }
  | { type: 'removeNodes'; ids?: string[] }
  | { type: 'moveNodes'; positions: Record<string, Point> }
  | { type: 'connect'; from: GraphEndpoint; to: GraphEndpoint }
  | { type: 'disconnect'; edge: GraphEdge }
  | { type: 'rewire'; edge: GraphEdge; from: GraphEndpoint; to: GraphEndpoint }
  | { type: 'select'; ids: string[]; shift?: boolean }
  | {
      type: 'boxSelect';
      rect: Rect;
      shift?: boolean;
      sizes?: Record<string, { width: number; height: number }>;
    }
  | { type: 'paste'; offset?: Point }
  | { type: 'setAttrs'; id: string; attrs: Attributes; replace?: boolean }
  | { type: 'insertExpression'; expression: ExpressionGraph; position: Point; to?: GraphEndpoint }
  | { type: 'setViewport'; viewport: Viewport };

const clone = <T>(value: T): T => structuredClone(value);
const protectedNode = (node: GraphNode) =>
  node.component === 'input' || node.component === 'output';
const snapshot = (state: EditorState): EditorSnapshot => ({
  graph: state.graph,
  selection: state.selection,
  nextId: state.nextId,
});
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const endpointEqual = (a: GraphEndpoint, b: GraphEndpoint) =>
  a.node === b.node && a.port === b.port;
const edgeEqual = (a: GraphEdge, b: GraphEdge) =>
  endpointEqual(a.from, b.from) && endpointEqual(a.to, b.to);
export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 2;
export function clampZoom(zoom: number): number {
  return Number.isFinite(zoom) ? Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom)) : 1;
}
export function screenToWorld(point: Point, viewport: Viewport): Point {
  const zoom = clampZoom(viewport.zoom);
  return { x: (point.x - viewport.x) / zoom, y: (point.y - viewport.y) / zoom };
}
export function worldToScreen(point: Point, viewport: Viewport): Point {
  const zoom = clampZoom(viewport.zoom);
  return { x: point.x * zoom + viewport.x, y: point.y * zoom + viewport.y };
}
/** Cursor coordinates are local to the canvas, not browser client coordinates. */
export function zoomAt(viewport: Viewport, cursor: Point, zoom: number): Viewport {
  const world = screenToWorld(cursor, viewport),
    next = clampZoom(zoom);
  return { x: cursor.x - world.x * next, y: cursor.y - world.y * next, zoom: next };
}
export function nodePosition(node: GraphNode): Point {
  return {
    x: typeof node.ui?.x === 'number' ? node.ui.x : 0,
    y: typeof node.ui?.y === 'number' ? node.ui.y : 0,
  };
}
export function initialGraph(level: CircuitLevel): GraphJSON {
  return {
    version: 1,
    nodes: [
      ...Object.keys(level.interface.inputs).map((name, i) => ({
        id: `input_${i}`,
        component: 'input',
        attrs: { name },
        ui: { x: 60, y: 80 + i * 140 },
      })),
      ...Object.keys(
        level.interface.outputs ?? { [level.interface.output]: level.interface.outputShape },
      ).map((name, i) => ({
        id: i === 0 ? 'output' : `output_${i}`,
        component: 'output',
        attrs: { name },
        // Output diagnostics can add several wrapped lines below a disconnected node.
        ui: { x: 660, y: 120 + i * 200 },
      })),
    ],
    edges: [],
  };
}
export function createEditorState(
  level: CircuitLevel,
  graph: GraphJSON = initialGraph(level),
  /** Palette list: level primitives plus player components (see availableComponents). */
  available: readonly string[] = level.available,
): EditorState {
  return {
    graph: clone(graph),
    available: [...available],
    selection: [],
    nextId: 1,
    viewport: { x: 0, y: 0, zoom: 1 },
    past: [],
    future: [],
    clipboard: null,
    transaction: null,
    error: null,
  };
}
/** Structural validation deliberately allows incomplete inputs while editing. */
export function connectionError(
  graph: GraphJSON,
  edge: GraphEdge,
  registry: ComponentRegistry = createDefaultRegistry(),
): string | null {
  const source = graph.nodes.find((n) => n.id === edge.from.node),
    target = graph.nodes.find((n) => n.id === edge.to.node);
  if (!source || !target) return 'missing_node';
  if (
    !registry.get(source.component)?.outputs.includes(edge.from.port) ||
    !registry.get(target.component)?.inputs.includes(edge.to.port)
  )
    return 'invalid_port';
  const seen = new Set<string>(),
    pending = [target.id];
  while (pending.length) {
    const id = pending.pop()!;
    if (id === source.id) return 'cycle';
    if (seen.has(id)) continue;
    seen.add(id);
    for (const existing of graph.edges)
      if (existing.from.node === id) pending.push(existing.to.node);
  }
  return null;
}
function connect(
  graph: GraphJSON,
  edge: GraphEdge,
  registry: ComponentRegistry,
): GraphJSON | string {
  const base = { ...graph, edges: graph.edges.filter((e) => !endpointEqual(e.to, edge.to)) };
  const error = connectionError(base, edge, registry);
  return error ?? { ...base, edges: [...base.edges, clone(edge)] };
}
function commit(
  state: EditorState,
  graph: GraphJSON,
  selection = state.selection,
  nextId = state.nextId,
): EditorState {
  if (same(graph, state.graph)) return { ...state, error: null };
  return {
    ...state,
    graph,
    selection,
    nextId,
    error: null,
    past: [...state.past, snapshot(state)],
    future: [],
  };
}
/** Pure reducer. Rejected edits preserve graph/history and expose a localization error key.
 * Supply the same registry used by the level/compiler (including custom components).
 * During a movement transaction only move/commit/cancel commands are accepted.
 */
export function editorReducer(
  state: EditorState,
  command: EditorCommand,
  registry: ComponentRegistry = createDefaultRegistry(),
): EditorState {
  const reject = (error: string): EditorState => ({ ...state, error });
  if (state.transaction && !['moveNodes', 'commitMove', 'cancelMove'].includes(command.type))
    return reject('movement_in_progress');
  switch (command.type) {
    case 'undo': {
      const previous = state.past.at(-1);
      return previous
        ? {
            ...state,
            ...previous,
            past: state.past.slice(0, -1),
            future: [...state.future, snapshot(state)],
            error: null,
          }
        : state;
    }
    case 'redo': {
      const next = state.future.at(-1);
      return next
        ? {
            ...state,
            ...next,
            past: [...state.past, snapshot(state)],
            future: state.future.slice(0, -1),
            error: null,
          }
        : state;
    }
    case 'beginMove':
      return { ...state, transaction: snapshot(state), error: null };
    case 'cancelMove':
      return state.transaction
        ? { ...state, ...state.transaction, transaction: null, error: null }
        : state;
    case 'commitMove': {
      if (!state.transaction) return state;
      return {
        ...state,
        transaction: null,
        error: null,
        past: same(state.graph, state.transaction.graph)
          ? state.past
          : [...state.past, state.transaction],
        future: same(state.graph, state.transaction.graph) ? state.future : [],
      };
    }
    case 'moveNodes': {
      if (
        Object.values(command.positions).some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y))
      )
        return reject('invalid_position');
      const graph = {
        ...state.graph,
        nodes: state.graph.nodes.map((n) =>
          command.positions[n.id] ? { ...n, ui: { ...n.ui, ...command.positions[n.id] } } : n,
        ),
      };
      return state.transaction ? { ...state, graph, error: null } : commit(state, graph);
    }
    case 'addNode': {
      if (
        !state.available.includes(command.component) ||
        !registry.get(command.component) ||
        ['input', 'output'].includes(command.component)
      )
        return reject('unavailable_component');
      if (!Number.isFinite(command.position.x) || !Number.isFinite(command.position.y))
        return reject('invalid_position');
      let nextId = state.nextId;
      while (state.graph.nodes.some((n) => n.id === `node_${nextId}`)) nextId++;
      const id = `node_${nextId++}`;
      return commit(
        state,
        {
          ...state.graph,
          nodes: [
            ...state.graph.nodes,
            {
              id,
              component: command.component,
              attrs: clone(command.attrs ?? {}),
              ui: { ...command.position },
            },
          ],
        },
        [id],
        nextId,
      );
    }
    case 'removeNodes': {
      const ids = new Set(command.ids ?? state.selection);
      const removed = new Set(
        state.graph.nodes.filter((n) => ids.has(n.id) && !protectedNode(n)).map((n) => n.id),
      );
      return commit(
        state,
        {
          ...state.graph,
          nodes: state.graph.nodes.filter((n) => !removed.has(n.id)),
          edges: state.graph.edges.filter(
            (e) => !removed.has(e.from.node) && !removed.has(e.to.node),
          ),
        },
        state.selection.filter((id) => !removed.has(id)),
      );
    }
    case 'connect':
    case 'rewire': {
      if (command.type === 'rewire' && !state.graph.edges.some((e) => edgeEqual(e, command.edge)))
        return reject('missing_edge');
      const base =
        command.type === 'rewire'
          ? { ...state.graph, edges: state.graph.edges.filter((e) => !edgeEqual(e, command.edge)) }
          : state.graph;
      const graph = connect(base, { from: command.from, to: command.to }, registry);
      return typeof graph === 'string' ? reject(graph) : commit(state, graph);
    }
    case 'disconnect':
      return commit(state, {
        ...state.graph,
        edges: state.graph.edges.filter((e) => !edgeEqual(e, command.edge)),
      });
    case 'select': {
      const ids = [...new Set(command.ids)].filter((id) =>
        state.graph.nodes.some((n) => n.id === id),
      );
      const selection = command.shift
        ? [
            ...state.selection.filter((id) => !ids.includes(id)),
            ...ids.filter((id) => !state.selection.includes(id)),
          ]
        : ids;
      return { ...state, selection, error: null };
    }
    case 'boxSelect': {
      const r = command.rect,
        left = Math.min(r.x, r.x + r.width),
        right = Math.max(r.x, r.x + r.width),
        top = Math.min(r.y, r.y + r.height),
        bottom = Math.max(r.y, r.y + r.height);
      const ids = state.graph.nodes
        .filter((n) => {
          const p = nodePosition(n),
            size = command.sizes?.[n.id] ?? { width: 180, height: 100 };
          return (
            p.x <= right && p.x + size.width >= left && p.y <= bottom && p.y + size.height >= top
          );
        })
        .map((n) => n.id);
      return {
        ...state,
        selection: command.shift ? [...new Set([...state.selection, ...ids])] : ids,
        error: null,
      };
    }
    case 'copy': {
      const nodes = state.graph.nodes.filter(
          (n) => state.selection.includes(n.id) && !protectedNode(n),
        ),
        ids = new Set(nodes.map((n) => n.id));
      return {
        ...state,
        clipboard: clone({
          version: 1,
          nodes,
          edges: state.graph.edges.filter((e) => ids.has(e.from.node) && ids.has(e.to.node)),
        }),
        error: null,
      };
    }
    case 'setAttrs': {
      const node = state.graph.nodes.find((n) => n.id === command.id);
      if (!node) return reject('missing_node');
      if (protectedNode(node)) return reject('protected_node');
      const attrs = command.replace
        ? clone(command.attrs)
        : { ...node.attrs, ...clone(command.attrs) };
      // Automatic parameters hide the weight ports; wires into ports that disappear are dropped.
      const hidden =
        attrs.automatic_parameters === true
          ? new Set(registry.get(node.component)?.generatedInputs ?? [])
          : new Set<string>();
      // Slice/Stack/Concat/Split show as many ports as their count/parts settings ask for.
      const liveInputs = tensorOpPorts(node.component, attrs, 'inputs');
      const liveOutputs = tensorOpPorts(node.component, attrs, 'outputs');
      return commit(state, {
        ...state.graph,
        nodes: state.graph.nodes.map((n) => (n.id === command.id ? { ...n, attrs } : n)),
        edges: state.graph.edges.filter(
          (e) =>
            (e.to.node !== command.id ||
              (!hidden.has(e.to.port) && (!liveInputs || liveInputs.includes(e.to.port)))) &&
            (e.from.node !== command.id || !liveOutputs || liveOutputs.includes(e.from.port)),
        ),
      });
    }
    case 'paste':
    case 'insertExpression': {
      const source = command.type === 'paste' ? state.clipboard : command.expression;
      if (!source) return state;
      if (new Set(source.nodes.map((n) => n.id)).size !== source.nodes.length)
        return reject('duplicate_node');
      const offset =
        command.type === 'paste' ? (command.offset ?? { x: 32, y: 32 }) : command.position;
      if (!Number.isFinite(offset.x) || !Number.isFinite(offset.y))
        return reject('invalid_position');
      const remap = new Map<string, string>(),
        occupied = new Set(state.graph.nodes.map((n) => n.id));
      let nextId = state.nextId;
      for (const node of source.nodes) {
        if (
          protectedNode(node) ||
          !state.available.includes(node.component) ||
          !registry.get(node.component)
        )
          return reject('unavailable_component');
        while (occupied.has(`node_${nextId}`)) nextId++;
        const id = `node_${nextId++}`;
        remap.set(node.id, id);
        occupied.add(id);
      }
      const endpoint = (e: GraphEndpoint): GraphEndpoint => ({
        ...e,
        node: remap.get(e.node) ?? e.node,
      });
      const nodes = source.nodes.map((n: GraphNode, i) => ({
        ...clone(n),
        id: remap.get(n.id)!,
        ui:
          command.type === 'paste'
            ? { ...n.ui, x: nodePosition(n).x + offset.x, y: nodePosition(n).y + offset.y }
            : { x: offset.x + i * 220, y: offset.y },
      }));
      let graph: GraphJSON = { ...state.graph, nodes: [...state.graph.nodes, ...nodes] };
      for (const e of source.edges) {
        // Imported expressions may reference existing sources, never existing targets.
        if (!remap.has(e.to.node) || (command.type === 'paste' && !remap.has(e.from.node)))
          return reject('invalid_fragment');
        const result = connect(graph, { from: endpoint(e.from), to: endpoint(e.to) }, registry);
        if (typeof result === 'string') return reject(result);
        graph = result;
      }
      if (command.type === 'insertExpression') {
        const output = endpoint(command.expression.output),
          node = graph.nodes.find((n) => n.id === output.node);
        if (!node || !registry.get(node.component)?.outputs.includes(output.port))
          return reject('invalid_port');
        if (command.to) {
          const result = connect(graph, { from: output, to: command.to }, registry);
          if (typeof result === 'string') return reject(result);
          graph = result;
        }
      }
      return commit(
        state,
        graph,
        nodes.map((n) => n.id),
        nextId,
      );
    }
    case 'setViewport':
      return Number.isFinite(command.viewport.x) && Number.isFinite(command.viewport.y)
        ? {
            ...state,
            viewport: { ...command.viewport, zoom: clampZoom(command.viewport.zoom) },
            error: null,
          }
        : reject('invalid_position');
  }
}
