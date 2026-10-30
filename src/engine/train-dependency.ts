import type { Attributes, ComponentRegistry, GraphJSON } from './graph';
import type { TrainLevel } from './levels';

/** Error key raised when a training graph's prediction ignores the data or has nothing to learn. */
export const TRAIN_DEPENDENCY_ERROR = 'levels.train_no_dependency';
/** Error key raised when a memory level's readout looks at the sequence other than via the final state. */
export const FINAL_STATE_ERROR = 'levels.train_final_state_only';

function componentTrains(
  registry: ComponentRegistry,
  component: string,
  attrs: Attributes = {},
  depth = 0,
): boolean {
  if (component === 'param') return true;
  const definition = registry.get(component);
  if (!definition || depth > 8) return false;
  if (definition.generatedInputs?.length && attrs.automatic_parameters !== false) return true;
  let inner: GraphJSON | undefined;
  try {
    inner =
      typeof definition.subgraph === 'function' ? definition.subgraph(attrs) : definition.subgraph;
  } catch {
    return false;
  }
  return !!inner?.nodes.some((node) =>
    componentTrains(registry, node.component, node.attrs, depth + 1),
  );
}

/** The endpoint whose value is the model's prediction: the locked loss's prediction port,
 * or the level output itself when the trainer applies the loss outside the graph. */
function predictionSink(level: TrainLevel, graph: GraphJSON): string | undefined {
  const lock = level.lockedLoss;
  const name = lock ? (level.lossOutput ?? level.interface.output) : level.interface.output;
  const output = graph.nodes.find(
    (n) => n.component === 'output' && (n.attrs?.name ?? 'out') === name,
  );
  const feeding = output && graph.edges.find((e) => e.to.node === output.id)?.from.node;
  if (!lock) return feeding;
  const loss = graph.nodes.find((n) => n.id === feeding && n.component === lock.component);
  return loss
    ? graph.edges.find((e) => e.to.node === loss.id && e.to.port === lock.predictionPort)?.from.node
    : undefined;
}

/**
 * Cheap structural anti-cheat: the prediction must be reachable from a data input and from at
 * least one trainable parameter. A constant (or the raw input) wired into the loss trains nothing.
 */
export function trainDependencyError(
  level: TrainLevel,
  graph: GraphJSON,
  registry: ComponentRegistry,
): string | null {
  const sink = predictionSink(level, graph);
  if (!sink) return level.lockedLoss ? TRAIN_DEPENDENCY_ERROR : null;
  const target = level.lockedLoss?.target;
  const seen = new Set<string>();
  const pending = [sink];
  let data = false;
  let trainable = false;
  while (pending.length) {
    const id = pending.pop()!;
    if (seen.has(id)) continue;
    seen.add(id);
    const node = graph.nodes.find((n) => n.id === id);
    if (!node) continue;
    if (node.component === 'input') {
      const name = String(node.attrs?.name ?? '');
      if (name !== target && name in level.interface.inputs) data = true;
      continue;
    }
    if (componentTrains(registry, node.component, node.attrs)) trainable = true;
    for (const edge of graph.edges) if (edge.to.node === id) pending.push(edge.from.node);
  }
  if (level.finalStateOnly && finalStateLeak(level, graph, seen)) return FINAL_STATE_ERROR;
  // Target-free objectives (no data inputs at all) only need something to train.
  const hasData = Object.keys(level.interface.inputs).some((name) => name !== target);
  return (data || !hasData) && trainable ? null : TRAIN_DEPENDENCY_ERROR;
}

/**
 * Memory levels: within the prediction's dependency cone, the sequence input may only enter an
 * RNN window's x port, and an RNN window's history may only leave through a slice of its last
 * step. Otherwise the readout could skip the recurrence, e.g. slice step 0 and decode it.
 */
function finalStateLeak(level: TrainLevel, graph: GraphJSON, cone: ReadonlySet<string>): boolean {
  const { input, recurrent } = level.finalStateOnly!;
  const length = level.interface.inputs[input]?.[1];
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  for (const edge of graph.edges) {
    if (!cone.has(edge.to.node)) continue;
    const from = byId.get(edge.from.node),
      to = byId.get(edge.to.node);
    if (!from || !to) continue;
    if (from.component === 'input' && from.attrs?.name === input) {
      if (to.component !== recurrent || edge.to.port !== 'x') return true;
    } else if (from.component === recurrent) {
      const a = to.attrs ?? {};
      const index = Number(a.index ?? 0);
      const last = index === -1 || (length !== undefined && index === length - 1);
      const axis = Number(a.axis ?? 1);
      if (
        to.component !== 'slice' ||
        (axis !== 1 && axis !== -2) ||
        !last ||
        Number(a.length ?? 1) !== 1
      )
        return true;
    }
  }
  return false;
}
