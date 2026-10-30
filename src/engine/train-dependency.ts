import type { Attributes, ComponentRegistry, GraphJSON } from './graph';
import type { TrainLevel } from './levels';

/** Error key raised when a training graph's prediction ignores the data or has nothing to learn. */
export const TRAIN_DEPENDENCY_ERROR = 'levels.train_no_dependency';

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
  // Target-free objectives (no data inputs at all) only need something to train.
  const hasData = Object.keys(level.interface.inputs).some((name) => name !== target);
  return (data || !hasData) && trainable ? null : TRAIN_DEPENDENCY_ERROR;
}
