import type { GraphJSON, GraphPlan } from '../engine/graph';
import type { TrainingBatchSource } from '../engine/training';
import type { Tensor } from '../engine/tensor';
import type { PlotData } from '../ui/DataPlot';

export const TRAINING_PLOT_INTERVAL_MS = 250;

/** Await inside the runtime queue, never launch a concurrent model execution/readback. */
export class TrainingPlotRefresh {
  private last = -Infinity;
  constructor(private readonly now: () => number = () => performance.now()) {}
  async refresh(capture: () => Promise<PlotData | undefined>, final = false) {
    const now = this.now();
    if (!final && now - this.last < TRAINING_PLOT_INTERVAL_MS) return undefined;
    this.last = now;
    return capture();
  }
}

/** Fixed coordinate datasets only. Tokens and resampled minibatches are not planar plots. */
export async function trainingPlot(
  graph: GraphJSON,
  plan: GraphPlan,
  batch: TrainingBatchSource,
  output: string,
): Promise<PlotData | undefined> {
  if (typeof batch === 'function') return undefined;
  const x = batch.inputs.x,
    target = batch.inputs.t ?? batch.targets;
  if (!x || !target) return undefined;
  const regression = x.shape.length === 1 && target.shape.join() === x.shape.join();
  const classes = x.shape.length === 2 && x.shape[1] === 2 && target.shape.length === 2;
  if (!regression && !classes) return undefined;
  const probes = new Map<string, Tensor>();
  const outputs = plan.execute(batch.inputs, probes);
  try {
    const edge = regression
      ? graph.edges.find(
          (e) =>
            e.to.port === 'p' && graph.nodes.find((n) => n.id === e.to.node)?.component === 'mse',
        )
      : undefined;
    const prediction = regression ? edge && probes.get(edge.from.node) : outputs[output];
    if (!prediction) return undefined;
    const xs = Array.from(await x.read());
    const ts = Array.from(await target.read());
    const values = Array.from(await prediction.read());
    if (!values.every(Number.isFinite)) return undefined;
    if (regression)
      return values.length === xs.length
        ? { kind: 'regression', x: xs, t: ts, p: values }
        : undefined;
    const n = x.shape[0],
      c = target.shape[1];
    if (prediction.shape.length !== 2 || prediction.shape[0] !== n || prediction.shape[1] !== c)
      return undefined;
    const labels = (data: number[]) =>
      Array.from({ length: n }, (_, i) => {
        let best = 0;
        for (let j = 1; j < c; j++) if (data[i * c + j] > data[i * c + best]) best = j;
        return best;
      });
    return {
      kind: 'classes',
      points: Array.from({ length: n }, (_, i) => [xs[2 * i], xs[2 * i + 1]]),
      label: labels(ts),
      predicted: labels(values),
    };
  } finally {
    plan.releaseExecution(outputs, [
      ...Object.values(batch.inputs),
      ...(batch.targets ? [batch.targets] : []),
    ]);
  }
}
