import type { GraphJSON, GraphPlan } from '../engine/graph';
import type { TrainingBatchSource } from '../engine/training';
import { tensor, type Tensor } from '../engine/tensor';
import type { ClassField, PlotData } from '../ui/DataPlot';
import type { TrainLevel } from '../engine/levels';
import { predictionSource, sequenceProbe } from '../levels/sequence-probe';

export const TRAINING_PLOT_INTERVAL_MS = 250;
/** Decision-field resolution; drawn under the points of two-feature classification plots. */
export const FIELD_NX = 32,
  FIELD_NY = 20;
/** The field costs ~nx·ny/n extra model runs, so it refreshes less often than the points. */
export const FIELD_INTERVAL_MS = 800;

/** Await inside the runtime queue, never launch a concurrent model execution/readback. */
export class TrainingPlotRefresh {
  private last = -Infinity;
  private lastField = -Infinity;
  private cached?: ClassField;
  constructor(private readonly now: () => number = () => performance.now()) {}
  /** Recompute the decision field at most every FIELD_INTERVAL_MS (always when final). */
  async field(compute: () => Promise<ClassField | undefined>, final = false) {
    const now = this.now();
    if (!final && now - this.lastField < FIELD_INTERVAL_MS) return this.cached;
    this.lastField = now;
    this.cached = await compute();
    return this.cached;
  }
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
  /** Gates the decision field (e.g. TrainingPlotRefresh.field); computed every call if omitted. */
  fieldGate: (compute: () => Promise<ClassField | undefined>) => Promise<ClassField | undefined> = (
    compute,
  ) => compute(),
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
    const points = Array.from({ length: n }, (_, i): [number, number] => [
      xs[2 * i],
      xs[2 * i + 1],
    ]);
    const field = await fieldGate(() => classField(plan, batch.inputs, x, points, output, c));
    return {
      kind: 'classes',
      points,
      label: labels(ts),
      predicted: labels(values),
      ...(field ? { field } : {}),
    };
  } finally {
    plan.releaseExecution(outputs, [
      ...Object.values(batch.inputs),
      ...(batch.targets ? [batch.targets] : []),
    ]);
  }
}

/** Same padding as the plot's axes, so the field fills the visible plane. */
function padded(values: number[]): [number, number] {
  const lo = Math.min(...values),
    hi = Math.max(...values);
  const pad = (hi - lo || 1) * 0.08;
  return [lo - pad, hi + pad];
}

/**
 * Evaluate the model over a grid of the plane. The plan was compiled for the dataset's batch
 * size, so the grid runs in chunks of exactly n rows (the last chunk padded); other inputs
 * (targets) are passed through unchanged and only the model output is read.
 */
async function classField(
  plan: GraphPlan,
  inputs: Record<string, Tensor>,
  x: Tensor,
  points: [number, number][],
  output: string,
  classes: number,
): Promise<ClassField | undefined> {
  const n = points.length;
  const [x0, x1] = padded(points.map((p) => p[0]));
  const [y0, y1] = padded(points.map((p) => p[1]));
  const total = FIELD_NX * FIELD_NY;
  const cell = (k: number): [number, number] => [
    x0 + (((k % FIELD_NX) + 0.5) / FIELD_NX) * (x1 - x0),
    y0 + ((Math.floor(k / FIELD_NX) + 0.5) / FIELD_NY) * (y1 - y0),
  ];
  // Launch every chunk first, then read them together: readbacks overlap instead of
  // each waiting for the GPU in turn.
  const runs: { chunk: Tensor; outputs: ReturnType<GraphPlan['execute']> }[] = [];
  const result: number[] = [],
    margin: number[] = [];
  try {
    for (let start = 0; start < total; start += n) {
      const data = new Float32Array(2 * n);
      for (let r = 0; r < n; r++) {
        const [a, b] = cell(Math.min(total - 1, start + r));
        data[2 * r] = a;
        data[2 * r + 1] = b;
      }
      const chunk = tensor(data, [n, 2], x.backend);
      runs.push({ chunk, outputs: plan.execute({ ...inputs, x: chunk }, new Map()) });
    }
    const ys = runs.map((run) => run.outputs[output]);
    if (ys.some((y) => !y || y.shape.length !== 2 || y.shape[0] !== n || y.shape[1] !== classes))
      return undefined;
    const all = await Promise.all(ys.map((y) => y!.read()));
    for (let k = 0; k < total; k++) {
      const values = all[Math.floor(k / n)],
        r = k % n;
      let best = 0;
      for (let j = 1; j < classes; j++)
        if (values[r * classes + j] > values[r * classes + best]) best = j;
      if (!Number.isFinite(values[r * classes + best])) return undefined;
      result.push(best);
      if (classes === 2) margin.push(values[r * 2 + 1] - values[r * 2]);
    }
  } finally {
    for (const run of runs) {
      plan.releaseExecution(run.outputs, Object.values(inputs));
      run.chunk.dispose();
    }
  }
  return {
    nx: FIELD_NX,
    ny: FIELD_NY,
    x0,
    x1,
    y0,
    y1,
    classes: result,
    ...(classes === 2 ? { margin } : {}),
  };
}

/**
 * Sequence levels: run the model on a fixed sweep of windows (in chunks of the plan's batch)
 * and pair the predictions with the target curve. Inside the runtime queue, like trainingPlot.
 */
export async function sequencePlot(
  level: TrainLevel,
  graph: GraphJSON,
  plan: GraphPlan,
): Promise<PlotData | undefined> {
  const kind = level.sequenceProbe;
  const shape = level.interface.inputs.x;
  const source = predictionSource(graph);
  if (!kind || !shape || shape.length !== 2 || !source) return undefined;
  const [batch, length] = shape;
  const probe = sequenceProbe(kind, length);
  const runs: { inputs: Record<string, Tensor>; outputs: ReturnType<GraphPlan['execute']> }[] = [];
  const traces: Map<string, Tensor>[] = [];
  try {
    for (let start = 0; start < probe.rows.length; start += batch) {
      const data = new Float32Array(batch * length);
      for (let r = 0; r < batch; r++)
        data.set(probe.rows[Math.min(probe.rows.length - 1, start + r)], r * length);
      const inputs: Record<string, Tensor> = {
        x: tensor(data, [batch, length]),
        ...(level.interface.inputs.t ? { t: tensor(new Float32Array(batch), [batch]) } : {}),
      };
      const trace = new Map<string, Tensor>();
      traces.push(trace);
      runs.push({ inputs, outputs: plan.execute(inputs, trace) });
    }
    const predictions = traces.map((trace) => trace.get(source));
    if (predictions.some((p) => !p || p.shape.reduce((a, b) => a * b, 1) !== batch))
      return undefined;
    const all = await Promise.all(predictions.map((p) => p!.read()));
    const p = probe.rows.map((_, k) => all[Math.floor(k / batch)][k % batch]);
    if (!p.every(Number.isFinite)) return undefined;
    return {
      kind: 'regression',
      x: probe.curve.x,
      t: probe.curve.t,
      fit: { x: probe.at, p },
      targetLine: probe.targetLine,
    };
  } finally {
    for (const run of runs) {
      plan.releaseExecution(run.outputs, Object.values(run.inputs));
      Object.values(run.inputs).forEach((t) => t.dispose());
    }
  }
}
