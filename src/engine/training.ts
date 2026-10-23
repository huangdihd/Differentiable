import * as T from './tensor';
import type { Tensor } from './tensor';
import { GraphError } from './graph';

/** Structured diagnostics retain the training category across worker/level boundaries. */
export class TrainingError extends GraphError {
  constructor(key: string, node?: string, params: Record<string, unknown> = {}) {
    super(key, node, undefined, params);
    this.message = key;
  }
}

export interface TrainingModel {
  parameters: Map<string, Tensor>;
  execute(inputs: Record<string, Tensor>, trace?: Map<string, Tensor>): Record<string, Tensor>;
  releaseExecution?(outputs: Record<string, Tensor>, keep?: readonly Tensor[]): void;
}
export interface TrainingBatch {
  inputs: Record<string, Tensor>;
  targets?: Tensor;
}
/** Returns fresh, session-owned tensors for the zero-based optimizer step.
 * Called once per update, and for evaluation when no explicit evaluation batch is supplied.
 * All distinct input/target tensors are disposed, including on failure. Return standalone
 * tensors, not model parameters or tensors whose tapes own other allocations.
 * A fixed TrainingBatch, in contrast, remains caller-owned.
 */
export type TrainingBatchFactory = (step: number) => TrainingBatch | Promise<TrainingBatch>;
export type TrainingBatchSource = TrainingBatch | TrainingBatchFactory;
function releaseBatch(batch: TrainingBatch): void {
  new Set([...Object.values(batch.inputs), batch.targets]).forEach((t) => t?.dispose());
}
export interface OptimizerOptions {
  kind: 'sgd' | 'adam';
  learningRate: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}
/** Serializable, bounded node snapshot; size records the full element count. */
export interface TrainingTraceValue {
  shape: number[];
  values: number[];
  size: number;
}
export interface TrainingStepTrace extends TrainingReport {
  forward: Record<string, TrainingTraceValue>;
  gradient: Record<string, TrainingTraceValue>;
  /** Values before/after the update, keyed by canvas parameter node ID. */
  parameters: Record<string, { before: number[]; after: number[] }>;
}
export interface TrainingReport {
  step: number;
  loss: number;
  accuracy?: number;
}
/** Retain graph-owned constants and parameters while releasing a single execution tape. */
export function disposeExecution(outputs: Tensor[], extra: Tensor[] = []): void {
  const leaves = new Set<Tensor>(extra),
    seen = new Set<Tensor>();
  const visit = (x: Tensor) => {
    if (seen.has(x)) return;
    seen.add(x);
    if (!x.tape) leaves.add(x);
    else x.tape.inputs.forEach(visit);
  };
  outputs.forEach(visit);
  T.disposeGraph(outputs, [...leaves]);
}
function disposeLoss(loss: Tensor, outputs: Tensor[], keep: Tensor[]): void {
  const seen = new Set<Tensor>();
  const visit = (x: Tensor) => {
    if (seen.has(x)) return;
    seen.add(x);
    if (!x.tape) keep.push(x);
    else x.tape.inputs.forEach(visit);
  };
  outputs.forEach(visit);
  T.disposeGraph([loss, ...outputs], keep);
}
/** Updates and Adam moments stay on the parameter backend without host readback. */
export class Optimizer {
  step = 0;
  private moments = new Map<string, { m: Tensor; v: Tensor }>();
  constructor(readonly options: OptimizerOptions) {
    if (!(options.learningRate > 0 && Number.isFinite(options.learningRate)))
      throw new TrainingError('training.learning_rate', undefined, {
        actual: String(options.learningRate),
        min: 0,
        max: Number.MAX_VALUE,
        minExclusive: true,
        rangeKey: 'training.learning_rate_range',
      });
  }
  /** Releases optimizer-owned state only; model parameters remain caller-owned. */
  dispose(): void {
    this.moments.forEach(({ m, v }) => {
      m.dispose();
      v.dispose();
    });
    this.moments.clear();
    this.step = 0;
  }
  async update(parameters: Map<string, Tensor>, gradients: Tensor[]): Promise<void> {
    if (parameters.size !== gradients.length) throw new Error('training.gradient_count');
    let index = 0;
    for (const parameter of parameters.values()) {
      const gradient = gradients[index++];
      parameter.assertLive();
      gradient.assertLive();
      if (parameter.backend !== gradient.backend) throw new Error('engine.backend.mismatch');
      if (
        parameter.shape.length !== gradient.shape.length ||
        parameter.shape.some((d, i) => d !== gradient.shape[i])
      )
        throw new Error('training.gradient_shape');
    }
    this.step++;
    index = 0;
    // Bias corrections change every step; passing them as data keeps the update kernel's code
    // (and thus the compiled GPU pipeline) identical across steps.
    const corrections = new Map<Tensor['backend'], Tensor>();
    const correctionsFor = (backend: Tensor['backend']) => {
      let value = corrections.get(backend);
      if (!value) {
        const { beta1 = 0.9, beta2 = 0.999 } = this.options;
        value = T.tensor([1 - beta1 ** this.step, 1 - beta2 ** this.step], [2], backend);
        corrections.set(backend, value);
      }
      return value;
    };
    try {
      for (const [key, parameter] of parameters) {
        const gradient = gradients[index++];
        const { kind, learningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = this.options;
        const allocated: Tensor[] = [];
        const allocate = () => {
          const t = new T.Tensor(parameter.backend, parameter.shape);
          allocated.push(t);
          return t;
        };
        try {
          const next = allocate();
          if (kind === 'adam') {
            const previous = this.moments.get(key);
            const state = previous ?? { m: allocate(), v: allocate() };
            const m = allocate(),
              v = allocate();
            parameter.backend.submit([
              { op: 'optimizer_m', inputs: [state.m, gradient], output: m, attrs: { beta1 } },
              { op: 'optimizer_v', inputs: [state.v, gradient], output: v, attrs: { beta2 } },
              {
                op: 'optimizer_adam',
                inputs: [parameter, m, v, correctionsFor(parameter.backend)],
                output: next,
                attrs: { learningRate, epsilon },
              },
            ]);
            this.moments.set(key, { m, v });
            state.m.dispose();
            state.v.dispose();
            allocated.splice(allocated.indexOf(m), 1);
            allocated.splice(allocated.indexOf(v), 1);
          } else {
            parameter.backend.submit([
              {
                op: 'optimizer_sgd',
                inputs: [parameter, gradient],
                output: next,
                attrs: { learningRate },
              },
            ]);
          }
          parameters.set(key, next);
          allocated.splice(allocated.indexOf(next), 1);
          parameter.dispose();
        } finally {
          allocated.forEach((t) => t.dispose());
        }
      }
    } finally {
      corrections.forEach((t) => t.dispose());
    }
  }
}
export type LossFunction = (prediction: Tensor, target: Tensor) => Tensor;
/** Mean cross entropy of one-hot labels and probabilities. */
export const crossEntropy: LossFunction = (prediction, target) => {
  const n = target.shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const eps = T.tensor([1e-7], [], prediction.backend),
    scale = T.tensor([-1 / n], [], prediction.backend);
  return T.mul(
    T.sum(
      T.mul(target, T.log(T.add(prediction, eps))),
      target.shape.map((_, i) => i),
    ),
    scale,
  );
};
export interface TrainingSessionOptions {
  lossOutput?: string;
  evaluationBatch?: TrainingBatch;
  /** Fresh evaluation plan borrowing the current model parameters; disposed after evaluation. */
  evaluationModel?: () => TrainingModel & { dispose(): void };
  /** Report fixed evaluation loss rather than sampled training loss. */
  evaluationReports?: boolean;
  accuracy?: boolean;
  /** Read and report training loss every N updates (default 10), and on the final update. */
  reportEvery?: number;
}
export class TrainingSession {
  readonly optimizer: Optimizer;
  constructor(
    readonly model: TrainingModel,
    readonly batch: TrainingBatchSource,
    readonly loss: LossFunction | undefined,
    options: OptimizerOptions,
    readonly output = 'y',
    readonly sessionOptions: TrainingSessionOptions = {},
  ) {
    this.optimizer = new Optimizer(options);
  }
  /** Releases optimizer state, not the model or borrowed batches. */
  dispose(): void {
    this.optimizer.dispose();
  }
  private getLoss(outputs: Record<string, Tensor>, batch: TrainingBatch): Tensor {
    if (!this.sessionOptions.lossOutput && !batch.targets)
      throw new Error('training.missing_targets');
    const loss = this.sessionOptions.lossOutput
      ? outputs[this.sessionOptions.lossOutput]
      : this.loss?.(outputs[this.output], batch.targets!);
    if (!loss) throw new Error('training.missing_loss');
    return this.sessionOptions.lossOutput &&
      loss.shape.length > 0 &&
      loss.shape.reduce((a, b) => a * b, 1) === 1
      ? T.reshape(loss, [])
      : loss;
  }
  private cleanup(
    outputs: Record<string, Tensor>,
    loss: Tensor | undefined,
    batch: TrainingBatch,
    parameters: Tensor[],
    model: TrainingModel = this.model,
  ): void {
    const keep = [
      ...parameters,
      ...Object.values(batch.inputs),
      ...(batch.targets ? [batch.targets] : []),
    ];
    try {
      if (loss) disposeLoss(loss, Object.values(outputs), keep);
      else disposeExecution(Object.values(outputs), keep);
    } finally {
      model.releaseExecution?.(outputs, keep);
    }
  }
  /** Failure-only replay: reuse the failing batch and never run backward or an update.
   * Probe insertion order is the model's dependency order (canvas nodes for GraphPlan).
   * Diagnosis is best effort: a replay/readback failure must not hide the original error.
   */
  private async diagnoseNonfinite(
    error: unknown,
    batch: TrainingBatch,
    model: TrainingModel = this.model,
  ): Promise<unknown> {
    if (!(error instanceof Error) || error.message !== 'training.nonfinite') return error;
    const probes = new Map<string, Tensor>();
    let outputs: Record<string, Tensor> | undefined;
    let diagnostic: unknown = new TrainingError('training.nonfinite');
    try {
      outputs = model.execute(batch.inputs, probes);
      for (const [node, tensor] of probes) {
        const values = await tensor.read();
        const index = values.findIndex((value) => !Number.isFinite(value));
        if (index !== -1) {
          diagnostic = new TrainingError('training.nonfinite', node, {
            stageKey: 'training.stage.forward',
            actual: String(values[index]),
            index,
          });
          break;
        }
      }
    } catch {
      // Preserve the original category even when the diagnostic replay cannot complete.
    } finally {
      if (outputs) {
        try {
          this.cleanup(outputs, undefined, batch, [...model.parameters.values()], model);
        } catch {
          // Cleanup of the original execution still runs in the caller's finally block.
        }
      }
    }
    return diagnostic;
  }
  /** Classification accuracy from the same pre-update forward pass as a step trace. */
  private async accuracy(
    outputs: Record<string, Tensor>,
    batch: TrainingBatch,
  ): Promise<number | undefined> {
    if (!(this.sessionOptions.accuracy ?? !this.sessionOptions.lossOutput)) return undefined;
    if (!batch.targets) throw new Error('training.missing_targets');
    const prediction = outputs[this.output];
    const actual = await prediction.read(),
      expected = await batch.targets.read();
    if (![...actual, ...expected].every(Number.isFinite)) throw new Error('training.nonfinite');
    const classes = prediction.shape.at(-1)!,
      count = actual.length / classes;
    if (!(classes > 0 && count > 0) || actual.length !== expected.length)
      throw new Error('training.accuracy_shape');
    let correct = 0;
    for (let i = 0; i < count; i++) {
      let a = 0,
        e = 0;
      for (let j = 1; j < classes; j++) {
        if (actual[i * classes + j] > actual[i * classes + a]) a = j;
        if (expected[i * classes + j] > expected[i * classes + e]) e = j;
      }
      if (a === e) correct++;
    }
    return correct / count;
  }
  /** One live update. Forward values, gradients and loss precede the update; step is 1-based.
   * Resampled sessions show the current sampled batch loss, not held-out evaluation loss.
   * GraphPlan supplies canvas-node probes without expanding composite internals.
   */
  async trainStepTrace(): Promise<TrainingStepTrace> {
    const batch =
      typeof this.batch === 'function' ? await this.batch(this.optimizer.step) : this.batch;
    const probes = new Map<string, Tensor>();
    let outputs: Record<string, Tensor> | undefined, loss: Tensor | undefined;
    let parameters: Tensor[] = [],
      gradients: Tensor[] = [];
    const snapshot = async (tensor: Tensor): Promise<TrainingTraceValue> => {
      const data = await tensor.read();
      if (!data.every(Number.isFinite)) throw new Error('training.nonfinite');
      return {
        shape: [...tensor.shape],
        values: Array.from(data.slice(0, 256)),
        size: data.length,
      };
    };
    try {
      outputs = this.model.execute(batch.inputs, probes);
      parameters = [...this.model.parameters.values()];
      loss = this.getLoss(outputs, batch);
      if (loss.shape.length !== 0) throw new Error('training.scalar_loss');
      const value = (await loss.read())[0];
      if (!Number.isFinite(value)) throw new Error('training.nonfinite');
      const nodes = [...probes.entries()];
      const ancestors = new Set<Tensor>();
      const visit = (tensor: Tensor): void => {
        if (ancestors.has(tensor)) return;
        ancestors.add(tensor);
        tensor.tape?.inputs.forEach(visit);
      };
      visit(loss);
      // Deduplicate aliased output/input probes before requesting reverse-mode gradients.
      const targets = [...new Set([...parameters, ...nodes.map(([, t]) => t)])];
      gradients = await T.backward(loss, targets);
      const byTensor = new Map(targets.map((t, i) => [t, gradients[i]]));
      const trace: TrainingStepTrace = {
        step: this.optimizer.step + 1,
        loss: value,
        forward: {},
        gradient: {},
        parameters: {},
      };
      const accuracy = await this.accuracy(outputs, batch);
      if (accuracy !== undefined) trace.accuracy = accuracy;
      for (const [id, tensor] of nodes) {
        trace.forward[id] = await snapshot(tensor);
        if (ancestors.has(tensor)) trace.gradient[id] = await snapshot(byTensor.get(tensor)!);
      }
      const parameterGroups = new Map<string, { keys: string[]; tensors: Tensor[] }>();
      for (const [key, tensor] of this.model.parameters) {
        const id = decodeURIComponent(key.split('/')[0]);
        if (!probes.has(id)) continue;
        const group = parameterGroups.get(id) ?? { keys: [], tensors: [] };
        group.keys.push(key);
        group.tensors.push(tensor);
        parameterGroups.set(id, group);
      }
      for (const [id, group] of parameterGroups)
        trace.parameters[id] = {
          before: (
            await Promise.all(group.tensors.map(async (tensor) => Array.from(await tensor.read())))
          ).flat(),
          after: [],
        };
      const parameterProbes = [...parameterGroups].flatMap(([id, group]) =>
        group.keys.map((key) => ({ id, key })),
      );
      const after = new Map<string, number[]>();
      for (const id of parameterGroups.keys()) after.set(id, []);
      await this.optimizer.update(
        this.model.parameters,
        parameters.map((p) => byTensor.get(p)!),
      );
      for (const { key, id } of parameterProbes)
        after.get(id)!.push(...(await this.model.parameters.get(key)!.read()));
      for (const [id, values] of after) trace.parameters[id].after = values;
      for (const parameter of Object.values(trace.parameters))
        if (!parameter.after.every(Number.isFinite)) throw new Error('training.nonfinite');
      return trace;
    } catch (error) {
      throw await this.diagnoseNonfinite(error, batch);
    } finally {
      gradients.forEach((g) => g.dispose());
      try {
        if (outputs) this.cleanup(outputs, loss, batch, parameters);
      } finally {
        if (typeof this.batch === 'function') releaseBatch(batch);
      }
    }
  }
  /** Alias for worker callers. */
  stepTrace(): Promise<TrainingStepTrace> {
    return this.trainStepTrace();
  }
  /** Resumes this session's parameter and optimizer state, yielding to the UI regularly. */
  async run(
    steps: number,
    options: {
      yieldEvery?: number;
      reportEvery?: number;
      onReport?: (report: TrainingReport) => void;
      signal?: AbortSignal;
    } = {},
  ): Promise<TrainingReport> {
    if (!Number.isInteger(steps) || steps < 0) throw new Error('training.steps');
    const reportEvery = options.reportEvery ?? this.sessionOptions.reportEvery ?? 10;
    if (!Number.isInteger(reportEvery) || reportEvery < 1) throw new Error('training.report_every');
    let evaluation: TrainingReport | undefined;
    for (let i = 0; i < steps; i++) {
      if (options.signal?.aborted) break;
      const batch =
        typeof this.batch === 'function' ? await this.batch(this.optimizer.step) : this.batch;
      const report = (this.optimizer.step + 1) % reportEvery === 0 || i === steps - 1;
      let outputs: Record<string, Tensor> | undefined;
      let parameters: Tensor[] = [],
        loss: Tensor | undefined,
        gradients: Tensor[] = [],
        value: number | undefined;
      try {
        outputs = this.model.execute(batch.inputs);
        parameters = [...this.model.parameters.values()];
        loss = this.getLoss(outputs, batch);
        if (loss.shape.length !== 0) throw new Error('training.scalar_loss');
        if (report && !this.sessionOptions.evaluationReports) {
          value = (await loss.read())[0];
          if (!Number.isFinite(value)) throw new Error('training.nonfinite');
        }
        gradients = await T.backward(loss, parameters);
        await this.optimizer.update(this.model.parameters, gradients);
      } catch (error) {
        throw await this.diagnoseNonfinite(error, batch);
      } finally {
        gradients.forEach((g) => g.dispose());
        try {
          if (outputs) this.cleanup(outputs, loss, batch, parameters);
        } finally {
          if (typeof this.batch === 'function') releaseBatch(batch);
        }
      }
      if (this.sessionOptions.evaluationReports && report) {
        evaluation = await this.evaluate();
        options.onReport?.(evaluation);
      }
      if (value !== undefined) options.onReport?.({ step: this.optimizer.step, loss: value });
      if ((i + 1) % (options.yieldEvery ?? 10) === 0)
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    return evaluation?.step === this.optimizer.step ? evaluation : this.evaluate();
  }
  async evaluate(
    source: TrainingBatchSource = this.sessionOptions.evaluationBatch ?? this.batch,
  ): Promise<TrainingReport> {
    const batch = typeof source === 'function' ? await source(this.optimizer.step) : source;
    let evaluationModel: (TrainingModel & { dispose(): void }) | undefined;
    let outputs: Record<string, Tensor> | undefined, loss: Tensor | undefined;
    try {
      evaluationModel = this.sessionOptions.evaluationModel?.();
      outputs = (evaluationModel ?? this.model).execute(batch.inputs);
      loss = this.getLoss(outputs, batch);
      if (loss.shape.length !== 0) throw new Error('training.scalar_loss');
      const value = (await loss.read())[0];
      if (!Number.isFinite(value)) throw new Error('training.nonfinite');
      for (const output of Object.values(outputs)) {
        if (!(await output.read()).every(Number.isFinite)) throw new Error('training.nonfinite');
      }
      const report: TrainingReport = { step: this.optimizer.step, loss: value };
      const accuracy = await this.accuracy(outputs, batch);
      if (accuracy !== undefined) report.accuracy = accuracy;
      return report;
    } catch (error) {
      throw await this.diagnoseNonfinite(error, batch, evaluationModel ?? this.model);
    } finally {
      try {
        if (outputs)
          this.cleanup(
            outputs,
            loss,
            batch,
            [...this.model.parameters.values()],
            evaluationModel ?? this.model,
          );
      } finally {
        evaluationModel?.dispose();
        if (typeof source === 'function') releaseBatch(batch);
      }
    }
  }
}
