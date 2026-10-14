import * as T from './tensor';
import type { Tensor } from './tensor';

export interface TrainingModel {
  parameters: Map<string, Tensor>;
  execute(inputs: Record<string, Tensor>): Record<string, Tensor>;
  releaseExecution?(outputs: Record<string, Tensor>, keep?: readonly Tensor[]): void;
}
export interface TrainingBatch {
  inputs: Record<string, Tensor>;
  targets: Tensor;
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
  new Set([...Object.values(batch.inputs), batch.targets]).forEach((t) => t.dispose());
}
export interface OptimizerOptions {
  kind: 'sgd' | 'adam';
  learningRate: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
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
/** CPU-reference optimizer. Reads synchronize accelerator buffers; kernels remain backend-independent. */
export class Optimizer {
  step = 0;
  private moments = new Map<string, { m: Float32Array; v: Float32Array }>();
  constructor(readonly options: OptimizerOptions) {
    if (!(options.learningRate > 0 && Number.isFinite(options.learningRate)))
      throw new Error('training.learning_rate');
  }
  async update(parameters: Map<string, Tensor>, gradients: Tensor[]): Promise<void> {
    if (parameters.size !== gradients.length) throw new Error('training.gradient_count');
    this.step++;
    let index = 0;
    for (const [key, parameter] of parameters) {
      const values = await parameter.read(),
        gradient = await gradients[index++].read();
      const { kind, learningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = this.options;
      let state = this.moments.get(key);
      if (!state) {
        state = { m: new Float32Array(values.length), v: new Float32Array(values.length) };
        this.moments.set(key, state);
      }
      const next = new Float32Array(values.length);
      for (let i = 0; i < values.length; i++) {
        let delta = gradient[i];
        if (kind === 'adam') {
          state.m[i] = beta1 * state.m[i] + (1 - beta1) * delta;
          state.v[i] = beta2 * state.v[i] + (1 - beta2) * delta * delta;
          delta =
            state.m[i] /
            (1 - beta1 ** this.step) /
            (Math.sqrt(state.v[i] / (1 - beta2 ** this.step)) + epsilon);
        }
        next[i] = values[i] - learningRate * delta;
      }
      parameters.set(key, T.tensor(next, parameter.shape, parameter.backend));
      parameter.dispose();
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
  accuracy?: boolean;
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
  private getLoss(outputs: Record<string, Tensor>, batch: TrainingBatch): Tensor {
    const loss = this.sessionOptions.lossOutput
      ? outputs[this.sessionOptions.lossOutput]
      : this.loss?.(outputs[this.output], batch.targets);
    if (!loss) throw new Error('training.missing_loss');
    return loss;
  }
  private cleanup(
    outputs: Record<string, Tensor>,
    loss: Tensor | undefined,
    batch: TrainingBatch,
    parameters: Tensor[],
  ): void {
    const keep = [...parameters, ...Object.values(batch.inputs), batch.targets];
    try {
      if (loss) disposeLoss(loss, Object.values(outputs), keep);
      else disposeExecution(Object.values(outputs), keep);
    } finally {
      this.model.releaseExecution?.(outputs, keep);
    }
  }
  /** Resumes this session's parameter and optimizer state, yielding to the UI regularly. */
  async run(
    steps: number,
    options: {
      yieldEvery?: number;
      onReport?: (report: TrainingReport) => void;
      signal?: AbortSignal;
    } = {},
  ): Promise<TrainingReport> {
    if (!Number.isInteger(steps) || steps < 0) throw new Error('training.steps');
    for (let i = 0; i < steps; i++) {
      if (options.signal?.aborted) break;
      const batch =
        typeof this.batch === 'function' ? await this.batch(this.optimizer.step) : this.batch;
      let outputs: Record<string, Tensor> | undefined;
      let parameters: Tensor[] = [],
        loss: Tensor | undefined,
        gradients: Tensor[] = [],
        value: number;
      try {
        outputs = this.model.execute(batch.inputs);
        parameters = [...this.model.parameters.values()];
        loss = this.getLoss(outputs, batch);
        if (loss.shape.length !== 0) throw new Error('training.scalar_loss');
        value = (await loss.read())[0];
        if (!Number.isFinite(value)) throw new Error('training.nonfinite');
        gradients = await T.backward(loss, parameters);
        await this.optimizer.update(this.model.parameters, gradients);
      } finally {
        gradients.forEach((g) => g.dispose());
        try {
          if (outputs) this.cleanup(outputs, loss, batch, parameters);
        } finally {
          if (typeof this.batch === 'function') releaseBatch(batch);
        }
      }
      options.onReport?.({ step: this.optimizer.step, loss: value });
      if ((i + 1) % (options.yieldEvery ?? 10) === 0)
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    return this.evaluate();
  }
  async evaluate(
    source: TrainingBatchSource = this.sessionOptions.evaluationBatch ?? this.batch,
  ): Promise<TrainingReport> {
    const batch = typeof source === 'function' ? await source(this.optimizer.step) : source;
    let outputs: Record<string, Tensor> | undefined, loss: Tensor | undefined;
    try {
      outputs = this.model.execute(batch.inputs);
      loss = this.getLoss(outputs, batch);
      if (loss.shape.length !== 0) throw new Error('training.scalar_loss');
      const value = (await loss.read())[0];
      if (!Number.isFinite(value)) throw new Error('training.nonfinite');
      for (const output of Object.values(outputs)) {
        if (!(await output.read()).every(Number.isFinite)) throw new Error('training.nonfinite');
      }
      const report: TrainingReport = { step: this.optimizer.step, loss: value };
      if (this.sessionOptions.accuracy ?? !this.sessionOptions.lossOutput) {
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
        report.accuracy = correct / count;
      }
      return report;
    } finally {
      try {
        if (outputs) this.cleanup(outputs, loss, batch, [...this.model.parameters.values()]);
      } finally {
        if (typeof source === 'function') releaseBatch(batch);
      }
    }
  }
}
