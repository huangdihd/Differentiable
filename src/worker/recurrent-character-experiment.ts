import { CpuBackend } from '../engine/cpu';
import type { Backend } from '../engine/backend';
import type { Progress } from '../engine/levels';
import type { Tensor } from '../engine/tensor';
import { TrainingSession, type TrainingBatch } from '../engine/training';
import { characterComparisonPositions } from '../levels/character-comparison-data';
import { VOCAB } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import {
  CharacterRecurrentModel,
  recurrentCharacterBatch,
  recurrentCharacterPredictionBatch,
  recurrentCharacterSeed,
  releaseRecurrentBatch,
} from './character-recurrent-model';
import type {
  FixedWindowPrediction,
  FixedWindowResponse,
  FixedWindowSnapshot,
} from './fixed-window-protocol';

type RecurrentCharacterProgress = Extract<FixedWindowResponse, { type: 'progress' }>;

/** Local calibration knobs, not a second general-purpose training configuration system. */
export interface RecurrentCharacterExperimentOptions {
  learningRate?: number;
  stepsPerRun?: number;
}

/** Ephemeral training of the passed player cell, with fixture-owned embedding/head weights. */
export class RecurrentCharacterExperiment {
  private readonly backend: Backend;
  private readonly cancellation = new AbortController();
  private readonly learningRate: number;
  private readonly stepsPerRun: number;
  private model?: CharacterRecurrentModel;
  private session?: TrainingSession;
  private batch?: TrainingBatch;
  private running = false;
  private disposed = false;
  private completedRuns = 0;
  private backendDisposed = false;

  constructor(
    private readonly progress: Progress,
    options: RecurrentCharacterExperimentOptions = {},
    backend: Backend = new CpuBackend(),
  ) {
    this.backend = backend;
    this.learningRate = options.learningRate ?? 0.01;
    this.stepsPerRun = options.stepsPerRun ?? 320;
    if (!(this.learningRate > 0 && Number.isFinite(this.learningRate)))
      throw new Error('training.learning_rate');
    if (!(Number.isInteger(this.stepsPerRun) && this.stepsPerRun > 0))
      throw new Error('training.steps');
  }

  private async initialize(): Promise<TrainingSession> {
    if (this.session) return this.session;
    this.batch = recurrentCharacterBatch(this.backend);
    this.model = new CharacterRecurrentModel(this.backend, { progress: this.progress });
    this.session = new TrainingSession(
      this.model,
      this.batch,
      undefined,
      { kind: 'adam', learningRate: this.learningRate },
      'probabilities',
      { accuracy: false, lossOutput: 'loss' },
    );
    await this.session.evaluate();
    this.assertActive();
    return this.session;
  }

  private assertActive(): void {
    if (this.disposed) throw new Error('worker.rnn_disposed');
  }

  private async predict(reset: boolean): Promise<FixedWindowPrediction[]> {
    if (!this.model) throw new Error('worker.rnn_uninitialized');
    const predictions: FixedWindowPrediction[] = [];
    for (const { history } of fixedWindowExamples) {
      this.assertActive();
      const input = reset ? [...history].slice(-2).join('') : history;
      const batch = recurrentCharacterPredictionBatch(input, this.backend);
      let outputs: Record<string, Tensor> | undefined;
      try {
        outputs = this.model.execute(batch.inputs);
        const output = outputs.probabilities;
        if (output.shape.length !== 2 || output.shape[0] !== 1 || output.shape[1] !== VOCAB.length)
          throw new Error('levels.output_shape');
        const probabilities = Array.from(await output.read());
        this.assertActive();
        if (probabilities.some((value) => !Number.isFinite(value) || value < 0))
          throw new Error('training.nonfinite');
        const total = probabilities.reduce((sum, value) => sum + value, 0);
        if (Math.abs(total - 1) > 1e-5) throw new Error('worker.rnn_probabilities');
        const prediction =
          VOCAB[
            probabilities.reduce(
              (best, value, index) => (value > probabilities[best] ? index : best),
              0,
            )
          ];
        predictions.push({ history, input, probabilities, prediction });
      } finally {
        if (outputs) this.model.releaseExecution(outputs, Object.values(batch.inputs));
        releaseRecurrentBatch(batch);
      }
    }
    return predictions;
  }

  async run(
    onProgress?: (progress: RecurrentCharacterProgress) => void,
  ): Promise<FixedWindowSnapshot> {
    this.assertActive();
    if (this.running) throw new Error('worker.training_busy');
    if (this.completedRuns >= 2) throw new Error('worker.rnn_complete');
    this.running = true;
    try {
      // Report immediately: the first run also pays for backend setup and kernel compilation.
      const firstStep = this.completedRuns * this.stepsPerRun;
      onProgress?.({ type: 'progress', step: firstStep, totalSteps: firstStep + this.stepsPerRun });
      const session = await this.initialize();
      const totalSteps = session.optimizer.step + this.stepsPerRun;
      let lastPost = 0;
      const report = await session.run(this.stepsPerRun, {
        signal: this.cancellation.signal,
        reportEvery: 10,
        onReport: ({ step, loss }) => {
          this.assertActive();
          lastPost = performance.now();
          onProgress?.({ type: 'progress', step, totalSteps, loss });
        },
        // Step-only updates keep the bar moving between loss reports (throttled, no GPU readback).
        onStep: (step) => {
          const now = performance.now();
          if (now - lastPost < 80) return;
          lastPost = now;
          onProgress?.({ type: 'progress', step, totalSteps });
        },
      });
      this.assertActive();
      const predictions = await this.predict(false);
      const resetPredictions = await this.predict(true);
      this.assertActive();
      this.completedRuns++;
      return {
        model: 'rnn',
        parameterCount: [...this.model!.parameters.values()].reduce(
          (total, parameter) =>
            total + parameter.shape.reduce((product, dimension) => product * dimension, 1),
          0,
        ),
        sampleCount: characterComparisonPositions.length,
        step: report.step,
        loss: report.loss,
        stepsPerRun: this.stepsPerRun,
        seed: recurrentCharacterSeed,
        learningRate: this.learningRate,
        predictions,
        resetPredictions,
      };
    } catch (error) {
      // An interrupted update is not transactional. Never retry a partly trained experiment.
      this.dispose();
      throw error;
    } finally {
      this.running = false;
      if (this.disposed) this.release();
    }
  }

  private release(): void {
    this.session?.dispose();
    this.session = undefined;
    this.model?.dispose();
    this.model = undefined;
    if (this.batch) releaseRecurrentBatch(this.batch);
    this.batch = undefined;
    if (!this.backendDisposed) {
      this.backend.dispose?.();
      this.backendDisposed = true;
    }
  }

  dispose(): void {
    this.disposed = true;
    this.cancellation.abort();
    // In-flight execution/readback owns its tape until the surrounding finally completes.
    if (!this.running) this.release();
  }
}
