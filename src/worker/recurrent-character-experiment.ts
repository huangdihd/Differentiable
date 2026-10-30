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
type Peek = Extract<FixedWindowResponse, { type: 'peek' }>;

/** Same cadence as the fixed window: a non-blocking look at both sentences every 40 steps. */
const PEEK_EVERY = 40;

/** Local calibration knobs, not a second general-purpose training configuration system. */
export interface RecurrentCharacterExperimentOptions {
  learningRate?: number;
  stepsPerRun?: number;
  clipNorm?: number;
  seed?: number;
}

/** Ephemeral training of the passed player cell, with fixture-owned embedding/head weights. */
export class RecurrentCharacterExperiment {
  private readonly backend: Backend;
  private readonly cancellation = new AbortController();
  private readonly learningRate: number;
  private readonly stepsPerRun: number;
  private readonly clipNorm: number;
  private readonly seed: number;
  private model?: CharacterRecurrentModel;
  private session?: TrainingSession;
  private batch?: TrainingBatch;
  private running = false;
  private disposed = false;
  private completedRuns = 0;
  private backendDisposed = false;
  private peeking = false;

  constructor(
    private readonly progress: Progress,
    options: RecurrentCharacterExperimentOptions = {},
    backend: Backend = new CpuBackend(),
  ) {
    this.backend = backend;
    this.learningRate = options.learningRate ?? 0.02;
    this.stepsPerRun = options.stepsPerRun ?? 320;
    this.clipNorm = options.clipNorm ?? 1;
    this.seed = options.seed ?? recurrentCharacterSeed;
    if (!(this.learningRate > 0 && Number.isFinite(this.learningRate)))
      throw new Error('training.learning_rate');
    if (!(Number.isInteger(this.stepsPerRun) && this.stepsPerRun > 0))
      throw new Error('training.steps');
  }

  private async initialize(): Promise<TrainingSession> {
    if (this.session) return this.session;
    this.batch = recurrentCharacterBatch(this.backend);
    this.model = new CharacterRecurrentModel(this.backend, {
      progress: this.progress,
      seed: this.seed,
    });
    this.session = new TrainingSession(
      this.model,
      this.batch,
      undefined,
      { kind: 'adam', learningRate: this.learningRate, clipNorm: this.clipNorm },
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

  private predict(reset: boolean): Promise<FixedWindowPrediction[]> {
    return this.startPredictions(reset, true);
  }

  /** Forward passes and readbacks are enqueued synchronously; only the results resolve later. */
  private startPredictions(reset: boolean, strict: boolean): Promise<FixedWindowPrediction[]> {
    const model = this.model;
    if (!model) return Promise.reject(new Error('worker.rnn_uninitialized'));
    const launched: {
      history: string;
      input: string;
      batch: ReturnType<typeof recurrentCharacterPredictionBatch>;
      outputs?: Record<string, Tensor>;
      read?: Promise<Float32Array>;
    }[] = [];
    const releaseAll = () => {
      for (const { outputs, batch } of launched) {
        try {
          if (outputs) model.releaseExecution(outputs, Object.values(batch.inputs));
          releaseRecurrentBatch(batch);
        } catch {
          // A disposed experiment already released everything.
        }
      }
    };
    try {
      for (const { history } of fixedWindowExamples) {
        this.assertActive();
        const input = reset ? [...history].slice(-2).join('') : history;
        const item: (typeof launched)[number] = {
          history,
          input,
          batch: recurrentCharacterPredictionBatch(input, this.backend),
        };
        launched.push(item);
        item.outputs = model.execute(item.batch.inputs);
        const output = item.outputs.probabilities;
        if (output.shape.length !== 2 || output.shape[0] !== 1 || output.shape[1] !== VOCAB.length)
          throw new Error('levels.output_shape');
        item.read = output.read();
        item.read.catch(() => undefined);
      }
    } catch (error) {
      releaseAll();
      return strict ? Promise.reject(error) : Promise.resolve([]);
    }
    return (async () => {
      try {
        const predictions: FixedWindowPrediction[] = [];
        for (const { history, input, read } of launched) {
          const probabilities = Array.from(await read!);
          this.assertActive();
          if (probabilities.some((value) => !Number.isFinite(value) || value < 0)) {
            if (!strict) return [];
            throw new Error('training.nonfinite');
          }
          const total = probabilities.reduce((sum, value) => sum + value, 0);
          if (Math.abs(total - 1) > 1e-5) {
            if (!strict) return [];
            throw new Error('worker.rnn_probabilities');
          }
          const prediction =
            VOCAB[
              probabilities.reduce(
                (best, value, index) => (value > probabilities[best] ? index : best),
                0,
              )
            ];
          predictions.push({ history, input, probabilities, prediction });
        }
        return predictions;
      } catch (error) {
        if (strict) throw error;
        return [];
      } finally {
        releaseAll();
      }
    })();
  }

  async run(
    onProgress?: (progress: RecurrentCharacterProgress) => void,
    onPeek?: (peek: Peek) => void,
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
      const report = await session.run(this.stepsPerRun, {
        signal: this.cancellation.signal,
        reportEvery: 10,
        onReport: ({ step, loss }) => {
          this.assertActive();
          onProgress?.({ type: 'progress', step, totalSteps, loss });
        },
        // Every step posts; the UI batches into animation frames. Loss reports arrive a few steps late.
        onStep: (step) => {
          onProgress?.({ type: 'progress', step, totalSteps });
          if (onPeek && step % PEEK_EVERY === 0 && step < totalSteps && !this.peeking) {
            this.peeking = true;
            void this.startPredictions(false, false)
              .then((predictions) => {
                if (predictions.length && !this.disposed)
                  onPeek?.({ type: 'peek', step, predictions });
              })
              .finally(() => {
                this.peeking = false;
              });
          }
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
        seed: this.seed,
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
