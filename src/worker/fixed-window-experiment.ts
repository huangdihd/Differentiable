// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { setTrainingDebug } from '../engine/training-debug';
import type { Backend } from '../engine/backend';
import { CpuBackend } from '../engine/cpu';
import { createBackend } from '../engine/backend-selection';
import { compileGraph, GraphError, type GraphPlan, type TensorPorts } from '../engine/graph';
import { trainingBatchSource, type ForwardLevel, type Progress } from '../engine/levels';
import { Random } from '../engine/random';
import { tensor, type Tensor } from '../engine/tensor';
import { crossEntropy, TrainingSession, type TrainingBatch } from '../engine/training';
import { gameLevels } from '../levels/catalog';
import { chapter4TrainLevels } from '../levels/chapter4';
import {
  characterComparisonLines,
  characterComparisonPositions,
} from '../levels/character-comparison-data';
import { encode, VOCAB } from '../levels/corpus';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import { referenceRegistry } from '../levels/level-helpers';
import { availableComponents } from '../levels/registry';
import { workerError } from './error';
import { fixedWindowBatchRegistry } from './fixed-window-batch';
import { playerExperimentGraph } from './player-experiment-graph';
import { RecurrentCharacterExperiment } from './recurrent-character-experiment';
import type {
  CharacterComparisonModel,
  FixedWindowPrediction,
  FixedWindowRequest,
  FixedWindowResponse,
  FixedWindowSnapshot,
} from './fixed-window-protocol';

type FixedWindowProgress = Extract<FixedWindowResponse, { type: 'progress' }>;
type Peek = Extract<FixedWindowResponse, { type: 'peek' }>;

/** Mid-training predictions: one non-blocking forward + readback at most every this many steps. */
export const PEEK_EVERY = 40;

function releaseBatch(batch: TrainingBatch): void {
  new Set([...Object.values(batch.inputs), batch.targets]).forEach((value) => value?.dispose());
}

/** The comparison owns all tensors and never changes the engine's default backend. */
function fixedComparisonBatch(backend: Backend): TrainingBatch {
  const count = characterComparisonPositions.length;
  const inputs: number[] = [];
  const targets = new Float32Array(count * VOCAB.length);
  characterComparisonPositions.forEach(({ lineIndex, offset }, row) => {
    const { tokens } = characterComparisonLines[lineIndex];
    inputs.push(...tokens.slice(offset - 2, offset));
    targets[row * VOCAB.length + tokens[offset]] = 1;
  });
  const x = tensor(inputs, [count, 2], backend);
  try {
    return {
      inputs: { x },
      targets: tensor(targets, [count, VOCAB.length], backend),
    };
  } catch (error) {
    x.dispose();
    throw error;
  }
}

/** Copy factory-owned tensors onto the experiment backend without changing the global backend. */
async function takeBatch(source: TrainingBatch, backend: Backend): Promise<TrainingBatch> {
  const copies = new Map<Tensor, Tensor>();
  try {
    for (const value of new Set([...Object.values(source.inputs), source.targets])) {
      if (value) copies.set(value, tensor(await value.read(), value.shape, backend));
    }
    return {
      inputs: Object.fromEntries(
        Object.entries(source.inputs).map(([name, value]) => [name, copies.get(value)!]),
      ),
      targets: source.targets ? copies.get(source.targets) : undefined,
    };
  } catch (error) {
    copies.forEach((value) => value.dispose());
    throw error;
  } finally {
    releaseBatch(source);
  }
}

/** 5-1 uses its original reference; post-build comparisons require the player's actual circuit. */
export class FixedWindowExperiment {
  private readonly backend: Backend;
  private readonly level = chapter4TrainLevels[1];
  private readonly cancellation = new AbortController();
  private trainingPlan?: GraphPlan;
  private session?: TrainingSession;
  private readonly batches: TrainingBatch[] = [];
  private running = false;
  private disposed = false;
  private completedRuns = 0;
  private backendDisposed = false;
  private peeking = false;

  private readonly learningRate: number;
  private readonly stepsPerRun: number;
  private readonly playerCircuit?: ReturnType<typeof playerExperimentGraph>;
  private readonly playerComponents?: string[];

  constructor(playerProgress?: Progress, backend: Backend = new CpuBackend()) {
    this.backend = backend;
    this.learningRate = playerProgress ? 0.02 : 0.03;
    this.stepsPerRun = playerProgress ? 320 : 160;
    if (!playerProgress) return;
    const player = playerExperimentGraph(playerProgress, this.level.id);
    this.playerComponents = availableComponents(this.level, playerProgress);
    this.playerCircuit = {
      graph: player.graph,
      registry: fixedWindowBatchRegistry(
        player.graph,
        player.registry,
        this.level.interface.inputs,
        this.playerComponents,
      ),
    };
  }

  private compile(
    inputShapes: Record<string, readonly number[]>,
    parameters?: Map<string, Tensor>,
  ) {
    if (this.playerCircuit)
      return compileGraph(this.playerCircuit.graph, {
        backend: this.backend,
        registry: this.playerCircuit.registry,
        preferNative: false,
        availableComponents: this.playerComponents,
        inputShapes,
        random: new Random(this.level.seed),
        parameters,
      });
    const levelIndex = gameLevels.findIndex((level) => level.id === this.level.id);
    if (levelIndex < 0 || !this.level.referenceGraph)
      throw new Error('worker.fixed_window_reference_missing');
    const preceding = gameLevels
      .slice(0, levelIndex)
      .filter((level): level is ForwardLevel => level.kind === 'forward');
    return compileGraph(this.level.referenceGraph, {
      backend: this.backend,
      registry: referenceRegistry(preceding),
      availableComponents: [
        ...new Set([
          ...this.level.available,
          ...preceding.flatMap((level) => level.unlockComponents ?? []),
        ]),
      ],
      inputShapes,
      random: new Random(this.level.seed),
      parameters,
    });
  }

  private async initialize(): Promise<TrainingSession> {
    if (this.session) return this.session;
    let batch: TrainingBatch;
    if (this.playerCircuit) batch = fixedComparisonBatch(this.backend);
    else {
      const source = trainingBatchSource(this.level);
      batch = await takeBatch(
        typeof source === 'function' ? await source(0) : source,
        this.backend,
      );
    }
    this.batches.push(batch);
    this.assertActive();
    const evaluationBatch =
      !this.playerCircuit && this.level.evaluationDataset
        ? await takeBatch(
            this.level.evaluationDataset(new Random(this.level.seed + 1)),
            this.backend,
          )
        : undefined;
    if (evaluationBatch) this.batches.push(evaluationBatch);
    this.assertActive();
    this.trainingPlan = this.compile({ x: batch.inputs.x.shape });
    this.session = new TrainingSession(
      this.trainingPlan,
      batch,
      crossEntropy,
      {
        ...this.level.optimizer,
        learningRate: this.learningRate,
        // Same global-norm clip as the player RNN run, so the comparison stays fair.
        ...(this.playerCircuit ? { clipNorm: 1 } : {}),
      },
      this.level.interface.output,
      { accuracy: false, evaluationBatch },
    );
    // GraphPlan owns parameters only after its first execution. Predictions must borrow them.
    await this.session.evaluate();
    this.assertActive();
    return this.session;
  }

  private assertActive(): void {
    if (this.disposed) throw new Error('worker.fixed_window_disposed');
  }

  private predict(): Promise<FixedWindowPrediction[]> {
    return this.startPredictions(true);
  }

  /**
   * Enqueues every forward pass and readback synchronously (no await before the reads start),
   * so a mid-training peek never interleaves with a training step. Only the mapped results
   * resolve later. Non-strict peeks never throw for odd outputs; they just resolve to [].
   */
  private startPredictions(strict: boolean): Promise<FixedWindowPrediction[]> {
    if (!this.trainingPlan?.parameters.size)
      return Promise.reject(new Error('worker.fixed_window_uninitialized'));
    // Predict with the exact plan that was trained. A separately compiled batch-1 plan can
    // silently re-initialise any parameter it does not find under the same id (or whose
    // shape depends on the batch), which made trained models look untrained (≈1/27 each).
    const plan = this.trainingPlan;
    const rows = this.batches[0].inputs.x.shape[0];
    const launched: {
      history: string;
      input: string;
      x: Tensor;
      outputs?: TensorPorts;
      read?: Promise<Float32Array>;
    }[] = [];
    const releaseAll = () => {
      for (const item of launched) {
        try {
          if (item.outputs) plan.releaseExecution(item.outputs);
          item.x.dispose();
        } catch {
          // A disposed experiment already released everything.
        }
      }
    };
    try {
      for (const { history } of fixedWindowExamples) {
        this.assertActive();
        const input = [...history].slice(-2).join('');
        const encoded = encode(input);
        const x = tensor(
          Array.from({ length: rows }, () => encoded).flat(),
          [rows, 2],
          this.backend,
        );
        const item: (typeof launched)[number] = { history, input, x };
        launched.push(item);
        item.outputs = plan.execute({ x });
        const output = item.outputs[this.level.interface.output];
        if (
          output.shape.length !== 2 ||
          output.shape[0] !== rows ||
          output.shape[1] !== VOCAB.length
        )
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
          const weights = Array.from(await read!).slice(0, VOCAB.length);
          this.assertActive();
          const outputSum = weights.reduce((sum, value) => sum + value, 0);
          const invalidIndex = weights.findIndex((value) => !Number.isFinite(value) || value < 0);
          if (invalidIndex !== -1 || !(outputSum > 0) || !Number.isFinite(outputSum)) {
            if (!strict) return [];
            const outputNode = this.playerCircuit?.graph.nodes.find(
              (node) =>
                node.component === 'output' && node.attrs?.name === this.level.interface.output,
            );
            throw new GraphError('worker.fixed_window_probabilities', outputNode?.id, 'x', {
              sum: String(outputSum),
              minimum: String(Math.min(...weights)),
              maximum: String(Math.max(...weights)),
              ...(invalidIndex !== -1 && {
                index: invalidIndex,
                value: String(weights[invalidIndex]),
              }),
            });
          }
          // Match generation at temperature 1: normalize the model's nonnegative weights.
          // This changes neither their ranking nor the graph and loss used for training.
          const probabilities = weights.map((value) => value / outputSum);
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
    onProgress?: (progress: FixedWindowProgress) => void,
    onPeek?: (peek: Peek) => void,
  ): Promise<FixedWindowSnapshot> {
    this.assertActive();
    if (this.running) throw new Error('worker.training_busy');
    if (this.completedRuns >= 2) throw new Error('worker.fixed_window_complete');
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
            void this.startPredictions(false)
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
      const predictions = await this.predict();
      this.assertActive();
      this.completedRuns++;
      return {
        model: 'fixed-window',
        parameterCount: [...this.trainingPlan!.parameters.values()].reduce(
          (total, parameter) =>
            total + parameter.shape.reduce((product, dimension) => product * dimension, 1),
          0,
        ),
        sampleCount: this.batches[0].inputs.x.shape[0],
        step: report.step,
        loss: report.loss,
        stepsPerRun: this.stepsPerRun,
        seed: this.level.seed,
        learningRate: this.learningRate,
        predictions,
      };
    } catch (error) {
      // A failed run may already have updated parameters; retry only with a fresh experiment.
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
    this.trainingPlan?.dispose();
    this.trainingPlan = undefined;
    this.batches.splice(0).forEach(releaseBatch);
    if (!this.backendDisposed) {
      this.backend.dispose?.();
      this.backendDisposed = true;
    }
  }

  dispose(): void {
    this.disposed = true;
    this.cancellation.abort();
    // In-flight engine work still owns its execution tape until its finally blocks finish.
    if (!this.running) this.release();
  }
}

/** Kept separate from the worker entry so importing tests never installs an onmessage handler. */
export class FixedWindowWorkerHandler {
  private experiment?: FixedWindowExperiment | RecurrentCharacterExperiment;
  private model?: CharacterComparisonModel;
  private initializing?: Promise<void>;
  private disposed = false;
  constructor(private readonly output: (message: FixedWindowResponse) => void) {}

  private async initialize(request: FixedWindowRequest): Promise<void> {
    if (request.debug) setTrainingDebug(true);
    const selection = await createBackend(request.backend ?? 'auto');
    if (this.disposed) {
      selection.backend.dispose?.();
      throw new Error('worker.disposed');
    }
    try {
      if (request.model === 'rnn') {
        playerExperimentGraph(request.playerProgress!, 'time-rnn');
        this.experiment = new RecurrentCharacterExperiment(
          request.playerProgress!,
          {},
          selection.backend,
        );
      } else {
        this.experiment = new FixedWindowExperiment(request.playerProgress, selection.backend);
      }
      this.model = request.model;
    } catch (error) {
      selection.backend.dispose?.();
      throw error;
    }
  }

  async handle(request: FixedWindowRequest): Promise<void> {
    try {
      if (
        this.disposed ||
        request?.type !== 'train' ||
        (request.model !== 'fixed-window' && request.model !== 'rnn') ||
        (request.backend !== undefined &&
          request.backend !== 'cpu' &&
          request.backend !== 'webgpu') ||
        (this.model &&
          (this.model !== request.model ||
            request.playerProgress !== undefined ||
            request.backend !== undefined)) ||
        (!this.experiment && this.initializing) ||
        (!this.experiment && request.model === 'rnn' && !request.playerProgress)
      )
        throw new Error(this.initializing ? 'worker.training_busy' : 'worker.invalid_request');
      if (!this.experiment) {
        this.initializing = this.initialize(request);
        try {
          await this.initializing;
        } finally {
          this.initializing = undefined;
        }
      }
      const snapshot = await this.experiment!.run(
        (progress) => this.output(progress),
        (peek) => this.output(peek),
      );
      this.output({ type: 'result', snapshot });
    } catch (error) {
      this.output({ type: 'error', ...workerError(error) });
    }
  }

  dispose(): void {
    this.disposed = true;
    this.experiment?.dispose();
  }
}
