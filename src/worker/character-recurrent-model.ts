// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import type { Backend } from '../engine/backend';
import {
  compileGraph,
  type ComponentRegistry,
  type GraphJSON,
  type GraphPlan,
} from '../engine/graph';
import type { Progress } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import type { TrainingBatch, TrainingModel } from '../engine/training';
import {
  characterComparisonLines,
  characterComparisonPositions,
} from '../levels/character-comparison-data';
import { encode, VOCAB } from '../levels/corpus';
import { playerExperimentGraph } from './player-experiment-graph';

export const recurrentCharacterSeed = 4105;
const embeddingWidth = 8;
const hiddenWidth = 24;

export function releaseRecurrentBatch(batch: TrainingBatch): void {
  new Set([...Object.values(batch.inputs), batch.targets]).forEach((value) => value?.dispose());
}

/** Each line advances once per character; finished rows leave rather than receive padding. */
export function recurrentCharacterBatch(backend: Backend): TrainingBatch {
  const inputs: Record<string, T.Tensor> = {};
  try {
    const positionsByOffset = new Map<number, typeof characterComparisonPositions>();
    for (const position of characterComparisonPositions) {
      const positions = positionsByOffset.get(position.offset) ?? [];
      positions.push(position);
      positionsByOffset.set(position.offset, positions);
    }
    let previousRows = positionsByOffset.get(2)!.map(({ lineIndex }) => lineIndex);
    inputs.initialHidden = T.tensor(
      new Float32Array(previousRows.length * hiddenWidth),
      [previousRows.length, hiddenWidth],
      backend,
    );
    inputs.lossScale = T.tensor([-1 / characterComparisonPositions.length], [], backend);
    inputs.tokens_0 = T.tensor(
      previousRows.map((lineIndex) => characterComparisonLines[lineIndex].tokens[0]),
      [previousRows.length],
      backend,
    );
    for (const [offset, positions] of positionsByOffset) {
      const step = offset - 1;
      const rows = positions.map(({ lineIndex }) => lineIndex);
      if (rows.length !== previousRows.length) {
        const previousIndices = new Map(previousRows.map((lineIndex, index) => [lineIndex, index]));
        inputs[`rows_${step}`] = T.tensor(
          rows.map((lineIndex) => previousIndices.get(lineIndex)!),
          [rows.length],
          backend,
        );
      }
      inputs[`tokens_${step}`] = T.tensor(
        rows.map((lineIndex) => characterComparisonLines[lineIndex].tokens[step]),
        [rows.length],
        backend,
      );
      const targets = new Float32Array(rows.length * VOCAB.length);
      positions.forEach(({ lineIndex, offset }, row) => {
        targets[row * VOCAB.length + characterComparisonLines[lineIndex].tokens[offset]] = 1;
      });
      inputs[`targets_${step}`] = T.tensor(targets, [rows.length, VOCAB.length], backend);
      previousRows = rows;
    }
    return { inputs };
  } catch (error) {
    releaseRecurrentBatch({ inputs });
    throw error;
  }
}

/** The target is never included; every inference starts with its own zero hidden state. */
export function recurrentCharacterPredictionBatch(
  history: string,
  backend: Backend,
): TrainingBatch {
  const tokens = encode(history);
  if (tokens.length < 2 || history.includes('\n')) throw new Error('worker.rnn_history');
  const inputs: Record<string, T.Tensor> = {};
  try {
    inputs.initialHidden = T.tensor(new Float32Array(hiddenWidth), [1, hiddenWidth], backend);
    tokens.forEach((token, step) => {
      inputs[`tokens_${step}`] = T.tensor([token], [1], backend);
    });
    return { inputs };
  } catch (error) {
    releaseRecurrentBatch({ inputs });
    throw error;
  }
}

export interface CharacterRecurrentModelOptions {
  progress: Progress;
  /** Initialization seed; defaults to recurrentCharacterSeed. */
  seed?: number;
}

interface RecurrentExecution {
  intermediates: T.Tensor[];
  cells: { plan: GraphPlan; outputs: Record<string, T.Tensor> }[];
}

/** Player-built trainable cell; embedding and character head are experiment-owned fixtures. */
export class CharacterRecurrentModel implements TrainingModel {
  readonly parameters = new Map<string, T.Tensor>();
  private readonly constants = new Map<number, T.Tensor>();
  private readonly cellPlans = new Map<number, GraphPlan>();
  private readonly cellParameters = new Map<string, T.Tensor>();
  private readonly random: Random;
  private readonly executions = new WeakMap<Record<string, T.Tensor>, RecurrentExecution>();
  private readonly cellGraph: GraphJSON;
  private readonly registry: ComponentRegistry;
  private disposed = false;

  constructor(
    private readonly backend: Backend,
    { progress, seed = recurrentCharacterSeed }: CharacterRecurrentModelOptions,
  ) {
    this.random = new Random(seed);
    if (
      !progress.passed.includes('time-rnn') ||
      !progress.unlockedComponents.includes('rnn_cell') ||
      !progress.passedSolutions['time-rnn']
    )
      throw new Error('worker.rnn_player_cell_required');
    const player = playerExperimentGraph(progress, 'time-rnn');
    this.cellGraph = player.graph;
    this.registry = player.registry;
    const random = this.random;
    const shapes: [string, number[]][] = [
      ['embedding', [VOCAB.length, embeddingWidth]],
      ['inputWeight', [embeddingWidth, hiddenWidth]],
      ['recurrentWeight', [hiddenWidth, hiddenWidth]],
      ['hiddenBias', [hiddenWidth]],
      ['outputWeight', [hiddenWidth, VOCAB.length]],
      ['outputBias', [VOCAB.length]],
    ];
    try {
      for (const [name, shape] of shapes) {
        const count = shape.reduce((product, dimension) => product * dimension, 1);
        this.parameters.set(
          name,
          T.tensor(
            Float32Array.from({ length: count }, () => random.normal() * 0.15),
            shape,
            backend,
          ),
        );
      }
      this.constants.set(1e-7, T.tensor([1e-7], [], backend));
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  private cellPlan(rows: number): GraphPlan {
    const cached = this.cellPlans.get(rows);
    if (cached) return cached;
    const plan = compileGraph(this.cellGraph, {
      backend: this.backend,
      registry: this.registry,
      preferNative: false,
      parameters: this.cellParameters,
      random: this.random,
      inputShapes: {
        x: [rows, embeddingWidth],
        h: [rows, hiddenWidth],
        wx: [embeddingWidth, hiddenWidth],
        wh: [hiddenWidth, hiddenWidth],
        b: [hiddenWidth],
      },
    });
    const shape = plan.outputShapes.hiddenState;
    if (shape?.length !== 2 || shape[0] !== rows || shape[1] !== hiddenWidth) {
      plan.dispose();
      throw new Error('worker.rnn_hidden_shape');
    }
    this.cellPlans.set(rows, plan);
    return plan;
  }

  private release(execution: RecurrentExecution, keep: readonly T.Tensor[] = []): void {
    for (const { plan, outputs } of execution.cells) plan.releaseExecution(outputs, keep);
    const retained = new Set(keep);
    // Never walk the cell tapes here: their private constants belong to cached plans.
    for (const value of execution.intermediates) if (!retained.has(value)) value.dispose();
  }

  execute(
    inputs: Record<string, T.Tensor>,
    trace?: Map<string, T.Tensor>,
  ): Record<string, T.Tensor> {
    if (this.disposed) throw new Error('worker.rnn_model_disposed');
    if (!inputs.initialHidden || !inputs.tokens_0) throw new Error('worker.rnn_inputs');
    // Adam replaces map entries, so resolve current parameters for each execution.
    for (const name of this.cellParameters.keys())
      this.cellParameters.set(name, this.parameters.get(`cell/${name}`)!);
    const parameter = (name: string) => this.parameters.get(name)!;
    const constant = (value: number) => this.constants.get(value)!;
    const execution: RecurrentExecution = { intermediates: [], cells: [] };
    const track = (value: T.Tensor): T.Tensor => {
      execution.intermediates.push(value);
      return value;
    };
    let hidden = inputs.initialHidden;
    let loss: T.Tensor | undefined;
    let probabilities: T.Tensor | undefined;
    const predictionOnly = !inputs.targets_1;
    try {
      for (let step = 0; inputs[`tokens_${step}`]; step++) {
        if (inputs[`rows_${step}`]) hidden = track(T.gather(hidden, inputs[`rows_${step}`]));
        const embedding = track(T.gather(parameter('embedding'), inputs[`tokens_${step}`]));
        const rows = embedding.shape[0];
        const plan = this.cellPlan(rows);
        let cellOutputs: Record<string, T.Tensor>;
        try {
          cellOutputs = plan.execute({
            x: embedding,
            h: hidden,
            wx: parameter('inputWeight'),
            wh: parameter('recurrentWeight'),
            b: parameter('hiddenBias'),
          });
        } finally {
          // Storage is lazy, even on a failing first execute. Register the same leaves
          // before TrainingSession collects parameters; preserve nested compiled paths.
          for (const [name, value] of this.cellParameters)
            this.parameters.set(`cell/${name}`, value);
        }
        execution.cells.push({ plan, outputs: cellOutputs });
        hidden = cellOutputs.hiddenState;
        if (
          hidden?.shape.length !== 2 ||
          hidden.shape[0] !== rows ||
          hidden.shape[1] !== hiddenWidth
        )
          throw new Error('worker.rnn_hidden_shape');

        const targets = inputs[`targets_${step}`];
        if (!targets && !(predictionOnly && !inputs[`tokens_${step + 1}`])) continue;
        const projectedOutput = track(T.matmul(hidden, parameter('outputWeight')));
        const logits = track(T.add(projectedOutput, parameter('outputBias')));
        probabilities = track(T.softmax(logits));
        trace?.set(`probabilities_${step}`, probabilities);
        if (!targets) continue;
        // Normalize once over all positions, not over rows or time steps. Epsilon and
        // lossScale are owned leaves: TrainingSession's lossOutput cleanup retains leaves.
        const stabilized = track(T.add(probabilities, constant(1e-7)));
        const logProbabilities = track(T.log(stabilized));
        const targetLogProbabilities = track(T.mul(targets, logProbabilities));
        const logLikelihood = track(T.sum(targetLogProbabilities, [0, 1]));
        loss = loss ? track(T.add(loss, logLikelihood)) : logLikelihood;
      }
      if (loss) {
        if (!inputs.lossScale) throw new Error('worker.rnn_loss_scale');
        const outputs = { loss: track(T.mul(loss, inputs.lossScale)) };
        this.executions.set(outputs, execution);
        return outputs;
      }
      if (!probabilities) throw new Error('worker.rnn_output');
      const outputs = { probabilities };
      this.executions.set(outputs, execution);
      return outputs;
    } catch (error) {
      // No output graph reaches the caller on failure. Track every operation separately,
      // including branches that fail before they can be attached to a later output.
      this.release(execution);
      throw error;
    }
  }

  releaseExecution(outputs: Record<string, T.Tensor>, keep: readonly T.Tensor[] = []): void {
    const execution = this.executions.get(outputs);
    if (!execution) return;
    this.release(execution, keep);
    this.executions.delete(outputs);
  }

  dispose(): void {
    this.disposed = true;
    this.cellPlans.forEach((plan) => plan.dispose());
    this.cellPlans.clear();
    // Adam replacements may not yet have been synchronized into the plans' shared map.
    this.parameters.forEach((value) => value.dispose());
    this.constants.forEach((value) => value.dispose());
    this.parameters.clear();
    this.cellParameters.clear();
    this.constants.clear();
  }
}
