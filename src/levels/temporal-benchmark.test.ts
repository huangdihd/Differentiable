// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { describe, expect, it } from 'vitest';
import { createDefaultRegistry } from '../engine/graph';
import { graphMetrics } from '../engine/levels';
import artifact from './temporal-benchmark.json';
import {
  temporalBenchmarkDelays,
  temporalBenchmarkGraph,
  temporalBenchmarkModels,
} from './temporal-benchmark';

describe('temporal benchmark artifact', () => {
  it('records the fixed CPU task, budget, models and single formal seed', () => {
    expect(artifact).toMatchObject({
      schemaVersion: 1,
      task: 'delayed-copy',
      models: temporalBenchmarkModels,
      delays: temporalBenchmarkDelays,
      seeds: [811],
      training: {
        backend: 'cpu',
        optimizer: 'adam',
        learningRate: 0.02,
        steps: 300,
        batchSize: 32,
        evaluationBatchSize: 128,
        hiddenFeatures: 8,
      },
    });
    expect(artifact.results.map((result) => result.model)).toEqual(temporalBenchmarkModels);
  });

  it('keeps every recorded mean tied to its generated run and graph contract', () => {
    const registry = createDefaultRegistry();
    for (const result of artifact.results) {
      expect(result.delays.map((measurement) => measurement.delay)).toEqual(
        temporalBenchmarkDelays,
      );
      for (const measurement of result.delays) {
        expect(measurement.runs).toHaveLength(artifact.seeds.length);
        expect(measurement.runs.map((run) => run.seed)).toEqual(artifact.seeds);
        expect(measurement.meanMse).toBe(
          measurement.runs.reduce((sum, run) => sum + run.mse, 0) / measurement.runs.length,
        );
        expect(Number.isFinite(measurement.meanMse)).toBe(true);
        expect(measurement.meanMse).toBeGreaterThanOrEqual(0);
        expect(
          graphMetrics(
            temporalBenchmarkGraph(
              result.model as (typeof temporalBenchmarkModels)[number],
              measurement.delay,
            ),
            registry,
          ).parameterCount,
        ).toBe(result.parameterCount);
      }
    }
    expect(
      Object.fromEntries(artifact.results.map((result) => [result.model, result.parameterCount])),
    ).toEqual({ 'fixed-window': 3, rnn: 89, 'rnn-attention': 97 });
  });

  it('records a valid attention distribution and its weighted context', () => {
    const { history, query, weights, context } = artifact.attentionProbe;
    expect(history).toHaveLength(weights.length);
    expect(history.every((row) => row.length === query.length)).toBe(true);
    expect(weights.every((weight) => weight >= 0 && weight <= 1)).toBe(true);
    expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 12);
    expect(context).toHaveLength(query.length);
    for (const [feature, value] of context.entries())
      expect(value).toBeCloseTo(
        history.reduce((sum, row, rowIndex) => sum + weights[rowIndex] * row[feature], 0),
        12,
      );
    expect(weights.indexOf(Math.max(...weights))).toBe(1);
  });
});
