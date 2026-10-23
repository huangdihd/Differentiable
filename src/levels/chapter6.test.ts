import { expect, it, vi } from 'vitest';
import { compileGraph, type GraphJSON } from '../engine/graph';
import { trainingBatchSource, trainingEvaluationModel, type TrainLevel } from '../engine/levels';
import { Random } from '../engine/random';
import { crossEntropy, TrainingSession } from '../engine/training';
import { chapter5TrainLevels } from './chapter5';
import {
  chapter6Levels,
  chapter6TrainLevels,
  gptConfig,
  gptEvaluationIndices,
  gptPreceding,
  gptSequences,
} from './chapter6';
import { referenceRegistry } from './level-helpers';
import { CORPUS, corpusTokens, encode, decode, sampleToken, vocabularySize } from './corpus';
import * as T from '../engine/tensor';

it('GPT windows cover the entire corpus with aligned next-character targets', async () => {
  const level = chapter6TrainLevels[0];
  const batch = level.evaluationDataset!(new Random(level.seed + 1));
  try {
    const inputs = await batch.inputs.x.read();
    const targets = await batch.targets!.read();
    const covered = new Set<number>();
    expect(gptEvaluationIndices).toHaveLength(400);
    expect(new Set(gptEvaluationIndices).size).toBe(400);
    gptEvaluationIndices.forEach((index, row) => {
      const sequence = gptSequences[index];
      const start = index * gptConfig.stride;
      expect(sequence).toEqual(corpusTokens.slice(start, start + gptConfig.T + 1));
      for (let t = 0; t < gptConfig.T; t++) {
        covered.add(start + t + 1);
        expect(inputs[row * gptConfig.T + t]).toBe(sequence[t]);
        const offset = (row * gptConfig.T + t) * 27;
        expect(targets[offset + sequence[t + 1]]).toBe(1);
        expect(targets.slice(offset, offset + 27).reduce((a, b) => a + b, 0)).toBe(1);
      }
    });
    expect(covered.size).toBe(corpusTokens.length - 1);
  } finally {
    batch.inputs.x.dispose();
    batch.targets!.dispose();
  }
});

const registry = referenceRegistry(gptPreceding);
// Slow CPU calibration: train once, then check generated continuation word coverage.
it('GPT calibration, seeded parameters, text quality and single-head control', async () => {
  const level = {
    ...chapter6TrainLevels[0],
    available: [
      ...chapter6TrainLevels[0].available,
      ...gptPreceding.flatMap((l) => l.unlockComponents ?? []),
    ],
  };
  const train = async (graph: GraphJSON, level: TrainLevel) => {
    const contextLength = level.interface.inputs.x.at(-1)!;
    const inputShape = level.interface.inputs.x.length === 2 ? [1, contextLength] : [contextLength];
    const start = performance.now();
    const plan = compileGraph(graph, {
      registry,
      random: new Random(level.seed),
      inputShapes: level.interface.inputs,
    });
    const source = trainingBatchSource(level);
    const batch =
      level.evaluationDataset?.(new Random(level.seed + 1)) ??
      (typeof source === 'function' ? await source(0) : source);
    const session = new TrainingSession(
      plan,
      source,
      crossEntropy,
      level.optimizer,
      level.interface.output,
      {
        accuracy: false,
        evaluationBatch: batch,
        evaluationModel: trainingEvaluationModel(level, graph, plan, batch, registry),
        evaluationReports: true,
      },
    );
    let evaluationMs = 0;
    const evaluate = session.evaluate.bind(session);
    vi.spyOn(session, 'evaluate').mockImplementation(async (...args) => {
      const start = performance.now();
      try {
        return await evaluate(...args);
      } finally {
        evaluationMs += performance.now() - start;
      }
    });
    let hits = 0;
    let words = 0;
    try {
      const initial = await session.evaluate();
      const values = await Promise.all([...plan.parameters.values()].map((p) => p.read()));
      for (let i = 0; i < values.length; i++)
        for (let j = 0; j < i; j++)
          expect(Array.from(values[i])).not.toEqual(Array.from(values[j]));
      const final = await session.run(level.steps, {
        yieldEvery: 1000,
        reportEvery: level.evaluationEvery ?? 250,
      });
      const runtimeMs = performance.now() - start;
      console.info(JSON.stringify({ loss: final.loss, runtimeMs }));
      {
        const inference = compileGraph(graph, {
          registry,
          parameters: plan.parameters,
          inputShapes: { x: inputShape },
        });
        try {
          expect(inference.outputShapes.probabilities).toEqual([...inputShape, vocabularySize]);
          const vocabulary = new Set(CORPUS.split(/[^a-zA-Z]+/).filter(Boolean));
          for (const prompt of ['little moth', 'penny wren', 'round the pond']) {
            const tokens = encode(prompt);
            const random = new Random(6404);
            for (let i = 0; i < 120; i++) {
              const context = tokens.slice(-contextLength);
              while (context.length < contextLength) context.unshift(encode(' ')[0]);
              const x = T.tensor(context, inputShape);
              const outputs = inference.execute({ x });
              try {
                const probabilities = (await outputs.probabilities.read()).slice(-vocabularySize);
                const maximum = Math.max(...probabilities);
                tokens.push(
                  sampleToken(
                    Array.from(probabilities, (p) => (p / maximum) ** 5),
                    random,
                  ),
                );
              } finally {
                inference.releaseExecution(outputs);
                x.dispose();
              }
            }
            const sample = decode(tokens);
            const continuation = sample.slice(prompt.length);
            const tokensInText = continuation.split(/[^a-zA-Z]+/).filter(Boolean);
            const matched = tokensInText.filter((word) => vocabulary.has(word)).length;
            hits += matched;
            words += tokensInText.length;
            console.info(
              JSON.stringify({ prompt, continuation, words: tokensInText.length, matched }),
            );
            expect(tokens.length).toBe(prompt.length + 120);
          }
        } finally {
          inference.dispose();
        }
      }
      return {
        initial: initial.loss,
        loss: final.loss,
        runtimeMs,
        evaluationMs,
        hits,
        words,
        wordHitRate: hits / words,
      };
    } finally {
      session.dispose();
      plan.dispose();
      const batches = typeof source === 'function' ? [batch] : [batch, source];
      new Set(batches.flatMap((b) => [...Object.values(b.inputs), b.targets])).forEach((t) =>
        t?.dispose(),
      );
    }
  };
  const reference = await train(level.referenceGraph!, level);
  // Compare the original 5-5 single-head/single-layer lesson, without changing its configuration.
  const baselineLevel = { ...chapter5TrainLevels[0], available: level.available };
  const baseline = await train(baselineLevel.referenceGraph!, baselineLevel);
  console.info(
    JSON.stringify({
      config: gptConfig,
      N: gptSequences.length,
      steps: level.steps,
      par: chapter6Levels.map((l) => [l.id, l.par]),
      reference,
      baseline,
    }),
  );
  expect.soft(reference.evaluationMs / reference.runtimeMs).toBeLessThan(0.15);
  // ~75 sampled words make this noisy (±0.05): float-level changes elsewhere moved it 0.74 → 0.66
  // at the same loss. The baseline sits near 0.1, so 0.6 still separates them clearly.
  expect.soft(reference.wordHitRate).toBeGreaterThanOrEqual(0.6);
  expect.soft(baseline.wordHitRate).toBeLessThan(0.5);
  expect.soft(reference.loss).toBeLessThan(level.threshold.loss);
  expect(baseline.loss).toBeGreaterThan(level.threshold.loss);
}, 360000);
