import { describe, expect, it } from 'vitest';
import { applyResult, evaluateLevel, graphMetrics, initialProgress, saveGraph } from './levels';
import { compileGraph } from './graph';
import { Random } from './random';
import { crossEntropy, Optimizer, TrainingSession } from './training';
import * as T from './tensor';
import {
  createExampleRegistry,
  sigmoidLevel,
  sigmoidSolution,
  xorLevel,
  xorSolution,
} from '../levels/examples';

describe('level runner and progression', () => {
  it('evaluates reproducible fixed and generated cases with structured metrics', async () => {
    const registry = createExampleRegistry();
    const a = await evaluateLevel(sigmoidLevel, sigmoidSolution, { registry });
    const b = await evaluateLevel(sigmoidLevel, sigmoidSolution, { registry });
    expect(a).toEqual(b);
    expect(a.passed).toBe(true);
    expect(a.tests).toHaveLength(4);
    expect(a.metrics).toEqual({ topNodes: 7, expandedPrimitives: 6, parameterCount: 0 });
  });
  it('saves graph snapshots and unlocks only on successful unlocked results', async () => {
    const initial = initialProgress(['sigmoid']);
    const saved = saveGraph(initial, 'sigmoid', sigmoidSolution);
    expect(saved.savedGraphs.sigmoid).not.toBe(sigmoidSolution);
    const result = await evaluateLevel(sigmoidLevel, sigmoidSolution, {
      registry: createExampleRegistry(),
    });
    expect(applyResult(initial, sigmoidLevel, { ...result, passed: false })).toBe(initial);
    expect(applyResult(initialProgress([]), sigmoidLevel, result).unlocked).toEqual([]);
    const next = applyResult(saved, sigmoidLevel, result);
    expect(next.unlocked).toEqual(['sigmoid', 'derivative']);
    expect(initial.passed).toEqual([]);
    expect(
      applyResult(next, sigmoidLevel, {
        ...result,
        metrics: { ...result.metrics, expandedPrimitives: 100 },
      }).bestScores,
    ).toEqual(next.bestScores);
  });
  it('reports compile errors without passing', async () => {
    const result = await evaluateLevel(sigmoidLevel, { version: 1, nodes: [], edges: [] });
    expect(result.passed).toBe(false);
    expect(result.errorKey).toBeTruthy();
  });
  it('counts nested primitives and trainable scalar values', () => {
    expect(graphMetrics(xorSolution, createExampleRegistry()).parameterCount).toBe(42);
  });
});

describe('optimizers and resumable training', () => {
  it('SGD makes the expected update', async () => {
    const parameters = new Map([['x', T.tensor([2], [])]]),
      gradient = T.tensor([3], []);
    const optimizer = new Optimizer({ kind: 'sgd', learningRate: 0.1 });
    await optimizer.update(parameters, [gradient]);
    expect((await parameters.get('x')!.read())[0]).toBeCloseTo(1.7);
    parameters.get('x')!.dispose();
    gradient.dispose();
  });
  it('split Adam runs equal continuous runs and report each step', async () => {
    const make = () => {
      const plan = compileGraph(xorSolution, {
        registry: createExampleRegistry(),
        inputShapes: xorLevel.interface.inputs,
        random: new Random(17),
      });
      const batch = xorLevel.dataset(new Random(17));
      return {
        plan,
        batch,
        session: new TrainingSession(plan, batch, crossEntropy, xorLevel.optimizer),
      };
    };
    const a = make(),
      b = make();
    const reports: number[] = [];
    await a.session.run(10, { onReport: (r) => reports.push(r.step), yieldEvery: 2 });
    const split = await a.session.run(10),
      whole = await b.session.run(20);
    expect(split).toEqual(whole);
    expect(reports).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    for (const x of [a, b]) {
      x.plan.dispose();
      x.batch.targets.dispose();
      Object.values(x.batch.inputs).forEach((t) => t.dispose());
    }
  });
});

describe('loss-only evaluation and component progression', () => {
  const graph = {
    version: 1 as const,
    nodes: [
      { id: 'x', component: 'input', attrs: { name: 'x' } },
      { id: 'out', component: 'output', attrs: { name: 'loss' } },
    ],
    edges: [{ from: { node: 'x', port: 'out' }, to: { node: 'out', port: 'x' } }],
  };
  it('uses seeded held-out data and scalar graph loss without accuracy', async () => {
    const seeds: number[] = [];
    const level = {
      ...xorLevel,
      available: [],
      interface: { inputs: { x: [] }, output: 'loss', outputShape: [] },
      steps: 1,
      lossOutput: 'loss',
      threshold: { loss: 2 },
      dataset: () => ({ inputs: { x: T.tensor([1], []) }, targets: T.tensor([0], []) }),
      evaluationDataset: (random: Random) => {
        seeds.push(random.uniform());
        return { inputs: { x: T.tensor([3], []) }, targets: T.tensor([0], []) };
      },
    };
    const a = await evaluateLevel(level, graph),
      b = await evaluateLevel(level, graph);
    expect(a.errorKey).toBeUndefined();
    expect(a.passed).toBe(false);
    expect(a.training?.loss).toBe(3);
    expect(a.training?.accuracy).toBeUndefined();
    expect(a).toEqual(b);
    expect(seeds[0]).toBe(seeds[1]);
    expect((await evaluateLevel({ ...level, threshold: { loss: 4 } }, graph)).passed).toBe(true);
  });
  it('rejects nonfinite evaluation and releases execution on training exceptions', async () => {
    const input = T.tensor([1], []),
      target = T.tensor([0], []);
    let released = 0;
    const model = {
      parameters: new Map<string, T.Tensor>(),
      execute: () => ({ y: T.tensor([NaN], []) }),
      releaseExecution: (outputs: Record<string, T.Tensor>) => {
        released++;
        Object.values(outputs).forEach((t) => t.dispose());
      },
    };
    try {
      const session = new TrainingSession(
        model,
        { inputs: { x: input }, targets: target },
        undefined,
        { kind: 'sgd', learningRate: 0.1 },
        'y',
        { lossOutput: 'y' },
      );
      await expect(session.evaluate()).rejects.toThrow('training.nonfinite');
      await expect(session.run(1)).rejects.toThrow('training.nonfinite');
      const failing = new TrainingSession(
        model,
        session.batch,
        () => {
          throw new Error('loss.failed');
        },
        { kind: 'sgd', learningRate: 0.1 },
      );
      await expect(failing.run(1)).rejects.toThrow('loss.failed');
      expect(released).toBe(3);
    } finally {
      input.dispose();
      target.dispose();
    }
  });
  it('unlocks component ids independently from level ids', async () => {
    const result = await evaluateLevel(sigmoidLevel, sigmoidSolution, {
      registry: createExampleRegistry(),
    });
    const next = applyResult(
      initialProgress(['sigmoid'], ['add']),
      { ...sigmoidLevel, unlockComponents: ['custom.component'] },
      result,
    );
    expect(next.unlockedComponents).toEqual(['add', 'custom.component']);
    expect(next.unlocked).toEqual(['sigmoid', 'derivative']);
  });
  it('preserves graph error key and node context', async () => {
    const result = await evaluateLevel({ ...sigmoidLevel, available: [] }, sigmoidSolution, {
      registry: createExampleRegistry(),
    });
    expect(result.errorKey).toBe('component_unavailable');
    expect(result.error?.key).toBe(result.errorKey);
    expect(result.error?.node).toBeTruthy();
    expect(result.error?.params.component).toBeTruthy();
  });
});
