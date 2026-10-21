import {
  GraphError,
  compileGraph,
  createDefaultRegistry,
  type ComponentRegistry,
  type GraphJSON,
  type ShapePorts,
} from './graph';
import type { DatasetUse } from '../datasets';
import { Random } from './random';
import * as T from './tensor';
import {
  crossEntropy,
  disposeExecution,
  TrainingSession,
  type OptimizerOptions,
  type TrainingBatch,
  type TrainingBatchSource,
  type TrainingReport,
} from './training';

export interface LevelMetadata {
  id: string;
  titleKey: string;
  descriptionKey: string;
  chapter?: string;
  hintsKeys?: readonly string[];
  available: readonly string[];
  forbidden?: readonly string[];
  presets?: readonly string[];
  referenceGraph?: GraphJSON;
  /** Recheck persisted solutions against current cases when loading progress. */
  revalidateOnLoad?: boolean;
  par?: number;
  unlockFeatures?: readonly string[];
  /** Shown on completion as a gradient graph: "here is what you just did by hand". */
  revealGraph?: GraphJSON;
  interface: { inputs: ShapePorts; output: string; outputShape: readonly number[] };
  unlock: readonly string[];
  unlockComponents?: readonly string[];
  /**
   * Labels for a matrix output's rows and columns, from the current inputs, so the expected
   * answer reads as "row for ID 2, column for class 3" instead of a bare grid.
   */
  outputAxes?: (inputs: Record<string, number[]>) => { rows?: string[]; cols?: string[] };
  /** One line under the expected answer saying how to read it. */
  outputNoteKey?: string;
}
export interface LevelCase {
  id: string;
  inputs: Record<string, number[]>;
}
export interface LevelCases {
  fixed: readonly LevelCase[];
  generated?: {
    seed: number;
    count: number;
    generate: (random: Random, index: number) => LevelCase;
  };
}
export interface ForwardLevel extends LevelMetadata, LevelCases {
  kind: 'forward' | 'derivative';
  shapeVariants?: readonly (LevelCases & { inputs: ShapePorts })[];
  reference: (inputs: Record<string, T.Tensor>) => T.Tensor;
  derivativeInput?: string;
  atol?: number;
  rtol?: number;
}
export interface TrainLevel extends LevelMetadata {
  kind: 'train';
  /** Allows seeded autoregressive text generation from the trained graph. */
  generates?: boolean;
  seed: number;
  steps: number;
  optimizer: OptimizerOptions;
  tunable?: {
    learningRate: { min: number; max: number; default: number };
    optimizer?: readonly OptimizerOptions['kind'][];
  };
  /** Opt in to a fresh seeded minibatch each update; requires evaluationDataset. */
  resample?: boolean;
  /** Fixed evaluation report interval for resampled training (default 50). */
  evaluationEvery?: number;
  dataset: (random: Random) => TrainingBatch;
  /** The named dataset behind `dataset`, for the player-facing card and plots. */
  data?: DatasetUse;
  evaluationDataset?: (random: Random) => TrainingBatch;
  lossOutput?: string;
  threshold: { loss: number; accuracy?: number };
}
export type Level = ForwardLevel | TrainLevel;
export interface GraphMetrics {
  topNodes: number;
  expandedPrimitives: number;
  parameterCount: number;
}
export interface TestResult {
  id: string;
  passed: boolean;
  actual: number[];
  expected: number[];
  errorKey?: string;
  errorParams?: Record<string, unknown>;
}
export interface LevelResult {
  levelId: string;
  passed: boolean;
  tests: TestResult[];
  metrics: GraphMetrics;
  training?: TrainingReport;
  errorKey?: string;
  error?: { key: string; node?: string; port?: string; params: Record<string, unknown> };
}
export function graphMetrics(
  graph: GraphJSON,
  registry: ComponentRegistry = createDefaultRegistry(),
): GraphMetrics {
  let expandedPrimitives = 0,
    parameterCount = 0;
  const visit = (g: GraphJSON, stack: string[]) => {
    for (const node of g.nodes) {
      if (node.component === 'param')
        parameterCount += ((node.attrs?.shape ?? []) as number[]).reduce((a, b) => a * b, 1);
      if (['input', 'output', 'param', 'const'].includes(node.component)) continue;
      const definition = registry.get(node.component);
      if (definition?.subgraph) {
        if (stack.includes(node.component)) throw new Error('levels.recursive_component');
        visit(
          typeof definition.subgraph === 'function'
            ? definition.subgraph(node.attrs ?? {})
            : definition.subgraph,
          [...stack, node.component],
        );
      } else expandedPrimitives++;
    }
  };
  visit(graph, []);
  return { topNodes: graph.nodes.length, expandedPrimitives, parameterCount };
}
/** Only declared tunables can override the level's optimizer configuration. */
export function trainingLearningRate(level: TrainLevel, requested?: number): number {
  const range = level.tunable?.learningRate;
  if (!range) return level.optimizer.learningRate;
  const value = requested !== undefined && Number.isFinite(requested) ? requested : range.default;
  return Math.min(range.max, Math.max(range.min, value));
}
/** Only explicitly allowed optimizer choices override the level configuration. */
export function trainingOptimizer(
  level: TrainLevel,
  requested?: OptimizerOptions['kind'],
): OptimizerOptions['kind'] {
  const allowed = level.tunable?.optimizer;
  return requested !== undefined && allowed?.includes(requested)
    ? requested
    : (allowed?.[0] ?? level.optimizer.kind);
}
/** Fixed batches remain caller-owned; factory batches are owned by TrainingSession. */
export function trainingBatchSource(level: TrainLevel): TrainingBatchSource {
  if (!level.resample) return level.dataset(new Random(level.seed));
  if (!level.evaluationDataset) throw new Error('training.missing_evaluation_dataset');
  const random = new Random(level.seed + 2);
  return () => level.dataset(random);
}
/** Compile against evaluation shapes while borrowing the live (optimizer-replaced) parameters. */
export function trainingEvaluationModel(
  level: TrainLevel,
  graph: GraphJSON,
  plan: ReturnType<typeof compileGraph>,
  batch: TrainingBatch,
  registry?: ComponentRegistry,
): () => ReturnType<typeof compileGraph> {
  return () => {
    // Initialize lazy parameters on their owning training plan before evaluation borrows them.
    if (plan.parameters.size === 0) {
      const inputs = Object.fromEntries(
        Object.entries(level.interface.inputs).map(([key, shape]) => [
          key,
          T.tensor(new Float32Array(shape.reduce((n, d) => n * d, 1)), [...shape]),
        ]),
      );
      try {
        const outputs = plan.execute(inputs);
        plan.releaseExecution(outputs);
      } finally {
        Object.values(inputs).forEach((value) => value.dispose());
      }
    }
    return compileGraph(graph, {
      registry,
      availableComponents: level.available,
      inputShapes: Object.fromEntries(
        Object.entries(batch.inputs).map(([key, value]) => [key, value.shape]),
      ),
      random: new Random(level.seed),
      parameters: plan.parameters,
    });
  };
}
/** Tests compile and execute the actual user graph, never a solution-specific evaluator. */
export async function evaluateLevel(
  level: Level,
  graph: GraphJSON,
  options: {
    registry?: ComponentRegistry;
    learningRate?: number;
    optimizer?: OptimizerOptions['kind'];
    onReport?: (report: TrainingReport) => void;
    signal?: AbortSignal;
  } = {},
): Promise<LevelResult> {
  const result: LevelResult = {
    levelId: level.id,
    passed: false,
    tests: [],
    metrics: { topNodes: graph.nodes.length, expandedPrimitives: 0, parameterCount: 0 },
  };
  let plan: ReturnType<typeof compileGraph> | undefined;
  try {
    result.metrics = graphMetrics(graph, options.registry);
    plan = compileGraph(graph, {
      registry: options.registry,
      availableComponents: level.available,
      inputShapes: level.interface.inputs,
      random: new Random(level.kind === 'train' ? level.seed : 1),
    });
    const shape = plan.outputShapes[level.interface.output];
    if (!shape || JSON.stringify(shape) !== JSON.stringify(level.interface.outputShape))
      throw new Error('levels.output_shape');
    if (level.kind === 'train') {
      const batch = trainingBatchSource(level);
      let evaluationBatch: TrainingBatch | undefined;
      let session: TrainingSession | undefined;
      try {
        evaluationBatch = level.evaluationDataset?.(new Random(level.seed + 1));
        session = new TrainingSession(
          plan,
          batch,
          crossEntropy,
          {
            ...level.optimizer,
            kind: trainingOptimizer(level, options.optimizer),
            learningRate: trainingLearningRate(level, options.learningRate),
          },
          level.interface.output,
          {
            lossOutput: level.lossOutput,
            evaluationBatch,
            evaluationModel:
              level.resample && evaluationBatch
                ? trainingEvaluationModel(level, graph, plan, evaluationBatch, options.registry)
                : undefined,
            evaluationReports: level.resample,
            accuracy: level.threshold.accuracy !== undefined,
          },
        );
        const initial = await session.evaluate();
        const checkDiverged = (report: TrainingReport) => {
          if (level.tunable && (!Number.isFinite(report.loss) || report.loss > initial.loss * 1000))
            throw new Error('levels.train_diverged');
        };
        result.training = await session.run(level.steps, {
          ...options,
          reportEvery: level.resample ? (level.evaluationEvery ?? 50) : 1,
          // Report first so the player sees the loss blow up before the run stops.
          onReport: (report) => {
            options.onReport?.(report);
            checkDiverged(report);
          },
        });
        checkDiverged(result.training);
        // Ending above where it started is also overshooting, just more slowly.
        if (level.tunable && result.training.loss > initial.loss)
          throw new Error('levels.train_diverged');
        result.passed =
          result.training.loss < level.threshold.loss &&
          (level.threshold.accuracy === undefined ||
            result.training.accuracy! >= level.threshold.accuracy);
        if (
          result.training.loss < level.threshold.loss &&
          level.threshold.accuracy !== undefined &&
          (result.training.accuracy ?? -Infinity) < level.threshold.accuracy
        ) {
          result.errorKey = 'levels.train_accuracy';
          result.error = { key: result.errorKey, params: { accuracy: result.training.accuracy } };
        } else if (!result.passed && level.tunable) {
          result.errorKey = 'levels.train_slow';
          result.error = { key: result.errorKey, params: { loss: result.training.loss } };
        }
        result.tests.push({
          id: 'evaluation',
          ...(result.errorKey
            ? { errorKey: result.errorKey, errorParams: result.error?.params }
            : {}),
          passed: result.passed,
          actual: [
            result.training.loss,
            ...(result.training.accuracy === undefined ? [] : [result.training.accuracy]),
          ],
          expected: [
            level.threshold.loss,
            ...(level.threshold.accuracy === undefined ? [] : [level.threshold.accuracy]),
          ],
        });
      } finally {
        session?.dispose();
        const tensors = new Set([
          ...(typeof batch === 'function' ? [] : [...Object.values(batch.inputs), batch.targets]),
          ...(evaluationBatch
            ? [...Object.values(evaluationBatch.inputs), evaluationBatch.targets]
            : []),
        ]);
        tensors.forEach((x) => x?.dispose());
      }
    } else {
      const groups = [
        { inputs: level.interface.inputs, fixed: level.fixed, generated: level.generated },
        ...(level.shapeVariants ?? []),
      ];
      for (const [groupIndex, group] of groups.entries()) {
        const variant = groupIndex > 0;
        const prefix = variant ? `shape-${groupIndex}/` : '';
        const errorParams = {
          shapes: Object.entries(group.inputs)
            .map(([name, shape]) => `${name} [${shape.join('×')}]`)
            .join(', '),
        };
        let groupPlan = variant ? undefined : plan;
        try {
          if (!groupPlan)
            groupPlan = compileGraph(graph, {
              registry: options.registry,
              availableComponents: level.available,
              inputShapes: group.inputs,
              random: new Random(1),
            });
          const cases = [...group.fixed];
          if (group.generated) {
            const random = new Random(group.generated.seed);
            for (let i = 0; i < group.generated.count; i++)
              cases.push(group.generated.generate(random, i));
          }
          for (const test of cases) {
            const inputs: Record<string, T.Tensor> = {};
            let execution: Record<string, T.Tensor> | undefined;
            let outputs: T.Tensor[] = [],
              reference: T.Tensor | undefined,
              gradient: T.Tensor | undefined;
            try {
              for (const [name, data] of Object.entries(test.inputs))
                inputs[name] = T.tensor(data, group.inputs[name]);
              execution = groupPlan.execute(inputs);
              outputs = Object.values(execution);
              const output = execution[level.interface.output];
              const actual = Array.from(await output.read());
              reference = level.reference(inputs);
              if (level.kind === 'derivative')
                [gradient] = await T.backward(reference, [inputs[level.derivativeInput ?? 'x']]);
              const expectedTensor = gradient ?? reference;
              const expected = Array.from(await expectedTensor.read());
              const passed =
                JSON.stringify(output.shape) === JSON.stringify(expectedTensor.shape) &&
                actual.length === expected.length &&
                actual.every(
                  (x, i) =>
                    Number.isFinite(x) &&
                    Number.isFinite(expected[i]) &&
                    Math.abs(x - expected[i]) <=
                      (level.atol ?? 1e-5) + (level.rtol ?? 1e-4) * Math.abs(expected[i]),
                );
              result.tests.push({
                id: prefix + test.id,
                passed,
                actual,
                expected,
                ...(variant && !passed ? { errorKey: 'levels.shape_variant', errorParams } : {}),
              });
            } catch (error) {
              if (!variant) throw error;
              result.tests.push({
                id: prefix + test.id,
                passed: false,
                actual: [],
                expected: [],
                errorKey: 'levels.shape_variant',
                errorParams,
              });
            } finally {
              disposeExecution(outputs, Object.values(inputs));
              if (execution) groupPlan.releaseExecution(execution, Object.values(inputs));
              if (reference) T.disposeGraph([reference], Object.values(inputs));
              gradient?.dispose();
              Object.values(inputs).forEach((x) => x.dispose());
            }
          }
        } catch (error) {
          if (!variant) throw error;
          result.tests.push({
            id: prefix + 'compile',
            passed: false,
            actual: [],
            expected: [],
            errorKey: 'levels.shape_variant',
            errorParams,
          });
        } finally {
          if (variant) groupPlan?.dispose();
        }
      }
      result.passed = result.tests.length > 0 && result.tests.every((t) => t.passed);
    }
  } catch (error) {
    if (error instanceof GraphError) {
      result.errorKey = error.key;
      result.error = { key: error.key, node: error.node, port: error.port, params: error.params };
    } else {
      result.errorKey = error instanceof Error ? error.message : 'levels.execution_error';
      if (level.kind === 'train' && result.errorKey === 'training.nonfinite')
        result.errorKey = 'levels.train_diverged';
      if (result.errorKey === 'levels.train_diverged')
        result.error = { key: result.errorKey, params: {} };
    }
  } finally {
    plan?.dispose();
  }
  return result;
}
export interface Progress {
  unlocked: string[];
  unlockedComponents: string[];
  passed: string[];
  savedGraphs: Record<string, GraphJSON>;
  passedSolutions: Record<string, GraphJSON>;
  bestScores: Record<string, GraphMetrics>;
  trainingSettings?: Record<string, { learningRate: number; optimizer?: 'sgd' | 'adam' }>;
}
export const initialProgress = (
  levels: readonly string[],
  components: readonly string[] = [],
): Progress => ({
  unlocked: [...levels],
  unlockedComponents: [...components],
  passed: [],
  savedGraphs: {},
  passedSolutions: {},
  bestScores: {},
});
export function saveGraph(progress: Progress, levelId: string, graph: GraphJSON): Progress {
  return {
    ...progress,
    savedGraphs: { ...progress.savedGraphs, [levelId]: structuredClone(graph) },
  };
}
/** Pure progression: only an unlocked, matching, passing level unlocks its successors. */
export function applyResult(
  progress: Progress,
  level: Level,
  result: LevelResult,
  graph?: GraphJSON,
): Progress {
  if (!result.passed || result.levelId !== level.id || !progress.unlocked.includes(level.id))
    return progress;
  const old = progress.bestScores[level.id],
    score = result.metrics;
  const better =
    !old ||
    score.expandedPrimitives < old.expandedPrimitives ||
    (score.expandedPrimitives === old.expandedPrimitives &&
      score.parameterCount < old.parameterCount);
  return {
    ...progress,
    unlockedComponents: [
      ...new Set([...(progress.unlockedComponents ?? []), ...(level.unlockComponents ?? [])]),
    ],
    unlocked: [...new Set([...progress.unlocked, ...level.unlock])],
    passed: [...new Set([...progress.passed, level.id])],
    passedSolutions: graph
      ? { ...progress.passedSolutions, [level.id]: structuredClone(graph) }
      : progress.passedSolutions,
    bestScores: better ? { ...progress.bestScores, [level.id]: { ...score } } : progress.bestScores,
  };
}
