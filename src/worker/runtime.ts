import { buildBackwardGraph, traceBackward } from '../engine/backward-graph';
import { createBackend } from '../engine/backend-selection';
import { CpuBackend } from '../engine/cpu';
import { BackendError } from '../engine/webgpu';
import {
  compileGraph,
  GraphError,
  type GraphPlan,
  type GraphJSON,
  type TensorPorts,
  type ComponentRegistry,
} from '../engine/graph';
import {
  evaluateLevel,
  trainingLearningRate,
  trainingOptimizer,
  initialProgress,
  type Progress,
  graphMetrics,
  type Level,
  type LevelResult,
  type TrainLevel,
} from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import {
  crossEntropy,
  TrainingSession,
  type TrainingBatch,
  type TrainingReport,
} from '../engine/training';
import { gameLevels } from '../levels/catalog';
import { VOCAB, encode, decode } from '../levels/corpus';
import { availableComponents, buildPlayerRegistry } from '../levels/registry';
import type { WorkerError, WorkerRequest, WorkerResponse, ReplyData } from './protocol';

/** Never send exception prose or stack traces across the player-facing boundary. */
export function workerError(error: unknown): WorkerError {
  if (error instanceof GraphError) {
    const { message: _message, ...params } = error.params;
    return { key: error.key, node: error.node, port: error.port, params };
  }
  if (error instanceof BackendError) return { key: error.key, params: error.params };
  const key =
    error instanceof Error && /^(engine|training|levels|worker)\.[a-z0-9_.]+$/.test(error.message)
      ? error.message
      : 'worker.execution_error';
  return { key, params: {} };
}
interface Run {
  request: Extract<WorkerRequest, { type: 'start' }>;
  level: TrainLevel;
  plan: GraphPlan;
  session: TrainingSession;
  batches: TrainingBatch[];
  result: LevelResult;
  paused: boolean;
  initialLoss: number;
}
interface StepRun {
  key: string;
  plan: GraphPlan;
  session: TrainingSession;
  batch: TrainingBatch;
}
/** Node-testable message handler. One runtime per worker; engine default backend is process-local. */
export class WorkerRuntime {
  private registry: ComponentRegistry;
  private initialized = false;
  private disposed = false;
  private revision = -1;
  private latestRevision = -1;
  private queue: Promise<void> = Promise.resolve();
  private run?: Run;
  private stepRun?: StepRun;
  // A single CPU snapshot: latest successful continuous completion/stop or step wins.
  // Never borrow tensors owned by a training plan, which release() will dispose.
  private generationParameters?: {
    key: string;
    values: Map<string, { shape: number[]; data: Float32Array }>;
  };
  private generationKey(levelId: string, graph: GraphJSON) {
    // Not tied to the revision: passing a level syncs progress, and the player expects the
    // model they just trained to write. Same level and graph means the same parameters.
    return JSON.stringify([levelId, graph]);
  }
  private async retainParameters(level: TrainLevel, graph: GraphJSON, plan: GraphPlan) {
    if (!level.generates) return;
    const values = new Map<string, { shape: number[]; data: Float32Array }>();
    for (const [id, value] of plan.parameters) {
      const data = new Float32Array(await value.read());
      if (!data.every(Number.isFinite)) throw new Error('training.nonfinite');
      values.set(id, { shape: [...value.shape], data });
    }
    this.generationParameters = { key: this.generationKey(level.id, graph), values };
  }
  private async generate(request: Extract<WorkerRequest, { type: 'generate' }>, level: TrainLevel) {
    const temperature = request.temperature === undefined ? 1 : request.temperature;
    if (
      typeof request.prompt !== 'string' ||
      !Number.isInteger(request.length) ||
      request.length < 0 ||
      request.length > 400 ||
      !Number.isFinite(temperature) ||
      temperature <= 0 ||
      (request.seed !== undefined && !Number.isSafeInteger(request.seed))
    )
      throw new Error('worker.invalid_request');
    const inputShape = level.interface.inputs.x;
    const rank = inputShape?.length;
    if (rank !== 1 && rank !== 2) throw new Error('worker.generation_unavailable');
    const context = rank === 1 ? 1 : inputShape[1];
    if (!Number.isSafeInteger(context) || context < 1)
      throw new Error('worker.generation_unavailable');
    const tokens = encode([...request.prompt].filter((char) => VOCAB.includes(char)).join(''));
    const promptLength = tokens.length;
    const padded = [...tokens];
    while (padded.length < context) padded.unshift(encode(' ')[0]);
    const parameters = new Map<string, T.Tensor>();
    const saved = this.generationParameters;
    if (saved?.key === this.generationKey(level.id, request.graph))
      for (const [id, value] of saved.values) parameters.set(id, T.tensor(value.data, value.shape));
    let plan: GraphPlan | undefined;
    try {
      const shape = rank === 1 ? [1] : [1, context];
      plan = compileGraph(request.graph, {
        registry: this.registry,
        availableComponents: level.available,
        inputShapes: { x: shape },
        random: new Random(level.seed),
        parameters,
      });
      const outputShape = plan.outputShapes[level.interface.output];
      if (
        !outputShape ||
        outputShape[0] !== 1 ||
        outputShape.at(-1) !== VOCAB.length ||
        !(
          outputShape.length === 2 ||
          (rank === 2 && outputShape.length === 3 && outputShape[1] === context)
        )
      )
        throw new Error('levels.output_shape');
      const random = new Random(request.seed ?? level.seed);
      for (let i = 0; i < request.length; i++) {
        const x = T.tensor(padded.slice(-context), shape);
        let outputs: TensorPorts | undefined;
        try {
          outputs = plan.execute({ x });
          const probabilities = (await outputs[level.interface.output].read()).slice(-VOCAB.length);
          if (
            probabilities.length !== VOCAB.length ||
            !probabilities.every((p) => Number.isFinite(p) && p >= 0)
          )
            throw new Error('worker.invalid_probabilities');
          const maximum = Math.max(...probabilities);
          if (!(maximum > 0)) throw new Error('worker.invalid_probabilities');
          // Scaling by max before the power is algebraically equivalent after normalization,
          // but avoids underflow of every weight at small positive temperatures.
          const weights = Array.from(probabilities, (p) =>
            p === maximum ? 1 : Math.pow(p / maximum, 1 / temperature),
          );
          const total = weights.reduce((sum, p) => sum + p, 0);
          let draw = random.uniform();
          let token = weights.length - 1;
          while (weights[token] === 0) token--;
          for (let j = 0; j < weights.length; j++) {
            draw -= weights[j] / total;
            if (draw < 0) {
              token = j;
              break;
            }
          }
          padded.push(token);
          tokens.push(token);
        } finally {
          if (outputs) plan.releaseExecution(outputs);
          x.dispose();
        }
      }
      this.reply(request, { text: decode(tokens.slice(0, promptLength + request.length)) });
    } finally {
      plan?.dispose();
      parameters.forEach((value) => value.dispose());
    }
  }
  private releaseStep() {
    const run = this.stepRun;
    this.stepRun = undefined;
    if (!run) return;
    run.session.dispose();
    run.plan.dispose();
    new Set([...Object.values(run.batch.inputs), run.batch.targets]).forEach((t) => t?.dispose());
  }
  private timer?: ReturnType<typeof setTimeout>;
  private timerGeneration = 0;
  constructor(
    private output: (message: WorkerResponse) => void,
    private levels: readonly Level[] = gameLevels,
    private progress: Progress = initialProgress([]),
    registry?: ComponentRegistry,
  ) {
    this.registry = registry ?? buildPlayerRegistry(levels, progress);
  }
  private emit(message: WorkerResponse) {
    if (message.type !== 'error' && message.revision < this.latestRevision) return;
    this.output(message);
  }
  handle(request: WorkerRequest): Promise<void> {
    if (Number.isSafeInteger(request.requestId) && Number.isSafeInteger(request.revision))
      this.latestRevision = Math.max(this.latestRevision, request.revision);
    const task = this.queue.then(() => this.dispatch(request));
    this.queue = task.catch(() => {});
    return task;
  }
  private reply(request: WorkerRequest, data: ReplyData[keyof ReplyData]) {
    this.emit({
      type: 'reply',
      requestId: request.requestId,
      revision: request.revision,
      command: request.type,
      data,
    });
  }
  private release() {
    this.releaseStep();
    this.timerGeneration++;
    clearTimeout(this.timer);
    const run = this.run;
    this.run = undefined;
    if (!run) return;
    run.session.dispose();
    run.plan.dispose();
    new Set(run.batches.flatMap((b) => [...Object.values(b.inputs), b.targets])).forEach((t) =>
      t?.dispose(),
    );
  }
  /** Await pending work before releasing owned tensors. */
  async dispose() {
    this.disposed = true;
    await this.queue;
    this.release();
    this.generationParameters = undefined;
  }
  private schedule() {
    const run = this.run;
    const generation = ++this.timerGeneration;
    this.timer = setTimeout(() => {
      this.queue = this.queue
        .then(() => {
          if (!this.disposed && this.run === run && generation === this.timerGeneration)
            return this.tick();
        })
        .catch(() => {});
    }, 0);
  }
  private async tick() {
    const run = this.run;
    if (!run || run.paused) return;
    try {
      const checkDiverged = (report: TrainingReport) => {
        if (
          run.level.tunable &&
          (!Number.isFinite(report.loss) || report.loss > run.initialLoss * 1000)
        ) {
          run.result.training = report;
          throw new GraphError('levels.train_diverged', undefined, undefined, {
            loss: report.loss,
          });
        }
      };
      const report = await run.session.run(1, { reportEvery: 1 });
      // Chart the step first so the player sees the loss blow up before the run stops.
      if (Number.isFinite(report.loss))
        this.emit({
          type: 'progress',
          requestId: run.request.requestId,
          revision: run.request.revision,
          report,
        });
      checkDiverged(report);
      if (report.step >= run.level.steps) {
        await this.retainParameters(run.level, run.request.graph, run.plan);
        this.finish(run, report, true);
        this.release();
      } else this.schedule();
    } catch (error) {
      run.result.error =
        run.level.tunable && error instanceof Error && error.message === 'training.nonfinite'
          ? { key: 'levels.train_diverged', params: { loss: Infinity } }
          : workerError(error);
      run.result.errorKey = run.result.error.key;
      this.emit({
        type: 'result',
        requestId: run.request.requestId,
        revision: run.request.revision,
        result: run.result,
      });
      this.release();
    }
  }
  private finish(run: Run, report: TrainingReport, completed: boolean) {
    const { threshold } = run.level;
    const passed =
      completed &&
      report.loss < threshold.loss &&
      (threshold.accuracy === undefined || (report.accuracy ?? -Infinity) >= threshold.accuracy);
    if (
      completed &&
      report.loss < threshold.loss &&
      threshold.accuracy !== undefined &&
      (report.accuracy ?? -Infinity) < threshold.accuracy
    ) {
      run.result.errorKey = 'levels.train_accuracy';
      run.result.error = { key: run.result.errorKey, params: { accuracy: report.accuracy } };
    } else if (completed && !passed && run.level.tunable) {
      // Ending above where it started is also overshooting, just more slowly.
      const key = report.loss > run.initialLoss ? 'levels.train_diverged' : 'levels.train_slow';
      run.result.errorKey = key;
      run.result.error = { key, params: { loss: report.loss } };
    }
    Object.assign(run.result, {
      training: report,
      passed,
      tests: [
        {
          id: 'evaluation',
          ...(run.result.errorKey
            ? { errorKey: run.result.errorKey, errorParams: run.result.error?.params }
            : {}),
          passed,
          actual: [report.loss, ...(report.accuracy === undefined ? [] : [report.accuracy])],
          expected: [
            threshold.loss,
            ...(threshold.accuracy === undefined ? [] : [threshold.accuracy]),
          ],
        },
      ],
    });
    this.emit({
      type: 'result',
      requestId: run.request.requestId,
      revision: run.request.revision,
      result: run.result,
    });
  }
  private async dispatch(request: WorkerRequest) {
    try {
      if (this.disposed) throw new Error('worker.disposed');
      if (
        !Number.isSafeInteger(request.requestId) ||
        !Number.isSafeInteger(request.revision) ||
        request.revision < 0
      )
        throw new Error('worker.invalid_request');
      if (request.revision < this.revision) throw new Error('worker.stale_revision');
      if (request.revision > this.revision) {
        this.release();
        this.revision = request.revision;
      }
      if (request.type === 'init') {
        if (this.initialized) throw new Error('worker.already_initialized');
        const selection =
          request.backend === 'cpu'
            ? { backend: new CpuBackend(), kind: 'cpu' as const }
            : await createBackend();
        T.setDefaultBackend(selection.backend);
        this.initialized = true;
        this.reply(request, {
          backend: selection.kind,
          ...('fallbackReason' in selection
            ? { fallback: { key: 'worker.cpu_fallback', params: {} } }
            : {}),
        });
        return;
      }
      if (request.type === 'cancel') {
        // The client cancels on every revision bump, including the sync after a pass;
        // a finished model stays usable (its key already pins the level and graph).
        this.release();
        this.reply(request, { state: 'stopped' });
        return;
      }
      if (!this.initialized) throw new Error('worker.not_initialized');
      if (request.type === 'trainReset') {
        if (this.run) throw new Error('worker.training_busy');
        this.releaseStep();
        this.generationParameters = undefined;
        this.reply(request, { state: 'reset' });
        return;
      }
      if ((request.type === 'trainStep' || request.type === 'generate') && this.run)
        throw new Error('worker.training_busy');
      if (request.type === 'syncProgress') {
        this.release();
        const progress = structuredClone(request.progress);
        const registry = buildPlayerRegistry(this.levels, progress);
        this.progress = progress;
        this.registry = registry;
        this.reply(request, { state: 'synced' });
        return;
      }
      if (request.type === 'pause' || request.type === 'resume' || request.type === 'stop') {
        const run = this.run;
        if (!run) throw new Error('worker.no_training');
        if (request.type === 'stop') {
          try {
            const report = await run.session.evaluate();
            await this.retainParameters(run.level, run.request.graph, run.plan);
            this.finish(run, report, false);
          } finally {
            this.release();
          }
          this.reply(request, { state: 'stopped' });
        } else if (request.type === 'pause') {
          run.paused = true;
          this.timerGeneration++;
          clearTimeout(this.timer);
          this.reply(request, { state: 'paused' });
        } else {
          if (run.paused) {
            run.paused = false;
            this.schedule();
          }
          this.reply(request, { state: 'running' });
        }
        return;
      }
      const sourceLevel = this.levels.find((l) => l.id === request.levelId);
      if (!sourceLevel) throw new Error('worker.unknown_level');
      const level = { ...sourceLevel, available: availableComponents(sourceLevel, this.progress) };
      if (request.type === 'generate') {
        if (level.kind !== 'train' || level.generates !== true)
          throw new Error('worker.generation_unavailable');
        await this.generate(request, level);
        return;
      }
      if (request.type === 'evaluate') {
        if (level.kind === 'train' && this.run) throw new Error('worker.training_busy');
        const result = await evaluateLevel(level, request.graph, {
          registry: this.registry,
          learningRate: request.learningRate,
          optimizer: request.optimizer,
        });
        if (result.error)
          result.error = workerError(
            new GraphError(
              result.error.key,
              result.error.node,
              result.error.port,
              result.error.params,
            ),
          );
        if (result.errorKey && !/^[a-z0-9_.]+$/.test(result.errorKey))
          result.errorKey = 'worker.execution_error';
        this.reply(request, result);
        return;
      }
      if (request.type === 'backward') {
        const options = {
          registry: this.registry,
          availableComponents: level.available,
          inputShapes: level.interface.inputs,
          output: level.interface.output,
          wrt:
            level.kind === 'derivative' && level.derivativeInput
              ? [level.derivativeInput]
              : Object.keys(level.interface.inputs),
        };
        const backward = buildBackwardGraph(request.graph, options);
        const trace = request.inputs
          ? await traceBackward(request.graph, { ...options, inputs: request.inputs })
          : undefined;
        this.reply(request, { backward, ...(trace === undefined ? {} : { trace }) });
        return;
      }
      if (request.type === 'trainStep') {
        if (level.kind !== 'train') throw new Error('worker.not_training_level');
        const learningRate = trainingLearningRate(level, request.learningRate);
        const kind = trainingOptimizer(level, request.optimizer);
        const key = JSON.stringify([level.id, request.graph, kind]);
        if (this.stepRun?.key !== key) this.releaseStep();
        if (!this.stepRun) {
          const plan = compileGraph(request.graph, {
            registry: this.registry,
            inputShapes: level.interface.inputs,
            availableComponents: level.available,
            random: new Random(level.seed),
          });
          let batch: TrainingBatch | undefined;
          try {
            if (
              JSON.stringify(plan.outputShapes[level.interface.output]) !==
              JSON.stringify(level.interface.outputShape)
            )
              throw new Error('levels.output_shape');
            batch = level.dataset(new Random(level.seed));
            const session = new TrainingSession(
              plan,
              batch,
              crossEntropy,
              { ...level.optimizer, kind, learningRate },
              level.interface.output,
              { lossOutput: level.lossOutput, accuracy: level.threshold.accuracy !== undefined },
            );
            this.stepRun = { key, plan, batch, session };
          } catch (error) {
            plan.dispose();
            if (batch)
              new Set([...Object.values(batch.inputs), batch.targets]).forEach((t) => t?.dispose());
            throw error;
          }
        }
        this.stepRun.session.optimizer.options.learningRate = learningRate;
        try {
          const trace = await this.stepRun.session.stepTrace();
          await this.retainParameters(level, request.graph, this.stepRun.plan);
          this.reply(request, trace);
        } catch (error) {
          this.releaseStep();
          throw error;
        }
        return;
      }
      let graph = request.graph;
      let probeName = '__worker_probe';
      if (request.type === 'probe') {
        while (graph.nodes.some((n) => n.id === probeName || n.attrs?.name === probeName))
          probeName += '_';
        // Probe only what feeds the endpoint (plus the level inputs), so a half-built graph still
        // shows values for the parts that are already wired.
        const upstream = new Set<string>();
        const pending = [request.endpoint.node];
        while (pending.length) {
          const id = pending.pop()!;
          if (upstream.has(id)) continue;
          upstream.add(id);
          for (const e of graph.edges) if (e.to.node === id) pending.push(e.from.node);
        }
        const kept = graph.nodes.filter((n) => upstream.has(n.id) || n.component === 'input');
        graph = {
          ...graph,
          nodes: [...kept, { id: probeName, component: 'output', attrs: { name: probeName } }],
          edges: [
            ...graph.edges.filter((e) => upstream.has(e.to.node)),
            { from: request.endpoint, to: { node: probeName, port: 'x' } },
          ],
        };
      }
      // A probe may run on a shape variant, so the player can see why a variant fails.
      const groups =
        level.kind === 'train'
          ? [level.interface.inputs]
          : [level.interface.inputs, ...(level.shapeVariants ?? []).map((v) => v.inputs)];
      const shapes =
        request.type === 'probe' && request.shapes
          ? groups.find((g) => JSON.stringify(g) === JSON.stringify(request.shapes))
          : level.interface.inputs;
      if (!shapes) throw new Error('worker.invalid_request');
      let plan: GraphPlan;
      try {
        plan = compileGraph(graph, {
          registry: this.registry,
          inputShapes: shapes,
          availableComponents: level.available,
          random: new Random(level.kind === 'train' ? level.seed : 1),
        });
        if (
          request.type !== 'probe' &&
          JSON.stringify(plan.outputShapes[level.interface.output]) !==
            JSON.stringify(level.interface.outputShape)
        ) {
          plan.dispose();
          throw new Error('levels.output_shape');
        }
      } catch (error) {
        if (request.type !== 'compile') throw error;
        this.reply(request, {
          outputShapes: {},
          metrics: { topNodes: graph.nodes.length, expandedPrimitives: 0, parameterCount: 0 },
          diagnostics: [workerError(error)],
        });
        return;
      }
      let retained = false;
      try {
        if (request.type === 'compile')
          this.reply(request, {
            outputShapes: plan.outputShapes,
            metrics: graphMetrics(graph, this.registry),
            diagnostics: [],
          });
        else if (request.type === 'probe') {
          const inputs: TensorPorts = {};
          let outputs: TensorPorts | undefined;
          try {
            for (const [name, shape] of Object.entries(shapes)) {
              if (!request.inputs[name]) throw new Error('worker.missing_input');
              inputs[name] = T.tensor(request.inputs[name], shape);
            }
            outputs = plan.execute(inputs);
            const value = outputs[probeName];
            this.reply(request, { shape: value.shape, values: Array.from(await value.read()) });
          } finally {
            if (outputs) plan.releaseExecution(outputs);
            Object.values(inputs).forEach((t) => t.dispose());
          }
        } else if (request.type === 'start') {
          if (level.kind !== 'train') throw new Error('worker.not_training_level');
          if (!Number.isSafeInteger(level.steps) || level.steps < 1)
            throw new Error('training.steps');
          const learningRate = trainingLearningRate(level, request.learningRate);
          const kind = trainingOptimizer(level, request.optimizer);
          this.release();
          const metrics = graphMetrics(graph, this.registry);
          const batches: TrainingBatch[] = [];
          try {
            batches.push(level.dataset(new Random(level.seed)));
            if (level.evaluationDataset)
              batches.push(level.evaluationDataset(new Random(level.seed + 1)));
            const session = new TrainingSession(
              plan,
              batches[0],
              crossEntropy,
              { ...level.optimizer, kind, learningRate },
              level.interface.output,
              {
                evaluationBatch: batches[1],
                lossOutput: level.lossOutput,
                accuracy: level.threshold.accuracy !== undefined,
              },
            );
            let initialLoss: number;
            try {
              initialLoss = level.tunable ? (await session.evaluate()).loss : Infinity;
            } catch (error) {
              if (level.tunable && error instanceof Error && error.message === 'training.nonfinite')
                initialLoss = Infinity;
              else {
                session.dispose();
                throw error;
              }
            }
            this.run = {
              initialLoss,
              request,
              level,
              plan,
              session,
              batches,
              paused: false,
              result: {
                levelId: level.id,
                passed: false,
                tests: [],
                metrics,
              },
            };
            retained = true;
            this.reply(request, { state: 'running' });
            this.schedule();
          } finally {
            if (!retained)
              new Set(batches.flatMap((b) => [...Object.values(b.inputs), b.targets])).forEach(
                (t) => t?.dispose(),
              );
          }
        }
      } finally {
        if (!retained) plan.dispose();
      }
    } catch (error) {
      this.emit({
        type: 'error',
        requestId: request.requestId,
        revision: request.revision,
        error: workerError(error),
      });
    }
  }
}
