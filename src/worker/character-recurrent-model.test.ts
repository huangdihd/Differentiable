import { afterEach, expect, it, vi } from 'vitest';
import { CpuBackend } from '../engine/cpu';
import { GraphPlan } from '../engine/graph';
import { evaluateLevel, initialProgress } from '../engine/levels';
import { Random } from '../engine/random';
import * as T from '../engine/tensor';
import { Optimizer, TrainingSession } from '../engine/training';
import {
  characterComparisonLines,
  characterComparisonPositions,
} from '../levels/character-comparison-data';
import { decode, encode, VOCAB } from '../levels/corpus';
import { gameLevels } from '../levels/catalog';
import { fixedWindowExamples } from '../levels/fixed-window-evidence';
import { availableComponents, buildPlayerRegistry } from '../levels/registry';
import {
  CharacterRecurrentModel,
  recurrentCharacterBatch,
  recurrentCharacterPredictionBatch,
  recurrentCharacterSeed,
  releaseRecurrentBatch,
} from './character-recurrent-model';
import {
  recurrentCharacterParameterTestProgress,
  recurrentCharacterTestProgress,
} from './recurrent-character-test-fixture';

afterEach(() => vi.restoreAllMocks());

it('reproduces all 1683 seeded normal parameters, including biases', async () => {
  const backend = new CpuBackend();
  const first = new CharacterRecurrentModel(backend, {
    progress: recurrentCharacterTestProgress(),
  });
  const replay = new CharacterRecurrentModel(backend, {
    progress: recurrentCharacterTestProgress(),
  });
  const random = new Random(recurrentCharacterSeed);
  try {
    let parameterCount = 0;
    for (const [name, value] of first.parameters) {
      const actual = await value.read();
      parameterCount += actual.length;
      expect(actual).toEqual(await replay.parameters.get(name)!.read());
      expect(actual).toEqual(
        Float32Array.from({ length: actual.length }, () => random.normal() * 0.15),
      );
    }
    expect(parameterCount).toBe(1683);
  } finally {
    first.dispose();
    replay.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
  expect(() => first.execute({})).toThrow('worker.rnn_model_disposed');
});

it('packs every shared position once without padding, cross-line inputs or target leakage', async () => {
  const backend = new CpuBackend();
  const batch = recurrentCharacterBatch(backend);
  try {
    expect(characterComparisonPositions).toHaveLength(1495);
    let previousRows = characterComparisonPositions
      .filter(({ offset }) => offset === 2)
      .map(({ lineIndex }) => lineIndex);
    expect(await batch.inputs.initialHidden.read()).toEqual(
      new Float32Array(previousRows.length * 24),
    );
    let tokenUpdates = 0;
    const actualPositions: typeof characterComparisonPositions = [];
    const newline = encode('\n')[0];
    for (let step = 0; batch.inputs[`tokens_${step}`]; step++) {
      const positions = characterComparisonPositions.filter(({ offset }) => offset === step + 1);
      const rows = step === 0 ? previousRows : positions.map(({ lineIndex }) => lineIndex);
      const tokens = Array.from(await batch.inputs[`tokens_${step}`].read());
      expect(batch.inputs[`tokens_${step}`].shape).toEqual([rows.length]);
      expect(tokens).toEqual(
        rows.map((lineIndex) => characterComparisonLines[lineIndex].tokens[step]),
      );
      expect(tokens).not.toContain(newline);
      tokenUpdates += rows.length;
      if (rows.length !== previousRows.length) {
        const retainedRows = Array.from(await batch.inputs[`rows_${step}`].read()).map(
          (index) => previousRows[index],
        );
        expect(retainedRows).toEqual(rows);
      } else expect(batch.inputs[`rows_${step}`]).toBeUndefined();
      if (step > 0) {
        const targets = await batch.inputs[`targets_${step}`].read();
        expect(targets.length).toBe(rows.length * VOCAB.length);
        positions.forEach((position, row) => {
          const label = targets.slice(row * VOCAB.length, (row + 1) * VOCAB.length);
          expect(label.reduce((sum, value) => sum + value, 0)).toBe(1);
          expect(label.indexOf(1)).toBe(
            characterComparisonLines[position.lineIndex].tokens[position.offset],
          );
          actualPositions.push(position);
        });
      } else expect(batch.inputs.targets_0).toBeUndefined();
      previousRows = rows;
    }
    expect(actualPositions).toEqual(characterComparisonPositions);
    expect(tokenUpdates).toBe(
      characterComparisonPositions.length + characterComparisonLines.length,
    );
    expect(tokenUpdates).toBeLessThan(
      characterComparisonPositions.reduce((sum, { offset }) => sum + offset, 0),
    );
    expect((await batch.inputs.lossScale.read())[0]).toBeCloseTo(-1 / 1495, 10);
  } finally {
    releaseRecurrentBatch(batch);
  }
  expect(backend.liveBuffers).toBe(0);
});

it('matches standalone causal prefixes and averages all 1495 target losses rather than line means', async () => {
  const backend = new CpuBackend();
  const model = new CharacterRecurrentModel(backend, {
    progress: recurrentCharacterTestProgress(),
  });
  const batch = recurrentCharacterBatch(backend);
  const trace = new Map<string, T.Tensor>();
  // Cached plans own constants for each packed row count and for standalone predictions.
  model.releaseExecution(model.execute(batch.inputs));
  const warmPrediction = recurrentCharacterPredictionBatch('th', backend);
  model.releaseExecution(model.execute(warmPrediction.inputs));
  releaseRecurrentBatch(warmPrediction);
  const baseline = backend.liveBuffers;
  let outputs: Record<string, T.Tensor> | undefined;
  try {
    outputs = model.execute(batch.inputs, trace);
    const lossesByLine = new Map<number, number[]>();
    let globalLoss = 0;
    let count = 0;
    const samples = [
      characterComparisonPositions[0],
      characterComparisonPositions.at(-1)!,
      ...fixedWindowExamples.map(({ history }) =>
        characterComparisonPositions.find(
          ({ lineIndex, offset }) =>
            decode(characterComparisonLines[lineIndex].tokens.slice(0, offset)) === history,
        )!,
      ),
    ];
    for (const [name, value] of trace) {
      const step = Number(name.slice('probabilities_'.length));
      const positions = characterComparisonPositions.filter(({ offset }) => offset === step + 1);
      const probabilities = await value.read();
      positions.forEach(({ lineIndex, offset }, row) => {
        const target = characterComparisonLines[lineIndex].tokens[offset];
        const loss = -Math.log(probabilities[row * VOCAB.length + target] + 1e-7);
        globalLoss += loss;
        const losses = lossesByLine.get(lineIndex) ?? [];
        losses.push(loss);
        lossesByLine.set(lineIndex, losses);
        count++;
      });
    }
    expect(count).toBe(1495);
    const actualLoss = (await outputs.loss.read())[0];
    const lineMean =
      [...lossesByLine.values()].reduce(
        (sum, losses) => sum + losses.reduce((total, loss) => total + loss, 0) / losses.length,
        0,
      ) / lossesByLine.size;
    expect(actualLoss).toBeCloseTo(globalLoss / count, 5);
    expect(Math.abs(actualLoss - lineMean)).toBeGreaterThan(1e-5);

    for (const sample of samples) {
      expect(sample).toBeDefined();
      const { lineIndex, offset } = sample;
      const history = decode(characterComparisonLines[lineIndex].tokens.slice(0, offset));
      const predictionBatch = recurrentCharacterPredictionBatch(history, backend);
      let prediction: Record<string, T.Tensor> | undefined;
      try {
        prediction = model.execute(predictionBatch.inputs);
        const values = await prediction.probabilities.read();
        const rows = characterComparisonPositions.filter((position) => position.offset === offset);
        const row = rows.findIndex((position) => position.lineIndex === lineIndex);
        const packed = await trace.get(`probabilities_${offset - 1}`)!.read();
        expect(
          Math.max(
            ...values.map((value, index) => Math.abs(value - packed[row * VOCAB.length + index])),
          ),
        ).toBeLessThan(1e-7);
      } finally {
        if (prediction) model.releaseExecution(prediction, Object.values(predictionBatch.inputs));
        releaseRecurrentBatch(predictionBatch);
      }
    }
  } finally {
    if (outputs) model.releaseExecution(outputs, Object.values(batch.inputs));
    expect(backend.liveBuffers).toBe(baseline);
    releaseRecurrentBatch(batch);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it.each(['matmul', 'exp', 'softmax', 'log'])(
  'releases partial tapes on %s failure but retains parameters, constants and inputs',
  async (operation) => {
    const backend = new CpuBackend();
    const model = new CharacterRecurrentModel(backend, {
      progress: recurrentCharacterTestProgress(),
    });
    const batch = recurrentCharacterBatch(backend);
    model.releaseExecution(model.execute(batch.inputs));
    const baseline = backend.liveBuffers;
    const submit = backend.submit.bind(backend);
    const failure = new Error(`injected ${operation} failure`);
    const dispatch = vi.spyOn(backend, 'submit').mockImplementation((commands) => {
      if (commands.some(({ op }) => op === operation)) throw failure;
      submit(commands);
    });
    try {
      expect(() => model.execute(batch.inputs)).toThrow(
        operation === 'exp' ? 'component_error' : failure.message,
      );
      expect(backend.liveBuffers).toBe(baseline);
      for (const value of [...model.parameters.values(), ...Object.values(batch.inputs)])
        expect(() => value.assertLive()).not.toThrow();
      dispatch.mockRestore();
      const output = model.execute(batch.inputs);
      expect(Number.isFinite((await output.loss.read())[0])).toBe(true);
      model.releaseExecution(output, Object.values(batch.inputs));
      expect(backend.liveBuffers).toBe(baseline);
    } finally {
      releaseRecurrentBatch(batch);
      model.dispose();
    }
    expect(backend.liveBuffers).toBe(0);
  },
);

it('cleans the complete unreturned tape when lossScale is missing', () => {
  const backend = new CpuBackend();
  const model = new CharacterRecurrentModel(backend, {
    progress: recurrentCharacterTestProgress(),
  });
  const batch = recurrentCharacterBatch(backend);
  const { lossScale, ...inputs } = batch.inputs;
  model.releaseExecution(model.execute(batch.inputs));
  const baseline = backend.liveBuffers;
  try {
    expect(() => model.execute(inputs)).toThrow('worker.rnn_loss_scale');
    expect(backend.liveBuffers).toBe(baseline);
    expect(() => lossScale.assertLive()).not.toThrow();
  } finally {
    releaseRecurrentBatch(batch);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it.each(['training batch', 'prediction batch', 'parameters', 'constants'])(
  'cleans partial allocations when creating %s fails',
  (kind) => {
    const backend = new CpuBackend();
    const allocate = backend.allocate.bind(backend);
    let allocations = 0;
    vi.spyOn(backend, 'allocate').mockImplementation((...arguments_) => {
      if (++allocations === (kind === 'constants' ? 7 : 4))
        throw new Error('injected allocation failure');
      return allocate(...arguments_);
    });
    const create = () => {
      if (kind === 'training batch') return recurrentCharacterBatch(backend);
      if (kind === 'prediction batch')
        return recurrentCharacterPredictionBatch('little moth', backend);
      return new CharacterRecurrentModel(backend, { progress: recurrentCharacterTestProgress() });
    };
    expect(create).toThrow('injected allocation failure');
    expect(backend.liveBuffers).toBe(0);
  },
);

it.each(['', 't', 'th\n', 'unknown!'])(
  'rejects invalid inference history %j before allocating',
  (history) => {
    const backend = new CpuBackend();
    expect(() => recurrentCharacterPredictionBatch(history, backend)).toThrow();
    expect(backend.liveBuffers).toBe(0);
  },
);

it.each(['empty', 'passed', 'unlocked', 'solution'])(
  'requires the real passed player cell (%s missing), never a reference fallback',
  (missing) => {
    const backend = new CpuBackend();
    const progress = missing === 'empty' ? initialProgress([]) : recurrentCharacterTestProgress();
    if (missing === 'passed') progress.passed = progress.passed.filter((id) => id !== 'time-rnn');
    if (missing === 'unlocked')
      progress.unlockedComponents = progress.unlockedComponents.filter(
        (name) => name !== 'rnn_cell',
      );
    if (missing === 'solution') delete progress.passedSolutions['time-rnn'];
    expect(() => new CharacterRecurrentModel(backend, { progress })).toThrow(
      'worker.rnn_player_cell_required',
    );
    expect(backend.liveBuffers).toBe(0);
  },
);

it('executes the player-authored extra identity and passes its hiddenState to the character head', async () => {
  const backend = new CpuBackend();
  const progress = recurrentCharacterTestProgress();
  const snapshot = structuredClone(progress);
  const model = new CharacterRecurrentModel(backend, { progress });
  const batch = recurrentCharacterPredictionBatch('th', backend);
  const execute = GraphPlan.prototype.execute;
  const traces: Map<string, T.Tensor>[] = [];
  const execution = vi.spyOn(GraphPlan.prototype, 'execute').mockImplementation(function (
    this: GraphPlan,
    inputs,
  ) {
    const trace = new Map<string, T.Tensor>();
    traces.push(trace);
    return execute.call(this, inputs, trace);
  });
  let outputs: Record<string, T.Tensor> | undefined;
  try {
    outputs = model.execute(batch.inputs);
    expect(execution).toHaveBeenCalledTimes(2);
    expect(new Set(execution.mock.contexts).size).toBe(1);
    execution.mock.calls.forEach(([inputs], step) => {
      expect(
        Object.fromEntries(Object.entries(inputs).map(([name, tensor]) => [name, tensor.shape])),
      ).toEqual({ x: [1, 8], h: [1, 24], wx: [8, 24], wh: [24, 24], b: [24] });
      expect(inputs.x.tape).toBeDefined();
      expect(inputs.wx).toBe(model.parameters.get('inputWeight'));
      expect(inputs.wh).toBe(model.parameters.get('recurrentWeight'));
      expect(inputs.b).toBe(model.parameters.get('hiddenBias'));
      const result = execution.mock.results[step].value as Record<string, T.Tensor>;
      expect(result.hiddenState).toBe(traces[step].get('player_identity'));
      if (step > 0) expect(inputs.h).toBe(execution.mock.results[step - 1].value.hiddenState);
    });
    const hidden = execution.mock.results.at(-1)!.value.hiddenState as T.Tensor;
    const projected = T.matmul(hidden, model.parameters.get('outputWeight')!);
    const logits = T.add(projected, model.parameters.get('outputBias')!);
    const expected = T.softmax(logits);
    expect(await outputs.probabilities.read()).toEqual(await expected.read());
    [projected, logits, expected].forEach((value) => value.dispose());
    expect(progress).toEqual(snapshot);
  } finally {
    if (outputs) model.releaseExecution(outputs);
    releaseRecurrentBatch(batch);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('a deliberately zero player graph actually changes predictions without any grading or built-in RNN', async () => {
  const backend = new CpuBackend();
  const progress = recurrentCharacterTestProgress();
  const player = new CharacterRecurrentModel(backend, { progress });
  const zeroProgress = recurrentCharacterTestProgress();
  const zeroGraph = zeroProgress.passedSolutions['time-rnn'];
  zeroGraph.nodes.find(({ id }) => id === 'player_one')!.attrs = { value: 0 };
  const zero = new CharacterRecurrentModel(backend, { progress: zeroProgress });
  const batch = recurrentCharacterPredictionBatch('little moth', backend);
  const outputs: { model: CharacterRecurrentModel; values: Record<string, T.Tensor> }[] = [];
  try {
    for (const model of [player, zero])
      outputs.push({ model, values: model.execute(batch.inputs) });
    const actual = await outputs[0].values.probabilities.read();
    const changed = await outputs[1].values.probabilities.read();
    expect(changed).not.toEqual(actual);
    const bias = await zero.parameters.get('outputBias')!.read();
    const exponentials = Array.from(bias, (value) => Math.exp(value));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    changed.forEach((value, index) => expect(value).toBeCloseTo(exponentials[index] / total, 7));
  } finally {
    for (const { model, values } of outputs) model.releaseExecution(values);
    releaseRecurrentBatch(batch);
    player.dispose();
    zero.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('executes the earned nested matrix circuit instead of its native override', async () => {
  const backend = new CpuBackend();
  const progress = recurrentCharacterTestProgress();
  const original = new CharacterRecurrentModel(backend, { progress });
  const provider = gameLevels.find((level) => level.unlockComponents?.includes('matmul'))!;
  const graph = progress.passedSolutions[provider.id];
  const output = graph.nodes.find(({ component }) => component === 'output')!;
  const edge = graph.edges.find(({ to }) => to.node === output.id)!;
  const source = edge.from;
  edge.from = { node: 'player_zero_matrix', port: 'out' };
  graph.nodes.push(
    { id: 'player_zero', component: 'const', attrs: { value: 0 } },
    { id: 'player_zero_matrix', component: 'mul' },
  );
  graph.edges.push(
    { from: source, to: { node: 'player_zero_matrix', port: 'a' } },
    { from: { node: 'player_zero', port: 'out' }, to: { node: 'player_zero_matrix', port: 'b' } },
  );
  const modified = new CharacterRecurrentModel(backend, { progress });
  const batch = recurrentCharacterPredictionBatch('th', backend);
  let originalOutputs: Record<string, T.Tensor> | undefined;
  let modifiedOutputs: Record<string, T.Tensor> | undefined;
  try {
    originalOutputs = original.execute(batch.inputs);
    modifiedOutputs = modified.execute(batch.inputs);
    expect(await modifiedOutputs.probabilities.read()).not.toEqual(
      await originalOutputs.probabilities.read(),
    );
  } finally {
    if (originalOutputs) original.releaseExecution(originalOutputs);
    if (modifiedOutputs) modified.releaseExecution(modifiedOutputs);
    releaseRecurrentBatch(batch);
    original.dispose();
    modified.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('caches exactly one plan per packed row count and retains constants across tape release', async () => {
  const backend = new CpuBackend();
  const model = new CharacterRecurrentModel(backend, {
    progress: recurrentCharacterTestProgress(),
  });
  const batch = recurrentCharacterBatch(backend);
  const execute = vi.spyOn(GraphPlan.prototype, 'execute');
  try {
    const first = model.execute(batch.inputs);
    const loss = await first.loss.read();
    const plans = new Set(execute.mock.contexts as GraphPlan[]);
    const rows = new Set(
      Object.entries(batch.inputs)
        .filter(([name]) => name.startsWith('tokens_'))
        .map(([, value]) => value.shape[0]),
    );
    expect(plans.size).toBe(rows.size);
    for (const plan of plans) expect(plan.parameters.size).toBe(0);
    model.releaseExecution(first);
    const baseline = backend.liveBuffers;
    for (let replay = 0; replay < 2; replay++) {
      const outputs = model.execute(batch.inputs);
      expect(await outputs.loss.read()).toEqual(loss);
      model.releaseExecution(outputs);
      model.releaseExecution(outputs);
      expect(backend.liveBuffers).toBe(baseline);
    }
    expect(new Set(execute.mock.contexts)).toEqual(plans);
  } finally {
    releaseRecurrentBatch(batch);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('accepts a grade-passing scalar parameter and shares its Adam-updated tensor across every row-count plan', async () => {
  const progress = recurrentCharacterParameterTestProgress();
  const level = gameLevels.find((candidate) => candidate.id === 'time-rnn')!;
  if (level.kind === 'observation') throw new Error('expected circuit level');
  const graded = await evaluateLevel(
    { ...level, available: availableComponents(level, progress) },
    progress.passedSolutions[level.id],
    { registry: buildPlayerRegistry(gameLevels, progress) },
  );
  expect(graded.passed, JSON.stringify(graded)).toBe(true);
  expect(graded.tests).toHaveLength(16);

  const backend = new CpuBackend();
  const model = new CharacterRecurrentModel(backend, { progress });
  const batch = recurrentCharacterBatch(backend);
  const prediction = recurrentCharacterPredictionBatch('th', backend);
  const session = new TrainingSession(
    model,
    batch,
    undefined,
    { kind: 'adam', learningRate: 0.01 },
    'probabilities',
    { accuracy: false, lossOutput: 'loss' },
  );
  const execute = vi.spyOn(GraphPlan.prototype, 'execute');
  try {
    expect(model.parameters.has('cell/embedding')).toBe(false);
    // No warmup: the first training forward must register the lazy parameter in time.
    await session.run(1);
    const first = model.parameters.get('cell/embedding')!;
    expect(first.shape).toEqual([]);
    expect((await first.read())[0]).not.toBe(1);
    expect(model.parameters.get('embedding')!.shape).toEqual([VOCAB.length, 8]);
    expect(model.parameters.size).toBe(7);
    model.releaseExecution(model.execute(prediction.inputs));
    const plans = new Set(execute.mock.contexts as GraphPlan[]);
    const rows = new Set(
      Object.entries(batch.inputs)
        .filter(([name]) => name.startsWith('tokens_'))
        .map(([, value]) => value.shape[0]),
    );
    rows.add(1);
    expect(plans.size).toBe(rows.size);
    expect(new Set([...plans].map((plan) => plan.parameters)).size).toBe(1);
    for (const plan of plans) expect(plan.parameters.get('embedding')).toBe(first);
    const baseline = backend.liveBuffers;

    await session.run(1);
    const second = model.parameters.get('cell/embedding')!;
    expect(second).not.toBe(first);
    expect(() => first.assertLive()).toThrow('engine.tensor.disposed');
    expect((await second.read())[0]).not.toBe(1);
    model.releaseExecution(model.execute(prediction.inputs));
    for (const plan of plans) expect(plan.parameters.get('embedding')).toBe(second);
    expect(new Set(execute.mock.contexts)).toEqual(plans);
    expect(backend.liveBuffers).toBe(baseline);
  } finally {
    session.dispose();
    releaseRecurrentBatch(batch);
    releaseRecurrentBatch(prediction);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
}, 20000);

it('keeps nested parameter paths separate and releases replacements even before another execute', async () => {
  const backend = new CpuBackend();
  const progress = recurrentCharacterParameterTestProgress();
  const provider = gameLevels.find((level) => level.unlockComponents?.includes('tanh'))!;
  const graph = progress.passedSolutions[provider.id];
  const output = graph.nodes.find(({ component }) => component === 'output')!;
  const edge = graph.edges.find(({ to }) => to.node === output.id)!;
  const source = edge.from;
  edge.from = { node: 'parameter_scale', port: 'out' };
  graph.nodes.push(
    { id: 'embedding', component: 'param', attrs: { shape: [], init: 'ones' } },
    { id: 'parameter_scale', component: 'mul' },
  );
  graph.edges.push(
    { from: source, to: { node: 'parameter_scale', port: 'a' } },
    { from: { node: 'embedding', port: 'out' }, to: { node: 'parameter_scale', port: 'b' } },
  );
  const model = new CharacterRecurrentModel(backend, { progress });
  const batch = recurrentCharacterBatch(backend);
  const optimizer = new Optimizer({ kind: 'adam', learningRate: 0.01 });
  let outputs: Record<string, T.Tensor> | undefined;
  let gradients: T.Tensor[] = [];
  try {
    outputs = model.execute(batch.inputs);
    const names = [...model.parameters.keys()];
    expect(names.filter((name) => name.startsWith('cell/')).sort()).toEqual([
      'cell/embedding',
      'cell/next/embedding',
    ]);
    expect(model.parameters.get('cell/embedding')).not.toBe(
      model.parameters.get('cell/next/embedding'),
    );
    gradients = await T.backward(outputs.loss, [...model.parameters.values()]);
    for (const name of ['cell/embedding', 'cell/next/embedding']) {
      const values = await gradients[names.indexOf(name)].read();
      expect(values.every(Number.isFinite)).toBe(true);
      expect(values.some((value) => value !== 0)).toBe(true);
    }
    await optimizer.update(model.parameters, gradients);
    for (const name of ['cell/embedding', 'cell/next/embedding'])
      expect((await model.parameters.get(name)!.read())[0]).not.toBe(1);
    // Do not execute again: the plan's map still refers to the old Adam tensors.
  } finally {
    gradients.forEach((gradient) => gradient.dispose());
    if (outputs) model.releaseExecution(outputs);
    optimizer.dispose();
    releaseRecurrentBatch(batch);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it.each(['allocation', 'forward'])(
  'owns lazy cell parameters and constants when the first %s fails',
  (failure) => {
    const backend = new CpuBackend();
    const progress = recurrentCharacterParameterTestProgress();
    const graph = progress.passedSolutions['time-rnn'];
    const scale = graph.nodes.find(({ id }) => id === 'embedding')!;
    // Initialize the parameter before the constants, then fail the next storage allocation.
    graph.nodes = [scale, ...graph.nodes.filter((node) => node !== scale)];
    const model = new CharacterRecurrentModel(backend, { progress });
    const batch = recurrentCharacterPredictionBatch('th', backend);
    const allocate = backend.allocate.bind(backend);
    const submit = backend.submit.bind(backend);
    if (failure === 'allocation') {
      const execute = GraphPlan.prototype.execute;
      vi.spyOn(GraphPlan.prototype, 'execute').mockImplementation(function (
        this: GraphPlan,
        inputs,
      ) {
        vi.spyOn(backend, 'allocate').mockImplementation((...arguments_) => {
          if (this.parameters.size > 0) throw new Error('lazy allocation failure');
          return allocate(...arguments_);
        });
        return execute.call(this, inputs);
      });
    } else
      vi.spyOn(backend, 'submit').mockImplementation((commands) => {
        if (commands.some(({ op }) => op === 'exp')) throw new Error('first forward failure');
        submit(commands);
      });
    try {
      expect(() => model.execute(batch.inputs)).toThrow(
        failure === 'allocation' ? 'lazy allocation failure' : 'component_error',
      );
      expect(model.parameters.get('cell/embedding')!.shape).toEqual([]);
      for (const value of [...model.parameters.values(), ...Object.values(batch.inputs)])
        expect(() => value.assertLive()).not.toThrow();
    } finally {
      releaseRecurrentBatch(batch);
      model.dispose();
    }
    expect(backend.liveBuffers).toBe(0);
  },
);

it.each(['shape', 'output'])(
  'rejects player cells with invalid %s and releases their resources',
  (invalid) => {
    const backend = new CpuBackend();
    const progress = recurrentCharacterTestProgress();
    const graph = progress.passedSolutions['time-rnn'];
    if (invalid === 'shape')
      graph.edges.find(({ to }) => to.node === 'output_hiddenState')!.from = {
        node: 'x',
        port: 'out',
      };
    if (invalid === 'output')
      graph.nodes.find(({ id }) => id === 'output_hiddenState')!.attrs = { name: 'wrong' };
    const model = new CharacterRecurrentModel(backend, { progress });
    const batch = recurrentCharacterPredictionBatch('th', backend);
    const baseline = backend.liveBuffers;
    try {
      expect(() => model.execute(batch.inputs)).toThrow('worker.rnn_hidden_shape');
      expect(backend.liveBuffers).toBe(baseline);
      for (const value of [...model.parameters.values(), ...Object.values(batch.inputs)])
        expect(() => value.assertLive()).not.toThrow();
    } finally {
      releaseRecurrentBatch(batch);
      model.dispose();
    }
    expect(backend.liveBuffers).toBe(0);
  },
);

it('releases earlier cell executions when a later player cell fails', () => {
  const backend = new CpuBackend();
  const model = new CharacterRecurrentModel(backend, {
    progress: recurrentCharacterTestProgress(),
  });
  const batch = recurrentCharacterPredictionBatch('little moth', backend);
  model.releaseExecution(model.execute(batch.inputs));
  const baseline = backend.liveBuffers;
  const submit = backend.submit.bind(backend);
  let exponentials = 0;
  vi.spyOn(backend, 'submit').mockImplementation((commands) => {
    if (commands.some(({ op }) => op === 'exp') && ++exponentials === 3)
      throw new Error('later cell failure');
    submit(commands);
  });
  try {
    expect(() => model.execute(batch.inputs)).toThrow('component_error');
    expect(exponentials).toBe(3);
    expect(backend.liveBuffers).toBe(baseline);
  } finally {
    releaseRecurrentBatch(batch);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('backpropagates through player cell edits and preserves borrowed input tapes on release', async () => {
  const backend = new CpuBackend();
  const batch = recurrentCharacterBatch(backend);
  const initialLeaf = batch.inputs.initialHidden;
  const one = T.tensor([1], [], backend);
  batch.inputs.initialHidden = T.mul(initialLeaf, one);
  try {
    for (const identityScale of [1, 0]) {
      const progress = recurrentCharacterTestProgress();
      progress.passedSolutions['time-rnn'].nodes.find(({ id }) => id === 'player_one')!.attrs = {
        value: identityScale,
      };
      const model = new CharacterRecurrentModel(backend, { progress });
      let outputs: Record<string, T.Tensor> | undefined;
      let gradients: T.Tensor[] = [];
      try {
        outputs = model.execute(batch.inputs);
        gradients = await T.backward(outputs.loss, [...model.parameters.values()]);
        const names = [...model.parameters.keys()];
        for (const [index, gradient] of gradients.entries()) {
          const values = await gradient.read();
          expect(values.every(Number.isFinite)).toBe(true);
          const shouldUpdate = identityScale === 1 || names[index] === 'outputBias';
          expect(
            values.some((value) => value !== 0),
            names[index],
          ).toBe(shouldUpdate);
        }
      } finally {
        gradients.forEach((gradient) => gradient.dispose());
        if (outputs) model.releaseExecution(outputs);
        model.dispose();
      }
      for (const value of [initialLeaf, one, ...Object.values(batch.inputs)])
        expect(() => value.assertLive()).not.toThrow();
    }
  } finally {
    releaseRecurrentBatch(batch);
    initialLeaf.dispose();
    one.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});

it('trains and predicts with earned dependencies whose old saved solutions are absent', async () => {
  const backend = new CpuBackend();
  const progress = recurrentCharacterTestProgress();
  for (const component of ['tanh', 'matmul']) {
    const provider = gameLevels.find((level) => level.unlockComponents?.includes(component))!;
    delete progress.passedSolutions[provider.id];
  }
  const before = JSON.stringify(progress);
  const model = new CharacterRecurrentModel(backend, { progress });
  const batch = recurrentCharacterBatch(backend);
  const prediction = recurrentCharacterPredictionBatch('th', backend);
  const session = new TrainingSession(
    model,
    batch,
    undefined,
    { kind: 'adam', learningRate: 0.01 },
    'probabilities',
    { accuracy: false, lossOutput: 'loss' },
  );
  try {
    const report = await session.run(1);
    expect(report.step).toBe(1);
    expect(Number.isFinite(report.loss)).toBe(true);
    const outputs = model.execute(prediction.inputs);
    try {
      expect(outputs.probabilities.shape).toEqual([1, VOCAB.length]);
      const probabilities = await outputs.probabilities.read();
      expect(probabilities.every(Number.isFinite)).toBe(true);
      expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 5);
    } finally {
      model.releaseExecution(outputs);
    }
    expect(JSON.stringify(progress)).toBe(before);
  } finally {
    session.dispose();
    releaseRecurrentBatch(batch);
    releaseRecurrentBatch(prediction);
    model.dispose();
  }
  expect(backend.liveBuffers).toBe(0);
});
