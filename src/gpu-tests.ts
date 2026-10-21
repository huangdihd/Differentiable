import { CpuBackend } from './engine/cpu';
import { WebGpuBackend } from './engine/webgpu';
import { type Backend, type Attributes, infer, size } from './engine/backend';
import { tensor, backward } from './engine/tensor';
import { compileGraph } from './engine/graph';
import { Random } from './engine/random';
import { TrainingSession } from './engine/training';
import { gptFixture, type GptConfig } from './engine/gpu-gpt-fixture';

async function report(type: string, data: Record<string, unknown> = {}) {
  const response = await fetch('/__gpu_report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, ...data }),
  });
  if (!response.ok) throw new Error('Failed to deliver test report');
}
function assert(ok: boolean, message: string): asserts ok {
  if (!ok) throw new Error(message);
}
function close(
  actual: Float32Array,
  expected: Float32Array,
  name: string,
  atol = 2e-4,
  rtol = 2e-3,
) {
  assert(actual.length === expected.length, `${name}: length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assert(
      Number.isFinite(actual[i]) &&
        Number.isFinite(expected[i]) &&
        Math.abs(actual[i] - expected[i]) <= atol + rtol * Math.abs(expected[i]),
      `${name}[${i}]: GPU=${actual[i]}, CPU=${expected[i]}`,
    );
  }
}
type Fixture = { op: string; shapes: number[][]; attrs?: Attributes; data?: number[][] };
async function operations(cpu: CpuBackend, gpu: WebGpuBackend) {
  const fixtures: Fixture[] = [];
  for (const shapes of [
    [[2, 3, 5], [5]],
    [
      [2, 1, 5],
      [1, 3, 1],
    ],
    [[], [2, 3]],
  ])
    for (const op of ['add', 'mul', 'max', 'ge']) fixtures.push({ op, shapes });
  for (const op of ['exp', 'log', 'recip', 'softmax', 'layernorm'])
    for (const shape of [
      [3, 7],
      [2, 3, 33],
      [2, 1],
      ...(op === 'softmax' || op === 'layernorm'
        ? [
            [3, 1024],
            [3, 1031],
          ]
        : []),
    ])
      fixtures.push({ op, shapes: [shape] });
  fixtures.push(
    { op: 'reshape', shapes: [[2, 3, 5]], attrs: { shape: [5, 6] } },
    { op: 'transpose', shapes: [[2, 3, 5]], attrs: { perm: [2, 0, 1] } },
  );
  for (const op of ['sum', 'reduce_max'])
    for (const axes of [[0, 2], [-1], []])
      for (const keepdims of [true, false])
        fixtures.push({ op, shapes: [[2, 3, 5]], attrs: { axes, keepdims } });
  for (const op of ['sum', 'reduce_max'])
    for (const width of [1024, 1031])
      for (const keepdims of [true, false])
        fixtures.push({ op, shapes: [[3, width]], attrs: { axes: [-1], keepdims } });
  for (const op of ['sum', 'reduce_max'])
    fixtures.push({ op, shapes: [[17, 3, 61]], attrs: { axes: [0, 2] } });
  fixtures.push({
    op: 'reduce_max',
    shapes: [[3, 1031]],
    attrs: { axes: [-1] },
    data: [
      Array.from({ length: 3 * 1031 }, (_, i) => (i % 1031 === 7 || i % 1031 === 900 ? 2 : 1)),
    ],
  });
  for (const shapes of [
    [
      [3, 7],
      [7, 5],
    ],
    [
      [2, 1, 3, 7],
      [1, 4, 7, 5],
    ],
    [
      [3, 7],
      [2, 7, 5],
    ],
  ])
    fixtures.push({ op: 'matmul', shapes });
  fixtures.push(
    {
      op: 'max',
      shapes: [[2, 3], [3]],
      data: [
        [1, 1, 2, 2, 1, 2],
        [1, 1, 2],
      ],
    },
    { op: 'reduce_max', shapes: [[2, 3]], attrs: { axes: [1] }, data: [[2, 2, 1, 3, 3, 3]] },
  );
  fixtures.push({
    op: 'gather',
    shapes: [
      [7, 5],
      [2, 4],
    ],
    data: [[], [1, 1, 6, 0, 1, 6, 6, 1]],
  });
  const rng = new Random(713);
  let count = 0;
  for (let trial = 0; trial < 3; trial++)
    for (const f of fixtures) {
      const attrs = f.attrs ?? {};
      const data = f.shapes.map((s, i) =>
        f.data?.[i]?.length
          ? f.data[i]
          : Array.from({ length: size(s) }, () =>
              f.op === 'log'
                ? 0.2 + rng.uniform()
                : f.op === 'recip'
                  ? (rng.uniform() < 0.5 ? -1 : 1) * (0.3 + rng.uniform())
                  : rng.normal() * 0.4,
            ),
      );
      const shape = infer(f.op, f.shapes, attrs);
      const seed = Float32Array.from({ length: size(shape) }, () => rng.normal());
      async function execute(backend: Backend) {
        const inputs = f.shapes.map((s, i) => tensor(data[i], s, backend));
        const output = tensor(new Float32Array(size(shape)), shape, backend);
        const g = tensor(seed, shape, backend);
        const owned = [...inputs, output, g];
        try {
          backend.submit([{ op: f.op, inputs, output, attrs }]);
          const result = [await output.read()];
          for (let i = 0; i < inputs.length; i++) {
            const dx = tensor(new Float32Array(size(inputs[i].shape)), inputs[i].shape, backend);
            owned.push(dx);
            backend.submit([
              {
                op: 'vjp',
                inputs: [...inputs, output, g],
                output: dx,
                attrs: { ...attrs, op: f.op, input: i, count: inputs.length },
              },
            ]);
            result.push(await dx.read());
          }
          return result;
        } finally {
          owned.forEach((t) => t.dispose());
        }
      }
      const expected = await execute(cpu),
        actual = await execute(gpu);
      actual.forEach((a, i) => close(a, expected[i], `${f.op} trial=${trial} result=${i}`));
      if (
        f.op === 'gather' ||
        (['softmax', 'layernorm', 'sum', 'reduce_max'].includes(f.op) &&
          f.shapes[0].at(-1)! >= 1024)
      )
        for (let repeat = 0; repeat < 5; repeat++) {
          const again = await execute(gpu);
          again.forEach((a, i) => {
            const bits = new Uint32Array(a.buffer, a.byteOffset, a.length);
            const reference = new Uint32Array(
              actual[i].buffer,
              actual[i].byteOffset,
              actual[i].length,
            );
            assert(
              bits.every((bit, j) => bit === reference[j]),
              `${f.op} result=${i} must be bitwise deterministic`,
            );
          });
        }
      count++;
    }
  await report('operations', {
    fixtures: count,
    randomVjpSeeds: true,
    gatherRepeatRuns: 5,
    longRowWidths: [1024, 1031],
    longRowForwardAndVjpRepeatRuns: 5,
  });
}
async function largeDispatches(cpu: CpuBackend, gpu: WebGpuBackend) {
  // 64 lanes per serial kernel: this requires 65,537 original x workgroups.
  // Width-one reductions avoid a >256 MiB long-row input; normalization separately
  // exercises the parallel row kernel with 65,536 original x workgroups.
  const length = 65536 * 64 + 1;
  const fixtures: Fixture[] = [
    { op: 'exp', shapes: [[length]] },
    { op: 'sum', shapes: [[length, 1]], attrs: { axes: [-1] } },
    { op: 'reduce_max', shapes: [[length, 1]], attrs: { axes: [-1] } },
    { op: 'softmax', shapes: [[65536, 2]] },
    { op: 'layernorm', shapes: [[65536, 2]] },
    // The original z dispatch exceeds the limit, without costly large matrices.
    {
      op: 'matmul',
      shapes: [
        [65536, 1, 1],
        [65536, 1, 1],
      ],
    },
    { op: 'optimizer_sgd', shapes: [[length], [length]], attrs: { learningRate: 0.003 } },
  ];
  for (const f of fixtures) {
    const attrs = f.attrs ?? {};
    const shape = f.op === 'optimizer_sgd' ? f.shapes[0] : infer(f.op, f.shapes, attrs);
    async function execute(backend: Backend) {
      const inputs = f.shapes.map((s, input) =>
        tensor(
          Float32Array.from(
            { length: size(s) },
            (_, i) => (((i * 17 + input * 7) % 101) - 50) / 100,
          ),
          s,
          backend,
        ),
      );
      const output = tensor(new Float32Array(size(shape)), shape, backend);
      try {
        backend.submit([{ op: f.op, inputs, output, attrs }]);
        return await output.read();
      } finally {
        [...inputs, output].forEach((t) => t.dispose());
      }
    }
    const expected = await execute(cpu);
    const actual = await execute(gpu);
    close(actual, expected, `large dispatch ${f.op}`);
    await report('large-dispatch', {
      op: f.op,
      shapes: f.shapes,
      originalWorkgroups: ['softmax', 'layernorm', 'matmul'].includes(f.op) ? 65536 : 65537,
      originalOversizedAxis: f.op === 'matmul' ? 'z' : 'x',
    });
  }
}
function model(backend: Backend, config: GptConfig) {
  const { graph, registry, shapes } = gptFixture(config);
  const plan = compileGraph(graph, {
    registry,
    inputShapes: shapes,
    backend,
    random: new Random(123),
    preferNative: true,
  });
  const { B, C, V } = config;
  const batch = (step: number) => {
    const rng = new Random(456 + step);
    const ids = Array.from({ length: B * C }, () => Math.floor(rng.uniform() * V));
    const labels = new Float32Array(B * C * V);
    ids.forEach((id, i) => {
      labels[i * V + ((id + 1) % V)] = 1;
    });
    const targets = tensor(labels, [B, C, V], backend);
    return { inputs: { tokens: tensor(ids, [B, C], backend), targets }, targets };
  };
  const session = new TrainingSession(
    plan,
    batch,
    undefined,
    { kind: 'adam', learningRate: 0.003 },
    'logits',
    { lossOutput: 'loss', accuracy: false },
  );
  return {
    plan,
    session,
    batch,
    dispose() {
      session.dispose();
      plan.dispose();
    },
  };
}
async function compareGpt(cpu: CpuBackend, gpu: WebGpuBackend) {
  const config = { B: 2, C: 6, D: 12, H: 3, V: 27, layers: 2 };
  const a = model(cpu, config),
    b = model(gpu, config);
  try {
    for (let step = 0; step < 4; step++) {
      const ba = a.batch(step),
        bb = b.batch(step);
      const oa = a.plan.execute(ba.inputs),
        ob = b.plan.execute(bb.inputs);
      let ga: Awaited<ReturnType<typeof backward>> = [],
        gb: typeof ga = [];
      try {
        close(await ob.logits.read(), await oa.logits.read(), `GPT logits ${step}`, 1e-3, 1e-2);
        close(await ob.loss.read(), await oa.loss.read(), `GPT loss ${step}`, 1e-4, 1e-3);
        ga = await backward(oa.loss, [...a.plan.parameters.values()]);
        gb = await backward(ob.loss, [...b.plan.parameters.values()]);
        for (let i = 0; i < ga.length; i++)
          close(await gb[i].read(), await ga[i].read(), `GPT VJP ${step}/${i}`, 1e-3, 1e-2);
      } finally {
        [...ga, ...gb].forEach((t) => t.dispose());
        a.plan.releaseExecution(oa);
        b.plan.releaseExecution(ob);
        [ba, bb].forEach((batch) =>
          new Set(Object.values(batch.inputs)).forEach((t) => t.dispose()),
        );
      }
      await a.session.run(1);
      await b.session.run(1);
      for (const [name, parameter] of a.plan.parameters)
        close(
          await b.plan.parameters.get(name)!.read(),
          await parameter.read(),
          `GPT updated ${step}/${name}`,
          2e-3,
          1e-2,
        );
    }
    await report('gpt-comparison', {
      config,
      steps: 4,
      initializationSeed: 123,
      batchSeed: 456,
      data: 'seeded synthetic next-token task; not corpus convergence',
    });
  } finally {
    a.dispose();
    b.dispose();
  }
}
async function memory(gpu: WebGpuBackend) {
  const baseline = gpu.stats.liveBuffers;
  const m = model(gpu, { B: 2, C: 4, D: 8, H: 2, V: 9, layers: 1 });
  try {
    await m.session.run(20);
    await gpu.flush();
    const warm = { ...gpu.stats };
    for (let step = 20; step < 300; step += 20) {
      await m.session.run(20);
      await gpu.flush();
      const now = gpu.stats;
      assert(
        now.liveBuffers === warm.liveBuffers,
        `live buffer growth at ${step + 20}: ${JSON.stringify(now)} vs ${JSON.stringify(warm)}`,
      );
      assert(
        now.allocatedBytes <= warm.allocatedBytes * 1.1 + 65536,
        `allocated bytes growth at ${step + 20}`,
      );
      assert(now.pooledBuffers <= warm.pooledBuffers + 16, `pool growth at ${step + 20}`);
    }
    await report('memory', { steps: 300, warm, final: { ...gpu.stats } });
  } finally {
    m.dispose();
  }
  await gpu.flush();
  assert(gpu.stats.liveBuffers === baseline, 'GPT model/optimizer disposal leaked GPU buffers');
}
async function benchmark(cpu: CpuBackend | undefined, gpu: WebGpuBackend, config: GptConfig) {
  const results: Record<string, unknown> = {};
  for (const [name, backend] of [
    ['cpu', cpu],
    ['gpu', gpu],
  ] as const) {
    if (!backend) continue;
    const m = model(backend, config);
    try {
      await m.session.run(1);
      if (name === 'gpu') await gpu.flush();
      const times: number[] = [];
      for (let i = 0; i < 3; i++) {
        const start = performance.now();
        await m.session.run(1);
        if (name === 'gpu') await gpu.flush();
        times.push(performance.now() - start);
      }
      const meanMs = times.reduce((a, b) => a + b) / times.length;
      results[name] = {
        stepMs: times,
        meanMs,
        tokensPerSecond: (config.B * config.C * 1000) / meanMs,
      };
    } finally {
      m.dispose();
    }
  }
  await report('benchmark', {
    config,
    warmupSteps: 1,
    measuredSteps: 3,
    timing: 'end-to-end forward + backward + Adam + loss readback; GPU completion fenced',
    results,
  });
}
export async function run() {
  const cpu = new CpuBackend();
  const gpu = await WebGpuBackend.create(); // No adapter is a failure, never a skip.
  try {
    await report('adapter', { info: gpu.adapterInfo });
    await operations(cpu, gpu);
    await largeDispatches(cpu, gpu);
    await compareGpt(cpu, gpu);
    await memory(gpu);
    for (const config of [
      { B: 16, C: 64, D: 64, H: 4, V: 27, layers: 2 },
      { B: 16, C: 128, D: 128, H: 4, V: 27, layers: 4 },
    ])
      await benchmark(cpu, gpu, config);
    await benchmark(undefined, gpu, { B: 16, C: 256, D: 256, H: 8, V: 96, layers: 6 });
    await gpu.flush();
    assert(gpu.stats.liveBuffers === 0, 'GPU live buffers remain after tests');
    assert(cpu.liveBuffers === 0, 'CPU live buffers remain after tests');
    await report('done', { ok: true, stats: gpu.stats });
  } finally {
    gpu.dispose();
  }
}
