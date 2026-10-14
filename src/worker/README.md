# Graph worker API

Import `WorkerClient` from `src/worker`. Its default constructor creates the Vite
module worker; an injected `WorkerTransport` supports Node protocol tests.

```ts
const client = new WorkerClient();
const backend = await client.init('auto'); // 'auto' | 'cpu' | 'webgpu'
await client.syncProgress(progress, 1); // Progress includes passedSolutions
client.setRevision(2); // monotonically increasing; bump on graph or level edits
const compiled = await client.compile(level.id, graph);
const probe = await client.probe(level.id, graph, { node: 'neg', port: 'out' }, inputs);
const { backward, trace } = await client.backward(level.id, graph, inputs);
const result = await client.evaluate(level.id, graph);
const unsubscribe = client.subscribe((event) => {
  if (event.type === 'progress') console.log(event.report);
  if (event.type === 'result') console.log(event.result);
});
const step = await client.trainStep('xor', trainingGraph, learningRate); // one optimizer update
await client.trainReset(); // discard only the independent step session
await client.start('xor', trainingGraph, learningRate); // acknowledgement, NOT final result
await client.pause();
await client.resume();
await client.stop();
unsubscribe();
client.dispose();
```

Use formal `gameLevels` IDs from `levels/catalog` and serializable `GraphJSON`.
Level callbacks and the player component registry stay inside the worker. Forward/derivative evaluation uses `evaluate`; interactive continuous training uses `start`. `evaluate(levelId, graph, learningRate?, optimizer?)` also supports bounded training evaluation, returning a final result directly (without streaming progress).
`compile` returns output shapes, graph metrics, and structured diagnostics.
`probe` returns shape/values from a fresh seeded execution, not live training
parameters. It adds a collision-free output to a copied graph and releases the
execution, inputs, and plan after readback.

`backward(levelId, graph, inputs?)` returns `{ backward, trace? }`. `backward`
contains the generated primitive graph and node `roles`, all serializable.
The differentiation targets are the derivative level's `derivativeInput` when
declared, otherwise all level inputs. The selected output is
`level.interface.output`. Supplying inputs requests a numeric trace; omitting them
only builds the read-only graph. Registry/availability come from current progress;
source graphs and player progress are not modified. Traces use CPU with fixed-seed
parameter initialization, not parameters copied from live training. Errors and revision
filtering follow the usual RPC rules.

`syncProgress(progress: Progress, revision?: number)` returns
`Promise<{ state: 'synced' }>`. Call it after `init`, before loading/evaluating a
persisted graph, and whenever progress changes. The optional revision must be
strictly newer than the client's current revision; omitted means current + 1.
UI code maintaining its own revision counter should pass `++revision`.
Synchronization rejects pending old promises, filters old training events, and
cancels training before rebuilding the registry from `passedSolutions`. Subsequent
compile/probe/evaluate/start requests derive availability from the current progress.
The runtime starts with empty progress, not an example solution registry.

`init` selects the worker-local backend exactly once. Both automatic and explicit
WebGPU preference fall back to CPU if unavailable, returning the key
`worker.cpu_fallback`; backend exceptions are not exposed. `cpu` skips WebGPU.

Every request/reply/event carries a request ID and graph revision. Replies resolve
only the corresponding promise. Progress/final events use the originating start
request ID. Incrementing the client revision rejects pending promises and ignores
old events; it also sends a `cancel` request immediately so the runtime discards old
training even when no compile/evaluate follows. Cancellation is serialized after any
in-flight step, releases paused sessions too, and emits no final result. `dispose` terminates the worker immediately. Errors reject with
`WorkerClientError.detail` (`key`, `params`, optional node/port), never player prose.

`trainStep(levelId, graph, learningRate?, optimizer?)` maintains an independent seeded session and
performs exactly one optimizer update. Its reply contains `step`, the pre-update `loss`,
pre-update `forward` and `gradient` node maps, and `parameters`. Forward/gradient values
have `{ shape, values, size }`; `values` is limited to 256 elements and `size` records the
full tensor length. Parameters are `Record<canvasParamId, { before: number[], after: number[] }>`
with full arrays, not snapshots; each `after` equals the next step's `before`.
Consecutive calls retain parameters and optimizer state. Changing the effective learning
rate updates the existing optimizer options without resetting the step count or parameters.
Changing the effective optimizer kind, graph or level rebuilds this session; revision changes,
progress synchronization, cancellation, starting continuous training, and disposal
release it. `trainReset()` returns `{ state: 'reset' }`, even without a step session;
the next step starts from the seeded initialization. Step replies never award a pass.
Both step and reset reject with `worker.training_busy` while a continuous session exists,
including while paused. Training evaluation also rejects while that session exists.

The optional numeric `learningRate` on `start`, `evaluate` and `trainStep` follows the
level's declared tuning bounds: finite values are clamped, missing/nonfinite values
use the declared default, and non-tunable levels ignore the override. The source level
and caller graph are never modified. Optional `optimizer?: 'sgd' | 'adam'` is accepted
by these three requests (the fourth argument on client methods). Only values listed in
`tunable.optimizer` are accepted; missing or invalid values use its first entry, or the
fixed optimizer if the list is absent. Both optimizers share the learning-rate range.
Classification step replies include pre-update `accuracy`; gradients include the output
node and differentiate the engine cross-entropy loss, not merely the sum of predictions.

For tunable levels, `start` measures the initial loss before any updates. A nonfinite
training loss or a step loss greater than ten times the initial loss ends training
with `levels.train_diverged`. Exhausting the level's step budget (60 for `3-4`)
without meeting its loss threshold yields `levels.train_slow`; insufficient accuracy
yields `levels.train_accuracy` with `params.accuracy` (fraction from 0 to 1). Structured errors expose
`params.loss` for interpolation (Infinity when the engine rejects nonfinite computation
before returning a report). Slow classification does not apply to manual stop or
non-tunable levels. `evaluate` uses the engine's corresponding classification rules.

Training preserves parameters and Adam moments across pause/resume. Control
requests are serialized between optimizer steps; paused sessions allocate no new
work. Each step emits an evaluated progress report. Completion emits a final
`LevelResult`; stop also emits a final result but never awards a pass. Starting a
new successfully compiled run replaces and disposes the previous session. A failed
stop evaluation still releases the session and rejects the stop request. Failed
training emits a nonpassing final result with structured error data. A graph revision change
silently disposes obsolete training rather than emitting an obsolete result.

## Character generation

`generate` requests carry `{ levelId, graph, prompt, length, seed?, temperature? }`
and return `{ text }`. The client method is
`generate(levelId, graph, prompt, length, seed?, temperature?)`.
Only training lessons marked `generates: true` support this command (currently
4-3 and 4-5); others return `worker.generation_unavailable`. A continuous training
session, including a paused one, rejects generation with `worker.training_busy`.

`length` is the number of **new** characters, an integer from 0 to 400.
Unknown prompt characters are dropped. Short contexts are internally left-padded
with spaces, but padding is not included in the returned text. The returned text
is the filtered prompt plus the requested new characters. Temperature defaults
to 1 and must be finite and positive; seed defaults to the lesson seed. Invalid
arguments return `worker.invalid_request`. Sampling uses the seedable PRNG and
normalized probability weights raised to `1 / temperature`.

The latest successful continuous completion/manual stop or single-step update
wins: generation copies its parameters into an independent inference plan.
Snapshots apply only to the identical level/graph/revision. Without a matching
snapshot, generation uses the lesson's seeded initial parameters. `evaluate`
does not provide generation parameters. Parameters are session-local, not saved
in browser storage. Revision changes, progress synchronization, `cancel` and
`trainReset` clear the snapshot. Generation never updates the training optimizer
or parameters.

`WorkerRuntime.handle` is the Node-testable protocol handler, with optional level, progress, and registry
injection for tests (registry overrides are replaced by progress synchronization). One runtime owns the engine default backend per worker/global
context; do not run independent runtimes concurrently in one JS global. The
protocol types are exported from `protocol.ts`; they contain no tensors/functions.
