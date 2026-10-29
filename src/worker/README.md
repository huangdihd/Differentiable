# Graph worker API

## Isolated character experiments

Displayed lesson 5-1 owns one instance of `fixed-window-worker.ts`, separate from
the graph Worker below. Its initial request is
`{ type: 'train', model: 'fixed-window' }`, without player progress; the UI also sends
the main game's selected `backend` (`cpu` or `webgpu`) when it is known. This retains
the independent 4-6 reference experiment: the original 512-example dataset, seed
4105 and Adam .03, with 160 updates followed by continuation to 320. No RNN is
trained in 5-1. After revealing the identical `th` inputs, the UI's “Build RNN”
action calls `acknowledgeObservation` and enters `time-rnn`.

After passing `time-rnn`, Next opens temporary `PlayerRnnComparison` with one
Worker per model. Each first request supplies
`{ type: 'train', model: 'fixed-window' | 'rnn', backend, playerProgress }`. The fixed-window
experiment compiles `passedSolutions['4-5']`; the RNN compiles and executes the
actual cell in `passedSolutions['time-rnn']`. Both use `compileGraph` and the player
registry from `buildPlayerRegistry`, including player component dependencies.
Compilation disables native shortcuts so saved nested implementations are executed
rather than substituted. Extra parameters inside a player RNN cell are shared
across time and batch shapes and included in the same optimizer under `cell/` keys.
A missing saved top-level circuit fails explicitly. Nested components follow the
same player registry rules as the editor: saved implementations are retained, and
already-earned components remain usable if their old solution was revoked. The
experiment does not require extra dependency saves beyond the original level.
The player built the RNN cell, not the whole character model: embedding
width 8, hidden width 24, the 27-character output head and fresh initial weights
belong to the experiment fixture.

`fixed-window-batch.ts` validates the fixed-window graph at the original 512-row
shape, then adapts only verified row-preserving reshape signatures for the new
training and prediction batch sizes. This includes nested components and resolved
dimension ports. It does not rewrite saved graphs, parameter shapes, constants or
feature widths; unrecognized shape incompatibilities retain their engine diagnostic.

The dedicated protocol is `fixed-window-protocol.ts`; its initial request may
include `backend` and `playerProgress`. `backend` selects the same backend type as the
main game, while the comparison Worker still creates its own backend instance. `playerProgress`
is required for the RNN and for the post-build fixed-window comparison. Send it only once per Worker. Later requests omit it and
continue the same parameters and Adam state. Switching a live Worker's model or
resupplying progress to replace its graph, duplicate runs, and extra runs are
rejected. Post-build checkpoints are 320/640 updates, unlike 5-1's 160/320.
Replies are `progress` (step, totalSteps, loss), `result` (snapshot), or `error`
(structured key, parameters and optional node/port; no exception prose or stack).
The comparison UI displays this diagnostic and identifies the affected model; a
training failure alone does not mean the player's circuit is invalid.
Snapshots include model identity, measured parameter/sample counts,
seed, learning rate, measured loss and two full next-character distributions. The
fixed-window prediction readback accepts finite nonnegative output weights and
normalizes them like text generation; it does not reject a numerically equivalent
distribution merely because its floating-point sum is not exactly 1.
RNN snapshots also include `resetPredictions`: same weights, zero state,
last-two-character input only, without retraining.

For the post-build comparison, `character-comparison-data.ts` supplies all 1,495
eligible within-line target positions to both models, with seed 4105 and Adam .01.
Training and evaluation use these same positions: losses are training-corpus
fitting losses, **not** held-out measurements. Same task and target positions do
not mean equal parameters or compute. Parameter counts are computed from actual
models; the conventional reference counts of 1,299 (fixed window) and 1,683 (RNN
fixture) are not hard-coded promises for player circuits. The original 4-6 lesson
and 5-1 experiment keep their datasets and hyperparameters. RNN execution packs
active lines, shares their causal prefixes, excludes targets from inputs, and
normalizes summed log probabilities by the total valid target count. Measured
predictions do not guarantee that both examples become correct or that the RNN
has unlimited memory.

Each model owns a separate backend instance, fresh weights and independent Adam state.
The backend type follows the main game (`cpu` or `webgpu`); independence means that the
Worker creates its own backend instead of borrowing or transferring the main Worker's
backend. CPU and WebGPU selection failures remain explicit rather than silently changing
the requested comparison backend.
Experiments borrow circuit definitions, not saved training weights; they do not
mutate player graphs or formal training and never read or write player storage.
A failed experiment must be recreated; retry restarts only the affected model.
Leaving terminates the Workers and discards experiment weights. This protocol
does not award completion. Only 5-1 uses observation acknowledgement; the post-build
comparison adds no level and persists neither view mode nor weights. Back returns
to `time-rnn`; Continue after both models' runs opens `time-rnn-window`.

## Player graph Worker

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
const reveal = await client.backward(level.id, forwardGraph, revealInputs, {
  inputShapes: { w: [], x: [], b: [], '∂f/∂z': [] },
  output: 'f',
  wrt: ['w', 'x', 'b'],
});
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

A level may declare multiple required outputs in `interface.outputs`. Their object
insertion order is semantic: `derivativeInputs[i]` names the input whose derivative
is expected at output `Object.keys(interface.outputs)[i]`. The arrays must have the
same length, derivative input names must be unique level inputs, and
`interface.output`/`outputShape` must match an entry in `interface.outputs`. Evaluation
reads, shape-checks, and compares every named output in that order.

`backward(levelId, graph, inputs?, options?)` returns `{ backward, trace? }`. `backward`
contains the generated primitive graph and node `roles`, all serializable. By default, this
single-output view differentiates the derivative level's `derivativeInput` when declared,
otherwise all level inputs, and selects `level.interface.output`. An optional independent
contract `{ inputShapes, output, wrt }` overrides only those differentiation coordinates;
level component availability and the current player registry still apply. This lets 1-11's
completion reveal send an independent `z=w·x+b`, `f=σ(z)` forward graph rather than
differentiating the player's multi-output backward graph. The generated graph is therefore
a first-order backward pass through Sigmoid and Linear.

Supplying inputs requests a numeric trace; omitting them only builds the read-only graph.
Source graphs and player progress are not modified. Traces use CPU with fixed-seed parameter
initialization for non-training levels. Errors and revision filtering follow the usual RPC
rules.

Chapter-3 training requests ignore supplied inputs and always trace a seeded dataset.
They additionally return `graph`, `parameterState: 'initial' | 'current'`, and
`attachedLoss`. Targets are all top-level parameter IDs. `lossOutput` selects the
circuit loss; otherwise a primitive cross-entropy segment is appended to the view
only, including training's `1e-7` epsilon and mean over all non-class dimensions.
The output is labeled `loss`. Parameters come from the latest successful step or
continuous completion/stop snapshot for the same level and graph, falling back to
training's seeded initialization. Reset clears the snapshot. Numeric training
previews retain at most 16 values per node. Later training chapters are rejected
with `worker.training_gradient_unavailable` after the 6-4 CPU benchmark exceeded
the 3-second budget. Active continuous training rejects with `worker.training_busy`.

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

Continuous coordinate training also attaches optional `plot` data to progress events.
Snapshots use live post-update parameters and fixed full datasets, not seeded probes.
Readback is awaited within the same serialized runtime queue as optimizer/control work:
first snapshot immediately, subsequent snapshots at most once per 250 ms, plus a forced
final snapshot. Regression follows the MSE prediction input; classification uses the
level output. Token models and resampled datasets are excluded. Existing request-ID and
revision filtering drops obsolete plot events together with obsolete progress.

## Character generation

`generate` requests carry `{ levelId, graph, prompt, length, seed?, temperature? }`
and return `{ text }`. The client method is
`generate(levelId, graph, prompt, length, seed?, temperature?)`.
Only training lessons marked `generates: true` support this command (currently
4-3, 4-5 and 5-5); others return `worker.generation_unavailable`. A continuous training
session, including a paused one, rejects generation with `worker.training_busy`.

`length` is the number of **new** characters, an integer from 0 to 400.
Unknown prompt characters are dropped. Short contexts are internally left-padded
with spaces, but padding is not included in the returned text. The returned text
is the filtered prompt plus the requested new characters. Temperature defaults
to 1 and must be finite and positive; seed defaults to the lesson seed. Invalid
arguments return `worker.invalid_request`. Sampling uses the seedable PRNG and
normalized probability weights raised to `1 / temperature`.

Inference uses a batch of one: `[N]` inputs become `[1]`, and `[N, T]`
inputs become `[1, T]` using the lesson's declared context length. Outputs must
be `[1, V]` or, for sequence inputs, `[1, T, V]`, where `V` is the corpus
vocabulary size. Sequence generation samples only the last position and slides
the context after each new character; the chapter 4 single-output path is unchanged.

The latest successful continuous completion/manual stop or single-step update
wins: generation copies its parameters into an independent inference plan.
Snapshots apply only to the identical level/graph. Without a matching
snapshot, generation uses the lesson's seeded initial parameters. `evaluate`
does not provide generation parameters. Parameters are session-local, not saved
in browser storage. Revision changes, progress synchronization and `cancel`
preserve the snapshot so a just-completed model remains available after a pass;
`trainReset` and disposal clear it. Generation never updates the training optimizer
or parameters.

`WorkerRuntime.handle` is the Node-testable protocol handler, with optional level, progress, and registry
injection for tests (registry overrides are replaced by progress synchronization). One runtime owns the engine default backend per worker/global
context; do not run independent runtimes concurrently in one JS global. The
protocol types are exported from `protocol.ts`; they contain no tensors/functions.
