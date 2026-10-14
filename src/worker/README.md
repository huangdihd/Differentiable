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
const result = await client.evaluate(level.id, graph);
const unsubscribe = client.subscribe((event) => {
  if (event.type === 'progress') console.log(event.report);
  if (event.type === 'result') console.log(event.result);
});
await client.start('xor', trainingGraph); // acknowledgement, NOT final result
await client.pause();
await client.resume();
await client.stop();
unsubscribe();
client.dispose();
```

Use formal `gameLevels` IDs from `levels/catalog` and serializable `GraphJSON`.
Level callbacks and the player component registry stay inside the worker. Forward/derivative evaluation uses `evaluate`; training uses `start`.
`compile` returns output shapes, graph metrics, and structured diagnostics.
`probe` returns shape/values from a fresh seeded execution, not live training
parameters. It adds a collision-free output to a copied graph and releases the
execution, inputs, and plan after readback.

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

Training preserves parameters and Adam moments across pause/resume. Control
requests are serialized between optimizer steps; paused sessions allocate no new
work. Each step emits an evaluated progress report. Completion emits a final
`LevelResult`; stop also emits a final result but never awards a pass. Starting a
new successfully compiled run replaces and disposes the previous session. A failed
stop evaluation still releases the session and rejects the stop request. Failed
training emits a nonpassing final result with structured error data. A graph revision change
silently disposes obsolete training rather than emitting an obsolete result.

`WorkerRuntime.handle` is the Node-testable protocol handler, with optional level, progress, and registry
injection for tests (registry overrides are replaced by progress synchronization). One runtime owns the engine default backend per worker/global
context; do not run independent runtimes concurrently in one JS global. The
protocol types are exported from `protocol.ts`; they contain no tensors/functions.
