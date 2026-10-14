# Differentiable

A neural-network construction game. This first increment contains a CPU reference engine and tests; the React entry point is intentionally an empty placeholder. `demos/` is untouched reference material, not application source.

## Development

Use a current Node release supported by Vitest 5 (Node 22.12+, 24+, or 26+) and npm.

```sh
npm install
npm run typecheck
npm test
npm run format:check
npm run build
npm run dev
# Apply Prettier (100 columns, single quotes) to sources:
npm run format
```

Tests use worker threads, not subprocess forks. `npm test` is non-watch. The GPT feasibility test takes roughly 5 seconds on the development machine and enforces a 60-second budget.

## Engine boundaries

- `src/engine/backend.ts`: opaque storage handles, shape utilities, ordered command submission, asynchronous readback, explicit release. Only f32 is supported, including integer-valued indices.
- `cpu.ts`: reference kernels and VJPs. Custom kernels can register both forward and VJP handlers.
- `tensor.ts`: backend-independent first-order reverse autodiff. `backward(y, inputs)` computes gradients of `sum(y)`; `ge` and gather indices have zero gradients. Tied `max` chooses its first operand; tied `reduce_max` chooses the first element in row-major reduction order.
- `graph.ts`: versioned JSON graphs, named ports, recursive component expansion, concrete shape inference and reusable execution plans. Node `ui` data is opaque to the engine. Component and error identifiers are keys, not translated text.
- `expression.ts`: configurable expression-to-graph lowering; no JavaScript evaluation.
- `training.ts`, `levels.ts`: seeded evaluation, SGD/Adam, resumable training, structural scores and pure progress data. Training can use a named scalar graph loss and an independent seeded evaluation dataset. `level-suite.ts` combines forward, derivative and training checks into a mixed test list.
- `src/levels/`: example sigmoid, derivative and XOR levels, not final game content.

### Tensor ownership

Uploads and readbacks copy host arrays. Callers own tensor inputs and returned gradients. Dispose each gradient after use. For eager expressions, use `disposeGraph(outputs, retainedInputsAndParameters)` after backward to release the tape's tensors. A compiled graph owns its constants and initialized parameters. After backward, call `plan.releaseExecution(outputs)` with the exact object returned by `execute`; this also frees disconnected intermediates while retaining plan storage and caller inputs. Dispose the plan when finished. Custom implementations must clean private allocations that are neither returned nor reachable through returned tapes, particularly when throwing. Never release a forward tensor before its backward computation completes. Backend implementations must defer storage reuse until queued users finish.

`TrainingSession` borrows fixed training/evaluation batches. A batch factory receives the zero-based optimizer step and returns fresh standalone tensors (synchronously or asynchronously); the session owns and releases all distinct inputs/targets, even on failure. Without an explicit evaluation batch, evaluation invokes that factory too. Factory-created tensors must not alias model parameters or own additional tape allocations.

### Native components

Native matmul supports rank >= 2 and broadcast batch dimensions; softmax and layernorm operate on the last axis. Layernorm is normalization only; gain and bias are ordinary multiply/add nodes. Canonical primitive-only GraphJSON answers and native implementations live together in `standard-components.ts`. These three operations are registered components, not primitives; `preferNative: false` expands their answers. Shape-dependent answers receive concrete input shapes at compilation (for reshape attributes and normalization width). The default primitive vocabulary is `input output const param add mul exp log max ge sum reduce_max reshape transpose gather arange`. Native component selection is separate from game unlock policy: a level's available list must represent what the player has unlocked. Availability only checks nodes placed directly in the top-level player graph, including disconnected nodes; input/output are structural exemptions. An unlocked component can use internal primitives absent from the level palette. Subgraphs with parameters deliberately bypass native shortcuts until an explicit native parameter-binding ABI is introduced. `safeCompileGraph` returns a discriminated result with structured diagnostics (currently the first error); `compileGraph` is its throwing counterpart.

The GPT test builds a complete GraphJSON model with nested registered linear, affine-normalization, three-head causal-attention, MLP and Transformer-block components. Its two blocks have independently scoped parameters. `TrainingSession` trains with native kernels for 100 seeded Adam steps using fresh batches from a contiguous 2,548-character public-domain Alice in Wonderland excerpt. An 80/20 positional split keeps validation text out of training; held-out loss is averaged over deterministic batches covering the validation suffix (the final partial batch overlaps) and must beat both full-corpus and validation unigram entropy. The trained parameters are then shared with a `preferNative: false` plan to compare logits, loss and every parameter gradient. This is a small numerical/generalization feasibility test, not a quality language model.

## WebGPU and backend selection

`createBackend()` from `src/engine/backend-selection.ts` asynchronously returns `{ backend, kind, fallbackReason? }`. Call `setDefaultBackend(backend)` from `tensor.ts` once **before** creating a game's graphs or datasets. Default arguments then use this backend, compiled plans capture it, and existing tensors retain their original backend. Mixed-backend operations are rejected. Explicit backend arguments remain available for concurrent independent games/tests. Node defaults to CPU. Creation failures fall back to CPU with a reason; failures during GPU execution are not silently retried on CPU.

`WebGpuBackend.create()` requires a real adapter and never falls back. Shape-specialized WGSL covers the built-in forward/VJP operations; matmul and both adjoints use 16×16 shared-memory tiles. Gather adjoints assign one invocation to each table-gradient element and scan indices in input order (no floating-point atomics). Commands share encoders, submitted on read/flush or every 512 dispatches to bound encoder retention. Released buffers are reused only after queue completion, with a bounded per-size pool. `stats` exposes live/pool counts and allocated storage bytes. Call `dispose()` on the backend at game shutdown. GPU errors expose `BackendError.key` and `params`; invalid GPU gather indices are reported asynchronously at read/flush. Device loss poisons subsequent operations.

SGD/Adam updates and Adam moments execute on either backend without readback. Call `TrainingSession.dispose()` (or standalone `Optimizer.dispose()`) to release moments when finished; model and batch ownership is unchanged. `reportEvery` in session/run options defaults to 10; the final update also reports. Evaluation always synchronizes. Resumable training requires keeping the session alive.

```sh
npm run test:gpu
# Optional installed browser override:
CHROME_PATH=/path/to/chrome npm run test:gpu
```

The dependency-free runner starts its own Vite server and installed headless Chrome, uses an OS temporary profile, and cleans both up. Missing adapters, browser launch failures and timeouts fail the command rather than skipping. It reports adapter/browser information, forward/VJP comparisons, deterministic gather, GPT comparisons, memory checks and the requested small/medium CPU/GPU benchmarks. Default timeout is 30 minutes (`GPU_TEST_TIMEOUT_MS`). Primitive comparisons use `2e-4 + 2e-3*abs(reference)` to accommodate f32 reduction order and transcendental differences; GPT losses use `1e-4 + 1e-3*abs(reference)`, with wider gradient/parameter tolerances for accumulated error. These thresholds still require validation on actual adapters.

**Current validation limitation:** Chrome crashes in this development sandbox before adapter acquisition (`bootstrap_check_in … Permission denied (1100)`, SIGSEGV); disabling crash reporting did not help. GPU shaders, numerical parity, memory stability and performance have therefore not yet been validated on hardware. No speedup is claimed. The browser GPT fixture preserves the nested model architecture but uses seeded synthetic data rather than the Node test's Alice corpus.

## Current limits

No editor, localization runtime, storage persistence or production level content is included. Primitive broadcast matmul materializes a large intermediate and is intended for correctness checks, not production GPT throughput. GPU normalization uses deterministic 64-lane workgroup reductions; sum/reduce_max use the same parallel path for reduction fibers of at least 256 elements (sum VJP is elementwise, reduce_max VJP selects the first tied index). Each reduction uses 256 bytes of shared scratch; reduce_max VJP additionally declares 256 bytes of integer winner scratch. No floating-point atomics are used. Oversized logical dispatch grids are remapped automatically to device-sized physical grids. Adam bias corrections remain input tensor data, not step-specialized shader code. Gather backward still scans indices per table element, a performance risk for large user-defined models.

The engine supports first-order gradients only. No symbolic shapes or mixed precision. The given primitive set suffices for this GPT architecture; reciprocal via `exp(-log(x))` is intentionally restricted to positive inputs.
