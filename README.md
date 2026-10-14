# Differentiable

A neural-network construction game. Includes the tensor engines, a functional React editor, chapters 0–4, including the XOR challenge and character language models. `demos/` is untouched reference material, not application source.

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

## Editor shell

Run `npm run dev` for chapter selection, graph editing, probes, tests and resumable training. Eight scalar lessons, ten derivative lessons, eight tensor lessons and nine learning lessons unlock sequentially, followed by the XOR challenge (3-10, stable ID `xor`) and five sequence lessons (4-1–4-5). Chapter 2 introduces broadcasting at 2-2 and rank-relative transpose at 2-6; these practice lessons do not award components. Passing a component-building lesson saves the player's circuit as its reusable component; later circuits resolve nested components against the latest passed implementations. Chinese/English localization and versioned auto-saves are enabled. See [editor controls and architecture](src/ui/README.md) and [Worker protocol](src/worker/README.md). Browser interaction and visual acceptance still require manual review.

## Engine boundaries

- `src/engine/backend.ts`: opaque storage handles, shape utilities, ordered command submission, asynchronous readback, explicit release. Only f32 is supported, including integer-valued indices.
- `cpu.ts`: reference kernels and VJPs. Custom kernels can register both forward and VJP handlers.
- `tensor.ts`: backend-independent first-order reverse autodiff. `backward(y, inputs)` computes gradients of `sum(y)`; `ge` and gather indices have zero gradients. Tied `max` chooses its first operand; tied `reduce_max` chooses the first element in row-major reduction order.
- `graph.ts`: versioned JSON graphs, named ports, recursive component expansion, concrete shape inference and reusable execution plans. Node `ui` data is opaque to the engine. Component and error identifiers are keys, not translated text.
- `backward-graph.ts`: symbolic reverse graphs made of ordinary primitives, with node roles and serializable per-node value/gradient traces for future visualization. Composite answers are expanded before differentiation; no hidden VJP kernels are used in the generated graph.
- `expression.ts`: configurable expression-to-graph lowering; no JavaScript evaluation.
- `training.ts`, `levels.ts`: seeded evaluation, SGD/Adam, resumable training, structural scores and pure progress data. Training can use a named single-element graph loss and an independent seeded evaluation dataset. `level-suite.ts` combines forward, derivative and training checks into a mixed test list.
- `src/levels/`: chapter 0/1/2/3/4 level data, seeded checks, reference circuits and par; the catalog includes the XOR training preview. `examples.ts` remains a test fixture, not the game catalog.
- `src/levels/registry.ts`: shared pure `buildPlayerRegistry(levels, progress)` and `availableComponents(level, progress)` used by the editor and Worker. Native kernels remain native while component inspection shows the passed circuit.
- Save format v3 remaps pre-v3 chapter-2 IDs once across all level-keyed progress fields, then unions unlocks from passed lessons. Component names are unchanged. It separates `passedSolutions` (latest passing circuit per level) from `savedGraphs` (unfinished edits). Legacy v1 saves are migrated asynchronously: validate saved circuits against the new interface and checks, otherwise use verified reference circuits. Legacy milestones credit prerequisites needed by the new linear route. Language v1 saves remain readable.

### Gradient-descent lessons and saves

The route after 2-8 is 3-1 → 3-2 → 3-3 → 3-4 → 3-5 (cross entropy) →
3-6 (classification) → 3-7 (momentum) → 3-8 (Adam) → 3-9 (hidden layer) →
`xor` (displayed as 3-10). Cross entropy is a player subgraph with `p`/`t` ports;
it reduces classes, then averages rows, including at new batch/class sizes.
MSE is the player's saved subgraph, not a native kernel. Lesson 3-4 predicts
`w*x+b` from zero-initialized scalar parameters and uses the graph's `loss` output
(shape `[1]`); batches need no separate targets when `lossOutput` is set.
`TrainLevel.tunable.learningRate` declares min/max/default; evaluation and worker
training clamp an explicit rate to that range and otherwise use the default.

`tunable.optimizer?: ('sgd' | 'adam')[]` declares allowed optimizers; the first is
the default. Unlisted/invalid requests are ignored. Both optimizers share the declared
learning-rate range. Worker `evaluate`, `start`, and `trainStep` requests accept
`optimizer?`; client methods accept it after the optional learning-rate argument.
Classification step traces include accuracy and gradients of the external cross-entropy
loss with respect to node outputs, including `y`.

Save v3 optionally includes `trainingSettings: Record<levelId, { learningRate: number;
optimizer?: 'sgd' | 'adam' }>`.
Rates must be finite and positive; optimizer, if present, must be `sgd` or `adam`. Older saves remain valid; loading unions new
successors with existing unlocks, so completing 2-8 unlocks 3-1 without removing
an already-unlocked XOR. UI code should persist its chosen rate in this field and
pass it to worker training requests; no learning-rate UI is added in this step.

### Sequence lessons (chapter 4)

The route continues `xor` → 4-1 → 4-2 → 4-3 → 4-4 → 4-5. Loading an old
save that passed XOR unions in the new 4-1 unlock. Expanded primitive par is
**4, 8, 10, 5, 29**, respectively.

- 4-1 builds `one_hot(idx, k)`: class count is the length of `k = [0,…,V−1]`.
  Tests include `[4]` IDs / 5 classes and `[6]` IDs / 7 classes.
- 4-2 builds `embedding(idx, table)` with one-hot and matmul, using a constant
  five-row ID vector. This earned circuit is specialized to the lesson's five-row
  table; passing also unlocks general-purpose native `gather`. Native unlocks do
  not get replaced with the lesson subgraph.
- 4-4 builds `sample(p, u, triangle)`, where `triangle` is the upper triangular
  all-ones matrix. Broadcast multiplication followed by summation computes
  `p × triangle`; counting cumulative boundaries below `u` gives the sampled ID.
  Tests use 5 and 8 classes, with draws strictly inside probability bins.

`src/levels/corpus.ts` contains an original 1,611-character English nursery rhyme,
with a sorted 27-character vocabulary. Its exact empirical bigram conditional
entropy is **2.0315743966 nats** (uniform guessing: `log(27) = 3.295837`).
All initialization and sampling are seeded.

| Level | Data | Steps | Loss threshold | Adam LR min–max (default) | Measured loss |
| --- | --- | --- | --- | --- | --- |
| 4-3 | All 1,610 adjacent pairs; parameter seed 4103 | 180 | 2.181574 (H + .15) | .001–.1 (.03) | 2.059719 |
| 4-5 | 512 contexts sampled with seed 4505; parameter seed 4105 | 160 | 1.639788 | .001–.1 (.03) | .738885 |

Both train and evaluate on their declared empirical distribution; these are not
held-out generalization claims. No accuracy threshold is imposed. The 4-5 model
uses embedding width 8 and a tanh hidden layer of width 24. On its exact 512-example
distribution, the optimal one-character loss is **1.789788**, while two-character
conditional entropy is **.727661**. Its threshold is .15 below that one-character
optimum, requiring additional context. Full reference training takes about 2 seconds
for 4-3 and 3–4 seconds for 4-5 on the development CPU in isolated tests; parallel
suite execution may take longer. Tests run the complete step budgets, not shortened
surrogates. Training rates are tunable; Adam is the default, with SGD also allowed.

Both models support the worker `generate` API documented in
[src/worker/README.md](src/worker/README.md). No generation panel or other UI is added.

### Classification lesson calibration (CPU, seeded reference circuits)

| Level | Data / seed | Steps | Threshold | LR min–max (default) | Optimizer default |
| --- | --- | --- | --- | --- | --- |
| 3-6 | 40 points, separated clusters / 3106 | 80 | loss < 0.15, accuracy 100% | 0.001–1 (0.001) | SGD |
| 3-8 | 40 points; label feature ×0.05, nuisance feature ×20 / 3108 | 600 | loss < 0.15, accuracy 100% | 0.001–0.1 (0.05) | SGD; Adam selectable |
| 3-9 | 64 points, radii near 0.6 and 2 / 3109 | 300 | loss < 0.15, accuracy ≥95% | 0.001–0.1 (0.05) | Adam; SGD selectable |

3-6: default loss 0.583232 / accuracy 100% fails; SGD 0.1 gives 0.021978 / 100%.
3-8: SGD rates 0.001, 0.005, 0.01, 0.05, 0.1 give losses 0.692122,
0.689022, 0.710369, 4.099624, 8.059049 (all 50%, all fail). Adam at the
default rate gives 0.119819 / 100%, passing. These are sampled rates, not an
exhaustive numerical search of the continuous interval. The small-feature weight
can move by at most 600 × 0.1 × 0.06 = 3.6 per class under SGD; its maximum
class logit separation is only 0.432, insufficient for the loss target even with
an optimal balanced nuisance coefficient. 3-9 uses a 16-unit sigmoid hidden layer
and reaches loss 0.003621 / 100%. New expanded pars (3-5…3-9): 12, 14, 4, 14, 25.

### Symbolic backward graphs

`buildBackwardGraph(graph, { registry, availableComponents?, inputShapes, output, wrt })`
returns `{ graph, roles }` without executing tensors. `output` names a forward output;
`wrt` names inputs. The resulting ordinary GraphJSON keeps forward outputs and adds
`grad_<name>` outputs, seeded with ones for non-scalar outputs. Roles identify forward,
backward, seed and explicit fan-out accumulation nodes; backward rule labels are i18n keys.
Composite internals use readable `instance/internal` paths (numeric suffixes resolve ID
collisions), with instance paths recorded in `origin`.

`await traceBackward(graph, { ...options, inputs })` returns `forward` and `gradient`
maps keyed by forward node ID, each value `{ shape, values }`. It probes the generated
graph in a fresh CPU execution with deterministic parameter initialization, then releases
all tensors. It does not observe live training parameters. Comparisons stop gradients;
max/reduce_max ties match the CPU rules. Unsupported primitives such as gather/arange
report `backward_unsupported`. This is a data API only; gradient views and animations
are not implemented here.

### Tensor chapter checks

Forward levels may declare `shapeVariants: { inputs: ShapePorts, fixed: LevelCase[], generated?: LevelCases['generated'] }[]`. Each group compiles independently and uses the same reference function; the primary shape must match the declared interface, while variants must match the reference's output shape. Variant test IDs are prefixed with `shape-N/`; failures carry `levels.shape_variant` and input shapes in `errorParams.shapes`. Chapter 2 uses these checks to reject fixed-width solutions before saving native-backed player components. Tensor inputs in the live panel are read-only fixtures with previous/next case controls; scalar inputs remain editable.

### Tensor ownership

Uploads and readbacks copy host arrays. Callers own tensor inputs and returned gradients. Dispose each gradient after use. For eager expressions, use `disposeGraph(outputs, retainedInputsAndParameters)` after backward to release the tape's tensors. A compiled graph owns its constants and initialized parameters. After backward, call `plan.releaseExecution(outputs)` with the exact object returned by `execute`; this also frees disconnected intermediates while retaining plan storage and caller inputs. Dispose the plan when finished. Custom implementations must clean private allocations that are neither returned nor reachable through returned tapes, particularly when throwing. Never release a forward tensor before its backward computation completes. Backend implementations must defer storage reuse until queued users finish.

`TrainingSession` borrows fixed training/evaluation batches. A batch factory receives the zero-based optimizer step and returns fresh standalone tensors (synchronously or asynchronously); the session owns and releases all distinct inputs/targets, even on failure. Without an explicit evaluation batch, evaluation invokes that factory too. Factory-created tensors must not alias model parameters or own additional tape allocations.

### Native components

Native matmul supports rank >= 2 and broadcast batch dimensions; softmax and layernorm operate on the last axis. Layernorm is normalization only; gain and bias are ordinary multiply/add nodes. Canonical primitive-only GraphJSON answers and native implementations live together in `standard-components.ts`. These three operations are registered components, not primitives; `preferNative: false` expands their answers. The matmul, softmax and layernorm answers are shape-independent: `unsqueeze` inserts a singleton dimension, and normalization counts the last-axis elements using a sum of ones. `unsqueeze` takes integer `axis` (negative axes count from the output rank), uses reshape and its VJP, and reports `invalid_axes` for out-of-range axes. `transpose` accepts an `axes` pair of distinct integer axes (negative indices are relative to input rank), swaps those dimensions, and reports `invalid_axes` for invalid pairs. The pair takes priority over legacy `perm`; without either attribute, transpose still reverses all dimensions. Its axis-pair VJP swaps the same axes. New UI transpose nodes default to `axes: [-2, -1]`, while legacy permutations remain editable. The default primitive vocabulary is `input output const param add mul exp log max ge sum reduce_max unsqueeze reshape transpose gather arange`. Native component selection is separate from game unlock policy: a level's available list must represent what the player has unlocked. Availability only checks nodes placed directly in the top-level player graph, including disconnected nodes; input/output are structural exemptions. An unlocked component can use internal primitives absent from the level palette. Subgraphs with parameters deliberately bypass native shortcuts until an explicit native parameter-binding ABI is introduced. `safeCompileGraph` returns a discriminated result with structured diagnostics (currently the first error); `compileGraph` is its throwing counterpart.

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

The functional editor, localization runtime, versioned local storage, chapters 0–4 are included. Chapter 3 teaches scalar updates, vector gradients, player-built MSE and cross entropy, regression, classification, momentum, Adam and hidden-layer networks. The learning-rate control and single-step animation UI are not included yet; the engine/worker APIs are ready for them. Chapter 4 adds one-hot encoding, embedding, bigram training, sampling and a two-character model. Text generation is available through the worker API; its UI is not included. Chapters 5 onward remain upcoming. Primitive broadcast matmul materializes a large intermediate and is intended for correctness checks, not production GPT throughput. GPU normalization uses deterministic 64-lane workgroup reductions; sum/reduce_max use the same parallel path for reduction fibers of at least 256 elements (sum VJP is elementwise, reduce_max VJP selects the first tied index). Each reduction uses 256 bytes of shared scratch; reduce_max VJP additionally declares 256 bytes of integer winner scratch. No floating-point atomics are used. Oversized logical dispatch grids are remapped automatically to device-sized physical grids. Adam bias corrections remain input tensor data, not step-specialized shader code. Gather backward still scans indices per table element, a performance risk for large user-defined models.

The engine supports first-order gradients only. No symbolic shapes or mixed precision. The given primitive set suffices for this GPT architecture; reciprocal via `exp(-log(x))` is intentionally restricted to positive inputs.
