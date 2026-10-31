# Engine architecture

How the engine, the graph format and the two backends fit together. The API surface described
here is what `src/engine/` implements; the game layer on top of it is covered in
[`levels.md`](levels.md).

## Engine boundaries

- `src/engine/backend.ts`: opaque storage handles, shape utilities, ordered command submission, asynchronous readback, explicit release. Only f32 is supported, including integer-valued indices.
- `cpu.ts`: reference kernels and VJPs. Custom kernels can register both forward and VJP handlers.
- `tensor.ts`: backend-independent first-order reverse autodiff. `backward(y, inputs)` computes gradients of `sum(y)`; `ge` and gather indices have zero gradients. Tied `max` chooses its first operand; tied `reduce_max` chooses the first element in row-major reduction order.
- `graph.ts`: versioned JSON graphs, named ports, recursive component expansion, concrete shape inference and reusable execution plans. Nested diagnostics carry operand provenance through component boundaries, so an internal shape error names the outer instance ports (for example `x`/`w`) rather than leaking primitive formal names (`a`/`b`). Node `ui` data is opaque to the engine. Component and error identifiers are keys, not translated text. `size(x, axis=-1)` returns a scalar shape dimension without reading tensor storage. Scalar constants, `size`, and scalar primitive arithmetic (including expanded player components) are evaluated at compile time. Optional `reshape.d0`–`d3` and `arange.n` wires override their attributes; reshape rank still comes from the `shape` array. Wired dimensions must be static and within 1e-3 of an integer; reshape retains its existing `-1` inference rules. Dimension wires and `size` have no gradients. Plans are recompiled for each concrete input-shape variant.
- `backward-graph.ts`: symbolic reverse graphs made of ordinary primitives, with node roles and serializable per-node value/gradient traces for the gradient view. Composite answers are expanded before differentiation; no hidden VJP kernels are used in the generated graph.
- `expression.ts`: configurable expression-to-graph lowering; no JavaScript evaluation.
- `training.ts`, `levels.ts`: seeded evaluation, SGD/Adam, resumable training, structural scores and pure progress data. Training can use a named single-element graph loss and an independent seeded evaluation dataset. `level-suite.ts` combines forward, derivative and training checks into a mixed test list.
- `src/levels/`: chapter 0–7 level data, including `chapter-temporal.ts`, seeded checks, reference circuits and par (expanded primitive count; the player-built `sub`/`div` count as one operation each, like `add`/`mul`, and the expression box inserts them for `-` and `/` once unlocked); the catalog includes the XOR training preview. `examples.ts` remains a test fixture, not the game catalog.
- `src/levels/registry.ts`: shared pure `buildPlayerRegistry(levels, progress)` and `availableComponents(level, progress)` used by the editor and Worker. Native kernels remain native while component inspection shows the passed circuit.
- Save format v9 idempotently repairs generic single-output `y` names in both `savedGraphs` and `passedSolutions`, including saves written by early v8 builds. The migration preserves node IDs, wiring, attributes, parameters, and editor `ui` data; chapter 0's intentionally generic `y` is unchanged. Old momentum drafts gain the newly required, initially disconnected `updatedMomentum` output; old single-output completion is revalidated and revoked while the editable circuit is retained. Regression drafts using the old automatic Linear defaults directly between `x` and MSE are narrowly updated to `1→1`; submitted passing solutions and unrelated Linear nodes remain untouched. It also unlocks the inserted `loss-intro` lesson for players who had already reached later chapter-3 content, without granting completion or removing later access. The reset introduced in v7 replaces 1-11 circuits from v2–v6 envelopes with the local backward step for `z=w·x+b`: incompatible drafts, solutions, scores, and settings are removed and completion is revoked, while previous graduates retain access to `axes` and later lessons. The v5 `1-12`→`1-11` merge, chapter-1 reorder, and pre-v3 chapter-2 ID migrations also remain in sequence. Component names are unchanged. Legacy v1 saves are migrated asynchronously: compatible saved circuits are validated against current interfaces and checks; unchanged contracts may use verified reference fallbacks, while changed contracts never receive automatic completion. Language envelopes from v1–v9 remain readable.

Progress loading fails closed: denied reads, malformed saves, unsupported versions, and migration exceptions show a localized error and disable autosaving. Only an absent store/key starts a new game. The app never presents or persists an empty fallback as successfully loaded progress; pending writes are gated while loading or after failure. Before an asynchronous migration returns or writes, it checks both the load's ownership and the original stored bytes; cancelled or stale loads cannot write their snapshot over newer progress. Failed autosaves retain the latest pending snapshot for explicit retry, show an unsaved-progress warning, and request confirmation before leaving while a write still fails.

### Signed arithmetic and save revalidation

Reciprocal is a native scalar primitive (`recip`, VJP `-g·y²`), so its former lesson 0-3 is
retired; later chapter-0 IDs keep their numbers and saves drop the retired entry on load. Division
tests positive and negative nonzero inputs; sqrt retains its positive-only cases. Levels marked
`revalidateOnLoad` recheck saved passing circuits with the current evaluator on async load.
Failed circuits lose completion and scores, but retain unlocks and earned components; an existing
draft is preserved, otherwise the revoked circuit becomes the draft. For the changed chapter-3 local-backprop
contracts, obsolete protected `t` inputs are renamed to `∂L/∂p` in both drafts and submitted solutions before
revalidation. Inputs removed from the new interface, such as the unused `w` in the vector-gradient lesson, are
removed with their incident wires so revoked work remains editable without pretending the old circuit is current.
Earned components without
passing solutions use their chapter reference circuit, while unearned presets keep their
self-contained built-ins. `loadProgressWithMigration(levels, source, onRevoked?)` optionally
reports the revoked level IDs to its third-argument callback; the home screen lists them with
links back to each level until dismissed.

Real-valued property inputs accept decimal fractions such as `1/3` and `[1/2, -1/4]`.
Nonfinite results (including division by zero) are invalid. Integer-only fields retain their
existing syntax; draft text keeps the player's original spelling.

### Parameter names

Parameter properties include an optional display-only name (`attrs.label`): whitespace is trimmed,
up to 16 Unicode code points are kept, and an empty name removes the attribute. Names can repeat;
`W_1` is displayed literally. Names appear on the canvas (including component inspection) and
in gradient labels; unnamed parameters keep the existing defaults. Shapes remain visible.
Edits use the normal undo/redo history and are saved with the graph; old saves need no migration.
Names do not change parameter IDs, computation, scores, or training-plan/snapshot cache keys.

### Configurable learned modules

Earned learned components (`Linear`, `Embedding`, `embed_pos`, `feed_forward`, and
`transformer_block`) retain the player-built subgraphs that define their computation. Fresh
instances default to an `nn.Module`-like declarative configuration whose dimensions and options
automatically generate instance-scoped trainable parameters. Legacy graphs without automatic
parameters retain their explicit parameter input wiring; automatic instances reject those wires to
avoid hidden overrides. `Linear` with `bias=false` generates no trainable bias. A one-input-feature
Linear also accepts rank-one `[N]` data as a batch of N scalar samples; one output feature is
flattened back to `[N]`, while ordinary vector and matrix inputs retain standard matmul semantics.
A `GraphPlan` uses its seeded `Random` to initialize generated parameters, and structural
metrics include those parameters in the parameter count. Compiler-only provenance on generated
parameters preserves each learned component's public formal input name for nested shape diagnostics.

### Training gradient view

After autodiff is unlocked, chapter-3 training circuits can open the gradient view
while idle. It differentiates `loss` with respect to explicit parameter node IDs and generated
module parameter paths, not data inputs. `lossOutput` selects an existing circuit loss; otherwise the view appends
primitive `-sum(t * log(y + 1e-7)) / rows`, matching training cross entropy.
These engine-added nodes are labeled separately and never saved into the player circuit.
Traces use `dataset(new Random(level.seed))`, with seed-initialized parameters or
the latest successful training step/completion/stop snapshot for the same circuit.
The description identifies initial versus current parameters. Training previews
are bounded to 16 values per node (the UI shows three).

Later training chapters are disabled in both toolbar and worker: the 6-4 CPU
benchmark exceeded the 3-second budget and its test worker exited after about
59 seconds. The skipped manual benchmark lives in `training-backward.test.ts`.
The helper supports arbitrary-rank cross entropy and seeded evaluation batches
for resampled levels, but those larger levels are not exposed yet.

### Symbolic backward graphs

`buildBackwardGraph(graph, { registry, availableComponents?, inputShapes, output, wrt })`
returns `{ graph, roles }` without executing tensors. `output` names one forward output;
`wrt` names inputs. The existing backward view follows this single-output contract and is not
shown on the multi-output gradient lesson. The resulting ordinary GraphJSON keeps forward outputs and adds
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

### Tensor ownership

Uploads and readbacks copy host arrays. Callers own tensor inputs and returned gradients. Dispose each gradient after use. For eager expressions, use `disposeGraph(outputs, retainedInputsAndParameters)` after backward to release the tape's tensors. A compiled graph owns its constants and initialized parameters. After backward, call `plan.releaseExecution(outputs)` with the exact object returned by `execute`; this also frees disconnected intermediates while retaining plan storage and caller inputs. Dispose the plan when finished. Custom implementations must clean private allocations that are neither returned nor reachable through returned tapes, particularly when throwing. Never release a forward tensor before its backward computation completes. Backend implementations must defer storage reuse until queued users finish.

`TrainingSession` borrows fixed training/evaluation batches. A batch factory receives the zero-based optimizer step and returns fresh standalone tensors (synchronously or asynchronously); the session owns and releases all distinct inputs/targets, even on failure. Without an explicit evaluation batch, evaluation invokes that factory too. Factory-created tensors must not alias model parameters or own additional tape allocations.

### Native components

Native matmul supports rank >= 2 and broadcast batch dimensions; softmax and layernorm operate on the last axis. Layernorm is normalization only; gain and bias are ordinary multiply/add nodes. Canonical primitive-only GraphJSON answers and native implementations live together in `standard-components.ts`. These three operations are registered components, not primitives; `preferNative: false` expands their answers. The matmul, softmax and layernorm answers are shape-independent: `unsqueeze` inserts a singleton dimension, and normalization counts the last-axis elements using a sum of ones. `unsqueeze` takes integer `axis` (negative axes count from the output rank), uses reshape and its VJP, and reports `invalid_axes` for out-of-range axes. `transpose` accepts an `axes` pair of distinct integer axes (negative indices are relative to input rank), swaps those dimensions, and reports `invalid_axes` for invalid pairs. The pair takes priority over legacy `perm`; without either attribute, transpose still reverses all dimensions. Its axis-pair VJP swaps the same axes. New UI transpose nodes default to `axes: [-2, -1]`, while legacy permutations remain editable. The default primitive vocabulary is `input output const param add mul exp log recip max ge sum reduce_max unsqueeze reshape transpose gather arange`. Native component selection is separate from game unlock policy: a level's available list must represent what the player has unlocked. Availability only checks nodes placed directly in the top-level player graph, including disconnected nodes; input/output are structural exemptions. An unlocked component can use internal primitives absent from the level palette. Subgraphs with parameters deliberately bypass native shortcuts until an explicit native parameter-binding ABI is introduced. `safeCompileGraph` returns a discriminated result with structured diagnostics (currently the first error); `compileGraph` is its throwing counterpart.

The GPT test builds a complete GraphJSON model with nested registered linear, affine-normalization, three-head causal-attention, MLP and Transformer-block components. Its two blocks have independently scoped parameters. `TrainingSession` trains with native kernels for 100 seeded Adam steps using fresh batches from a contiguous 2,548-character public-domain Alice in Wonderland excerpt. An 80/20 positional split keeps validation text out of training; held-out loss is averaged over deterministic batches covering the validation suffix (the final partial batch overlaps) and must beat both full-corpus and validation unigram entropy. The trained parameters are then shared with a `preferNative: false` plan to compare logits, loss and every parameter gradient. This is a small numerical/generalization feasibility test, not a quality language model.

## WebGPU and backend selection

`createBackend(preference)` from `src/engine/backend-selection.ts` asynchronously returns `{ backend, kind, fallbackReason? }`. Call `setDefaultBackend(backend)` from `tensor.ts` once **before** creating a game's graphs or datasets. Default arguments then use this backend, compiled plans capture it, and existing tensors retain their original backend. Mixed-backend operations are rejected. Explicit backend arguments remain available for concurrent independent games/tests. `auto` falls back to CPU with a reason; explicit `webgpu` fails instead of silently switching to CPU. Failures during GPU execution are not silently retried on CPU.

`WebGpuBackend.create()` requires a real adapter and never falls back. Shape-specialized WGSL covers the built-in forward/VJP operations; matmul and both adjoints use 16×16 shared-memory tiles. Gather adjoints assign one invocation to each table-gradient element and scan indices in input order (no floating-point atomics). Commands share encoders, submitted on read/flush or every 512 dispatches to bound encoder retention. Released buffers are reused only after queue completion, with a 64 MiB total free-buffer budget and least-recently-used eviction across sizes. Oversize free buffers are destroyed rather than cached; live/in-flight storage is not subject to this cache budget. `stats` exposes live/pool counts, pooled bytes and allocated storage bytes. Call `dispose()` on the backend at game shutdown. GPU errors expose `BackendError.key` and `params`; invalid GPU gather indices are reported asynchronously at read/flush. Device loss poisons subsequent operations.

SGD/Adam updates and Adam moments execute on either backend without readback. Call `TrainingSession.dispose()` (or standalone `Optimizer.dispose()`) to release moments when finished; model and batch ownership is unchanged. `reportEvery` in session/run options defaults to 10; the final update also reports. Evaluation always synchronizes. Resumable training requires keeping the session alive.

```sh
npm run test:gpu
# Optional installed browser override:
CHROME_PATH=/path/to/chrome npm run test:gpu
```

The dependency-free runner starts its own Vite server and installed headless Chrome, uses an OS temporary profile, and cleans both up. Missing adapters, browser launch failures and timeouts fail the command rather than skipping. It reports adapter/browser information, forward/VJP comparisons, deterministic gather, GPT comparisons, memory checks and the requested small/medium CPU/GPU benchmarks. Default timeout is 30 minutes (`GPU_TEST_TIMEOUT_MS`). Primitive comparisons use `2e-4 + 2e-3*abs(reference)` to accommodate f32 reduction order and transcendental differences; GPT losses use `1e-4 + 1e-3*abs(reference)`, with wider gradient/parameter tolerances for accumulated error. These thresholds still require validation on actual adapters.

**Current validation limitation:** Chrome crashes in this development sandbox before adapter acquisition (`bootstrap_check_in … Permission denied (1100)`, SIGSEGV); disabling crash reporting did not help. GPU shaders, numerical parity, memory stability and performance have therefore not yet been validated on hardware. No speedup is claimed. The browser GPT fixture preserves the nested model architecture but uses seeded synthetic data rather than the Node test's Alice corpus.
