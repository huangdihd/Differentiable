# Differentiable

A neural-network construction game. Includes the tensor engines, a functional React editor, chapters 0–7, including the XOR challenge, time-series forecasting and character language models. `demos/` is untouched reference material, not application source.

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

Tests use worker threads, not subprocess forks. `npm test` is non-watch. The GPT calibration test takes roughly 67 seconds on the development machine (reference plus control), with a 360-second timeout.

## Demo deployment

The GitHub repository is private. `main` holds development history; the `demo` branch is the only
source branch that triggers `.github/workflows/deploy-demo.yml`. Each push to `demo` installs the
locked dependencies, runs `npm run build:demo` to build and obfuscate `dist/`, then uploads that
untracked output to the existing `differentiable` Cloudflare Pages production project. The workflow
requires repository Actions secrets `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`; the latter
must be scoped to Account → Cloudflare Pages → Edit. Neither the token nor generated `dist/` belongs
in Git.

To release an approved `main` commit without committing generated assets:

```sh
git push origin main:demo
```

## Editor shell

Run `npm run dev` for chapter selection, graph editing, probes, tests and resumable training. Eight scalar lessons, eleven derivative lessons, nine tensor lessons and twelve learning lessons unlock sequentially, followed by the XOR challenge (3-13, stable ID `xor`), six sequence lessons (4-1–4-6), seven time-series lessons (5-1–5-7), five Attention lessons (6-1–6-5), and four GPT lessons (7-1–7-4). Attention retains stable IDs `5-*` and GPT retains stable IDs `6-*` so existing saves remain compatible. Derivative lesson 1-10 builds a gradient by wiring the named `∂a` and `∂b` outputs. Lesson 1-11 uses the forward context `z=w·x+b`, `f=σ(z)`: the player receives `w`, `x`, and the upstream gradient `∂f/∂z`, then backpropagates only through the multiply-and-add calculation `z=w·x+b` to output `∂f/∂w`, `∂f/∂x`, and `∂f/∂b`. The value of `b` is not needed because `∂z/∂b=1`. Passing reveals an engine-generated first-order backward graph for the independent full `z=w·x+b → f=σ(z)` calculation and unlocks the existing autodiff and expression tools; the player's backward circuit is never differentiated. Chapter 2 follows with shapes and axes at 2-1 (stable ID `axes`), broadcasting at 2-3 and rank-relative transpose at 2-7; these practice lessons do not award components. Passing a component-building lesson saves the player's circuit as its reusable component; later circuits resolve nested components against the latest passed implementations. Lesson outputs with domain meaning use semantic interface names such as `probabilities`, `loss`, and `attentionScores`; player-facing labels are localized while graph keys remain stable identifiers. Chinese/English localization and versioned auto-saves are enabled. See [editor controls and architecture](src/ui/README.md) and [Worker protocol](src/worker/README.md).

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

### Gradient-descent lessons and saves

The route after 2-9 starts with 3-1 (squared-loss bowl, stable ID `loss-intro`) →
3-2 (one gradient-descent step, stable ID `3-1`) → 3-3 (`neuron`, one neuron) →
3-4 (vector gradient, stable ID `3-2`) → 3-5 (`layer`, a layer) → 3-6 (MSE, stable ID `3-3`) →
3-7 (regression, stable ID `3-4`) → 3-8 (cross entropy, stable ID `3-5`) →
3-9 (classification) → 3-10 (momentum, outputting both updated weight and updated momentum) → 3-11 (Adam) → 3-12 (hidden layer) →
`xor` (displayed as 3-13). In 3-1 the player drags prediction `p` and builds `p−t → square → loss`;
in 3-2 the forward context is the simplest predictor `p=w·x`, while the playable circuit consumes
the loss side's supplied `∂L/∂p`, carries it through the local multiply, and subtracts
`0.1·(∂L/∂p)·x` from `w`. Dragging `w` keeps the same hidden target and updates the supplied
upstream gradient, so prediction, loss, the optimum, uphill gradient, target descent step, and the
circuit's updated position remain consistent. Display 3-3 expands one neuron as
`p=w₁x₁+w₂x₂+b`, identifies the weighted sum with the dot product `w·x`, and connects it to
batched `Xw+b` and multi-neuron `XW+b`. Display 3-4 then reuses that model with bias fixed at zero:
it receives scalar `∂L/∂p` directly and broadcasts `(∂L/∂p)·x` to compute each weight's loss
gradient, without rebuilding the squared loss from `t`. Because `∂p/∂w=x`, this local VJP needs the feature vector but not the current weight values, so its interface omits `w`. Display codes and stable IDs remain unchanged; old 3-2/3-4 circuits are revalidated against the new upstream-gradient contracts, with their protected `t` input renamed to `∂L/∂p` and the now-unused vector-gradient `w` input removed so revoked drafts remain editable. The neuron circuit uses the player's dot + add for a point or batch (par 3).
Display 3-5 combines several neurons, names the resulting matrix multiply plus bias a Linear layer,
and awards the player's matmul + add circuit as `linear` with x/w/b ports (par 5).
Loading a save that had already unlocked stable ID `3-2` restores access to `neuron`; passing `3-2`
or reaching stable ID `3-3` also restores `layer`. In the new route, passing `neuron` unlocks `3-2`,
passing `3-2` unlocks `layer`, and passing `layer` unlocks `3-3`. These migration rules grant unlocks
only, never pass credit. Cross entropy is a player subgraph with `p`/`t` ports;
it reduces classes, then averages rows, including at new batch/class sizes. Its row-selectable
stage trace shows `p`, `ln(p)`, one-hot `t`, `t·ln(p)`, the per-row negative sum, and the final
batch mean.
MSE is the player's saved subgraph, not a native kernel. Its row-selectable stage trace shows
`p`, `t`, `p−t`, `(p−t)²`, and the last-axis Mean, explicitly joining display 3-1's squared loss to
display 2-4's Mean for vector and `[2,4]` fixtures. Lesson 3-7 (stable ID `3-4`) predicts
`w*x+b` and uses the graph's `loss` output (shape `[1]`); a freshly placed Linear defaults to
one input and one output feature in this lesson, so `[16]` is treated as 16 scalar samples and can
feed MSE directly. Old drafts whose automatic Linear directly connects `x` to MSE are migrated from
the former one- or two-feature defaults to `1→1`. The equivalent two-scalar-Parameter circuit remains valid; batches need no
separate targets when `lossOutput` is set.
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
successors with existing unlocks, so completing 2-9 (stable ID `2-8`) unlocks 3-1 without removing
an already-unlocked XOR. UI code should persist its chosen rate in this field and
pass it to worker training requests; no learning-rate UI is added in this step.

### Sequence lessons (chapter 4)

The route continues `xor` → 4-1 → 4-2 → 4-3 → 4-4 → 4-5 (`reshape`) →
4-6 (stable ID `4-5`). Loading an old save that passed XOR unions in the new
4-1 unlock. Expanded primitive par for 4-1–4-4 and 4-6 is
**4, 10, 6, 5, 22**, respectively.

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

- 4-5 (stable ID `reshape`) practices [N,2,D] → [N,2D] without changing
  row-major element order or element count, unlike swapping axes with transpose.
  Wire `size(x, axis=0)` to `reshape.d0` with shape `[1,-1]`, or
  `size(x, axis=2) * 2` to `d1` with shape `[-1,1]`. Shape `-1` infers a
  dimension; axis `-1` selects the last axis. N and D vary across checks.
  This practice lesson awards no component and prepares the two-character model.

`src/levels/corpus.ts` contains an original 1,611-character English nursery rhyme,
with a sorted 27-character vocabulary. Its exact empirical bigram conditional
entropy is **2.0315743966 nats** (uniform guessing: `log(27) = 3.295837`).
All initialization and sampling are seeded.

| Level | Data                                                     | Steps | Loss threshold     | Adam LR min–max (default) | Measured loss |
| ----- | -------------------------------------------------------- | ----- | ------------------ | ------------------------- | ------------- |
| 4-3   | All 1,610 adjacent pairs; parameter seed 4103            | 180   | 2.181574 (H + .15) | .001–.1 (.03)             | 2.059719      |
| 4-6   | 512 contexts sampled with seed 4505; parameter seed 4105 | 160   | 1.639788           | .001–.1 (.03)             | .738885       |

Both train and evaluate on their declared empirical distribution; these are not
held-out generalization claims. No accuracy threshold is imposed. The 4-6 model
uses embedding width 8 and a tanh hidden layer of width 24. On its exact 512-example
distribution, the optimal one-character loss is **1.789788**, while two-character
conditional entropy is **.727661**. Its threshold is .15 below that one-character
optimum, requiring additional context. Full reference training takes about 2 seconds
for 4-3 and 3–4 seconds for 4-6 on the development CPU in isolated tests; parallel
suite execution may take longer. Tests run the complete step budgets, not shortened
surrogates. Training rates are tunable; Adam is the default, with SGD also allowed.

Both models support the worker `generate` API documented in
[src/worker/README.md](src/worker/README.md). No generation panel or other UI is added.

### Time-series forecasting lessons (chapter 5)

The route continues 4-6 (stable ID `4-5`) → `time-window-limit` → `time-rnn` →
`time-rnn-window` → `time-rnn-delay` → `time-recall-bottleneck` →
`time-attention` → `time-forecast`, then enters Attention at stable ID `5-1`
(displayed 6-1). Existing saves that already contain stable Attention or GPT IDs
keep that access unchanged; loading a save that passed `4-5` adds
`time-window-limit` without deleting later unlocks. Save version remains **10**.
Loading also repairs current-version saves: it removes only this observation's old
`savedGraphs`, `bestScores` and `passedSolutions` entries while preserving its
completed/read status, later unlocks and other lessons' data.

The route is causal rather than a component parade. Chapter 4-6 predicts from a
fixed two-character context and shows that two characters beat one. Displayed 5-1
(`time-window-limit`, `kind: 'observation'`) is now a dedicated method-diagnosis
page, not a canvas or a cropping circuit. Two real corpus samples, `little moth` →
space and `tap a tune along th` → `e`, share the last two characters `th`. The same
model receives the same input and therefore produces the same probability
distribution; deterministic top-1 must choose the same character for both and can
get at most one of this pair right. This is not a global 50% accuracy claim.
Displayed 5-1 retains its independent fixed-window reference experiment: an isolated
Worker trains the original 4-6 reference on its 512-example dataset, with seed 4105
and Adam .03, for 160 updates and then continues the same parameters and Adam state
to 320. Its Worker creates an independent backend instance of the type selected by
the main game. After revealing the identical `th` inputs, “Build RNN” immediately
calls `acknowledgeObservation` and enters `time-rnn`; there is no prebuilt RNN training
in 5-1. The observation has no graph, score or saved solution. Random sampling cannot
recover history absent from the input.

After the player builds and passes `time-rnn`, Next opens the temporary
`PlayerRnnComparison` view. It executes the actual recurrent cell in
`passedSolutions['time-rnn']` through `compileGraph` and `buildPlayerRegistry`, and
retrains the actual fixed-window circuit in `passedSolutions['4-5']` with its player
component dependencies. The player built an RNN **cell**, not a complete character
model: the 8-wide embedding, 24-wide hidden state, 27-character output head and fresh
initial weights are experiment fixtures. A missing saved top-level circuit fails
explicitly; nested components follow the editor's existing player-registry rules,
including earned components whose old saved solution was revoked. Saved player
implementations are executed with native shortcuts disabled.

Both post-build models use all 1,495 shared within-line target positions with at least
two preceding characters, including line-ending newlines, seed 4105 and Adam .01.
Each trains to 320 updates, then continues its own weights and Adam state to 640.
The original 4-6 lesson and 5-1 experiment keep their dataset and hyperparameters.
Parameter counts come from the actual models; the conventional reference counts are
1,299 for the fixed window and 1,683 for the RNN fixture, not hard-coded promises for
player circuits. This is the same task and target positions, **not** equal parameter
or compute budgets. RNN state resets at each line; packed execution shares causal
prefixes and removes finished rows without padding. Its cross entropy weights every
valid target equally, not every line equally.

For each displayed sentence, fixed-window and RNN predictions are shown together
at the same checkpoint. A reset-to-`th` toggle runs the same RNN weights from zero
state on only the last two characters; it is an input ablation, not a third training
run. All probabilities, errors and losses are measured, never canned. Training can
regress on a displayed example even as aggregate loss decreases: no checkpoint is
promised to predict both examples correctly. This is **training-corpus fitting**,
not held-out generalization or a universal RNN advantage. Neither success on these
two examples nor recurrent state implies unlimited memory.

Each model owns an isolated Worker, a fresh backend instance, fresh weights and independent
Adam state. The backend type follows the main game: a CPU game creates an independent CPU
backend, and a WebGPU game creates an independent WebGPU backend. Workers and backend
instances are never shared across the game and comparison models. Explicit WebGPU
initialization failure is reported instead of silently changing the comparison to CPU.
The fixed-window prediction readback normalizes finite nonnegative output weights like text generation,
so f32 rounding does not turn a valid model into a false failure. The first
post-build request supplies `playerProgress` once; later requests continue
the same session without resending it, and model or graph switches are rejected.
Experiments do not mutate player graphs, formal training or saves, and do not read
or write player storage. Leaving terminates the Workers and discards weights;
failures offer a fresh restart of only the affected model, never fake results.
The post-build comparison adds no level and persists neither its view mode nor its
weights. Back returns to `time-rnn`; Continue after the comparison opens
`time-rnn-window`, without another observation acknowledgement.

`time-rnn` builds one recurrent step;
`time-rnn-window` then exposes a six-step static unroll whose cells share `wx`, `wh`
and `b`. The unlocked native `rnn_window` performs the same recurrence for dynamic
sequence lengths and emits the complete `[B,T,H]` state history. `time-rnn-delay`
trains that RNN on a six-step delayed-copy task and grades a separately seeded
batch. `time-recall-bottleneck` then gives two different histories with one
identical final state, making the final-state information bottleneck concrete.
`time-attention` adds an explicit `[B,H]` query over the retained history; no causal
mask is needed because every entry is already observed.

`time-forecast` composes `rnn_window → temporal_attention → linear → mse`. The
final RNN state supplies the query, attention can revisit any of the six hidden
states, and the Linear head predicts the next value of a noiseless sine fragment
with varied phase, frequency and amplitude. Adam trains 400 steps at default
learning rate .02; grading uses a separately seeded 16-row batch and requires MSE
below .02. The reference has 89 parameters and passes deterministic CPU
calibration.

The checked-in delayed-copy evidence is generated, not hand-authored:

```sh
npm run benchmark:temporal
```

Every cell uses the same CPU engine, Adam learning rate .02, 300 updates, batch 32,
held-out batch 128, hidden width 8 and formal seed 811. A sequence has `delay + 1`
values: the first, amplitude-marked value is sampled from `[0.5,1.5)` and is the
target; later distractors come from `[-0.2,0.2)`. The fixed-window model sees only
the final two values. The generated `src/levels/temporal-benchmark.json` currently
records:

| model                    | parameters | delay 2 | delay 6 | delay 16 | delay 32 |
| ------------------------ | ---------: | ------: | ------: | -------: | -------: |
| fixed two-value window   |          3 | .078288 | .076351 |  .081022 |  .076636 |
| RNN final state          |         89 | .000139 | .000633 |  .082844 |  .076160 |
| RNN + temporal attention |         97 | .000639 | .000129 |  .000795 |  .000365 |

These are reproducible outcomes for one fixed seed and budget, not confidence
intervals or a universal claim that one architecture always wins. The marked first
event also makes this a retrieval probe, not an arbitrary unmarked sequence task.

### Attention lessons (displayed chapter 6; stable IDs `5-*`)

The displayed route continues 5-7 → 6-1 → 6-2 → 6-3 → 6-4 → 6-5. The lessons
retain internal IDs `5-1` through `5-5`; old saves keep those IDs rather than being
rewritten. A new player reaches stable `5-1` by completing `time-forecast`.
Forward lessons award `attn_scores(q,k)`, `causal_mask(scores)`,
`attention(q,k,v)` and `embed_pos(idx,tokens,positions)`. Each checks two additional
shapes, including a batch axis; sequence lengths and vector widths vary. `size`
wires the sequence length into `arange`, and supplies the score scaling width.
5-2 compares scores with absolute tolerance 1e-4 and zero relative tolerance:
visible scores stay strict and masked f32 values must be -1e9, as the lesson states.

5-5 trains on 128 seeded corpus windows of length 8 (1024 next-character targets),
with vocabulary 27 and embedding width 12. Training and grading use the same fixed
empirical distribution, not a held-out generalization test. Adam runs 200 steps,
with learning rate .001–.1 (default .03), seed 5505 and loss threshold 1.75.
The CPU reference reaches approximately 1.63 while the
`embed_pos → linear → softmax` control remains near 1.85 and fails. Cross entropy already
supports `[N,T,V]`: it sums all entries and averages over N×T, without reshaping.

Sequence generation feeds the last T characters as `[1,T]`, left-padding with
spaces, then samples the last output position of `[1,T,V]`. Padding is not shown.
Seeded sampling and the existing one-/two-character model behavior are tested.
The existing `generates` panel includes 5-5 without UI changes. Generated prose
remains rudimentary. Symbolic gradient inspection still lacks `gather`/`arange`
support; ordinary tensor autodiff and training work without that inspector.

### GPT lessons (displayed chapter 7; calibrated word coverage)

The displayed route is 6-5 → 7-1 → 7-2 → 7-3 → 7-4. Its preserved internal IDs are
5-5 → 6-1 (`multi_head`, q/k/v) → 6-2 (`feed_forward`, x/w1/b1/w2/b2) →
6-3 (`transformer_block`) → 6-4. The block has x plus eight weight/bias ports:
wq, wk, wv, wo, w1, b1, w2, b2. It uses two pre-LN residual branches without
learnable normalization gains/biases. Existing Attention graduates with stable ID `5-5` still unlock `6-1` on load.

The GPT forward inputs are uniformly [B,T,D]: primary [2,4,8],
variants [3,5,12] and [1,3,4], with FFN widths 16, 20 and 7.
The multi-head circuit reshapes to [B,T,2,D/2], swaps axes 1 and 2,
applies attention with batch axes [B,2], swaps back and reshapes to [B,T,D].
B, T and D come from size(q,0), size(q,1) and size(q,-1); D/2 uses static
scalar multiplication by 1/2. No engine API was changed.

6-4 samples N=8 windows with replacement from all 1595 stride-1 starts,
T=16, on every update. The model has D=24, H=48 and two blocks; Adam runs
1500 steps with seed 6404 and learning rate 0.01 (tunable 0.001–0.05).
The loss threshold is 1.1, approximately 1.2 times reference evaluation loss.

Evaluation uses 400 fixed windows: index `floor(i * (1595 - 1) / 399)` for
`i=0..399`. This includes both endpoints, spreads starts across the corpus,
and the overlapping windows cover every next-character target. Evaluate initially,
every 250 updates and at completion. Standalone CPU reference training measured
59.51 seconds including 4.32 seconds of evaluation (7.26%); evaluation loss was
0.925595 (full-window loss in the earlier run: 0.919865). The faster of the two
specified candidates passes, so D=32/H=64 was not retried. Player descriptions
round this measurement to about 1 minute; GPU speed was not measured in this step.

Text quality uses three prompts (`little moth`, `penny wren`, `round the pond`),
temperature 0.2, sampling seed 6404 reset per prompt, and 120 new characters each.
Exclude prompts, split each continuation on nonletters, discard empty tokens,
and count corpus-vocabulary hits over all word occurrences in the three samples
(including truncated words). Reference coverage is 57/77 = 74.03% (required ≥70%).
The original 5-5 single-head/single-layer lesson, using its unchanged T=8, D=12,
N=128, seed 5505, 200 steps and lr=.03, reaches loss 1.632982 and coverage
8/79 = 10.13% (required <50%); its loss also exceeds the 6-4 threshold.
This is an original-lesson control, not an equal-budget architecture comparison.
An exploratory equal-budget T=16/1500-step control repeated “the” and reached
78.13% word coverage despite loss 1.984947: vocabulary coverage alone does not
exclude repetition or establish sentence quality. No extra quality criterion was added.

`TrainLevel.resample` uses a persistent `Random(seed + 2)` training stream;
fixed evaluation uses `Random(seed + 1)`. Other levels keep their existing path.
`evaluationEvery` selects the resampled evaluation interval (default 50), shared
by level evaluation and continuous Worker training. Separate evaluation graphs
borrow live parameters. Step traces still show sampled pre-update batch loss.
Tests cover default/custom cadence and the final partial interval, aligned window
targets, seeded initialization, reference/control loss and continuation coverage.
Worker generation tests retain their short parameter-retention smoke check.
Tests run files serially. Browser/GPU validation is not part of this step.

Step 12d's exact-complete-line standard is superseded by word coverage; no corpus,
kernels, backward pass or parameter initialization was changed for this calibration.

### Classification lesson calibration (CPU, seeded reference circuits; stable level IDs)

| Level | Data / seed                                                 | Steps | Threshold                  | LR min–max (default) | Optimizer default    |
| ----- | ----------------------------------------------------------- | ----- | -------------------------- | -------------------- | -------------------- |
| 3-6   | 40 points, separated clusters / 3106                        | 80    | loss < 0.15, accuracy 100% | 0.001–1 (0.001)      | SGD                  |
| 3-8   | 40 points; label feature ×0.05, nuisance feature ×20 / 3108 | 600   | loss < 0.15, accuracy 100% | 0.001–0.1 (0.05)     | SGD; Adam selectable |
| 3-9   | 64 points, radii near 0.6 and 2 / 3109                      | 300   | loss < 0.15, accuracy ≥95% | 0.001–0.1 (0.05)     | Adam; SGD selectable |

3-6: default loss 0.583232 / accuracy 100% fails; SGD 0.1 gives 0.021978 / 100%.
Display 3-11 (stable ID `3-8`) keeps the Adam introduction at gameplay level:
the player trains with SGD, resets to the seeded initial state, and compares Adam on
the ×0.05 signal / ×20 nuisance data. Optimizer internals are intentionally left out
because this level does not ask the player to build the update rule.
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

### Tensor chapter checks

Forward levels may declare `shapeVariants: { inputs: ShapePorts, fixed: LevelCase[], generated?: LevelCases['generated'] }[]`. Each group compiles independently and uses the same reference function; the primary shape must match the declared interface, while variants must match the reference's output shape. Variant test IDs are prefixed with `shape-N/`; failures carry `levels.shape_variant` and input shapes in `errorParams.shapes`. Chapter 2 uses these checks to reject fixed-width solutions before saving native-backed player components. Tensor inputs in the live panel are read-only fixtures with previous/next case controls; scalar inputs remain editable.

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

## Current limits

The functional editor, localization runtime, versioned local storage, and chapters 0–7 are included. Chapter 3 teaches scalar updates, vector gradients, player-built MSE and cross entropy, regression, classification, momentum, Adam and hidden-layer networks. Training levels include learning-rate controls, single-step forward/backward/update animation, loss and data plots, and player-facing dataset cards. Scalar function lessons plot their target and explicit asymptotes without bridging discontinuities; derivative lessons switch between the derivative curve and the original function with target/player tangents. Single-output multivariable derivative lessons visualize a labeled one-dimensional slice along the differentiated input while holding the other inputs fixed. The gradient lesson uses a rotatable 3D function surface with target (yellow) and player (cyan) gradient arrows in the input plane; matching arrows turn green. It introduces the gradient vector and fastest-ascent geometry only; loss, negative gradients and gradient descent wait until chapter 3. Separate modes drag the input point or rotate the view. Chapter 4 adds one-hot encoding, embedding, bigram training, sampling, reshape practice and a two-character model. Chapter 5 exposes a fixed-window failure, builds and unrolls an RNN, verifies short-delay training, demonstrates the final-state bottleneck, then adds temporal attention and next-value forecasting. Chapter 6 adds scaled scores, causal masks, attention, positional embeddings and a single-head language model. Text generation is available through the worker API and the existing `generates`-gated panel. Chapter 7 adds multi-head attention, a feed-forward network, residual blocks and a two-block GPT training level. Its CPU calibration passes loss and continuation vocabulary coverage; this does not guarantee complete nursery-rhyme sentences. Primitive broadcast matmul materializes a large intermediate and is intended for correctness checks, not production GPT throughput. GPU normalization uses deterministic 64-lane workgroup reductions; sum/reduce_max use the same parallel path for reduction fibers of at least 256 elements (sum VJP is elementwise, reduce_max VJP selects the first tied index). Each reduction uses 256 bytes of shared scratch; reduce_max VJP additionally declares 256 bytes of integer winner scratch. No floating-point atomics are used. Oversized logical dispatch grids are remapped automatically to device-sized physical grids. Adam bias corrections remain input tensor data, not step-specialized shader code. Gather backward still scans indices per table element, a performance risk for large user-defined models.

The engine supports first-order gradients only. No symbolic shapes or mixed precision. The given primitive set suffices for this GPT architecture.
