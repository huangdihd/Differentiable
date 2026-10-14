# Differentiable

A neural-network construction game. This first increment contains a CPU reference engine and tests; the React entry point is intentionally an empty placeholder. `demos/` is untouched reference material, not application source.

## Development

Use a current Node release supported by Vitest 5 (Node 22.12+, 24+, or 26+) and npm.

```sh
npm install
npm run typecheck
npm test
npm run build
npm run dev
```

Tests use worker threads, not subprocess forks. `npm test` is non-watch. The GPT feasibility test takes roughly 20 seconds on the development machine and enforces a 60-second budget.

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

### Native components

Native matmul supports rank >= 2 and broadcast batch dimensions; softmax and layernorm operate on the last axis. Layernorm is normalization only; gain and bias are ordinary multiply/add nodes. Equivalent primitive compositions live in `composites.ts`. Native component selection is separate from game unlock policy: a level's available list must represent what the player has unlocked. Availability is checked recursively, including underlying primitives; input/output are structural exemptions. Subgraphs with parameters deliberately bypass native shortcuts until an explicit native parameter-binding ABI is introduced. `safeCompileGraph` returns a discriminated result with structured diagnostics (currently the first error); `compileGraph` is its throwing counterpart.

The GPT test constructs token/position embeddings, four-head causal attention, residual MLP blocks, two Transformer layers and cross entropy using primitive tensor operations. It compares logits, loss and every parameter gradient against native matmul/softmax/layernorm, then trains the primitive version for 16 seeded Adam steps. It is a numerical sufficiency test, not a serialized editor/save-game GPT fixture or a quality language model.

## Future GPU work / current limits

No WebGPU implementation, editor, localization runtime, storage persistence or production level content is included. Backend `submit` accepts command batches; eager execution currently sends individual commands, so a GPU implementation should queue them and fuse/flush at synchronization boundaries. Allocation/release must be fence-aware. CPU optimizer updates currently read parameters and gradients back; GPU-resident optimizer kernels and moments will be needed for practical accelerator training. Error reporting for asynchronous shader faults, device loss, dispatch limits and f32 reduction tolerances still need GPU-specific treatment. Primitive broadcast matmul materializes a large intermediate and is intended for correctness checks, not production GPT throughput.

The engine supports first-order gradients only. No symbolic shapes or mixed precision. The given primitive set suffices for this GPT architecture; reciprocal via `exp(-log(x))` is intentionally restricted to positive inputs.
