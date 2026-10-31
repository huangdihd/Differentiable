# Development

Local setup, how the public demo is built and deployed, and what the project deliberately does
not do yet.

## Local setup

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

The repository is public, so the `LICENSE` file and this source tree are the corresponding source
for every published demo build. `main` holds development history; the `demo` branch is the only
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

## Current limits

The functional editor, localization runtime, versioned local storage, and chapters 0–7 are included. Chapter 3 teaches scalar updates, vector gradients, player-built MSE and cross entropy, regression, classification, momentum, Adam and hidden-layer networks. Training levels include learning-rate controls, single-step forward/backward/update animation, loss and data plots, and player-facing dataset cards. Scalar function lessons plot their target and explicit asymptotes without bridging discontinuities; derivative lessons switch between the derivative curve and the original function with target/player tangents. Single-output multivariable derivative lessons visualize a labeled one-dimensional slice along the differentiated input while holding the other inputs fixed. The gradient lesson uses a rotatable 3D function surface with target (yellow) and player (cyan) gradient arrows in the input plane; matching arrows turn green. It introduces the gradient vector and fastest-ascent geometry only; loss, negative gradients and gradient descent wait until chapter 3. Separate modes drag the input point or rotate the view. Chapter 4 adds one-hot encoding, embedding, bigram training, sampling, reshape practice and a two-character model. Chapter 5 exposes a fixed-window failure, builds and unrolls an RNN, verifies short-delay training, demonstrates the final-state bottleneck, then adds temporal attention and next-value forecasting. Chapter 6 adds scaled scores, causal masks, attention, positional embeddings and a single-head language model. Text generation is available through the worker API and the existing `generates`-gated panel. Chapter 7 adds multi-head attention, a feed-forward network, residual blocks and a two-block GPT training level. Its CPU calibration passes loss and continuation vocabulary coverage; this does not guarantee complete nursery-rhyme sentences. Primitive broadcast matmul materializes a large intermediate and is intended for correctness checks, not production GPT throughput. GPU normalization uses deterministic 64-lane workgroup reductions; sum/reduce_max use the same parallel path for reduction fibers of at least 256 elements (sum VJP is elementwise, reduce_max VJP selects the first tied index). Each reduction uses 256 bytes of shared scratch; reduce_max VJP additionally declares 256 bytes of integer winner scratch. No floating-point atomics are used. Oversized logical dispatch grids are remapped automatically to device-sized physical grids. Adam bias corrections remain input tensor data, not step-specialized shader code. Gather backward still scans indices per table element, a performance risk for large user-defined models.

The engine supports first-order gradients only. No symbolic shapes or mixed precision. The given primitive set suffices for this GPT architecture.
