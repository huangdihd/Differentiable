# Differentiable

**English** · [简体中文](README.zh-CN.md)

A neural-network construction game that runs in the browser. You start with `a + b` and finish by
training a GPT — every part you wire yourself becomes a component you can reuse in the next level.

[**Play**](https://differentiable.pages.dev) · [Trailer](https://www.bilibili.com/video/BV13BJt6SEVx) ·
[Architecture](docs/architecture.md) · [Level design](docs/levels.md)

[![Deploy demo](https://github.com/huangdihd/Differentiable/actions/workflows/deploy-demo.yml/badge.svg)](https://github.com/huangdihd/Differentiable/actions/workflows/deploy-demo.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

## Why this exists

Every other autodiff tutorial hands you `loss.backward()`. Here the backward pass is a circuit you
wire by hand before the autodiff tool exists, so the chain rule is something you built rather than
something you invoked. The same applies to softmax, layer normalization, attention and the
Transformer block: each one is a level you solve, then keep.

The engine under the game is a real one — typed tensors, a versioned JSON graph, first-order reverse
autodiff, a CPU reference backend and a WebGPU backend — not a teaching toy that special-cases the
answers.

## What you build

| Chapter | Topic | Ends with |
| --- | --- | --- |
| 0 | Scalars | `sigmoid`, `tanh` |
| 1 | Derivatives | hand-wired VJPs, then autodiff |
| 2 | Tensors | broadcasting, matmul, softmax, LayerNorm |
| 3 | Learning | MSE, cross entropy, SGD, momentum, Adam, MLP — XOR |
| 4 | Sequences | embeddings, bigram, a two-character model |
| 5 | Time-series | fixed windows, RNN, the final-state bottleneck |
| 6 | Attention | attention, causal masks, positional embeddings |
| 7 | GPT | multi-head attention, feed-forward, residual blocks, a trainable GPT |

62 levels in total. Roughly 3,500 tests cover the engine, the level catalogue and the training
paths.

## Quick start

Node 22.12+, 24+ or 26+.

```sh
npm install
npm run dev        # http://localhost:5173
```

```sh
npm run typecheck
npm test           # non-watch; the GPT calibration test alone takes ~67s
npm run build
npm run format:check
```

Full setup, the demo deployment pipeline and the known limits are in
[docs/development.md](docs/development.md).

## Project layout

```
src/engine/    tensors, graph compilation, autodiff, training, backends (CPU + WebGPU)
src/levels/    the level catalogue, reference circuits and seeded checks
src/editor/    graph document state, undo/redo, save migration
src/ui/        React editor shell, lesson panels, plots, figures
src/worker/    off-main-thread training, generation and evaluation
src/locales/   flat-key dictionaries for en and zh-CN
demos/         original prototypes kept as reference, not application source
```

## Documentation

- [Engine architecture](docs/architecture.md) — backend contract, graph format, autodiff,
  component expansion, WebGPU selection and limits
- [Level design](docs/levels.md) — the route, stable IDs, save compatibility and per-chapter
  calibration
- [Development](docs/development.md) — environment, scripts, demo deployment, current limits
- [Editor internals](src/ui/README.md) · [Worker protocol](src/worker/README.md) ·
  [Audio](src/audio/README.md) · [Analytics](analytics/README.md)

## Contributing

Issues and pull requests are welcome. Two rules matter more than the rest:

- **Level IDs are a save format.** Display numbers may change; the `id` field of a level may not.
  Renaming one silently breaks every existing save.
- **Behaviour changes need a test that fails first.** The suite is the specification for shape
  inference, gradient edge cases and training determinism; `npm run typecheck` and `npm test` must
  pass before a PR is considered.

Run `npm run format` before committing. Formatting is Prettier at 100 columns, single quotes.

## License

Differentiable is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

It is distributed in the hope that it will be useful, but **without any warranty**; without even
the implied warranty of merchantability or fitness for a particular purpose. See [`LICENSE`](LICENSE)
for the full terms.

Every source file carries `SPDX-License-Identifier: GPL-3.0-or-later`. Third-party packages and
fonts, all under licenses compatible with the GPLv3, are listed in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

The public demo bundle is obfuscated by `npm run build:demo`. Obfuscation does not make it less
free: GPLv3 section 6 requires that anyone who receives that bundle can obtain its corresponding
source, and this repository is exactly that. Keep the repository publicly readable for as long as
the demo build is publicly served.
