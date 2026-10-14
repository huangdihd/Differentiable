# Real-browser GPU tests

Run `node scripts/test-gpu.mjs` (also intended as `npm run test:gpu`). Requires an installed Chrome and the existing Vite dev dependency; no browser automation package or downloaded browser is used. `CHROME_PATH` overrides executable discovery. Chrome gets a fresh profile under Node's `os.tmpdir()`, removed on completion. `GPU_TEST_TIMEOUT_MS` overrides the 30-minute timeout. Temporary-directory or browser sandbox failures are failures, not skips. The runner does not disable the Chrome sandbox, request a software adapter, or fall back to CPU. Adapter information is printed before testing; verify it describes the hardware expected for any published timing.

The HTML page reports JSON lines over a loopback Vite middleware endpoint. Module load errors, uncaught browser errors, no adapter, Chrome exit, failed comparisons and timeout cause a nonzero exit. No CDP/Playwright connection is needed.

Coverage:

- Three seeded randomized trials of every forward tensor operation and each input VJP, with nonuniform random upstream gradients, broadcast/rank variants, reduction axes/keepdims, ties and repeated gather indices. Gather forward/VJPs must repeat bit-for-bit across five extra runs.
- Four CPU/GPU GraphJSON GPT steps comparing logits, loss, all parameter VJPs and Adam-updated parameters. The nested fixture mirrors `gpt.test.ts` without modifying/importing its Vitest test. The comparison uses a smaller 2-layer, 3-head model and reproducible synthetic next-token data, not the corpus convergence assertion.
- 300 small GPT training steps checking stable live buffers and bounded pool/allocated-byte growth after 20 warmup steps, plus disposal checks.
- CPU and GPU end-to-end training benchmarks: 2 layers / d64 / h4 / ctx64 / B16 and 4 layers / d128 / h4 / ctx128 / B16, vocabulary 27. One warmup and three timed forward/backward/Adam steps each; loss readback and explicit GPU completion fences are included. Model compilation/initialization and warmup are excluded. Output includes step milliseconds and tokens/second, not a speedup assertion.

Timing is a short feasibility sample, not a statistically rigorous performance study. CPU runs on the browser main thread and large CPU configurations may be slow. Direct optimizer command edge cases remain the responsibility of engine optimizer tests; this harness exercises Adam through GPT training. Maintain the mirrored fixture if the CPU GPT architecture changes.
