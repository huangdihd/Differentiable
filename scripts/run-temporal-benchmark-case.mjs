// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { createServer } from 'vite';

const [model, delayText, seedText = '811'] = process.argv.slice(2);
if (!model || !delayText) throw new Error('Usage: node scripts/run-temporal-benchmark-case.mjs MODEL DELAY [SEED]');

const server = await createServer({
  appType: 'custom',
  logLevel: 'error',
  server: { hmr: false, middlewareMode: true },
});

try {
  const { runTemporalBenchmarkCase, temporalBenchmarkDelays, temporalBenchmarkModels } =
    await server.ssrLoadModule('/src/levels/temporal-benchmark.ts');
  const delay = Number(delayText);
  const seed = Number(seedText);
  if (!temporalBenchmarkModels.includes(model))
    throw new Error(`Unknown model: ${model}`);
  if (!temporalBenchmarkDelays.includes(delay))
    throw new Error(`Unknown delay: ${delayText}`);
  if (!Number.isSafeInteger(seed)) throw new Error(`Invalid seed: ${seedText}`);
  const result = await runTemporalBenchmarkCase(model, delay, seed);
  console.log(JSON.stringify({ model, delay, ...result }));
} finally {
  await server.close();
}
