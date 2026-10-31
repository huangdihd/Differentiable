// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const server = await createServer({
  appType: 'custom',
  logLevel: 'error',
  server: { hmr: false, middlewareMode: true },
});

try {
  const { generateTemporalBenchmark } = await server.ssrLoadModule(
    '/src/levels/temporal-benchmark.ts',
  );
  const artifact = await generateTemporalBenchmark();
  const output = resolve('src/levels/temporal-benchmark.json');
  await writeFile(output, `${JSON.stringify(artifact, null, 2)}\n`);
  console.log(`Wrote ${output}`);
} finally {
  await server.close();
}
