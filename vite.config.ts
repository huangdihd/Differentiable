// Copyright (C) 2026 huangdihd
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of Differentiable. Full terms in LICENSE at the repository root.
import { defineConfig, configDefaults } from 'vitest/config';
export default defineConfig({
  test: {
    pool: 'threads',
    fileParallelism: false,
    reporters: ['default'],
    // promo/ has its own node:test suite (npm test inside promo/).
    exclude: [...configDefaults.exclude, 'promo/**'],
  },
});
