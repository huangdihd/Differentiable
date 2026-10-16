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
