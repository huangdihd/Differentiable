import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: { pool: 'threads', fileParallelism: false, reporters: ['default'] },
});
