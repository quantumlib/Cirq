import {defineConfig} from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*_test.ts'],
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    coverage: {
      provider: 'v8',
    },
  },
});
