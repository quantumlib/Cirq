import {defineConfig} from 'vitest/config';

export default defineConfig({
  test: {
    include: ['e2e/**/*.ts'],
    environment: 'node',
    testTimeout: 30000,
  },
});


