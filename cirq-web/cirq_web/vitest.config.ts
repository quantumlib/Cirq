import {defineConfig} from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*_test.ts'],
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    coverage: {
      provider: 'v8',
      all: true,
      include: ['**/*.ts'],
      exclude: [
        'coverage/**',
        'node_modules/**',
        '**/*_test.ts',
        'dist/**',
        'build/**',
        'utils/**',
        'e2e/**',
        '**/*.config.ts',
        'src/**/main.ts',
        'src/**/scene.ts'
      ],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      },
    },
  },
});
