module.exports = [
  ...require('gts'),
  {
    ignores: [
      'build/',
      '**/node_modules/',
      '**/dist/',
      'vitest.config.ts',
      'vitest.setup.ts',
      'vite.config.ts',
      'vitest.e2e.config.ts',
    ],
  },
  {
    files: ['**/*.ts'],
    rules: {
      'n/no-unpublished-import': 'off',
      'max-len': ['error', 100],
    },
  },
];
