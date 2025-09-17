import {defineConfig} from 'vite';

const common = {
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    minify: 'terser' as const,
    sourcemap: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        format: 'iife',
      },
    },
  },
};

export default [
  defineConfig({
    build: {
      ...common.build,
      lib: {
        entry: 'src/bloch_sphere/main.ts',
        formats: ['iife'],
        name: 'BlochSphereBundle',
        fileName: () => 'bloch_sphere.bundle.js',
      },
    },
  }),
  defineConfig({
    build: {
      ...common.build,
      lib: {
        entry: 'src/circuit/main.ts',
        formats: ['iife'],
        name: 'CircuitBundle',
        fileName: () => 'circuit.bundle.js',
      },
    },
  }),
];
