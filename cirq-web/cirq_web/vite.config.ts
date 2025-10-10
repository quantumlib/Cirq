import {defineConfig} from 'vite';

const common = {
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    minify: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        format: 'iife',
      },
    },
  },
};

export default defineConfig(({mode}) => {
  if (mode === 'bloch_sphere') {
    return {
      build: {
        ...common.build,
        lib: {
          entry: 'src/bloch_sphere/main.ts',
          formats: ['iife'],
          name: 'BlochSphereBundle',
          fileName: () => 'bloch_sphere.bundle.js',
        },
      },
    };
  }

  if (mode === 'circuit') {
    return {
      build: {
        ...common.build,
        lib: {
          entry: 'src/circuit/main.ts',
          formats: ['iife'],
          name: 'CircuitBundle',
          fileName: () => 'circuit.bundle.js',
        },
      },
    };
  }
});
