// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  root: 'public',
  build: { outDir: '../dist', emptyOutDir: true },
  assetsInclude: ['**/*.wasm'],
  server: {
    mimeTypes: { 'application/wasm': ['wasm'] },
    host: true,
    port: 5173,
  },
  optimizeDeps: {
    // Prevent esbuild from touching the backend; we import the dist file directly.
    exclude: [
      '@tensorflow/tfjs-backend-wasm',
      '@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js',
    ],
  },
});