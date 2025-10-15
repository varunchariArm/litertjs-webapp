// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  root: 'public',
  build: { outDir: '../dist', emptyOutDir: true },
  assetsInclude: ['**/*.wasm'],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    },
    mimeTypes: { 'application/wasm': ['wasm'] },
    host: true,
    port: 5173,
  },
});