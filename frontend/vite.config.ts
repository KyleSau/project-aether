import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

// mode "site": self-contained static build into ../site with relative base
export default defineConfig(({ mode }) => ({
  base: './',
  plugins: [react()],
  build: {
    outDir: mode === 'site' ? resolve(import.meta.dirname, '../site') : 'dist',
    emptyOutDir: true,
    target: 'es2022',
    chunkSizeWarningLimit: 4000,
    assetsInlineLimit: 0,
    sourcemap: false,
  },
  server: { port: 5280, strictPort: true },
}));
