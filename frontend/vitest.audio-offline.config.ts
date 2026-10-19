import { defineConfig } from 'vitest/config';
// Offline validation renders of the browser audio graph (node-web-audio-api); see tests/offline/render.offline.ts
export default defineConfig({ test: { include: ['tests/offline/*.offline.ts'], testTimeout: 900000, fileParallelism: false } });
