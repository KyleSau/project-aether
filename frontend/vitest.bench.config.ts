import { defineConfig } from 'vitest/config';
export default defineConfig({ test: { include: ['bench/**/*.bench.test.ts'], testTimeout: 600000 } });
