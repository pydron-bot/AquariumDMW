import { defineConfig } from 'vite';

export default defineConfig({
  base: process.env.TANK_BASE ?? '/',
  server: { host: true, port: 5175, allowedHosts: ['hive1', '.ts.net'] },
  preview: { host: true, port: 4175 },
  build: { chunkSizeWarningLimit: 900 },
});
