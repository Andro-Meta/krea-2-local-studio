import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8200',
      '/ws': { target: 'ws://localhost:8200', ws: true },
      '/outputs': 'http://localhost:8200',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
