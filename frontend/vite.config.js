import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend (demo_api.py) runs on :8000. Proxy API routes so the frontend can use
// relative paths and avoid any CORS friction during local development.
const backend = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/detect': backend,
      '/recommend': backend,
      '/health': backend,
    },
  },
})
