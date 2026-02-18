import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    // Proxy API calls from the Vite dev server (5174) to the FastAPI backend (5001)
    // so the frontend can call `/api/...` without hardcoding ports or hitting CORS.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5001",
        changeOrigin: true,
        secure: false
      }
    }
  },
  test: {
    environment: "jsdom"
  }
})
