import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to the Python backend (python demo/agent/server.py, port 8600).
// `npm run build` writes dist/, which the backend serves at http://localhost:8600.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8600" } },
  build: { chunkSizeWarningLimit: 900 },
});
