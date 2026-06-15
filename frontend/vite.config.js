import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build into the path FastAPI serves (frontend/dist). During `npm run dev`,
// /api and /uploads are proxied to the FastAPI backend on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.PORT) || 5173,
    host: true,
    proxy: {
      "/api": "http://localhost:8000",
      "/uploads": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
