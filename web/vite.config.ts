import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Endpoints proxied to the local API during development.
const apiTarget = process.env.VITE_DEV_API_TARGET ?? "http://localhost:7860";
const proxied = ["/healthz", "/models", "/samples", "/predict", "/jobs"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      proxied.map((path) => [
        path,
        { target: apiTarget, changeOrigin: true },
      ]),
    ),
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
