# Oil Spill Detection — Web

React + Vite + TypeScript front-end with three views:

- **Quick Detect** — drag-drop or pick a preloaded sample, choose a model,
  run `/predict`, and compare the original against the segmentation overlay
  (with an opacity slider, a per-class legend showing `class_percentages`, and
  a mask download).
- **Scene Monitor** — a MapLibre map; define an AOI bounding box and date
  range, submit a job to `/jobs/scene`, poll `/jobs/{id}`, and render the
  returned oil polygons with total area and a GeoJSON download.
- **Models** — a table of `/models` with honest per-class IoU / precision /
  recall / F1. Oil-class IoU is highlighted as the headline metric; pixel
  accuracy is labelled secondary.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
```

In dev, `/healthz`, `/models`, `/samples`, `/predict`, and `/jobs` are proxied
to the API at `http://localhost:7860` (override with `VITE_DEV_API_TARGET`).

## Build

```bash
npm run build    # outputs to web/dist
npm run preview  # serve the production build
```

## API base URL

Requests default to the same origin (so the build can be served from the API's
static mount). To point at a separately hosted API, set `VITE_API_BASE`:

```bash
VITE_API_BASE=https://api.example.com npm run build
```

## Map tiles

The map uses the key-free OpenStreetMap raster tile service
(`https://tile.openstreetmap.org/{z}/{x}/{y}.png`) with the required
"© OpenStreetMap contributors" attribution. No API key or secret is needed.

## End-to-end test

A Playwright smoke test (`e2e/quick-detect.spec.ts`) builds the app, serves it
with `vite preview`, mocks all API endpoints via route interception (no backend
needed), then exercises Quick Detect: it picks a sample, runs detect, and
asserts the overlay image and per-class legend percentages appear.

```bash
npx playwright install chromium   # one-time browser download
npm run e2e
```
