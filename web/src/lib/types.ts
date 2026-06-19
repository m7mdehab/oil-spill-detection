// Shapes mirror the backend API contract exactly.
import type { FeatureCollection, Polygon } from "./geojson";

export interface PerClassMetric {
  iou: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  oil_iou: number;
  oil_recall: number;
  mean_iou: number;
  macro_f1: number;
  pixel_accuracy: number;
  per_class: Record<string, PerClassMetric>;
  available: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface Sample {
  id: string;
  url: string;
}

export interface SamplesResponse {
  samples: Sample[];
}

export type RGB = [number, number, number];

export interface PredictResponse {
  model: string;
  width: number;
  height: number;
  class_percentages: Record<string, number>;
  legend: Record<string, RGB>;
  mask_png: string;
  overlay_png: string;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface JobResult {
  num_oil_polygons: number;
  total_oil_area_km2: number;
  geojson: FeatureCollection;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  detail?: string;
  result?: JobResult;
}

export interface SceneJobRequest {
  aoi: Polygon;
  start: string;
  end: string;
  model?: string;
}
