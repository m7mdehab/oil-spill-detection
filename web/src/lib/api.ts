import type {
  Job,
  ModelsResponse,
  PredictResponse,
  SamplesResponse,
  SceneJobRequest,
} from "./types";

// Same-origin by default so the app works when served from the API's static
// mount. Override with VITE_API_BASE (e.g. for a separately hosted API).
export const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

function url(path: string): string {
  return `${API_BASE}${path}`;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export async function getHealth(): Promise<{ status: string }> {
  return asJson(await fetch(url("/healthz")));
}

export async function getModels(): Promise<ModelsResponse> {
  return asJson(await fetch(url("/models")));
}

export async function getSamples(): Promise<SamplesResponse> {
  return asJson(await fetch(url("/samples")));
}

// Resolve a sample URL against the API base (sample URLs may be root-relative).
export function sampleSrc(sampleUrl: string): string {
  if (/^https?:\/\//.test(sampleUrl)) return sampleUrl;
  return `${API_BASE}${sampleUrl.startsWith("/") ? "" : "/"}${sampleUrl}`;
}

export async function predict(
  file: Blob,
  model: string,
  filename = "image.png",
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  form.append("model", model);
  return asJson(await fetch(url("/predict"), { method: "POST", body: form }));
}

export async function createSceneJob(
  body: SceneJobRequest,
): Promise<{ job_id: string; status: string }> {
  return asJson(
    await fetch(url("/jobs/scene"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function getJob(id: string): Promise<Job> {
  return asJson(await fetch(url(`/jobs/${id}`)));
}
