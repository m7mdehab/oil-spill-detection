import { useEffect, useMemo, useRef, useState } from "react";
import {
  getModels,
  getSamples,
  predict,
  sampleSrc,
} from "../lib/api";
import type { ModelInfo, PredictResponse, Sample } from "../lib/types";
import Legend from "../components/Legend";

interface Selection {
  src: string; // displayable URL or object URL for the original image
  blob: Blob; // bytes to POST to /predict
  filename: string;
}

export default function QuickDetect() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [opacity, setOpacity] = useState(0.65);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const objectUrl = useRef<string | null>(null);

  useEffect(() => {
    getModels()
      .then((r) => {
        const available = r.models.filter((m) => m.available);
        const list = available.length ? available : r.models;
        setModels(list);
        if (list.length) setModelId(list[0].id);
      })
      .catch((e) => setError(String(e.message ?? e)));
    getSamples()
      .then((r) => setSamples(r.samples))
      .catch(() => {
        /* samples are optional; uploading still works */
      });
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  function setUploadedFile(file: File) {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    const url = URL.createObjectURL(file);
    objectUrl.current = url;
    setActiveSampleId(null);
    setResult(null);
    setSelection({ src: url, blob: file, filename: file.name });
  }

  async function selectSample(s: Sample) {
    setError(null);
    setResult(null);
    setActiveSampleId(s.id);
    const src = sampleSrc(s.url);
    try {
      const res = await fetch(src);
      const blob = await res.blob();
      const filename = s.url.split("/").pop() || `${s.id}.png`;
      setSelection({ src, blob, filename });
    } catch (e) {
      setError(`Could not load sample: ${(e as Error).message}`);
    }
  }

  async function runDetect() {
    if (!selection || !modelId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predict(selection.blob, modelId, selection.filename);
      setResult(res);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setLoading(false);
    }
  }

  function downloadMask() {
    if (!result) return;
    const a = document.createElement("a");
    a.href = result.mask_png;
    a.download = "oil-spill-mask.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  const canDetect = useMemo(
    () => !!selection && !!modelId && !loading,
    [selection, modelId, loading],
  );

  return (
    <section>
      <div className="view-head">
        <h1>Quick Detect</h1>
        <p>
          Run segmentation on a single image. Pick a built-in sample for an
          instant result, or drop in your own satellite scene.
        </p>
      </div>

      <div className="detect-grid">
        <div className="card">
          <div className="field" style={{ marginBottom: "1rem" }}>
            <label htmlFor="model">Model</label>
            <select
              id="model"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              data-testid="model-select"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name}
                  {m.available ? "" : " (unavailable)"}
                </option>
              ))}
            </select>
          </div>

          <div
            className={`dropzone${dragOver ? " over" : ""}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer.files?.[0];
              if (f) setUploadedFile(f);
            }}
            data-testid="dropzone"
          >
            <strong>Drop an image</strong>
            <div className="faint" style={{ fontSize: "0.85rem" }}>
              or click to browse
            </div>
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setUploadedFile(f);
              }}
            />
          </div>

          {samples.length > 0 && (
            <>
              <div
                className="muted"
                style={{ fontSize: "0.85rem", marginTop: "1rem" }}
              >
                Or pick a sample
              </div>
              <div className="samples">
                {samples.map((s) => (
                  <button
                    key={s.id}
                    className={activeSampleId === s.id ? "active" : ""}
                    onClick={() => selectSample(s)}
                    data-testid={`sample-${s.id}`}
                    title={s.id}
                  >
                    <img src={sampleSrc(s.url)} alt={s.id} loading="lazy" />
                  </button>
                ))}
              </div>
            </>
          )}

          <button
            className="primary"
            style={{ width: "100%", marginTop: "1rem" }}
            disabled={!canDetect}
            onClick={runDetect}
            data-testid="detect-btn"
          >
            {loading ? (
              <>
                <span className="spinner" /> Detecting…
              </>
            ) : (
              "Detect"
            )}
          </button>

          {error && (
            <div className="error" style={{ marginTop: "1rem" }}>
              {error}
            </div>
          )}
        </div>

        <div className="card">
          {!selection ? (
            <p className="muted">Select a sample or upload an image to begin.</p>
          ) : (
            <>
              <div className="compare">
                <figure>
                  <figcaption>Original</figcaption>
                  <div className="image-frame">
                    <img src={selection.src} alt="Original input" />
                  </div>
                </figure>
                <figure>
                  <figcaption>Detection overlay</figcaption>
                  <div className="image-frame">
                    <img src={selection.src} alt="Base" />
                    {result && (
                      <img
                        className="overlay"
                        src={result.overlay_png}
                        alt="Segmentation overlay"
                        style={{ opacity }}
                        data-testid="overlay-img"
                      />
                    )}
                  </div>
                </figure>
              </div>

              {result && (
                <>
                  <div className="field" style={{ marginTop: "1rem" }}>
                    <label htmlFor="opacity">
                      Overlay opacity — {Math.round(opacity * 100)}%
                    </label>
                    <input
                      id="opacity"
                      className="slider"
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={opacity}
                      onChange={(e) => setOpacity(Number(e.target.value))}
                      data-testid="opacity-slider"
                    />
                  </div>

                  <Legend
                    percentages={result.class_percentages}
                    legend={result.legend}
                  />

                  <button
                    style={{ marginTop: "1rem" }}
                    onClick={downloadMask}
                    data-testid="download-mask"
                  >
                    Download mask (PNG)
                  </button>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
