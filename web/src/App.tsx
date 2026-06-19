import { useEffect, useState } from "react";
import QuickDetect from "./views/QuickDetect";
import SceneMonitor from "./views/SceneMonitor";
import Models from "./views/Models";
import { getHealth } from "./lib/api";

type ViewId = "detect" | "scene" | "models";

const VIEWS: { id: ViewId; label: string }[] = [
  { id: "detect", label: "Quick Detect" },
  { id: "scene", label: "Scene Monitor" },
  { id: "models", label: "Models" },
];

export default function App() {
  const [view, setView] = useState<ViewId>("detect");
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    getHealth()
      .then((h) => active && setApiOk(h.status === "ok"))
      .catch(() => active && setApiOk(false));
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          Oil Spill Detection
        </div>
        <nav className="nav" aria-label="Primary">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              className={view === v.id ? "active" : ""}
              onClick={() => setView(v.id)}
              data-testid={`nav-${v.id}`}
            >
              {v.label}
            </button>
          ))}
        </nav>
        <div style={{ marginLeft: "auto" }}>
          {apiOk === null ? (
            <span className="badge off">connecting…</span>
          ) : apiOk ? (
            <span className="badge ok">API online</span>
          ) : (
            <span className="badge off">API offline</span>
          )}
        </div>
      </header>

      <main className="content">
        {view === "detect" && <QuickDetect />}
        {view === "scene" && <SceneMonitor />}
        {view === "models" && <Models />}
      </main>
    </div>
  );
}
