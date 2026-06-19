import { CLASSES, OIL_CLASS_NAME, colorForClass, rgbToCss } from "../lib/classes";
import type { RGB } from "../lib/types";

interface Props {
  // class key -> percentage, as returned by /predict class_percentages
  percentages: Record<string, number>;
  legend?: Record<string, RGB>;
}

// Order rows by the canonical class list, then append any extra keys the API
// returned that we don't recognise, so nothing is silently dropped.
function orderedKeys(percentages: Record<string, number>): string[] {
  const known = CLASSES.map((c) => c.name).filter(
    (name) => name in percentages,
  );
  const extra = Object.keys(percentages).filter(
    (k) => !CLASSES.some((c) => c.name === k),
  );
  return [...known, ...extra];
}

export default function Legend({ percentages, legend }: Props) {
  const keys = orderedKeys(percentages);
  return (
    <div className="legend" data-testid="legend">
      {keys.map((key) => {
        const pct = percentages[key] ?? 0;
        const isOil = key === OIL_CLASS_NAME;
        return (
          <div
            key={key}
            className={`legend-row${isOil ? " oil" : ""}`}
            data-testid="legend-row"
          >
            <span
              className="legend-swatch"
              style={{ background: rgbToCss(colorForClass(key, legend)) }}
            />
            <span className="legend-name">{key}</span>
            <span className="legend-pct">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
