import type { RGB } from "./types";

// The five segmentation classes, in label order. Colors match the model legend
// but the live legend from /predict takes precedence when rendering results.
export interface ClassDef {
  id: number;
  name: string;
  color: RGB;
}

export const CLASSES: ClassDef[] = [
  { id: 0, name: "Sea Surface", color: [0, 0, 0] },
  { id: 1, name: "Oil Spill", color: [0, 255, 255] },
  { id: 2, name: "Look-alike", color: [255, 0, 0] },
  { id: 3, name: "Ship", color: [153, 76, 0] },
  { id: 4, name: "Land", color: [0, 153, 0] },
];

export const OIL_CLASS_NAME = "Oil Spill";

export function rgbToCss([r, g, b]: RGB): string {
  return `rgb(${r}, ${g}, ${b})`;
}

// Resolve a display color for a class key, preferring the legend returned by the
// API and falling back to the canonical palette by name.
export function colorForClass(
  key: string,
  legend?: Record<string, RGB>,
): RGB {
  if (legend && legend[key]) return legend[key];
  const byName = CLASSES.find((c) => c.name === key || String(c.id) === key);
  return byName ? byName.color : [128, 128, 128];
}
