// Minimal GeoJSON types (subset of the spec) so we avoid an extra dependency.

export type Position = [number, number];

export interface Polygon {
  type: "Polygon";
  coordinates: Position[][];
}

export interface MultiPolygon {
  type: "MultiPolygon";
  coordinates: Position[][][];
}

export type Geometry = Polygon | MultiPolygon;

export interface Feature<G extends Geometry = Geometry> {
  type: "Feature";
  geometry: G;
  properties: Record<string, unknown> | null;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Feature[];
}
