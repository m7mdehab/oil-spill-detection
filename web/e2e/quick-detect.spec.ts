import { test, expect } from "@playwright/test";

// 1x1 transparent PNG, used as a stand-in for sample/overlay/mask images so the
// test is fully self-contained and deterministic with no real backend.
const PNG_1PX =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";
const PNG_DATA_URL = `data:image/png;base64,${PNG_1PX}`;
const PNG_BYTES = Buffer.from(PNG_1PX, "base64");

const MODELS = {
  models: [
    {
      id: "unet-r34",
      name: "U-Net (ResNet-34)",
      oil_iou: 0.71,
      oil_recall: 0.83,
      mean_iou: 0.64,
      macro_f1: 0.72,
      pixel_accuracy: 0.95,
      per_class: {
        "Sea Surface": { iou: 0.96, precision: 0.97, recall: 0.98, f1: 0.97 },
        "Oil Spill": { iou: 0.71, precision: 0.78, recall: 0.83, f1: 0.8 },
        "Look-alike": { iou: 0.42, precision: 0.55, recall: 0.5, f1: 0.52 },
        Ship: { iou: 0.6, precision: 0.7, recall: 0.65, f1: 0.67 },
        Land: { iou: 0.88, precision: 0.9, recall: 0.91, f1: 0.9 },
      },
      available: true,
    },
  ],
};

const SAMPLES = {
  samples: [{ id: "scene-01", url: "/samples/scene-01.png" }],
};

const PREDICT = {
  model: "unet-r34",
  width: 1,
  height: 1,
  class_percentages: {
    "Sea Surface": 72.4,
    "Oil Spill": 12.6,
    "Look-alike": 3.1,
    Ship: 0.4,
    Land: 11.5,
  },
  legend: {
    "Sea Surface": [0, 0, 0],
    "Oil Spill": [0, 255, 255],
    "Look-alike": [255, 0, 0],
    Ship: [153, 76, 0],
    Land: [0, 153, 0],
  },
  mask_png: PNG_DATA_URL,
  overlay_png: PNG_DATA_URL,
};

test("quick detect: sample → detect → overlay + legend", async ({ page }) => {
  // Mock every backend endpoint the app touches.
  await page.route("**/healthz", (r) =>
    r.fulfill({ json: { status: "ok" } }),
  );
  await page.route("**/models", (r) => r.fulfill({ json: MODELS }));
  await page.route("**/samples", (r) => r.fulfill({ json: SAMPLES }));
  await page.route("**/samples/scene-01.png", (r) =>
    r.fulfill({ contentType: "image/png", body: PNG_BYTES }),
  );
  await page.route("**/predict", (r) => r.fulfill({ json: PREDICT }));

  await page.goto("/");

  // App lands on Quick Detect; ensure we're there.
  await page.getByTestId("nav-detect").click();
  await expect(page.getByRole("heading", { name: "Quick Detect" })).toBeVisible();

  // Model picker populated from /models.
  await expect(page.getByTestId("model-select")).toContainText(
    "U-Net (ResNet-34)",
  );

  // Pick the preloaded sample, then run detect.
  await page.getByTestId("sample-scene-01").click();
  await page.getByTestId("detect-btn").click();

  // Overlay image appears.
  const overlay = page.getByTestId("overlay-img");
  await expect(overlay).toBeVisible();
  await expect(overlay).toHaveAttribute("src", PNG_DATA_URL);

  // Legend shows the five classes with percentages.
  await expect(page.getByTestId("legend")).toBeVisible();
  const rows = page.getByTestId("legend-row");
  await expect(rows).toHaveCount(5);
  await expect(page.getByTestId("legend")).toContainText("Oil Spill");
  await expect(page.getByTestId("legend")).toContainText("12.6%");

  // Mask download control is offered.
  await expect(page.getByTestId("download-mask")).toBeVisible();
});
