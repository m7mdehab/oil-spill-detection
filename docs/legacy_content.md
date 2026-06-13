# Legacy Content — Original 2024 Graduation Project

This document preserves the user-facing content of the original Streamlit web
application (2024 graduation project) that previously lived in this repository.
The application itself has been removed from `main`; its full source remains on
the `legacy-archive` branch. Text below is reproduced from the original app
pages and README so it can be reused in the new web frontend and documentation.

## Original project description

From the original README:

> Welcome to the Oil Spill Detector Web Application! This application is
> designed to help users detect oil spills using satellite imagery, leveraging
> advanced machine learning and deep learning models.
>
> The Oil Spill Detector Web Application introduces users to the field of
> satellite image analysis for environmental monitoring. It uses deep learning
> models to identify and classify oil spills, enhancing efforts towards marine
> and coastal conservation.

From the welcome screen:

> This Webapp uses deep learning models to detect oil spills in satellite
> images. It allows users to upload Synthetic Aperture Radar (SAR) images,
> select from various detection models, and visualize the detection results
> with annotated areas showing detected oil spills.

## How it works (original explanation)

The application used deep learning models for semantic segmentation of
satellite images. The user workflow was:

1. Upload a SAR image (JPEG/PNG) or pick one of five bundled sample images.
2. Choose a detection model (DeepLabV3+ was marked "Recommended", U-Net, FCN,
   or SegNet).
3. Run detection. The model predicts a class for each pixel.
4. View the output: a color-coded image where each pixel's color corresponds
   to its classified category, plus the percentage of each class present in
   the image as a quantitative measure.

The original sample images are preserved in `data/samples/`.

## Model descriptions (original wording)

- **DeepLabV3+** — "Utilizes advanced techniques for semantic image
  segmentation. Known for its high accuracy and ability to deal with complex
  segmentation tasks, making it recommended for challenging scenes."
- **U-Net** — "Primarily used for precise segmentation tasks. It's effective
  in distinguishing complex features in images."
- **FCN (Fully Convolutional Network)** — "Adapts classical neural networks
  for pixel-wise segmentation."
- **SegNet** — "Known for its efficiency in segmenting image pixels into
  categorically distinct classes."

### Original comparison narrative

- **DeepLabV3+**: excelled in segmentation accuracy and efficiency; showed
  exceptional results adapting to complex and diverse datasets; the
  recommended choice for high-stakes applications.
- **U-Net**: consistently excellent in tasks requiring precise segmentation;
  high effectiveness in both training and testing; excels where pixel-level
  accuracy is crucial.
- **FCN**: versatile in handling different image sizes and efficient in
  processing, but slightly less accurate than U-Net; may lack fine-grained
  precision in localizing boundaries.
- **SegNet**: parameter-efficient thanks to pooling indices used in
  up-sampling; performed comparably well; attractive where computational
  resources are limited or real-time processing is needed.
- **Generalization**: all models generalized well to test data; DeepLabV3+
  adapted best to unseen data, closely followed by U-Net.

## Class color legend (5 classes)

The models classify image pixels into five classes, each rendered with a fixed
color in the output mask:

| Class ID | Class name  | Color | RGB             |
|----------|-------------|-------|-----------------|
| 0        | Sea Surface | Black | (0, 0, 0)       |
| 1        | Oil Spill   | Cyan  | (0, 255, 255)   |
| 2        | Look-alike  | Red   | (255, 0, 0)     |
| 3        | Ship        | Brown | (165, 42, 42)   |
| 4        | Land        | Green | (0, 128, 0)     |

Original descriptions of the classes:

- **Sea Surface** (Black): water surfaces without any contamination.
- **Oil Spill** (Cyan): the presence of oil spills.
- **Look-alike** (Red): pixels that may resemble oil spills but are not.
- **Ship** (Brown): pixels that represent ships.
- **Land** (Green): pixels representing land surfaces.

## Original reported metrics (2024)

> **Note:** these are the numbers reported in the original 2024 graduation
> project, reproduced exactly as published in the app's "Model Comparison and
> Conclusions" page. They have not been re-validated against the current
> codebase and are kept here for historical reference and later citation.

| Model          | Test Loss | Test Accuracy | Precision | Recall | IOU    |
|----------------|-----------|---------------|-----------|--------|--------|
| **DeeplabV3+** | 0.115     | 96.25%        | 96.25%    | 96.25% | 92.77% |
| **U-Net**      | 0.215     | 93.55%        | 94.07%    | 92.99% | 40.00% |
| **FCN**        | 0.174     | 93.23%        | 93.23%    | 93.23% | 87.32% |
| **SegNet**     | 0.223     | 92.87%        | 92.87%    | 92.87% | 86.70% |

## Research abstract (original documentation page)

> This research paper delves into the application of computational
> methodologies for detecting oil spills in marine environments. Through the
> analysis of SAR data and pattern recognition, the study employs machine
> learning and deep learning techniques such as U-Net, SegNet, FCN, and
> DeeplabV3+ to enhance oil spill detection. By reviewing extensive literature
> from fourteen papers, the paper assesses various approaches and datasets to
> advance marine ecosystem conservation. The focus is on developing more
> accurate and efficient methodologies to detect oil spills, contributing
> significantly to environmental protection efforts.

The page linked the full paper:
`https://drive.google.com/uc?export=download&id=1HzFIu3hNLMeeY__CL66ybPbM32SIHZ17`

## Legacy implementation notes

Recorded for reproducibility of the original inference behavior:

- Stack: Streamlit multi-page app, TensorFlow/Keras 2.15.0.
- Model files (kept on `legacy-archive`, not on `main`): `deeplab_model.tflite`
  (TFLite), `Unet_model.h5`, `SegNet_model.h5`, `FCN_model.h5` (Keras).
- Preprocessing: resize to 256x256, convert to float32, scale to [0, 1],
  force 3 channels (grayscale stacked, alpha dropped), add batch dimension.
- Postprocessing: argmax over the class dimension, per-class pixel
  percentages computed over the whole image.
- Detection rule: "Oil Spill Detected" was reported whenever the Oil Spill
  class covered more than 0% of pixels.
- The dataset itself was not documented in the app beyond the five-class
  scheme above; the raw archives are stored under `data/raw/`.

## Other legacy pages

- **Solar Cell Site Selection App**: an unrelated companion Android app for
  evaluating locations for solar PV installation (solar irradiation, slope,
  proximity to transmission lines), distributed as an APK from the repository.
- **Contact**: listed `M7mdehab999@gmail.com` as the contact address.
