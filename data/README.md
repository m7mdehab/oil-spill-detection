# Data

This directory is gitignored except for this file, `checksums.sha256`, and
`samples/`. Datasets are not committed; obtain them as described below and place
the archives in `data/raw/`, then extract with the project's data tooling.

## Primary dataset — Oil Spill Detection Dataset (segmentation)

Sentinel-1 SAR oil-spill semantic segmentation dataset (MKLab / m4d), 5 classes.

- **Splits:** 1002 training samples, 110 testing samples (the official split).
- **`images/`** — SAR images, `.jpg`, 1250×650, stored as 3-channel.
- **`labels/`** — RGB ground-truth masks, `.png`.
- **`labels_1D/`** — class-index ground-truth masks, `.png`, values `0..4`
  (this is what the dataset loader reads for training/eval).

Classes and canonical RGB legend (from the dataset's own `README.txt`, which is
authoritative for mask colors):

| ID | Class       | RGB           |
|----|-------------|---------------|
| 0  | Sea Surface | (0, 0, 0)     |
| 1  | Oil Spill   | (0, 255, 255) |
| 2  | Look-alike  | (255, 0, 0)   |
| 3  | Ship        | (153, 76, 0)  |
| 4  | Land        | (0, 153, 0)   |

Expected layout after extraction:

```
data/datasets/oil_spill/
  README.txt
  train/{images,labels,labels_1D}/
  test/{images,labels,labels_1D}/
```

Archive: `data/raw/oil_spill_dataset.zip` — SHA-256 in `checksums.sha256`.

## Auxiliary dataset — Oil Spill Classification Dataset

Binary oil / no-oil image classification set (`Images_Oil`, `Images_No_Oil`).
Its role in this project (e.g. auxiliary pretraining, look-alike hard-negative
mining, or extra evaluation) is assessed in the data report.

Archive: `data/raw/oil_spill_classification_dataset.zip` — SHA-256 in
`checksums.sha256`.

## Samples

`data/samples/` holds five SAR images preserved from the original project for UI
demos and quick smoke tests; these are committed.
