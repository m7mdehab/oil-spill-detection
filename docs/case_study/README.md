# Case study: MV Wakashio oil spill (Mauritius, August 2020)

This case study runs the end-to-end Sentinel-1 detection pipeline on a real,
documented oil-spill event and discusses the result honestly — including where the
detection is good and where it falls short.

![Sentinel-1 VV and detected oil over the Wakashio AOI](wakashio_detection.png)

*Left: Sentinel-1B VV backscatter (10 August 2020) over the spill area, south-east
Mauritius. Right: oil pixels detected by the SegFormer model (cyan).*

## The event (verified facts)

The bulk carrier **MV Wakashio** ran aground on a coral reef off **Pointe d'Esny,
south-east Mauritius** (~20.44° S, 57.74° E) on **25 July 2020** and began leaking
fuel oil on about **6 August 2020**, releasing roughly **1,000 tonnes of Very Low
Sulfur Fuel Oil (VLSFO)** before the vessel broke apart in mid-August. These
figures are from the International Maritime Organization and Cedre (see Sources).

The spill was independently mapped from space in the peer-reviewed literature.
Rajendran et al. (2021, *Environmental Pollution* 274) analysed Sentinel-1 VV
C-band SAR acquired between 5 July and 3 September 2020 and reported the oil
appearing as **dark warped patches**, with an overall SAR oil-spill mapping
accuracy of about **91.7 %** (Kappa 0.84). That study establishes both that
Sentinel-1 VV is an appropriate sensor for this event and what the spill looks
like in SAR.

> Note on quantitative comparison: we deliberately do **not** quote a single
> "official" spilled-area figure in km². A specific area value surfaced in
> secondary summaries could not be confirmed against the primary source, so it is
> excluded rather than cited unverified. We therefore compare our detection
> qualitatively (location, morphology, plausibility) against the published
> dark-patch mapping, and report our own measured area as a pipeline output.

## What the pipeline did

- **Scene** — `S1B_IW_GRDH_1SDV_20200810T013755_..._02B625` (Sentinel-1B, IW, GRDH,
  dual-pol; VV used), acquired **10 August 2020**, found and downloaded directly
  from the Copernicus Data Space Ecosystem via this project's `ingest` module.
- **Preprocess** — read the VV measurement, cropped to the Pointe d'Esny AOI
  (≈ 28 × 33 km), Lee speckle filter, conversion to (relative) decibels, and a
  percentile-based normalisation into the model's input range.
- **Inference** — tiled inference with the selected best model (SegFormer mit-b2,
  exported to ONNX), logit-averaged over overlapping tiles.
- **Vectorise** — oil-class pixels converted to polygons with per-polygon area
  (km²) and confidence.

Reproduce with: `python scripts/run_case_study.py` (after downloading the scene to
`data/scenes/`).

## Result

| Quantity | Value |
|---|---|
| Oil polygons detected (AOI) | 28 |
| Total detected oil area (AOI) | 6.92 km² |
| Mean polygon confidence | ~0.90 |
| Detection location | along the SE Mauritius coast / Blue Bay lagoon — consistent with the documented spill |

The model places oil in the **right location** — hugging the south-east coastline
and lagoon around Pointe d'Esny, exactly where the Wakashio oil grounded and
spread — with high per-polygon confidence. As a qualitative detection on a real,
previously-unseen Sentinel-1 scene, this works.

## Honest discussion of errors and limitations

The detected area should be read as a lower-bound, approximate figure, for several
documented reasons:

1. **Uncalibrated radiometry.** Full sigma-nought calibration via `xarray-sentinel`
   failed on this product (a library bug reading the product's GCP annotation), so
   the pipeline fell back to reading the raw measurement (digital-number
   amplitude → relative intensity). The decibel scale is therefore *relative*, and
   the normalisation window was fitted from scene percentiles rather than matched
   to the absolute statistics of the training data.
2. **Domain gap.** The model was trained on the MKLab dataset's preprocessed 8-bit
   JPEG SAR chips, not on raw calibrated Sentinel-1 scenes. The appearance of oil
   (contrast, speckle, dynamic range) differs between the two domains; thin sheen
   and the faint edges of the slick are the first thing missed under this shift,
   which biases the measured area downward.
3. **Geolocation is approximate.** Georeferencing uses an affine fitted to the
   product's ground-control points, which only approximates the true range/azimuth
   geometry of a GRD product (good to within a small number of pixels, not exact),
   so polygon areas carry a corresponding uncertainty.
4. **Coastal complexity.** The spill is nearshore and partly inside a lagoon, where
   land, surf, and shallow-water effects make SAR oil discrimination harder than in
   the open-ocean scenes the model was trained on.

The right way to close this gap is documented in `docs/metrics.md` and the
preprocessing notes: calibrate to true sigma-nought, and fit the dB→model window
against the training histogram. Those are the natural next steps; this case study
shows the pipeline runs end-to-end on real Copernicus data and detects the event
in the correct place, while being candid that an exact area match would require the
radiometric work above.

## Sources

- International Maritime Organization — *Responding to MV Wakashio oil spill* (FAQ): https://www.imo.org/en/MediaCentre/HotTopics/Pages/Wakashio-FAQ.aspx
- Cedre — *Wakashio* spill page: https://wwz.cedre.fr/en/Resources/Spills/Spills/Wakashio
- Rajendran, S., Vethamony, P., Sadooni, F., Al-Kuwari, H., Al-Khayat, J., Seegobin, V., Govil, H., Nasir, S. (2021). *Detection of Wakashio oil spill off Mauritius using Sentinel-1 and 2 data: Capability of sensors, image transformation methods and mapping.* **Environmental Pollution, 274**, 116618. https://doi.org/10.1016/j.envpol.2021.116618
- Scene: Copernicus Data Space Ecosystem (Sentinel-1B, 10 August 2020). Contains modified Copernicus Sentinel data 2020.
