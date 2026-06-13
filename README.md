# Oil Spill Detection from Sentinel-1 SAR

Semantic segmentation of oil spills in Sentinel-1 SAR imagery. The project
covers the full stack: model training and evaluation, a scene-processing
pipeline (download, calibration, tiling, inference), and a web API with a
frontend for interactive detection.

Status: under active development — modernization of the original 2024
graduation project (see the `legacy-archive` branch).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and GNU Make.

```sh
uv sync
make check
```

`make check` runs lint (ruff), formatting check, type check (pyright), and the
fast test suite.

## License

MIT — see [LICENSE](LICENSE).
