# Tile Annotation Pipeline (De-identified)

A portable, **de-identified** Python pipeline for:
- Splitting whole-slide images (WSI) into tiles
- Running HoVer-Net inference on tiles (external dependency)
- Rendering tile-level and WSI-level cell annotations
- Generating WSI / tile cell-type composition plots
- Extracting per-tile morphology features and aggregating to WSI-level features

> This repository intentionally contains **no real paths, institutions, patient/sample identifiers, usernames, or internal project names**.
> All dataset paths and model paths must be supplied via CLI arguments or a config file.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

1) Prepare a config (see `configs/example.yaml`).

2) Run:
```bash
tile-anno run --config configs/example.yaml
```

You can also run individual stages:
```bash
tile-anno segment --config configs/example.yaml
tile-anno analyze  --config configs/example.yaml
```

## Notes

- HoVer-Net itself is **not** included. You must provide:
  - `hovernet_infer_script` (e.g., `run_infer.py`)
  - `model_path` and `type_info_path`
- The pipeline stores step status in `meta.json` per sample to support resume.

## License

MIT (see `LICENSE`).
