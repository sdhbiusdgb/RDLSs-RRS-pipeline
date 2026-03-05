# 3D Feature Extraction (Generic, De-identified)

This repository provides a **generic** PyTorch pipeline to:
- Load paired **3D volumes** (e.g., NIfTI) and optional **masks**
- Build a **3D ResNet** backbone
- **Extract fixed-length features** (default: 2048) and save them per-sample

> ⚠️ **De-identification notice**  
> This repo is intentionally scrubbed of any real file paths, institutions, project names, or private identifiers.
> You must provide your own data paths and (optionally) pretrained weights.

## Project structure

```
.
├── src/
│   └── radiomics3d/
│       ├── datasets/
│       │   └── brain_s18.py
│       ├── models/
│       │   ├── model_factory.py
│       │   └── resnet3d.py
│       └── utils/
│           └── io.py
├── tools/
│   └── extract_features.py
├── configs/
│   └── default.yaml
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
```

## Data format

Provide a CSV file with (at minimum) these columns:

- `image_path`: path to a 3D image volume (e.g., `.nii` / `.nii.gz`)
- `voi_path`:   path to a 3D mask volume (same shape as image)

Example:

```csv
image_path,voi_path
/path/to/sampleA_image.nii.gz,/path/to/sampleA_mask.nii.gz
/path/to/sampleB_image.nii.gz,/path/to/sampleB_mask.nii.gz
```

## Feature extraction

```bash
python tools/extract_features.py   --csv /path/to/list.csv   --output_dir /path/to/features   --model_depth 50   --input_d 14 --input_h 28 --input_w 28   --pretrained /path/to/weights.pth   --device cuda:0
```

Outputs: one `*.pt` file per sample (PyTorch tensor of shape `(2048,)` by default).

## Notes

- This repo focuses on **feature extraction**, not segmentation training.
- The default model definition includes a segmentation head in the original sources, but for extraction we use the backbone + global average pooling.

## License

MIT. See `LICENSE`.
