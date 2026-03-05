# Tissue Classification (De-identified)

GitHub-ready refactor of three modules:

- `TIFDataset` (PyTorch Dataset for `.tif` tiles arranged by class folders). fileciteturn1file0
- `ResNet50_CCL` feature extractor wrapper (loads a checkpoint and exposes conv features). fileciteturn1file2
- `TissueClassifyModel` factory function (builds a classifier head). fileciteturn1file1

**De-identification**
- No real usernames, institutions, server paths, or private download links.
- All checkpoint paths and class mappings are passed as arguments.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Expected data layout

```
data_root/
  CLASS_A/*.tif
  CLASS_B/*.tif
```

## Usage

```python
from tissue_classifier.data.dataset import TIFDataset
from tissue_classifier.models.factory import build_tissue_classifier

ds = TIFDataset("data_root")
model = build_tissue_classifier(num_classes=9, checkpoint_path="weights/resnet50_ccl.pth")
```

## License
MIT.
