from __future__ import annotations

import torch.nn as nn
from .resnet50_ccl import ResNet50CCLFeatureExtractor


def build_tissue_classifier(
    num_classes: int,
    checkpoint_path: str,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a ResNet50 classifier initialized from a checkpoint.

    This is the de-identified refactor of `TissueClassifyModel()`:
    - load ResNet50 weights from `checkpoint_path`
    - replace `fc` to output `num_classes`
    """
    if num_classes <= 1:
        raise ValueError("num_classes must be > 1")

    helper = ResNet50CCLFeatureExtractor(
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        strict=False,
    )
    model = helper.model
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
