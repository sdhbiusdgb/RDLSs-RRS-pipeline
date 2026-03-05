from __future__ import annotations

import os
from typing import Any, Dict, Union

import torch
import torch.nn as nn
from torchvision import models


def _normalize_checkpoint(obj: Any) -> Dict[str, torch.Tensor]:
    """Convert common checkpoint formats to a plain state_dict."""
    if isinstance(obj, dict):
        # already a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                if any(k.startswith("module.") for k in sd.keys()):
                    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                return sd

    raise ValueError("Unrecognized checkpoint format (expected a state_dict-like object).")


class ResNet50CCLFeatureExtractor(nn.Module):
    """ResNet50 feature extractor that outputs 2048-d vectors.

    - Loads weights from `checkpoint_path`
    - Exposes convolutional backbone features (no final FC)
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze_backbone: bool = True,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
    ) -> None:
        super().__init__()
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model = models.resnet50(weights=None)

        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state_dict = _normalize_checkpoint(ckpt)
        self.model.load_state_dict(state_dict, strict=strict)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)
