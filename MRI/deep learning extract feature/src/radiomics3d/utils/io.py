from __future__ import annotations

import os
from typing import Dict, Any

import torch


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_pretrained_state_dict(path: str) -> Dict[str, Any]:
    """Load a checkpoint from disk.

    Supports:
    - {'state_dict': ...} style checkpoints
    - raw state_dict checkpoints
    """
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format.")
