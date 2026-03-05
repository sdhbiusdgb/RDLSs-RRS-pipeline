from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from . import resnet3d
from ..utils.io import load_pretrained_state_dict


@dataclass
class ModelConfig:
    name: str = "resnet"
    depth: int = 50
    shortcut: str = "B"
    input_D: int = 14
    input_H: int = 28
    input_W: int = 28
    num_seg_classes: int = 2
    no_cuda: bool = False


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.name != "resnet":
        raise ValueError("Only 'resnet' is supported in this repo.")

    kwargs = dict(
        sample_input_W=cfg.input_W,
        sample_input_H=cfg.input_H,
        sample_input_D=cfg.input_D,
        shortcut_type=cfg.shortcut,
        no_cuda=cfg.no_cuda,
        num_seg_classes=cfg.num_seg_classes,
    )

    if cfg.depth == 10:
        return resnet3d.resnet10(**kwargs)
    if cfg.depth == 18:
        return resnet3d.resnet18(**kwargs)
    if cfg.depth == 34:
        return resnet3d.resnet34(**kwargs)
    if cfg.depth == 50:
        return resnet3d.resnet50(**kwargs)
    if cfg.depth == 101:
        return resnet3d.resnet101(**kwargs)
    if cfg.depth == 152:
        return resnet3d.resnet152(**kwargs)
    if cfg.depth == 200:
        return resnet3d.resnet200(**kwargs)

    raise ValueError(f"Unsupported depth: {cfg.depth}")


def load_weights_into_model(model: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    state_dict = load_pretrained_state_dict(ckpt_path)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")


def build_feature_extractor(backbone: nn.Module) -> nn.Module:
    stem = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    extractor = nn.Sequential(
        stem,
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
    )
    return extractor
