#!/usr/bin/env python
"""CLI: extract per-sample 3D features and save as .pt files (generic)."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from radiomics3d.datasets.brain_s18 import BrainS18Dataset
from radiomics3d.models.model_factory import (
    ModelConfig,
    build_model,
    load_weights_into_model,
    build_feature_extractor,
)
from radiomics3d.utils.io import safe_makedirs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Extract 3D features (de-identified, generic).")
    p.add_argument("--csv", required=True, help="CSV with columns: image_path, voi_path")
    p.add_argument("--output_dir", required=True, help="Directory to save per-sample .pt features")
    p.add_argument("--pretrained", default=None, help="Path to pretrained checkpoint (optional)")
    p.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu")

    p.add_argument("--model_depth", type=int, default=50, choices=[10, 18, 34, 50, 101, 152, 200])
    p.add_argument("--shortcut", default="B", choices=["A", "B"])
    p.add_argument("--num_seg_classes", type=int, default=2)

    p.add_argument("--input_d", type=int, default=14)
    p.add_argument("--input_h", type=int, default=28)
    p.add_argument("--input_w", type=int, default=28)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    safe_makedirs(args.output_dir)
    device = torch.device(args.device)

    ds = BrainS18Dataset(
        csv_path=args.csv,
        input_D=args.input_d,
        input_H=args.input_h,
        input_W=args.input_w,
        phase="train",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    cfg = ModelConfig(
        name="resnet",
        depth=args.model_depth,
        shortcut=args.shortcut,
        input_D=args.input_d,
        input_H=args.input_h,
        input_W=args.input_w,
        num_seg_classes=args.num_seg_classes,
        no_cuda=(device.type == "cpu"),
    )
    model = build_model(cfg)

    if args.pretrained:
        load_weights_into_model(model, args.pretrained, strict=False)

    model.to(device)
    model.eval()

    feature_extractor = build_feature_extractor(model).to(device)
    feature_extractor.eval()

    for _, batch in tqdm(enumerate(loader), total=len(loader)):
        sample_name, image, _mask = batch
        image = image.to(device)

        with torch.no_grad():
            feats = feature_extractor(image)  # (B, 2048)

        for i in range(feats.shape[0]):
            name = str(sample_name[i])
            out_path = os.path.join(args.output_dir, f"{name}.pt")
            torch.save(feats[i].detach().cpu(), out_path)

    print(f"Done. Saved features to: {args.output_dir}")


if __name__ == "__main__":
    main()
