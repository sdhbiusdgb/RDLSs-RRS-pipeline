from __future__ import annotations
import os, json, sys
import numpy as np
import cv2
import tqdm
from typing import Dict, Any
from .paths import json_dir, tile_anno_dir, wsi_anno_dir

def _load_type_info(type_info_path: str) -> Dict[str, Any]:
    with open(type_info_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_tile_and_compose_wsi(out_root: str, sample_name: str, image_width: int, image_height: int, type_info_path: str):
    type_info = _load_type_info(type_info_path)
    jdir = json_dir(out_root, sample_name)
    tdir = tile_anno_dir(out_root, sample_name)
    wdir = wsi_anno_dir(out_root, sample_name)
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(wdir, exist_ok=True)

    for fn in tqdm.tqdm(os.listdir(jdir), ncols=100, file=sys.stdout, desc="Tile annotation"):
        if not fn.endswith(".json"):
            continue
        stem = os.path.splitext(fn)[0]
        out_png = os.path.join(tdir, stem + "_cell_annotation.png")
        if os.path.exists(out_png):
            continue

        tile_w = tile_h = 1024
        try:
            coord_str = stem.replace("tile_", "")
            left, top, right, bottom = map(int, coord_str.split("_"))
            tile_w = max(1, right - left)
            tile_h = max(1, bottom - top)
        except Exception:
            pass

        with open(os.path.join(jdir, fn), "r", encoding="utf-8") as f:
            res = json.load(f)
        cells = res.get("nuc", {})
        if not isinstance(cells, dict) or not cells:
            continue

        tile_img = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        for _, cdata in cells.items():
            if not isinstance(cdata, dict):
                continue
            ctype = str(cdata.get("type", "0"))
            if ctype not in type_info:
                continue
            rgb = type_info[ctype][1]
            color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            contour = cdata.get("contour")
            if contour is None:
                continue
            pts = np.asarray(contour, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                continue
            xs = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, tile_w - 1)
            ys = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, tile_h - 1)
            poly = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(tile_img, [poly], True, color, 2)

        cv2.imwrite(out_png, tile_img)

    canvas = np.full((image_height, image_width, 3), (255, 255, 255), dtype=np.uint8)
    for fname in tqdm.tqdm(os.listdir(tdir), ncols=100, file=sys.stdout, desc="WSI compose"):
        if not fname.endswith(".png"):
            continue
        coords_str = fname.replace("tile_", "").replace("_cell_annotation.png", "")
        try:
            left, top, right, bottom = map(int, coords_str.split("_"))
        except Exception:
            continue
        if left < 0 or top < 0 or right > image_width or bottom > image_height:
            continue
        tile = cv2.imread(os.path.join(tdir, fname))
        if tile is None:
            continue
        tile_h, tile_w = bottom - top, right - left
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
            tile = cv2.resize(tile, (tile_w, tile_h))
        canvas[top:bottom, left:right] = tile

    out_jpg = os.path.join(wdir, f"{sample_name}_cell_annotation.jpg")
    cv2.imwrite(out_jpg, canvas)
    return out_jpg
