from __future__ import annotations
import json, os
from typing import Any, Dict
from .paths import meta_path

DEFAULT_META = {
    "segmentation_done": False,
    "cell_infer_done": False,
    "wsi_compose_done": False,
    "wsi_pie_done": False,
    "tile_pie_done": False,
    "image_width": None,
    "image_height": None,
    "cell_features_done": False,
}

def load_meta(out_root: str, sample_name: str) -> Dict[str, Any]:
    p = meta_path(out_root, sample_name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        merged = dict(DEFAULT_META)
        merged.update(m)
        return merged
    return dict(DEFAULT_META)

def save_meta(out_root: str, sample_name: str, meta: Dict[str, Any]) -> None:
    p = meta_path(out_root, sample_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
