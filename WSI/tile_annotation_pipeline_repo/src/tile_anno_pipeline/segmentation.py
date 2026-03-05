from __future__ import annotations
import os, sys
import tqdm
from PIL import Image

try:
    import openslide
except Exception:
    openslide = None

def read_slide_as_rgb(slide_path: str, level: int = 0) -> Image.Image:
    if openslide is not None:
        try:
            slide = openslide.OpenSlide(slide_path)
            level = max(0, min(level, slide.level_count - 1))
            return slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
        except Exception:
            pass
    return Image.open(slide_path).convert("RGB")

def split_to_tiles(slide_path: str, out_tile_dir: str, tile_size: int = 1024, level: int = 0) -> tuple[int, int]:
    os.makedirs(out_tile_dir, exist_ok=True)
    img = read_slide_as_rgb(slide_path, level=level)
    w, h = img.size
    for y in tqdm.tqdm(range(0, h, tile_size), ncols=100, file=sys.stdout, desc="Tiling"):
        for x in range(0, w, tile_size):
            left, top = x, y
            right = min(x + tile_size, w)
            bottom = min(y + tile_size, h)
            tile = img.crop((left, top, right, bottom))
            tile.save(os.path.join(out_tile_dir, f"tile_{left}_{top}_{right}_{bottom}.tif"), format="TIFF")
    return w, h

def tiles_exist(tile_dir: str) -> bool:
    return os.path.isdir(tile_dir) and any(fn.startswith("tile_") and fn.endswith(".tif") for fn in os.listdir(tile_dir))

def recover_wh_from_tiles(tile_dir: str) -> tuple[int | None, int | None]:
    max_r, max_b = 0, 0
    if not os.path.isdir(tile_dir):
        return None, None
    for fn in os.listdir(tile_dir):
        if not (fn.startswith("tile_") and fn.endswith(".tif")):
            continue
        try:
            stem = fn.split(".")[0]
            _, l, t, r, b = stem.split("_")
            max_r = max(max_r, int(r))
            max_b = max(max_b, int(b))
        except Exception:
            continue
    return (max_r or None), (max_b or None)
