from __future__ import annotations
import os, json, sys
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import tqdm
from .paths import json_dir, wsi_plot_dir, tile_plot_dir

def _load_type_info(type_info_path: str):
    with open(type_info_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_wsi_celltype_distribution(out_root: str, sample_name: str, type_info_path: str):
    type_info = _load_type_info(type_info_path)
    outdir = wsi_plot_dir(out_root, sample_name)
    os.makedirs(outdir, exist_ok=True)
    jdir = json_dir(out_root, sample_name)

    cell_type_names = []
    for tile_json in os.listdir(jdir):
        if not tile_json.endswith(".json"):
            continue
        with open(os.path.join(jdir, tile_json), "r", encoding="utf-8") as f:
            res = json.load(f)
        for v in (res.get("nuc", {}) or {}).values():
            if isinstance(v, dict) and "type" in v:
                t = str(v["type"])
                if t in type_info:
                    cell_type_names.append(type_info[t][0])

    if not cell_type_names:
        return "", ""

    counts = Counter(cell_type_names)
    labels = list(counts.keys())
    sizes = list(counts.values())
    total = sum(sizes)
    percentages = [c / total * 100 for c in sizes]
    color_map = {type_info[k][0]: [v / 255 for v in type_info[k][1]] for k in type_info.keys()}
    colors = [color_map.get(lbl, [0.5, 0.5, 0.5]) for lbl in labels]

    plt.figure(figsize=(6, 6))
    wedges, _ = plt.pie(sizes, startangle=90, colors=colors, wedgeprops={"edgecolor": "white"})
    plt.legend(wedges, [f"{l} ({p:.1f}%)" for l, p in zip(labels, percentages)],
               title="Cell Type", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.title("Cell Type Composition")
    plt.tight_layout()
    pie_path = os.path.join(outdir, f"{sample_name}_pie.pdf")
    plt.savefig(pie_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, percentages, color=colors, edgecolor="black")
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xlabel("Cell Type")
    plt.ylabel("Percentage (%)")
    plt.title("Cell Type Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    bar_path = os.path.join(outdir, f"{sample_name}_bar_chart.pdf")
    plt.savefig(bar_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    return pie_path, bar_path

def plot_tile_pies_and_compose(out_root: str, sample_name: str, image_height: int, image_width: int, type_info_path: str):
    type_info = _load_type_info(type_info_path)
    outdir = tile_plot_dir(out_root, sample_name)
    os.makedirs(outdir, exist_ok=True)
    jdir = json_dir(out_root, sample_name)

    color_map = {type_info[k][0]: [v / 255 for v in type_info[k][1]] for k in type_info.keys()}

    for tile_json in os.listdir(jdir):
        if not tile_json.endswith(".json"):
            continue
        sample_id = os.path.splitext(tile_json)[0]
        save_path = os.path.join(outdir, f"{sample_id}_pie.png")
        if os.path.exists(save_path):
            continue
        with open(os.path.join(jdir, tile_json), "r", encoding="utf-8") as f:
            res = json.load(f)
        cells = res.get("nuc", {}) or {}
        if not cells:
            continue
        types = []
        for v in cells.values():
            if isinstance(v, dict) and "type" in v:
                t = str(v["type"])
                if t in type_info:
                    types.append(type_info[t][0])
        if not types:
            continue

        counts = Counter(types)
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = [color_map.get(lbl, [0.5, 0.5, 0.5]) for lbl in labels]

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, startangle=90, colors=colors, wedgeprops={"edgecolor": "white"})
        plt.tight_layout()
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        Image.open(save_path).resize((1024, 1024), Image.LANCZOS).save(save_path, format="PNG")

    canvas = np.full((image_height, image_width, 3), (255, 255, 255), dtype=np.uint8)
    for fname in tqdm.tqdm(os.listdir(outdir), ncols=100, file=sys.stdout, desc="Compose tile pies"):
        if not fname.endswith(".png"):
            continue
        coords_str = fname.replace("tile_", "").replace("_pie.png", "")
        try:
            left, top, right, bottom = map(int, coords_str.split("_"))
        except Exception:
            continue
        tile = cv2.imread(os.path.join(outdir, fname))
        if tile is None:
            continue
        tile_h, tile_w = bottom - top, right - left
        if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
            tile = cv2.resize(tile, (tile_w, tile_h))
        if 0 <= top < bottom <= image_height and 0 <= left < right <= image_width:
            canvas[top:bottom, left:right] = tile

    out_path = os.path.join(outdir, f"{sample_name}_tile_pie_composed.jpg")
    cv2.imwrite(out_path, canvas)
    return out_path
