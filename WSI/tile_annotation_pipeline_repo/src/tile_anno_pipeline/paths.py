from __future__ import annotations
import os

def sample_root(out_root: str, sample_name: str) -> str:
    return os.path.join(out_root, sample_name)

def tile_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(sample_root(out_root, sample_name), "0.tile_1024")

def hover_root(out_root: str, sample_name: str) -> str:
    return os.path.join(sample_root(out_root, sample_name), "1.cell_hovernet")

def json_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(hover_root(out_root, sample_name), "json")

def tile_anno_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(hover_root(out_root, sample_name), "0.annotation_tile")

def wsi_anno_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(hover_root(out_root, sample_name), "0.annotation_wsi")

def wsi_plot_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(hover_root(out_root, sample_name), "1.pie_chart_wsi_plot")

def tile_plot_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(hover_root(out_root, sample_name), "2.pie_chart_tile_plot")

def features_dir(out_root: str, sample_name: str) -> str:
    return os.path.join(hover_root(out_root, sample_name), "3.cell_features")

def meta_path(out_root: str, sample_name: str) -> str:
    return os.path.join(sample_root(out_root, sample_name), "meta.json")
