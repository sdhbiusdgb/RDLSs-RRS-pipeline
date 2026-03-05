from __future__ import annotations
import os, sys
import tqdm
from .config import AppConfig
from .io.slide_resolver import SlidePathResolver
from .paths import tile_dir, json_dir, features_dir
from .meta import load_meta, save_meta
from .segmentation import split_to_tiles, tiles_exist, recover_wh_from_tiles
from .hovernet import run_hovernet_infer
from .annotation import render_tile_and_compose_wsi
from .plots import plot_wsi_celltype_distribution, plot_tile_pies_and_compose
from .morphology import compute_cell_features_for_sample, build_wsi_morphology_features_for_sample

def _json_non_empty(jdir: str) -> bool:
    return os.path.isdir(jdir) and any(fn.endswith(".json") for fn in os.listdir(jdir))

def _features_non_empty(fdir: str) -> bool:
    return os.path.isdir(fdir) and any(fn.endswith(".csv") for fn in os.listdir(fdir))

def iter_samples(raw_image_dir: str):
    for name in sorted(os.listdir(raw_image_dir)):
        p = os.path.join(raw_image_dir, name)
        if os.path.isdir(p):
            yield name

def segment_dataset(cfg: AppConfig) -> None:
    ds = cfg.dataset
    pl = cfg.pipeline
    resolver = SlidePathResolver(cfg.io.resolver, cfg.io.slide_filename)
    for sample_name in tqdm.tqdm(list(iter_samples(ds.raw_image_dir)), ncols=100, file=sys.stdout, desc=f"{ds.dataset_name}-segment"):
        meta = load_meta(ds.CellAnnotation_dir, sample_name)
        tdir = tile_dir(ds.CellAnnotation_dir, sample_name)
        if meta.get("segmentation_done") and tiles_exist(tdir):
            continue
        if tiles_exist(tdir):
            meta["segmentation_done"] = True
            if not (meta.get("image_width") and meta.get("image_height")):
                w, h = recover_wh_from_tiles(tdir)
                meta["image_width"], meta["image_height"] = w, h
            save_meta(ds.CellAnnotation_dir, sample_name, meta)
            continue
        slide_path = resolver.resolve(ds.raw_image_dir, sample_name, ds.data_type)
        if not os.path.exists(slide_path):
            continue
        w, h = split_to_tiles(slide_path, tdir, tile_size=pl.tile_size, level=0)
        meta["segmentation_done"] = True
        meta["image_width"] = int(w); meta["image_height"] = int(h)
        save_meta(ds.CellAnnotation_dir, sample_name, meta)

def analyze_dataset(cfg: AppConfig) -> None:
    ds = cfg.dataset
    pl = cfg.pipeline
    for sample_name in tqdm.tqdm(list(iter_samples(ds.raw_image_dir)), ncols=100, file=sys.stdout, desc=f"{ds.dataset_name}-analyze"):
        meta = load_meta(ds.CellAnnotation_dir, sample_name)
        tdir = tile_dir(ds.CellAnnotation_dir, sample_name)
        if not tiles_exist(tdir):
            continue
        if not (meta.get("image_width") and meta.get("image_height")):
            w, h = recover_wh_from_tiles(tdir)
            if not (w and h):
                continue
            meta["image_width"], meta["image_height"] = int(w), int(h)
            save_meta(ds.CellAnnotation_dir, sample_name, meta)

        jdir = json_dir(ds.CellAnnotation_dir, sample_name)
        if not (meta.get("cell_infer_done") and _json_non_empty(jdir)):
            if _json_non_empty(jdir):
                meta["cell_infer_done"] = True
                save_meta(ds.CellAnnotation_dir, sample_name, meta)
            else:
                run_hovernet_infer(cfg.hovernet, ds.CellAnnotation_dir, sample_name, ds.gpu_id)
                if not _json_non_empty(jdir):
                    raise RuntimeError(f"No HoVer-Net JSON outputs found for {sample_name}: {jdir}")
                meta["cell_infer_done"] = True
                save_meta(ds.CellAnnotation_dir, sample_name, meta)

        fdir = features_dir(ds.CellAnnotation_dir, sample_name)
        if not (meta.get("cell_features_done") and _features_non_empty(fdir)):
            compute_cell_features_for_sample(ds.CellAnnotation_dir, sample_name, min_cells=pl.min_cells_for_features)
            meta["cell_features_done"] = True
            save_meta(ds.CellAnnotation_dir, sample_name, meta)

        build_wsi_morphology_features_for_sample(ds.CellAnnotation_dir, sample_name)

        if not meta.get("wsi_compose_done"):
            render_tile_and_compose_wsi(ds.CellAnnotation_dir, sample_name, int(meta["image_width"]), int(meta["image_height"]), cfg.hovernet.type_info_path)
            meta["wsi_compose_done"] = True
            save_meta(ds.CellAnnotation_dir, sample_name, meta)

        if not meta.get("wsi_pie_done"):
            plot_wsi_celltype_distribution(ds.CellAnnotation_dir, sample_name, cfg.hovernet.type_info_path)
            meta["wsi_pie_done"] = True
            save_meta(ds.CellAnnotation_dir, sample_name, meta)

        if not meta.get("tile_pie_done"):
            plot_tile_pies_and_compose(ds.CellAnnotation_dir, sample_name, int(meta["image_height"]), int(meta["image_width"]), cfg.hovernet.type_info_path)
            meta["tile_pie_done"] = True
            save_meta(ds.CellAnnotation_dir, sample_name, meta)

def run_all(cfg: AppConfig) -> None:
    segment_dataset(cfg)
    analyze_dataset(cfg)
