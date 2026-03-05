from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class DatasetConfig:
    dataset_name: str
    raw_image_dir: str
    CellAnnotation_dir: str
    data_type: str = "tif"
    gpu_id: int = 0


@dataclass
class HoverNetConfig:
    hovernet_infer_script: str
    model_path: str
    type_info_path: str
    model_mode: str = "fast"
    nr_types: int = 6
    batch_size: int = 1
    mem_usage: float = 0.1
    nr_inference_workers: int = 0
    nr_post_proc_workers: int = 0
    draw_dot: bool = True
    save_raw_map: bool = False


@dataclass
class PipelineConfig:
    tile_size: int = 1024
    min_cells_for_features: int = 10


@dataclass
class IOConfig:
    resolver: str = "simple"
    slide_filename: str = "slide"


@dataclass
class AppConfig:
    dataset: DatasetConfig
    hovernet: HoverNetConfig
    pipeline: PipelineConfig = PipelineConfig()
    io: IOConfig = IOConfig()


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return AppConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        hovernet=HoverNetConfig(**raw.get("hovernet", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        io=IOConfig(**raw.get("io", {})),
    )
