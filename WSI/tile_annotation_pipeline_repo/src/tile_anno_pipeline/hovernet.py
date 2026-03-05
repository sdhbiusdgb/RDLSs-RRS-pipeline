from __future__ import annotations
import os, subprocess, shutil
from .config import HoverNetConfig
from .paths import tile_dir, hover_root

def run_hovernet_infer(cfg: HoverNetConfig, out_root: str, sample_name: str, gpu_id: int) -> None:
    in_dir = tile_dir(out_root, sample_name)
    out_dir = hover_root(out_root, sample_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", cfg.hovernet_infer_script,
        f"--gpu={gpu_id}",
        f"--nr_types={cfg.nr_types}",
        f"--type_info_path={cfg.type_info_path}",
        f"--model_path={cfg.model_path}",
        f"--model_mode={cfg.model_mode}",
        f"--nr_inference_workers={cfg.nr_inference_workers}",
        f"--nr_post_proc_workers={cfg.nr_post_proc_workers}",
        f"--batch_size={cfg.batch_size}",
        "tile",
        f"--input_dir={in_dir}",
        f"--output_dir={out_dir}",
        f"--mem_usage={cfg.mem_usage}",
    ]
    if cfg.draw_dot:
        cmd.append("--draw_dot")
    if cfg.save_raw_map:
        cmd.append("--save_raw_map")

    log_file = os.path.join(out_dir, f"{sample_name}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        p = subprocess.run(" ".join(cmd), shell=True, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"HoVer-Net inference failed for {sample_name}. See log: {log_file}")

    mat_dir = os.path.join(out_dir, "mat")
    if os.path.isdir(mat_dir):
        shutil.rmtree(mat_dir)
