from __future__ import annotations
import os, re, glob, json
import numpy as np
import pandas as pd
from skimage.draw import polygon as draw_polygon
from skimage.measure import regionprops

_TILE_PAT = re.compile(r"tile_(\d+)_(\d+)_(\d+)_(\d+)\.(?:json|tif|png|jpg)$")

def _parse_tile_box(name: str):
    m = _TILE_PAT.search(os.path.basename(name))
    if not m:
        return None
    return tuple(map(int, m.groups()))

def _read_tile_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cells = []
    def add(it):
        if not isinstance(it, dict):
            return
        pts = it.get("contour") or it.get("points") or it.get("poly")
        if pts is None:
            return
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
            return
        ctype = it.get("type", None)
        try:
            ctype = int(ctype) if ctype is not None else None
        except Exception:
            ctype = None
        cells.append({"contour": arr, "type": ctype})
    if isinstance(data, dict):
        items = data.get("nuc") or data.get("instances") or data.get("cells")
        if isinstance(items, dict):
            for v in items.values():
                add(v)
        elif isinstance(items, list):
            for v in items:
                add(v)
    elif isinstance(data, list):
        for v in data:
            add(v)
    return cells

def _instances_from_cells(cells, tile_h, tile_w):
    label_map = np.zeros((tile_h, tile_w), dtype=np.uint16)
    label_type = {}
    cur = 1
    for cell in cells:
        arr = cell["contour"]
        ctype = cell.get("type", None)
        xs = np.clip(np.round(arr[:, 0]).astype(np.int32), 0, tile_w - 1)
        ys = np.clip(np.round(arr[:, 1]).astype(np.int32), 0, tile_h - 1)
        rr, cc = draw_polygon(ys, xs, shape=(tile_h, tile_w))
        mask = (label_map[rr, cc] == 0)
        if not np.any(mask):
            continue
        label_map[rr[mask], cc[mask]] = cur
        label_type[cur] = ctype
        cur += 1
    return label_map, label_type

def compute_tile_features(json_path: str, default_tile_size: int = 1024):
    box = _parse_tile_box(json_path)
    if box is not None:
        l, t, r, b = box
        tile_w = max(1, r - l); tile_h = max(1, b - t)
    else:
        tile_w = tile_h = default_tile_size
    cells = _read_tile_json(json_path)
    if not cells:
        return None
    lbl, label_type = _instances_from_cells(cells, tile_h, tile_w)
    props = regionprops(lbl)
    if not props:
        return None

    rows = []
    for p in props:
        ctype = label_type.get(p.label, None)
        try:
            ctype = int(ctype) if ctype is not None else np.nan
        except Exception:
            ctype = np.nan
        eig = getattr(p, "inertia_tensor_eigvals", None)
        if eig is None or not np.all(np.isfinite(eig)):
            eig_x = np.nan; eig_y = np.nan
        else:
            eig_x = float(eig[0]); eig_y = float(eig[1])
        rows.append({
            "label": int(p.label),
            "cell_type": ctype,
            "area": float(p.area),
            "bbox_area": float((p.bbox[2]-p.bbox[0])*(p.bbox[3]-p.bbox[1])),
            "convex_area": float(getattr(p, "convex_area", np.nan)),
            "eccentricity": float(getattr(p, "eccentricity", np.nan)),
            "equivalent_diameter": float(getattr(p, "equivalent_diameter", np.nan)),
            "euler_number": float(getattr(p, "euler_number", np.nan)),
            "extent": float(getattr(p, "extent", np.nan)),
            "filled_area": float(getattr(p, "filled_area", np.nan)),
            "inertia_tensor_eigvals_x": eig_x,
            "inertia_tensor_eigvals_y": eig_y,
            "major_axis_length": float(getattr(p, "major_axis_length", np.nan)),
            "minor_axis_length": float(getattr(p, "minor_axis_length", np.nan)),
            "perimeter": float(getattr(p, "perimeter", np.nan)),
            "solidity": float(getattr(p, "solidity", np.nan)),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
        })
    return pd.DataFrame(rows)

WSI_BASE_FEATURES = [
    "area","bbox_area","convex_area","eccentricity","equivalent_diameter","euler_number","extent","filled_area",
    "inertia_tensor_eigvals_x","inertia_tensor_eigvals_y","major_axis_length","minor_axis_length","perimeter","solidity",
]

def compute_cell_features_for_sample(out_root: str, sample_name: str, min_cells: int = 10):
    jdir = os.path.join(out_root, sample_name, "1.cell_hovernet", "json")
    out_dir = os.path.join(out_root, sample_name, "1.cell_hovernet", "3.cell_features")
    os.makedirs(out_dir, exist_ok=True)
    for jpath in sorted(glob.glob(os.path.join(jdir, "tile_*.json"))):
        out_csv = os.path.join(out_dir, os.path.splitext(os.path.basename(jpath))[0] + ".csv")
        if os.path.exists(out_csv):
            continue
        df = compute_tile_features(jpath)
        if df is None or len(df) < min_cells:
            continue
        df.to_csv(out_csv, index=False, encoding="utf-8")

def build_wsi_morphology_features_for_sample(out_root: str, sample_name: str):
    feat_dir = os.path.join(out_root, sample_name, "1.cell_hovernet", "3.cell_features")
    out_csv = os.path.join(out_root, sample_name, "1.cell_hovernet", f"{sample_name}_wsi_morphology_350d.csv")
    if os.path.exists(out_csv):
        return out_csv
    csv_list = sorted(glob.glob(os.path.join(feat_dir, "tile_*.csv")))
    if not csv_list:
        return ""
    dfs = []
    for p in csv_list:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    if not dfs:
        return ""
    df_all = pd.concat(dfs, ignore_index=True)
    if "cell_type" not in df_all.columns:
        return ""
    df_all["cell_type"] = pd.to_numeric(df_all["cell_type"], errors="coerce")
    df_all = df_all.dropna(subset=["cell_type"])
    df_all["cell_type"] = df_all["cell_type"].astype(int)

    rows = {}
    for ctype in sorted(df_all["cell_type"].unique()):
        sub = df_all[df_all["cell_type"] == ctype]
        for feat in WSI_BASE_FEATURES:
            if feat not in sub.columns:
                continue
            v = pd.to_numeric(sub[feat], errors="coerce").dropna()
            if v.empty:
                continue
            rows[f"type{ctype}_{feat}_mean"] = float(v.mean())
            rows[f"type{ctype}_{feat}_median"] = float(v.median())
            rows[f"type{ctype}_{feat}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rows[f"type{ctype}_{feat}_q25"] = float(v.quantile(0.25))
            rows[f"type{ctype}_{feat}_q75"] = float(v.quantile(0.75))
    pd.DataFrame([rows]).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv
