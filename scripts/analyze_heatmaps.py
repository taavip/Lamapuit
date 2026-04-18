#!/usr/bin/env python3
"""Analyze generated heatmaps for correlation with CHM tile heights.
Reads meta.json in output/heatmap_debug/*, loads CHM tile and heatmaps, computes stats.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window

OUT = Path("output/heatmap_debug")
CHM_DIR = Path("chm_max_hag")

methods = ["HiResCAM", "GradCAM+", "IntGrad", "RISE"]

results = []
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    meta_f = d / "meta.json"
    if not meta_f.exists():
        continue
    meta = json.loads(meta_f.read_text())
    raster = meta["raster"]
    row_off = int(meta["row_off"])
    col_off = int(meta["col_off"])
    chm_path = CHM_DIR / raster
    if not chm_path.exists():
        print("missing chm", chm_path)
        continue
    with rasterio.open(chm_path) as src:
        raw = src.read(
            1, window=Window(col_off, row_off, 128, 128), boundless=True, fill_value=0
        ).astype(float)
    # normalize raw
    rn = np.clip(raw, 0.0, 20.0) / 20.0
    rn_flat = rn.ravel()
    for m in methods:
        hm_f = d / f"heat_{m}.png"
        if not hm_f.exists():
            continue
        hm = cv2.imread(str(hm_f), cv2.IMREAD_GRAYSCALE).astype(float)
        # ensure same shape
        if hm.shape != rn.shape:
            print("shape mismatch", d.name, m, hm.shape, rn.shape)
        hm_flat = hm.ravel() / 255.0
        # compute pearson corr where raw has >0
        mask = ~np.isnan(rn_flat)
        if mask.sum() == 0:
            corr = None
        else:
            try:
                corr = np.corrcoef(rn_flat[mask], hm_flat[mask])[0, 1]
            except Exception:
                corr = None
        results.append(
            {
                "tile": d.name,
                "method": m,
                "mean_hm": float(hm.mean()),
                "max_hm": float(hm.max()),
                "corr": None if corr is None else float(corr),
            }
        )

# print summary
from statistics import mean

by_method = {}
for r in results:
    by_method.setdefault(r["method"], []).append(r)
for m, arr in by_method.items():
    corrs = [a["corr"] for a in arr if a["corr"] is not None]
    print(
        m,
        "n=",
        len(arr),
        "mean_hm=",
        sum(a["mean_hm"] for a in arr) / len(arr),
        "mean_max=",
        sum(a["max_hm"] for a in arr) / len(arr),
        "mean_corr=",
        None if not corrs else sum(corrs) / len(corrs),
    )

# print low-activity tiles
for r in results:
    if r["mean_hm"] < 1.0:  # very low heatmap values
        print("low activity", r)
