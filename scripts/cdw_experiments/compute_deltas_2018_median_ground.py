#!/usr/bin/env python3
"""
Compute delta rasters for CHMs in median_ground 2018 directory versus baseline.
"""
from pathlib import Path
import sys
import numpy as np
import rasterio

NODATA = -9999.0
# Directory created by the focused median_ground CHM run
target_dir = Path("data/lamapuit/cdw_experiments_436646_2018_ground_last_vs_last2/median_ground")
baseline_dir = Path("data/lamapuit/chm_max_hag_13_drop")

if not target_dir.exists():
    print("Target dir missing:", target_dir)
    sys.exit(1)

count = 0
for chm in sorted(target_dir.glob("*_exp_return_chm13_*.tif")):
    name = chm.name
    laz_stem = name.split("_exp_return_chm13")[0]
    parts = name.split("_")
    if len(parts) < 2:
        print("Skipping (unexpected name):", name)
        continue
    mode = parts[-2]
    res_token = parts[-1]
    if not res_token.endswith("cm.tif"):
        print("Skipping (unexpected res token):", name)
        continue
    res_cm = res_token.replace(".tif", "").replace("cm", "")

    baseline = baseline_dir / (laz_stem + "_chm_max_hag_" + res_cm + "cm.tif")
    delta_path = target_dir / (laz_stem + "_delta_vs_original_" + target_dir.name + "_" + mode + "_" + res_cm + "cm.tif")

    if not baseline.exists():
        print("Baseline missing, skipping:", baseline)
        continue

    with rasterio.open(chm) as src_chm, rasterio.open(baseline) as src_base:
        if src_chm.width != src_base.width or src_chm.height != src_base.height:
            print("Size mismatch, skipping:", chm.name)
            continue
        if src_chm.transform != src_base.transform:
            print("Transform mismatch, skipping:", chm.name)
            continue

        chm_arr = src_chm.read(1).astype("float32")
        base_arr = src_base.read(1).astype("float32")
        chm_nd = src_chm.nodata if src_chm.nodata is not None else NODATA
        base_nd = src_base.nodata if src_base.nodata is not None else NODATA

        valid = (chm_arr != chm_nd) & (base_arr != base_nd)
        delta = np.full_like(chm_arr, chm_nd, dtype="float32")
        delta[valid] = chm_arr[valid] - base_arr[valid]

        profile = src_chm.profile.copy()
        profile.update(dtype="float32", count=1, nodata=chm_nd, compress="lzw", tiled=True, blockxsize=256, blockysize=256)
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(delta_path, "w", **profile) as dst:
            dst.write(delta, 1)

    print("Wrote delta:", delta_path)
    count += 1

print("Done. Deltas written:", count)
