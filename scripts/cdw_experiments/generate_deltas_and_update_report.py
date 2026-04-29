#!/usr/bin/env python3
"""Generate delta GeoTIFFs for original-method CHMs and update the experiment report.

Writes delta rasters next to the CHMs in `data/lamapuit/chm_original_filtered/` and
appends corresponding rows into `analysis/cdw_experiments_436646/cdw_experiment_report.json`
and `analysis/cdw_experiments_436646/cdw_experiment_report.csv`.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


CHM_DIR = Path("data/lamapuit/chm_original_filtered")
BASELINE = Path("data/lamapuit/chm_max_hag_13_drop/436646_2018_madal_chm_max_hag_20cm.tif")
REPORT_JSON = Path("analysis/cdw_experiments_436646/cdw_experiment_report.json")
REPORT_CSV = Path("analysis/cdw_experiments_436646/cdw_experiment_report.csv")


def find_chms(target_dir: Path) -> list[Path]:
    if not target_dir.exists():
        return []
    return sorted([p for p in target_dir.glob("*_chm_max_hag_*_*.tif") if p.is_file()])


def compute_and_write_delta(chm_path: Path, baseline_path: Path) -> Path | None:
    with rasterio.open(chm_path) as src_chm, rasterio.open(baseline_path) as src_base:
        if src_chm.width != src_base.width or src_chm.height != src_base.height:
            print("Size mismatch, skipping:", chm_path.name)
            return None
        if src_chm.transform != src_base.transform:
            print("Transform mismatch, skipping:", chm_path.name)
            return None

        chm_arr = src_chm.read(1).astype("float32")
        base_arr = src_base.read(1).astype("float32")

        chm_nd = src_chm.nodata if src_chm.nodata is not None else -9999.0
        base_nd = src_base.nodata if src_base.nodata is not None else -9999.0

        valid = (chm_arr != chm_nd) & (base_arr != base_nd) & np.isfinite(chm_arr) & np.isfinite(base_arr)
        delta = np.full_like(chm_arr, chm_nd, dtype="float32")
        delta[valid] = chm_arr[valid] - base_arr[valid]

        # derive mode and resolution tokens
        name = chm_path.name
        m = re.search(r"_(all|last2|last)\.tif$", name)
        mode = m.group(1) if m else "unknown"
        m2 = re.search(r"_(\d+)cm_", name)
        res_cm = m2.group(1) if m2 else str(int(abs(src_chm.transform.a) * 100))

        delta_name = chm_path.stem + f"_delta_vs_original_original_method_{mode}_{res_cm}cm.tif"
        delta_path = chm_path.parent / delta_name

        profile = src_chm.profile.copy()
        profile.update(dtype="float32", count=1, nodata=chm_nd, compress="lzw", tiled=True, blockxsize=256, blockysize=256)
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(delta_path, "w", **profile) as dst:
            dst.write(delta, 1)

    print("Wrote delta:", delta_path)
    return delta_path


def load_report(json_path: Path) -> dict:
    if not json_path.exists():
        return {"generated_at": None, "rows": [], "total_rows": 0, "ok_rows": 0, "error_rows": 0}
    return json.loads(json_path.read_text(encoding="utf-8"))


def append_report_rows(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    report = load_report(json_path)
    existing = report.get("rows", [])
    # avoid duplicates by single_chm_tif
    existing_chms = {r.get("single_chm_tif"): r for r in existing}

    for r in rows:
        key = r.get("single_chm_tif")
        if key in existing_chms:
            # update delta if present
            existing_chms[key]["delta_vs_original_tif"] = r.get("delta_vs_original_tif", "")
            continue
        existing.append(r)

    report["rows"] = existing
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["total_rows"] = len(existing)
    report["ok_rows"] = sum(1 for rr in existing if rr.get("status") == "ok")
    report["error_rows"] = sum(1 for rr in existing if rr.get("status") == "error")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Append to CSV (preserve header order if available)
    header = None
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
    else:
        header = [
            "status",
            "tile_id",
            "year",
            "dtm_variant",
            "return_mode",
            "input_laz",
            "single_chm_tif",
            "intensity_tif",
            "density_tif",
            "split_rgb_tif",
            "baseline_tif",
            "delta_vs_original_tif",
            "reference_surface_tif",
            "fused_surface_tif",
            "fallback_mask_tif",
            "fallback_trigger_fraction",
            "single_chm_valid_px",
            "band1_valid_px",
            "band2_valid_px",
            "band3_valid_px",
            "band4_valid_px",
            "band5_valid_px",
            "intensity_points_used",
            "intensity_min",
            "intensity_max",
            "error",
        ]

    # Write header if CSV didn't exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    # Append new rows (only those not duplicated)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        for r in rows:
            # If this single_chm already existed in CSV, skip append to avoid duplicate lines
            skip = False
            # crude check: search for line containing the single_chm path
            with csv_path.open("r", encoding="utf-8") as fr:
                if any(r.get("single_chm_tif", "") in line for line in fr):
                    skip = True
            if skip:
                continue
            # ensure all header fields exist in r
            out = {k: r.get(k, "") for k in header}
            writer.writerow(out)


def main() -> int:
    chms = find_chms(CHM_DIR)
    if not chms:
        print("No CHMs found in:", CHM_DIR)
        return 1
    if not BASELINE.exists():
        print("Baseline missing:", BASELINE)
        return 2

    new_rows = []
    for chm in chms:
        delta = compute_and_write_delta(chm, BASELINE)
        if delta is None:
            continue

        # compute some basic stats for CSV/JSON
        with rasterio.open(chm) as src:
            arr = src.read(1)
            chm_nd = src.nodata if src.nodata is not None else -9999.0
            valid_px = int((arr != chm_nd).sum())

        # derive return mode
        m = re.search(r"_(all|last2|last)\.tif$", chm.name)
        mode = m.group(1) if m else "unknown"

        row = {
            "status": "ok",
            "tile_id": "436646",
            "year": "2018",
            "dtm_variant": "original_method",
            "return_mode": mode,
            "input_laz": "data/lamapuit/laz/436646_2018_madal.laz",
            "single_chm_tif": str(chm),
            "intensity_tif": "",
            "density_tif": "",
            "split_rgb_tif": "",
            "baseline_tif": str(BASELINE),
            "delta_vs_original_tif": str(delta),
            "reference_surface_tif": "",
            "fused_surface_tif": "",
            "fallback_mask_tif": "",
            "fallback_trigger_fraction": 0.0,
            "single_chm_valid_px": valid_px,
            "band1_valid_px": "",
            "band2_valid_px": "",
            "band3_valid_px": "",
            "band4_valid_px": "",
            "band5_valid_px": "",
            "intensity_points_used": "",
            "intensity_min": "",
            "intensity_max": "",
            "error": "",
        }
        new_rows.append(row)

    if new_rows:
        append_report_rows(new_rows, REPORT_JSON, REPORT_CSV)
        print(f"Appended {len(new_rows)} rows to report: {REPORT_JSON} and {REPORT_CSV}")
    else:
        print("No new rows to append")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
