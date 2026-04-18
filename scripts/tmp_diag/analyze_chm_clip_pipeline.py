#!/usr/bin/env python3
"""Deep audit: verify CHM clipping integrity (0..1.3 m) and LAZ/CHM consistency."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

HAG_FIELDS = [
    "HeightAboveGround",
    "height_above_ground",
    "heightaboveground",
    "HAG",
    "hag",
]


@dataclass
class ChmStats:
    raster: str
    width: int
    height: int
    nodata: float | None
    valid_pixels: int
    min_val: float | None
    max_val: float | None
    p95: float | None
    p99: float | None
    above_clip_count: int
    above_clip_pct: float
    tag_hag_max: str


def _safe_pct(a: np.ndarray, q: float) -> float | None:
    if a.size == 0:
        return None
    return float(np.percentile(a, q))


def analyze_chm_file(chm_path: Path, clip_max: float) -> ChmStats:
    with rasterio.open(chm_path) as src:
        arr = src.read(1, masked=True)
        valid = arr.compressed().astype(np.float32)
        tags = src.tags() or {}
        above = int(np.sum(valid > clip_max)) if valid.size else 0
        pct = (100.0 * above / valid.size) if valid.size else 0.0
        return ChmStats(
            raster=chm_path.name,
            width=int(src.width),
            height=int(src.height),
            nodata=float(src.nodata) if src.nodata is not None else None,
            valid_pixels=int(valid.size),
            min_val=float(valid.min()) if valid.size else None,
            max_val=float(valid.max()) if valid.size else None,
            p95=_safe_pct(valid, 95),
            p99=_safe_pct(valid, 99),
            above_clip_count=above,
            above_clip_pct=pct,
            tag_hag_max=str(tags.get("HAG_MAX", "")),
        )


def _find_laz_for_chm(chm_name: str, laz_dir: Path) -> Path | None:
    suffix = "_chm_max_hag_20cm.tif"
    stem = chm_name[:-len(suffix)] if chm_name.endswith(suffix) else chm_name
    candidate = laz_dir / f"{stem}.laz"
    return candidate if candidate.exists() else None


def analyze_laz_hag(laz_path: Path, clip_max: float, sample_step: int) -> dict:
    import laspy

    with laspy.open(str(laz_path)) as fh:
        las = fh.read()

    fields = {name.lower(): name for name in list(las.point_format.dimension_names)}
    selected = None
    for f in HAG_FIELDS:
        if f.lower() in fields:
            selected = fields[f.lower()]
            break

    out: dict = {
        "laz": str(laz_path),
        "hag_field": selected or "",
        "n_points": int(len(las.x)),
    }

    if not selected:
        out["message"] = "No HAG-like extra dimension found in LAZ"
        return out

    hag = np.asarray(las[selected], dtype=np.float32)
    if sample_step > 1:
        hag = hag[::sample_step]

    out.update(
        {
            "hag_min_raw": float(np.min(hag)) if hag.size else None,
            "hag_max_raw": float(np.max(hag)) if hag.size else None,
            "hag_p95_raw": _safe_pct(hag, 95),
            "hag_p99_raw": _safe_pct(hag, 99),
            "hag_above_clip_count_raw": int(np.sum(hag > clip_max)),
            "hag_above_clip_pct_raw": float(100.0 * np.mean(hag > clip_max)) if hag.size else 0.0,
            "hag_max_after_clip": float(np.max(np.clip(hag, 0.0, clip_max))) if hag.size else None,
        }
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Audit CHM clipping integrity")
    p.add_argument("--chm-dir", type=Path, required=True)
    p.add_argument("--laz-dir", type=Path, required=True)
    p.add_argument("--pattern", default="*_chm_max_hag_20cm.tif")
    p.add_argument("--clip-max", type=float, default=1.3)
    p.add_argument("--sample-raster", default="")
    p.add_argument("--sample-step", type=int, default=10)
    p.add_argument("--out-json", type=Path, default=Path("tmp/chm_clip_audit_report.json"))
    p.add_argument("--out-csv", type=Path, default=Path("tmp/chm_clip_audit_report.csv"))
    args = p.parse_args()

    rasters = sorted(args.chm_dir.glob(args.pattern))
    if not rasters:
        raise FileNotFoundError(f"No CHM files in {args.chm_dir} with pattern {args.pattern}")

    stats = [analyze_chm_file(r, args.clip_max) for r in rasters]

    global_max = max((s.max_val for s in stats if s.max_val is not None), default=None)
    any_above = sum(1 for s in stats if s.above_clip_count > 0)
    total_above = sum(s.above_clip_count for s in stats)

    deep_check = {}
    if args.sample_raster:
        laz = _find_laz_for_chm(args.sample_raster, args.laz_dir)
        deep_check["sample_raster"] = args.sample_raster
        deep_check["sample_laz"] = str(laz) if laz else ""
        if laz:
            deep_check["laz_hag_audit"] = analyze_laz_hag(laz, args.clip_max, args.sample_step)
        else:
            deep_check["message"] = "Could not map sample raster to LAZ"

    report = {
        "clip_max_expected": args.clip_max,
        "n_rasters": len(stats),
        "global_chm_max": global_max,
        "rasters_with_values_above_clip": any_above,
        "total_pixels_above_clip": total_above,
        "deep_check": deep_check,
        "rows": [s.__dict__ for s in stats],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].__dict__.keys()))
        w.writeheader()
        for s in stats:
            w.writerow(s.__dict__)

    print(f"n_rasters={len(stats)}")
    print(f"global_chm_max={global_max}")
    print(f"rasters_with_values_above_clip={any_above}")
    print(f"total_pixels_above_clip={total_above}")
    print(f"report_json={args.out_json}")
    print(f"report_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
