#!/usr/bin/env python3
"""Build a CHM dataset from all valid LAZ files using last returns only.

Methodology is aligned to the validated workflow used for:
`2018_harmonized_dem_last_gauss_chm.tif`

Pipeline per tile:
1) CSF ground classification per year (class=2).
2) Yearly DEM from per-cell minimum class-2 z.
3) Multi-year harmonized DEM (lowest valid; MAD-guarded).
4) CHM from last returns only with HAG filter 0..1.3 m (drop mode).
5) Gaussian smoothing on CHM.

Outputs are saved into one folder tree under --out-dir:
- chm/: all CHM rasters (raw + gauss)
- labels/: copied label CSVs used for processed tile-years
- harmonized_dem/: harmonized DEM per tile
- reports/: manifests, summaries, and methodology notes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter

NODATA = -9999.0


@dataclass
class YearInput:
    tile: str
    year: int
    laz_path: Path
    labels_csv: Path
    baseline_chm: Path


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _snap_origin(minx: float, miny: float, resolution: float) -> Tuple[float, float]:
    ox = math.floor(minx / resolution) * resolution
    oy = math.floor(miny / resolution) * resolution
    return ox, oy


def _write_pdal_pipeline(pipeline: list, path: Path) -> None:
    path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")


def _sanitize_laz_for_pdal(input_laz: Path, output_las: Path, chunk_size: int = 1_000_000) -> Path:
    """Rewrite LAZ to a clean LAS (point format 6) for PDAL compatibility.

    Some source files carry malformed ExtraBytes metadata that can make
    `readers.las` fail. This rewrite preserves the required geometric and
    return fields while dropping problematic extra-dimension declarations.
    """
    output_las.parent.mkdir(parents=True, exist_ok=True)
    with laspy.open(str(input_laz)) as src:
        hdr = laspy.LasHeader(point_format=6, version="1.4")
        hdr.scales = src.header.scales
        hdr.offsets = src.header.offsets
        try:
            crs = src.header.parse_crs()
            if crs is not None:
                hdr.add_crs(crs)
        except Exception:
            pass

        with laspy.open(str(output_las), mode="w", header=hdr) as dst:
            for pts in src.chunk_iterator(chunk_size):
                rec = laspy.ScaleAwarePointRecord.zeros(len(pts.x), header=hdr)
                rec.x = pts.x
                rec.y = pts.y
                rec.z = pts.z
                for dim in [
                    "intensity",
                    "return_number",
                    "number_of_returns",
                    "classification",
                    "scan_angle",
                    "user_data",
                    "point_source_id",
                    "gps_time",
                ]:
                    if hasattr(pts, dim):
                        setattr(rec, dim, getattr(pts, dim))
                dst.write_points(rec)
    return output_las


def _pdal_csf_reclassify(input_laz: Path, output_laz: Path, work_dir: Path, epsg: int = 3301) -> None:
    output_laz.parent.mkdir(parents=True, exist_ok=True)

    def _run_csf_pipeline(source_path: Path, pipe_path: Path) -> None:
        pipeline = [
            {
                "type": "readers.las",
                "filename": str(source_path),
                "override_srs": f"EPSG:{epsg}",
            },
            {"type": "filters.assign", "value": "Classification = 1"},
            {
                "type": "filters.csf",
                "resolution": 0.5,
                "rigidness": 3,
                "step": 0.65,
                "threshold": 0.5,
                "hdiff": 0.3,
            },
            {"type": "writers.las", "filename": str(output_laz), "compression": "true"},
        ]
        _write_pdal_pipeline(pipeline, pipe_path)
        _run(["pdal", "pipeline", str(pipe_path)])

    primary_pipe = work_dir / f"pdal_csf_{input_laz.stem}.json"
    try:
        _run_csf_pipeline(input_laz, primary_pipe)
        return
    except subprocess.CalledProcessError as exc:
        print(
            f"[csf-fallback] primary PDAL read failed for {input_laz.name}; "
            f"rewriting sanitized LAS and retrying. error={exc}",
            flush=True,
        )

    sanitized_las = work_dir / f"{input_laz.stem}_pdal_sanitized.las"
    _sanitize_laz_for_pdal(input_laz, sanitized_las)
    fallback_pipe = work_dir / f"pdal_csf_{input_laz.stem}_sanitized.json"
    _run_csf_pipeline(sanitized_las, fallback_pipe)


def _laz_bounds(laz_path: Path) -> Tuple[float, float, float, float]:
    with laspy.open(str(laz_path)) as fh:
        hdr = fh.header
        minx = float(hdr.mins[0])
        miny = float(hdr.mins[1])
        maxx = float(hdr.maxs[0])
        maxy = float(hdr.maxs[1])
    return minx, miny, maxx, maxy


def _global_bounds(paths: Iterable[Path]) -> Tuple[float, float, float, float]:
    mins = []
    maxs = []
    for p in paths:
        minx, miny, maxx, maxy = _laz_bounds(p)
        mins.append((minx, miny))
        maxs.append((maxx, maxy))
    minx = float(min(v[0] for v in mins))
    miny = float(min(v[1] for v in mins))
    maxx = float(max(v[0] for v in maxs))
    maxy = float(max(v[1] for v in maxs))
    return minx, miny, maxx, maxy


def _build_year_ground_dem(
    csf_laz: Path,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
    chunk_size: int,
) -> Tuple[np.ndarray, dict]:
    dem = np.full((dem_h, dem_w), np.nan, dtype=np.float32)
    n_points = 0
    n_ground = 0

    with laspy.open(str(csf_laz)) as fh:
        for pts in fh.chunk_iterator(chunk_size):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float32)
            n_points += int(len(x))

            try:
                cls = np.asarray(pts.classification)
                keep = cls == 2
            except Exception:
                keep = np.zeros(len(x), dtype=bool)

            if not np.any(keep):
                continue

            x = x[keep]
            y = y[keep]
            z = z[keep]
            n_ground += int(len(x))

            c = ((x - ox) / dem_resolution).astype(np.int32)
            r = ((maxy - y) / dem_resolution).astype(np.int32)
            valid = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
            if not np.any(valid):
                continue

            rr = r[valid]
            cc = c[valid]
            zz = z[valid]
            flat = rr * dem_w + cc
            order = np.argsort(flat)
            flat = flat[order]
            zz = zz[order]

            uniq, idx = np.unique(flat, return_index=True)
            mins = np.minimum.reduceat(zz, idx)

            current = dem.ravel()[uniq]
            merged = np.where(np.isfinite(current), np.minimum(current, mins), mins)
            dem.ravel()[uniq] = merged

    valid_cells = int(np.isfinite(dem).sum())
    stats = {
        "points_total": n_points,
        "points_ground_class2": n_ground,
        "ground_ratio_pct": float(100.0 * n_ground / n_points) if n_points else 0.0,
        "dem_valid_cells": valid_cells,
        "dem_valid_pct": float(100.0 * valid_cells / dem.size) if dem.size else 0.0,
    }
    return dem, stats


def _harmonize_year_dems(
    dems_by_year: Dict[int, np.ndarray],
    mad_factor: float = 2.5,
    floor_m: float = 0.15,
) -> Tuple[np.ndarray, dict]:
    years = sorted(dems_by_year.keys())
    stack = np.stack([dems_by_year[y] for y in years], axis=0).astype(np.float32)

    with np.errstate(invalid="ignore"):
        median = np.nanmedian(stack, axis=0)
        abs_dev = np.abs(stack - median[np.newaxis, :, :])
        mad = 1.4826 * np.nanmedian(abs_dev, axis=0)

    lower = median - np.maximum(floor_m, mad_factor * mad)
    finite = np.isfinite(stack)
    valid = finite & (stack >= lower[np.newaxis, :, :])

    with np.errstate(invalid="ignore"):
        filtered = np.where(valid, stack, np.nan)
        harm = np.nanmin(filtered, axis=0)
        fallback = np.nanmin(stack, axis=0)

    harm = np.where(np.isfinite(harm), harm, fallback).astype(np.float32)

    stats = {
        "years": years,
        "mad_factor": float(mad_factor),
        "floor_m": float(floor_m),
        "valid_cells": int(np.isfinite(harm).sum()),
        "valid_pct": float(100.0 * np.isfinite(harm).sum() / harm.size) if harm.size else 0.0,
        "median_mad": float(np.nanmedian(mad)) if np.isfinite(mad).any() else None,
        "p95_mad": float(np.nanpercentile(mad, 95)) if np.isfinite(mad).any() else None,
    }
    return harm, stats


def _read_grid_spec(chm_path: Path) -> dict:
    with rasterio.open(chm_path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "transform": src.transform,
            "crs": src.crs,
            "nodata": float(src.nodata) if src.nodata is not None else NODATA,
            "ox": float(src.transform.c),
            "maxy": float(src.transform.f),
            "res": float(abs(src.transform.a)),
        }


def _write_gtiff(path: Path, arr: np.ndarray, transform, crs, nodata: float, tags: dict) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def _smooth_masked(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: float | None = None,
    max_clip: float | None = None,
    use_cuda_if_available: bool = False,
) -> Tuple[np.ndarray, str]:
    if sigma <= 0:
        out = arr.copy()
        if min_clip is not None or max_clip is not None:
            valid = out != nodata
            lo = min_clip if min_clip is not None else -np.inf
            hi = max_clip if max_clip is not None else np.inf
            out[valid] = np.clip(out[valid], lo, hi)
        return out, "none"

    valid = arr != nodata
    if not np.any(valid):
        return arr.copy(), "none"

    if use_cuda_if_available:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter

            data_gpu = cp.asarray(np.where(valid, arr, 0.0).astype(np.float32))
            weight_gpu = cp.asarray(valid.astype(np.float32))
            smooth_data_gpu = gpu_gaussian_filter(data_gpu, sigma=sigma, mode="nearest")
            smooth_weight_gpu = gpu_gaussian_filter(weight_gpu, sigma=sigma, mode="nearest")
            smooth_data = cp.asnumpy(smooth_data_gpu)
            smooth_weight = cp.asnumpy(smooth_weight_gpu)

            out = np.full(arr.shape, nodata, dtype=np.float32)
            keep = smooth_weight > 1e-6
            out[keep] = smooth_data[keep] / smooth_weight[keep]
            if min_clip is not None or max_clip is not None:
                valid_out = out != nodata
                lo = min_clip if min_clip is not None else -np.inf
                hi = max_clip if max_clip is not None else np.inf
                out[valid_out] = np.clip(out[valid_out], lo, hi)
            return out, "cuda"
        except Exception:
            # Reliability-first fallback to CPU.
            pass

    data = np.where(valid, arr, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    smooth_data = gaussian_filter(data, sigma=sigma, mode="nearest")
    smooth_weight = gaussian_filter(weight, sigma=sigma, mode="nearest")

    out = np.full(arr.shape, nodata, dtype=np.float32)
    keep = smooth_weight > 1e-6
    out[keep] = smooth_data[keep] / smooth_weight[keep]
    if min_clip is not None or max_clip is not None:
        valid_out = out != nodata
        lo = min_clip if min_clip is not None else -np.inf
        hi = max_clip if max_clip is not None else np.inf
        out[valid_out] = np.clip(out[valid_out], lo, hi)

    return out, "cpu"


def _discover_inputs(laz_root: Path, label_dir: Path, baseline_dir: Path) -> Tuple[Dict[str, List[YearInput]], dict]:
    groups: Dict[str, List[YearInput]] = defaultdict(list)

    all_laz = sorted(laz_root.rglob("*.laz"))
    total = len(all_laz)
    matched = 0
    kept = 0
    skipped_no_label = 0
    skipped_no_baseline = 0
    skipped_bad_name = 0

    for laz in all_laz:
        name = laz.name
        parts = name.split("_")
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit() or parts[2] != "madal.laz":
            skipped_bad_name += 1
            continue

        tile = parts[0]
        year = int(parts[1])
        matched += 1
        label = label_dir / f"{tile}_{year}_madal_chm_max_hag_20cm_labels.csv"
        baseline = baseline_dir / f"{tile}_{year}_madal_chm_max_hag_20cm.tif"

        if not label.exists():
            skipped_no_label += 1
            continue
        if not baseline.exists():
            skipped_no_baseline += 1
            continue

        groups[tile].append(
            YearInput(tile=tile, year=year, laz_path=laz, labels_csv=label, baseline_chm=baseline)
        )
        kept += 1

    for tile in groups:
        groups[tile] = sorted(groups[tile], key=lambda x: x.year)

    summary = {
        "all_laz": total,
        "matched_tile_year_pattern": matched,
        "kept_with_labels_and_baseline": kept,
        "unique_tiles": len(groups),
        "skipped_bad_name": skipped_bad_name,
        "skipped_no_label": skipped_no_label,
        "skipped_no_baseline": skipped_no_baseline,
    }
    return dict(groups), summary


def _count_valid(arr: np.ndarray, nodata: float) -> Tuple[int, float, float, float]:
    vals = arr[arr != nodata]
    if vals.size == 0:
        return 0, math.nan, math.nan, math.nan
    return int(vals.size), float(vals.min()), float(vals.max()), float(vals.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Build last-return harmonized CHM dataset for all valid LAZ files")
    ap.add_argument("--laz-root", type=Path, default=Path("data"))
    ap.add_argument("--labels-dir", type=Path, default=Path("output/onboarding_labels_v2_drop13"))
    ap.add_argument("--baseline-chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/chm_dataset_lastreturns_hag0_1p3"),
        help="Single root folder for CHMs, copied labels, and reports.",
    )
    ap.add_argument("--dem-resolution", type=float, default=1.0)
    ap.add_argument("--hag-max", type=float, default=1.3)
    ap.add_argument("--chm-clip-min", type=float, default=0.0)
    ap.add_argument("--hag-upper-mode", choices=["drop", "clip"], default="drop")
    ap.add_argument("--gaussian-sigma", type=float, default=0.3)
    ap.add_argument("--chunk-size", type=int, default=800_000)
    ap.add_argument("--point-sample-rate", type=float, default=1.0)
    ap.add_argument("--epsg", type=int, default=3301)
    ap.add_argument("--mad-factor", type=float, default=2.5)
    ap.add_argument("--mad-floor", type=float, default=0.15)
    ap.add_argument("--reuse-csf", action="store_true")
    ap.add_argument("--use-cuda-if-available", action="store_true")
    ap.add_argument(
        "--tiles",
        default="",
        help="Optional comma-separated tile list for focused runs; empty means all discovered tiles.",
    )
    args = ap.parse_args()

    t0 = time.time()
    if not (0.0 < args.point_sample_rate <= 1.0):
        raise ValueError("--point-sample-rate must be in (0,1].")

    out_dir = args.out_dir
    chm_dir = out_dir / "chm"
    labels_out_dir = out_dir / "labels"
    scratch_dir = out_dir / "scratch"
    harmonized_dir = out_dir / "harmonized_dem"
    reports_dir = out_dir / "reports"
    for d in (out_dir, chm_dir, labels_out_dir, scratch_dir, harmonized_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    groups, discovery_summary = _discover_inputs(args.laz_root, args.labels_dir, args.baseline_chm_dir)
    selected_tiles = [t for t in args.tiles.split(",") if t.strip()] if args.tiles else []
    if selected_tiles:
        groups = {k: v for k, v in groups.items() if k in set(selected_tiles)}

    if not groups:
        raise RuntimeError("No valid LAZ+label+baseline combinations found for processing.")

    manifest_rows: List[dict] = []
    failed_tiles: List[dict] = []
    processed_tiles = 0
    processed_pairs = 0

    print(f"discovery_summary={json.dumps(discovery_summary)}", flush=True)
    print(f"tiles_to_process={len(groups)}", flush=True)

    for tile, entries in sorted(groups.items()):
        tile_t0 = time.time()
        print(f"[tile={tile}] years={[e.year for e in entries]}", flush=True)
        tile_scratch = scratch_dir / tile
        tile_scratch.mkdir(parents=True, exist_ok=True)

        try:
            # CSF classification per year (reused when enabled).
            csf_laz_by_year: Dict[int, Path] = {}
            for yi in entries:
                csf_out = tile_scratch / f"{yi.laz_path.stem}_csf_ground.laz"
                if args.reuse_csf and csf_out.exists():
                    print(f"[tile={tile}] reuse_csf={csf_out}", flush=True)
                else:
                    _pdal_csf_reclassify(yi.laz_path, csf_out, tile_scratch, epsg=args.epsg)
                csf_laz_by_year[yi.year] = csf_out

            # Common DEM grid for tile across available years.
            minx, miny, maxx, maxy = _global_bounds([e.laz_path for e in entries])
            ox, oy = _snap_origin(minx, miny, args.dem_resolution)
            dem_w = int(math.ceil((maxx - ox) / args.dem_resolution))
            dem_h = int(math.ceil((maxy - oy) / args.dem_resolution))
            dtm_transform = from_origin(ox, maxy, args.dem_resolution, args.dem_resolution)
            dtm_crs = f"EPSG:{args.epsg}"

            # Yearly DEMs + harmonization.
            dems_by_year: Dict[int, np.ndarray] = {}
            csf_stats_by_year: Dict[str, dict] = {}
            for yi in entries:
                dem, stats = _build_year_ground_dem(
                    csf_laz=csf_laz_by_year[yi.year],
                    ox=ox,
                    maxy=maxy,
                    dem_resolution=args.dem_resolution,
                    dem_h=dem_h,
                    dem_w=dem_w,
                    chunk_size=args.chunk_size,
                )
                dems_by_year[yi.year] = dem
                csf_stats_by_year[str(yi.year)] = stats

            harmonized_dem, harmonization_stats = _harmonize_year_dems(
                dems_by_year,
                mad_factor=args.mad_factor,
                floor_m=args.mad_floor,
            )

            harmonized_path = harmonized_dir / f"{tile}_harmonized_ground_dem_{int(args.dem_resolution*100)}cm.tif"
            _write_gtiff(
                harmonized_path,
                np.where(np.isfinite(harmonized_dem), harmonized_dem, NODATA).astype(np.float32),
                dtm_transform,
                dtm_crs,
                NODATA,
                {
                    "METHOD": "harmonized_lowest_valid_ground",
                    "TILE_ID": tile,
                    "DEM_RES": args.dem_resolution,
                    "MAD_FACTOR": args.mad_factor,
                    "MAD_FLOOR": args.mad_floor,
                    "YEARS": ",".join(str(e.year) for e in entries),
                },
            )

            # Generate CHM per year from last returns only.
            for yi in entries:
                spec = _read_grid_spec(yi.baseline_chm)
                width = spec["width"]
                height = spec["height"]
                grid_transform = spec["transform"]
                out_crs = spec["crs"] if spec["crs"] is not None else dtm_crs
                nodata = spec["nodata"]
                ox_grid = spec["ox"]
                maxy_grid = spec["maxy"]
                res_grid = spec["res"]

                dtm_resampled = np.full((height, width), np.nan, dtype=np.float32)
                reproject(
                    source=harmonized_dem.astype(np.float32),
                    destination=dtm_resampled,
                    src_transform=dtm_transform,
                    src_crs=dtm_crs,
                    dst_transform=grid_transform,
                    dst_crs=out_crs,
                    resampling=Resampling.bilinear,
                )

                chm_raw = np.full((height, width), nodata, dtype=np.float32)
                rng = np.random.default_rng(20260415 + yi.year)

                n_pts_total = 0
                n_last = 0
                n_valid_grid = 0
                n_valid_dtm = 0
                n_kept_hag = 0

                with laspy.open(str(yi.laz_path)) as fh:
                    for pts in fh.chunk_iterator(args.chunk_size):
                        x = np.asarray(pts.x, dtype=np.float64)
                        y = np.asarray(pts.y, dtype=np.float64)
                        z = np.asarray(pts.z, dtype=np.float32)
                        if x.size == 0:
                            continue
                        n_pts_total += int(x.size)

                        try:
                            rn = np.asarray(pts.return_number)
                            nr = np.asarray(pts.number_of_returns)
                            last_mask = rn == nr
                        except Exception:
                            # Reliability fallback if return fields are unavailable.
                            last_mask = np.ones(x.size, dtype=bool)

                        if args.point_sample_rate < 1.0:
                            samp = rng.random(x.size) < args.point_sample_rate
                            keep_pts = last_mask & samp
                        else:
                            keep_pts = last_mask

                        if not np.any(keep_pts):
                            continue

                        x = x[keep_pts]
                        y = y[keep_pts]
                        z = z[keep_pts]
                        n_last += int(x.size)

                        col = ((x - ox_grid) / res_grid).astype(np.int32)
                        row = ((maxy_grid - y) / res_grid).astype(np.int32)
                        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
                        if not np.any(valid):
                            continue

                        row = row[valid]
                        col = col[valid]
                        z = z[valid]
                        n_valid_grid += int(z.size)

                        flat = row * width + col
                        dtm_z = dtm_resampled[row, col]
                        valid_dtm = np.isfinite(dtm_z)
                        if not np.any(valid_dtm):
                            continue

                        flat = flat[valid_dtm]
                        hag = z[valid_dtm] - dtm_z[valid_dtm]
                        n_valid_dtm += int(hag.size)

                        if args.hag_upper_mode == "clip":
                            keep = hag >= args.chm_clip_min
                        else:
                            keep = (hag >= args.chm_clip_min) & (hag <= args.hag_max)
                        if not np.any(keep):
                            continue

                        valid_flat = flat[keep]
                        valid_hag = np.clip(hag[keep], args.chm_clip_min, args.hag_max).astype(np.float32)
                        n_kept_hag += int(valid_hag.size)
                        np.maximum.at(chm_raw.ravel(), valid_flat, valid_hag)

                chm_gauss, smooth_backend = _smooth_masked(
                    chm_raw,
                    nodata=nodata,
                    sigma=args.gaussian_sigma,
                    min_clip=args.chm_clip_min,
                    max_clip=args.hag_max,
                    use_cuda_if_available=args.use_cuda_if_available,
                )

                raw_name = f"{yi.tile}_{yi.year}_madal_harmonized_dem_last_raw_chm.tif"
                gauss_name = f"{yi.tile}_{yi.year}_madal_harmonized_dem_last_gauss_chm.tif"
                raw_path = chm_dir / raw_name
                gauss_path = chm_dir / gauss_name

                common_tags = {
                    "SOURCE_LAZ": yi.laz_path.name,
                    "TILE_ID": yi.tile,
                    "YEAR": yi.year,
                    "METHOD": "harmonized_dem_last",
                    "HAG_MAX": args.hag_max,
                    "CHM_CLIP_MIN": args.chm_clip_min,
                    "HAG_UPPER_MODE": args.hag_upper_mode,
                    "FILTER_MODE": "last_return_only",
                    "POINT_SAMPLE_RATE": args.point_sample_rate,
                }

                _write_gtiff(
                    raw_path,
                    chm_raw,
                    grid_transform,
                    out_crs,
                    nodata,
                    {**common_tags, "POST_FILTER": "none"},
                )
                _write_gtiff(
                    gauss_path,
                    chm_gauss,
                    grid_transform,
                    out_crs,
                    nodata,
                    {**common_tags, "POST_FILTER": f"gaussian_sigma_{args.gaussian_sigma}", "SMOOTH_BACKEND": smooth_backend},
                )

                label_copy = labels_out_dir / yi.labels_csv.name
                shutil.copy2(yi.labels_csv, label_copy)

                raw_count, raw_min, raw_max, raw_mean = _count_valid(chm_raw, nodata)
                gauss_count, gauss_min, gauss_max, gauss_mean = _count_valid(chm_gauss, nodata)

                manifest_rows.append(
                    {
                        "tile": yi.tile,
                        "year": yi.year,
                        "laz_path": str(yi.laz_path),
                        "label_csv": str(yi.labels_csv),
                        "label_copy": str(label_copy),
                        "baseline_chm": str(yi.baseline_chm),
                        "harmonized_dem": str(harmonized_path),
                        "chm_raw": str(raw_path),
                        "chm_gauss": str(gauss_path),
                        "points_total": n_pts_total,
                        "points_last_after_sampling": n_last,
                        "points_in_grid": n_valid_grid,
                        "points_with_finite_dtm": n_valid_dtm,
                        "points_after_hag_filter": n_kept_hag,
                        "raw_valid_count": raw_count,
                        "raw_min": raw_min,
                        "raw_max": raw_max,
                        "raw_mean": raw_mean,
                        "gauss_valid_count": gauss_count,
                        "gauss_min": gauss_min,
                        "gauss_max": gauss_max,
                        "gauss_mean": gauss_mean,
                        "smooth_backend": smooth_backend,
                        "csf_ground_ratio_pct": csf_stats_by_year[str(yi.year)]["ground_ratio_pct"],
                        "harmonized_valid_pct": harmonization_stats["valid_pct"],
                    }
                )

                processed_pairs += 1

            processed_tiles += 1
            print(
                f"[tile={tile}] done years={len(entries)} elapsed_sec={time.time() - tile_t0:.1f}",
                flush=True,
            )

        except Exception as exc:
            failed_tiles.append({"tile": tile, "error": str(exc)})
            print(f"[tile={tile}] FAILED error={exc}", flush=True)

    manifest_csv = reports_dir / "dataset_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        if manifest_rows:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        else:
            f.write("")

    summary = {
        "discovery": discovery_summary,
        "processed_tiles": processed_tiles,
        "processed_tile_year_pairs": processed_pairs,
        "failed_tiles": failed_tiles,
        "manifest_csv": str(manifest_csv),
        "out_dir": str(out_dir),
        "elapsed_seconds": time.time() - t0,
        "parameters": {
            "dem_resolution": args.dem_resolution,
            "hag_max": args.hag_max,
            "chm_clip_min": args.chm_clip_min,
            "hag_upper_mode": args.hag_upper_mode,
            "gaussian_sigma": args.gaussian_sigma,
            "point_sample_rate": args.point_sample_rate,
            "mad_factor": args.mad_factor,
            "mad_floor": args.mad_floor,
            "reuse_csf": bool(args.reuse_csf),
            "use_cuda_if_available": bool(args.use_cuda_if_available),
            "epsg": args.epsg,
        },
    }
    summary_json = reports_dir / "dataset_summary.json"
    _write_json(summary_json, summary)

    print(f"manifest_csv={manifest_csv}")
    print(f"summary_json={summary_json}")
    print(f"processed_tile_year_pairs={processed_pairs}")
    print(f"failed_tiles={len(failed_tiles)}")
    print(f"elapsed_seconds={summary['elapsed_seconds']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
