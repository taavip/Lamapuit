#!/usr/bin/env python3
"""Build a reproducible LAZ -> CHM dataset.

Method:
1) CSF ground classification for each tile-year LAZ.
2) Per-year DEM from class-2 ground points (per-cell minimum z).
3) Multi-year harmonized DEM (MAD-guarded lowest valid ground).
4) CHM generation from harmonized DEM (no IDW/TIN/TPS interpolation).
5) Save only raw and gaussian CHM outputs, plus copied labels and reports.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter

NODATA = -9999.0
DEFAULT_SEED = 20260416


@dataclass(frozen=True)
class LazEntry:
    tile: str
    year: int
    campaign: str
    laz_path: Path
    baseline_chm: Optional[Path]
    label_csv: Optional[Path]


def _setup_logging(reports_dir: Path) -> logging.Logger:
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("laz_to_chm_harmonized")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(reports_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_error(errors_jsonl: Path, payload: dict) -> None:
    errors_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with errors_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _run_cmd(cmd: List[str], logger: logging.Logger) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _write_pdal_pipeline(pipeline: list, path: Path) -> None:
    path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")


def _sanitize_laz_for_pdal(input_laz: Path, output_las: Path, chunk_size: int = 1_000_000) -> Path:
    """Rewrite LAZ to clean LAS for PDAL if metadata is malformed."""
    output_las.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(input_laz)) as src:
        # Use LAS 1.2 / point format 3 for maximum PDAL compatibility.
        # This avoids WKT global-encoding edge cases seen with some malformed inputs.
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = src.header.scales
        header.offsets = src.header.offsets
        try:
            crs = src.header.parse_crs()
            if crs is not None:
                header.add_crs(crs)
        except Exception:
            pass

        with laspy.open(str(output_las), mode="w", header=header) as dst:
            for pts in src.chunk_iterator(chunk_size):
                out = laspy.ScaleAwarePointRecord.zeros(len(pts.x), header=header)
                out.x = pts.x
                out.y = pts.y
                out.z = pts.z

                for dim in (
                    "intensity",
                    "classification",
                    "user_data",
                    "point_source_id",
                    "gps_time",
                ):
                    if hasattr(pts, dim) and hasattr(out, dim):
                        setattr(out, dim, getattr(pts, dim))

                # Point format 3 allows return_number/number_of_returns in [1, 7].
                # Clamp malformed values and enforce return_number <= number_of_returns.
                if (
                    hasattr(pts, "return_number")
                    and hasattr(pts, "number_of_returns")
                    and hasattr(out, "return_number")
                    and hasattr(out, "number_of_returns")
                ):
                    rn = np.clip(np.asarray(pts.return_number, dtype=np.int16), 1, 7)
                    nr = np.clip(np.asarray(pts.number_of_returns, dtype=np.int16), 1, 7)
                    rn = np.minimum(rn, nr)
                    out.return_number = rn.astype(np.uint8, copy=False)
                    out.number_of_returns = nr.astype(np.uint8, copy=False)
                else:
                    if hasattr(pts, "return_number") and hasattr(out, "return_number"):
                        rn = np.clip(np.asarray(pts.return_number, dtype=np.int16), 1, 7)
                        out.return_number = rn.astype(np.uint8, copy=False)
                    if hasattr(pts, "number_of_returns") and hasattr(out, "number_of_returns"):
                        nr = np.clip(np.asarray(pts.number_of_returns, dtype=np.int16), 1, 7)
                        out.number_of_returns = nr.astype(np.uint8, copy=False)

                # LAS 1.2 uses scan_angle_rank in PF3.
                if hasattr(pts, "scan_angle") and hasattr(out, "scan_angle_rank"):
                    out.scan_angle_rank = np.asarray(pts.scan_angle, dtype=np.int8)
                elif hasattr(pts, "scan_angle_rank") and hasattr(out, "scan_angle_rank"):
                    out.scan_angle_rank = pts.scan_angle_rank

                dst.write_points(out)

    return output_las


def _pdal_csf_reclassify(
    input_laz: Path,
    output_laz: Path,
    work_dir: Path,
    logger: logging.Logger,
    epsg: int = 3301,
) -> None:
    output_laz.parent.mkdir(parents=True, exist_ok=True)

    def _run_pipeline(source_path: Path, pipeline_path: Path) -> None:
        pipeline = [
            {
                "type": "readers.las",
                "filename": str(source_path),
                "nosrs": True,
                "default_srs": f"EPSG:{epsg}",
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
            {
                "type": "writers.las",
                "filename": str(output_laz),
                "compression": "true",
                "minor_version": 4,
                "dataformat_id": 6,
                "a_srs": f"EPSG:{epsg}",
            },
        ]
        _write_pdal_pipeline(pipeline, pipeline_path)
        _run_cmd(["pdal", "pipeline", str(pipeline_path)], logger)

    primary_path = work_dir / f"pdal_csf_{input_laz.stem}.json"

    try:
        _run_pipeline(input_laz, primary_path)
        return
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "PDAL read failed for %s. Retry with sanitized LAS. error=%s",
            input_laz.name,
            exc,
        )

    sanitized_las = work_dir / f"{input_laz.stem}_sanitized_for_pdal.las"
    _sanitize_laz_for_pdal(input_laz, sanitized_las)
    fallback_path = work_dir / f"pdal_csf_{input_laz.stem}_sanitized.json"
    _run_pipeline(sanitized_las, fallback_path)


def _parse_laz_filename(name: str) -> Optional[Tuple[str, int, str]]:
    match = re.match(r"^(\d+)_(\d{4})_(.+)\.laz$", name)
    if not match:
        return None
    tile = match.group(1)
    year = int(match.group(2))
    campaign = match.group(3)
    return tile, year, campaign


def _discover_entries(
    laz_dir: Path,
    labels_dir: Path,
    baseline_dir: Path,
) -> Tuple[Dict[str, List[LazEntry]], dict]:
    grouped: Dict[str, List[LazEntry]] = defaultdict(list)

    laz_files = sorted(laz_dir.glob("*.laz"))

    parsed = 0
    with_baseline = 0
    with_label = 0

    for laz_path in laz_files:
        parsed_info = _parse_laz_filename(laz_path.name)
        if parsed_info is None:
            continue

        tile, year, campaign = parsed_info
        parsed += 1

        baseline = baseline_dir / f"{tile}_{year}_{campaign}_chm_max_hag_20cm.tif"
        label = labels_dir / f"{tile}_{year}_{campaign}_chm_max_hag_20cm_labels.csv"

        baseline_path = baseline if baseline.exists() else None
        label_path = label if label.exists() else None

        if baseline_path is not None:
            with_baseline += 1
        if label_path is not None:
            with_label += 1

        grouped[tile].append(
            LazEntry(
                tile=tile,
                year=year,
                campaign=campaign,
                laz_path=laz_path,
                baseline_chm=baseline_path,
                label_csv=label_path,
            )
        )

    for tile in grouped:
        grouped[tile] = sorted(grouped[tile], key=lambda e: (e.year, e.campaign))

    summary = {
        "total_laz": len(laz_files),
        "parsed_laz": parsed,
        "unique_tiles": len(grouped),
        "with_baseline_chm": with_baseline,
        "with_label_csv": with_label,
    }
    return dict(grouped), summary


def _laz_bounds(laz_path: Path) -> Tuple[float, float, float, float]:
    with laspy.open(str(laz_path)) as fh:
        header = fh.header
        minx = float(header.mins[0])
        miny = float(header.mins[1])
        maxx = float(header.maxs[0])
        maxy = float(header.maxs[1])
    return minx, miny, maxx, maxy


def _global_bounds(paths: Iterable[Path]) -> Tuple[float, float, float, float]:
    mins = []
    maxs = []
    for p in paths:
        minx, miny, maxx, maxy = _laz_bounds(p)
        mins.append((minx, miny))
        maxs.append((maxx, maxy))

    return (
        float(min(v[0] for v in mins)),
        float(min(v[1] for v in mins)),
        float(max(v[0] for v in maxs)),
        float(max(v[1] for v in maxs)),
    )


def _snap_origin(minx: float, miny: float, resolution: float) -> Tuple[float, float]:
    ox = math.floor(minx / resolution) * resolution
    oy = math.floor(miny / resolution) * resolution
    return ox, oy


def _build_year_ground_dem(
    csf_laz: Path,
    ox: float,
    maxy: float,
    dem_resolution: float,
    dem_h: int,
    dem_w: int,
    chunk_size: int,
) -> Tuple[np.ndarray, dict]:
    # Use +inf sentinel during reduction so we can update in-place with np.minimum.at
    # without per-chunk sort/unique grouping.
    dem = np.full((dem_h, dem_w), np.inf, dtype=np.float32)
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

            cols = ((x - ox) / dem_resolution).astype(np.int32)
            rows = ((maxy - y) / dem_resolution).astype(np.int32)
            valid = (rows >= 0) & (rows < dem_h) & (cols >= 0) & (cols < dem_w)
            if not np.any(valid):
                continue

            rows = rows[valid]
            cols = cols[valid]
            z = z[valid]

            flat = rows * dem_w + cols
            np.minimum.at(dem.ravel(), flat, z)

    dem = np.where(np.isfinite(dem), dem, np.nan).astype(np.float32)

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
    mad_factor: float,
    mad_floor: float,
) -> Tuple[np.ndarray, dict]:
    years = sorted(dems_by_year.keys())
    stack = np.stack([dems_by_year[y] for y in years], axis=0).astype(np.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        abs_dev = np.abs(stack - median[np.newaxis, :, :])
        mad = 1.4826 * np.nanmedian(abs_dev, axis=0)

    lower = median - np.maximum(mad_floor, mad_factor * mad)
    valid = np.isfinite(stack) & (stack >= lower[np.newaxis, :, :])

    filtered = np.where(valid, stack, np.nan)
    safe_filtered = np.where(np.isfinite(filtered), filtered, np.inf)
    harm = np.min(safe_filtered, axis=0)
    harm = np.where(np.isfinite(harm), harm, np.nan)

    safe_stack = np.where(np.isfinite(stack), stack, np.inf)
    fallback = np.min(safe_stack, axis=0)
    fallback = np.where(np.isfinite(fallback), fallback, np.nan)

    harm = np.where(np.isfinite(harm), harm, fallback).astype(np.float32)

    stats = {
        "years": years,
        "mad_factor": float(mad_factor),
        "mad_floor": float(mad_floor),
        "valid_cells": int(np.isfinite(harm).sum()),
        "valid_pct": float(100.0 * np.isfinite(harm).sum() / harm.size) if harm.size else 0.0,
        "median_mad": float(np.nanmedian(mad)) if np.isfinite(mad).any() else None,
        "p95_mad": float(np.nanpercentile(mad, 95)) if np.isfinite(mad).any() else None,
    }
    return harm, stats


def _grid_spec_from_raster(path: Path, fallback_epsg: int) -> dict:
    with rasterio.open(path) as src:
        nodata = float(src.nodata) if src.nodata is not None else NODATA
        crs = src.crs if src.crs is not None else f"EPSG:{fallback_epsg}"
        return {
            "width": int(src.width),
            "height": int(src.height),
            "transform": src.transform,
            "crs": crs,
            "nodata": nodata,
            "ox": float(src.transform.c),
            "maxy": float(src.transform.f),
            "res": float(abs(src.transform.a)),
        }


def _grid_spec_from_laz(laz_path: Path, resolution: float, epsg: int) -> dict:
    minx, miny, maxx, maxy = _laz_bounds(laz_path)
    ox, _ = _snap_origin(minx, miny, resolution)
    width = int(math.ceil((maxx - ox) / resolution))
    height = int(math.ceil((maxy - miny) / resolution))
    transform = from_origin(ox, maxy, resolution, resolution)

    return {
        "width": width,
        "height": height,
        "transform": transform,
        "crs": f"EPSG:{epsg}",
        "nodata": NODATA,
        "ox": float(transform.c),
        "maxy": float(transform.f),
        "res": float(abs(transform.a)),
    }


def _write_gtiff(path: Path, arr: np.ndarray, spec: dict, tags: dict) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(spec["height"]),
        "width": int(spec["width"]),
        "count": 1,
        "dtype": "float32",
        "transform": spec["transform"],
        "crs": spec["crs"],
        "nodata": float(spec["nodata"]),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def _smooth_masked_cpu(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: Optional[float],
    max_clip: Optional[float],
) -> np.ndarray:
    if sigma <= 0:
        out = arr.copy()
    else:
        valid = arr != nodata
        if not np.any(valid):
            return arr.copy()

        data = np.where(valid, arr, 0.0).astype(np.float32)
        weight = valid.astype(np.float32)
        smooth_data = gaussian_filter(data, sigma=sigma, mode="nearest")
        smooth_weight = gaussian_filter(weight, sigma=sigma, mode="nearest")

        out = np.full(arr.shape, nodata, dtype=np.float32)
        keep = smooth_weight > 1e-6
        out[keep] = smooth_data[keep] / smooth_weight[keep]

    if min_clip is not None or max_clip is not None:
        valid = out != nodata
        lo = min_clip if min_clip is not None else -np.inf
        hi = max_clip if max_clip is not None else np.inf
        out[valid] = np.clip(out[valid], lo, hi)

    return out


def _smooth_masked_cuda_cupy(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: Optional[float],
    max_clip: Optional[float],
) -> np.ndarray:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter

    if sigma <= 0:
        out = arr.copy()
    else:
        valid = arr != nodata
        if not np.any(valid):
            return arr.copy()

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
        valid = out != nodata
        lo = min_clip if min_clip is not None else -np.inf
        hi = max_clip if max_clip is not None else np.inf
        out[valid] = np.clip(out[valid], lo, hi)

    return out


def _smooth_masked_cuda_torch(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: Optional[float],
    max_clip: Optional[float],
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        out = arr.copy()
    else:
        valid = arr != nodata
        if not np.any(valid):
            return arr.copy()

        # Build a compact Gaussian kernel from sigma (about +/-3 sigma support).
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device="cuda", dtype=torch.float32)
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / torch.sum(kernel_1d)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / torch.sum(kernel_2d)
        kernel = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1])

        data_t = torch.from_numpy(np.where(valid, arr, 0.0).astype(np.float32)).to("cuda")
        weight_t = torch.from_numpy(valid.astype(np.float32)).to("cuda")

        data_4d = data_t.unsqueeze(0).unsqueeze(0)
        weight_4d = weight_t.unsqueeze(0).unsqueeze(0)
        smooth_data = F.conv2d(data_4d, kernel, padding=radius).squeeze(0).squeeze(0)
        smooth_weight = F.conv2d(weight_4d, kernel, padding=radius).squeeze(0).squeeze(0)

        out_t = torch.full_like(smooth_data, float(nodata))
        keep = smooth_weight > 1e-6
        out_t[keep] = smooth_data[keep] / smooth_weight[keep]
        out = out_t.detach().cpu().numpy().astype(np.float32)

    if min_clip is not None or max_clip is not None:
        valid = out != nodata
        lo = min_clip if min_clip is not None else -np.inf
        hi = max_clip if max_clip is not None else np.inf
        out[valid] = np.clip(out[valid], lo, hi)

    return out


def _has_cupy_gpu() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _has_torch_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _choose_smoothing_backend(gpu_mode: str, sigma: float, logger: logging.Logger) -> str:
    if sigma <= 0:
        logger.info("Gaussian sigma <= 0, smoothing backend set to cpu (no smoothing effect).")
        return "cpu"

    if gpu_mode == "off":
        logger.info("GPU mode is off. Using cpu smoothing.")
        return "cpu"

    gpu_backends: List[str] = []
    if _has_cupy_gpu():
        gpu_backends.append("cuda-cupy")
    if _has_torch_gpu():
        gpu_backends.append("cuda-torch")

    if not gpu_backends:
        msg = (
            "No GPU smoothing backend available (CuPy missing/no device and torch CUDA unavailable). "
            "Falling back to cpu smoothing."
        )
        if gpu_mode == "force":
            raise RuntimeError(msg)
        logger.info(msg)
        return "cpu"

    if gpu_mode == "force":
        forced = gpu_backends[0]
        logger.info("GPU mode is force. Using %s smoothing.", forced)
        return forced

    # Auto mode: benchmark cpu and all available GPU backends once.
    rng = np.random.default_rng(12345)
    trial = rng.random((1024, 1024), dtype=np.float32)
    mask = rng.random((1024, 1024)) > 0.2
    trial = np.where(mask, trial, NODATA).astype(np.float32)

    timings: Dict[str, float] = {}

    cpu_t0 = time.perf_counter()
    _smooth_masked_cpu(trial, NODATA, sigma=sigma, min_clip=0.0, max_clip=1.3)
    timings["cpu"] = time.perf_counter() - cpu_t0

    if "cuda-cupy" in gpu_backends:
        cupy_t0 = time.perf_counter()
        _smooth_masked_cuda_cupy(trial, NODATA, sigma=sigma, min_clip=0.0, max_clip=1.3)
        timings["cuda-cupy"] = time.perf_counter() - cupy_t0

    if "cuda-torch" in gpu_backends:
        torch_t0 = time.perf_counter()
        _smooth_masked_cuda_torch(trial, NODATA, sigma=sigma, min_clip=0.0, max_clip=1.3)
        timings["cuda-torch"] = time.perf_counter() - torch_t0

    backend = min(timings, key=timings.get)
    timing_str = ", ".join(f"{k}={v:.4f}s" for k, v in sorted(timings.items()))
    logger.info("Smoothing benchmark %s -> selected=%s", timing_str, backend)
    return backend


def _smooth_masked(
    arr: np.ndarray,
    nodata: float,
    sigma: float,
    min_clip: Optional[float],
    max_clip: Optional[float],
    backend: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, str]:
    if backend == "cuda-cupy":
        try:
            return _smooth_masked_cuda_cupy(arr, nodata, sigma, min_clip, max_clip), "cuda-cupy"
        except Exception as exc:
            logger.warning("CuPy CUDA smoothing failed, fallback to cpu. error=%s", exc)
            return _smooth_masked_cpu(arr, nodata, sigma, min_clip, max_clip), "cpu"

    if backend == "cuda-torch":
        try:
            return _smooth_masked_cuda_torch(arr, nodata, sigma, min_clip, max_clip), "cuda-torch"
        except Exception as exc:
            logger.warning("Torch CUDA smoothing failed, fallback to cpu. error=%s", exc)
            return _smooth_masked_cpu(arr, nodata, sigma, min_clip, max_clip), "cpu"

    return _smooth_masked_cpu(arr, nodata, sigma, min_clip, max_clip), "cpu"


def _count_valid(arr: np.ndarray, nodata: float) -> Tuple[int, float, float, float]:
    vals = arr[arr != nodata]
    if vals.size == 0:
        return 0, math.nan, math.nan, math.nan
    return int(vals.size), float(vals.min()), float(vals.max()), float(vals.mean())


def _mem_available_gib() -> Optional[float]:
    """Best-effort MemAvailable reader from /proc/meminfo."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return float(kb) / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def _apply_cpu_worker_safety_cap(workers: int, max_safe_workers: int, logger: logging.Logger) -> int:
    """Cap aggressive CPU parallelism to avoid abrupt OOM-like container exits."""
    effective = workers

    if max_safe_workers > 0 and effective > max_safe_workers:
        logger.warning(
            "Requested workers=%d exceeds max-safe-workers=%d. Reducing for stability.",
            effective,
            max_safe_workers,
        )
        effective = max_safe_workers

    mem_available_gib = _mem_available_gib()
    if mem_available_gib is not None and effective > 1:
        # Each tile worker can hold multiple DEM grids plus point chunks in memory.
        # Use a conservative memory budget to avoid host/container instability.
        mem_budget_gib = max(2.0, mem_available_gib * 0.50)
        mem_per_worker_gib = 4.0
        max_by_mem = max(1, int(mem_budget_gib // mem_per_worker_gib))
        if effective > max_by_mem:
            logger.warning(
                "Reducing workers from %d to %d based on memory budget "
                "(MemAvailable=%.1f GiB, budget=%.1f GiB).",
                effective,
                max_by_mem,
                mem_available_gib,
                mem_budget_gib,
            )
            effective = max_by_mem

    return max(1, effective)


def _process_tile_job(tile: str, entries_serialized: List[dict], cfg: dict, out_dir_str: str, smoothing_backend: str) -> dict:
    """Process a single tile task. Returns manifest_rows, failures, processed_pairs."""
    from pathlib import Path

    out_dir_local = Path(out_dir_str)
    raw_dir_local = out_dir_local / "chm_raw"
    gauss_dir_local = out_dir_local / "chm_gauss"
    labels_dir_local = out_dir_local / "labels"
    reports_dir_local = out_dir_local / "reports"
    work_dir_local = out_dir_local / "_work"

    # Initialize worker-level logger
    import logging
    logger_local = logging.getLogger(f"laz_{tile}")
    logger_local.setLevel(logging.INFO)
    if not logger_local.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(reports_dir_local / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger_local.addHandler(sh)
        logger_local.addHandler(fh)

    manifest_rows_local: List[dict] = []
    failures_local: List[dict] = []
    processed_pairs_local = 0

    tile_work = work_dir_local / tile
    tile_work.mkdir(parents=True, exist_ok=True)

    logger_local.info("[tile=%s] start years=%s", tile, [e["year"] for e in entries_serialized])

    try:
        # Reconstruct LazEntry objects locally
        entries_local = [
            LazEntry(
                tile=e["tile"],
                year=int(e["year"]),
                campaign=e["campaign"],
                laz_path=Path(e["laz_path"]),
                baseline_chm=Path(e["baseline_chm"]) if e["baseline_chm"] else None,
                label_csv=Path(e["label_csv"]) if e["label_csv"] else None,
            )
            for e in entries_serialized
        ]

        # CSF classification for each tile-year.
        csf_by_year: Dict[int, Path] = {}
        for entry in entries_local:
            csf_path = tile_work / f"{entry.laz_path.stem}_csf_ground.laz"
            cache_hit = False
            if cfg["reuse_csf"] and csf_path.exists():
                logger_local.info("[tile=%s year=%s] reuse local CSF: %s", tile, entry.year, csf_path)
                cache_hit = True
            elif cfg["reuse_csf"] and cfg.get("csf_cache_dir") is not None:
                cached = Path(cfg.get("csf_cache_dir")) / tile / f"{entry.laz_path.stem}_csf_ground.laz"
                if cached.exists():
                    csf_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached, csf_path)
                    logger_local.info("[tile=%s year=%s] reused external CSF cache: %s", tile, entry.year, cached)
                    cache_hit = True

            if not cache_hit:
                _pdal_csf_reclassify(
                    input_laz=entry.laz_path,
                    output_laz=csf_path,
                    work_dir=tile_work,
                    logger=logger_local,
                    epsg=cfg["epsg"],
                )
            csf_by_year[entry.year] = csf_path

        # Shared DEM grid for multi-year harmonization at configured resolution.
        minx, miny, maxx, maxy = _global_bounds([e.laz_path for e in entries_local])
        ox, _ = _snap_origin(minx, miny, cfg["dem_resolution"])
        dem_w = int(math.ceil((maxx - ox) / cfg["dem_resolution"]))
        dem_h = int(math.ceil((maxy - miny) / cfg["dem_resolution"]))
        dem_transform = from_origin(ox, maxy, cfg["dem_resolution"], cfg["dem_resolution"])
        dem_crs = f"EPSG:{cfg['epsg']}"

        dems_by_year: Dict[int, np.ndarray] = {}
        csf_stats_by_year: Dict[int, dict] = {}
        for entry in entries_local:
            year_dem, year_stats = _build_year_ground_dem(
                csf_laz=csf_by_year[entry.year],
                ox=ox,
                maxy=maxy,
                dem_resolution=cfg["dem_resolution"],
                dem_h=dem_h,
                dem_w=dem_w,
                chunk_size=cfg["chunk_size"],
            )
            dems_by_year[entry.year] = year_dem
            csf_stats_by_year[entry.year] = year_stats

        harmonized_dem, harmonized_stats = _harmonize_year_dems(
            dems_by_year=dems_by_year,
            mad_factor=cfg["mad_factor"],
            mad_floor=cfg["mad_floor"],
        )

        # Many baseline rasters share identical geometry; cache DEM reprojections per grid.
        dtm_cache: Dict[Tuple[int, int, Tuple[float, ...], str], np.ndarray] = {}

        for entry in entries_local:
            year_t0 = time.time()
            try:
                # Keep baseline geometry for label alignment when available.
                if entry.baseline_chm is not None:
                    grid = _grid_spec_from_raster(entry.baseline_chm, fallback_epsg=cfg["epsg"])
                    grid_source = "baseline_chm"
                else:
                    grid = _grid_spec_from_laz(
                        laz_path=entry.laz_path,
                        resolution=cfg["fallback_grid_resolution"],
                        epsg=cfg["epsg"],
                    )
                    grid_source = "fallback_from_laz"

                cache_key = (
                    int(grid["width"]),
                    int(grid["height"]),
                    tuple(grid["transform"]),
                    str(grid["crs"]),
                )
                dtm_resampled = dtm_cache.get(cache_key)
                if dtm_resampled is None:
                    dtm_resampled = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
                    reproject(
                        source=harmonized_dem.astype(np.float32),
                        destination=dtm_resampled,
                        src_transform=dem_transform,
                        src_crs=dem_crs,
                        dst_transform=grid["transform"],
                        dst_crs=grid["crs"],
                        resampling=Resampling.bilinear,
                    )
                    dtm_cache[cache_key] = dtm_resampled

                chm_raw = np.full((grid["height"], grid["width"]), grid["nodata"], dtype=np.float32)
                rng = np.random.default_rng(DEFAULT_SEED + entry.year + int(entry.tile[-3:]))

                points_total = 0
                points_kept_returns = 0
                points_in_grid = 0
                points_with_dtm = 0
                points_after_hag = 0

                with laspy.open(str(entry.laz_path)) as fh:
                    for pts in fh.chunk_iterator(cfg["chunk_size"]):
                        x = np.asarray(pts.x, dtype=np.float64)
                        y = np.asarray(pts.y, dtype=np.float64)
                        z = np.asarray(pts.z, dtype=np.float32)

                        if x.size == 0:
                            continue

                        points_total += int(x.size)

                        if cfg["return_mode"] == "last":
                            try:
                                rn = np.asarray(pts.return_number)
                                nr = np.asarray(pts.number_of_returns)
                                keep = rn == nr
                            except Exception:
                                keep = np.ones(x.size, dtype=bool)
                        else:
                            keep = np.ones(x.size, dtype=bool)

                        if cfg["point_sample_rate"] < 1.0:
                            sample_mask = rng.random(x.size) < cfg["point_sample_rate"]
                            keep = keep & sample_mask

                        if not np.any(keep):
                            continue

                        x = x[keep]
                        y = y[keep]
                        z = z[keep]
                        points_kept_returns += int(x.size)

                        cols = ((x - grid["ox"]) / grid["res"]).astype(np.int32)
                        rows = ((grid["maxy"] - y) / grid["res"]).astype(np.int32)

                        valid = (
                            (rows >= 0)
                            & (rows < grid["height"])
                            & (cols >= 0)
                            & (cols < grid["width"])
                        )
                        if not np.any(valid):
                            continue

                        rows = rows[valid]
                        cols = cols[valid]
                        z = z[valid]
                        points_in_grid += int(z.size)

                        flat = rows * grid["width"] + cols
                        dtm_z = dtm_resampled[rows, cols]
                        finite_dtm = np.isfinite(dtm_z)
                        if not np.any(finite_dtm):
                            continue

                        flat = flat[finite_dtm]
                        hag = z[finite_dtm] - dtm_z[finite_dtm]
                        points_with_dtm += int(hag.size)

                        if cfg["hag_upper_mode"] == "clip":
                            keep_hag = hag >= cfg["chm_clip_min"]
                        else:
                            keep_hag = (hag >= cfg["chm_clip_min"]) & (hag <= cfg["hag_max"])

                        if not np.any(keep_hag):
                            continue

                        selected_flat = flat[keep_hag]
                        selected_hag = np.clip(
                            hag[keep_hag],
                            cfg["chm_clip_min"],
                            cfg["hag_max"],
                        ).astype(np.float32)

                        points_after_hag += int(selected_hag.size)
                        np.maximum.at(chm_raw.ravel(), selected_flat, selected_hag)

                chm_gauss, used_backend = _smooth_masked(
                    arr=chm_raw,
                    nodata=grid["nodata"],
                    sigma=cfg["gaussian_sigma"],
                    min_clip=cfg["chm_clip_min"],
                    max_clip=cfg["hag_max"],
                    backend=smoothing_backend,
                    logger=logger_local,
                )

                out_name = f"{entry.tile}_{entry.year}_{entry.campaign}_harmonized_dem_{cfg['return_mode']}"
                raw_path = raw_dir_local / f"{out_name}_raw_chm.tif"
                gauss_path = gauss_dir_local / f"{out_name}_gauss_chm.tif"

                common_tags = {
                    "TILE_ID": entry.tile,
                    "YEAR": entry.year,
                    "CAMPAIGN": entry.campaign,
                    "SOURCE_LAZ": entry.laz_path.name,
                    "METHOD": "csf_multi_year_harmonized_ground",
                    "DEM_RESOLUTION_M": cfg["dem_resolution"],
                    "RETURN_MODE": cfg["return_mode"],
                    "HAG_MAX_M": cfg["hag_max"],
                    "CHM_CLIP_MIN_M": cfg["chm_clip_min"],
                    "HAG_UPPER_MODE": cfg["hag_upper_mode"],
                    "MAD_FACTOR": cfg["mad_factor"],
                    "MAD_FLOOR_M": cfg["mad_floor"],
                    "GRID_REFERENCE": grid_source,
                }

                _write_gtiff(
                    raw_path,
                    chm_raw,
                    grid,
                    {**common_tags, "POST_FILTER": "none"},
                )
                _write_gtiff(
                    gauss_path,
                    chm_gauss,
                    grid,
                    {
                        **common_tags,
                        "POST_FILTER": f"gaussian_sigma_{cfg['gaussian_sigma']}",
                        "SMOOTH_BACKEND": used_backend,
                    },
                )

                label_copy = None
                if entry.label_csv is not None:
                    label_copy = labels_dir_local / entry.label_csv.name
                    label_copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(entry.label_csv, label_copy)

                raw_count, raw_min, raw_max, raw_mean = _count_valid(chm_raw, grid["nodata"])
                gauss_count, gauss_min, gauss_max, gauss_mean = _count_valid(chm_gauss, grid["nodata"])

                manifest_rows_local.append(
                    {
                        "tile": entry.tile,
                        "year": entry.year,
                        "campaign": entry.campaign,
                        "laz_path": str(entry.laz_path),
                        "baseline_chm": str(entry.baseline_chm) if entry.baseline_chm else "",
                        "label_src": str(entry.label_csv) if entry.label_csv else "",
                        "label_copy": str(label_copy) if label_copy else "",
                        "grid_source": grid_source,
                        "raw_chm": str(raw_path),
                        "gauss_chm": str(gauss_path),
                        "smooth_backend": used_backend,
                        "points_total": points_total,
                        "points_after_return_filter": points_kept_returns,
                        "points_in_grid": points_in_grid,
                        "points_with_dtm": points_with_dtm,
                        "points_after_hag_filter": points_after_hag,
                        "raw_valid_count": raw_count,
                        "raw_min": raw_min,
                        "raw_max": raw_max,
                        "raw_mean": raw_mean,
                        "gauss_valid_count": gauss_count,
                        "gauss_min": gauss_min,
                        "gauss_max": gauss_max,
                        "gauss_mean": gauss_mean,
                        "csf_ground_ratio_pct": csf_stats_by_year[entry.year]["ground_ratio_pct"],
                        "harmonized_valid_pct": harmonized_stats["valid_pct"],
                        "elapsed_sec": float(time.time() - year_t0),
                        "status": "ok",
                        "error": "",
                    }
                )

                processed_pairs_local += 1
                logger_local.info(
                    "[tile=%s year=%s] done status=ok elapsed=%.1fs",
                    entry.tile,
                    entry.year,
                    time.time() - year_t0,
                )

            except Exception as exc:
                err = {
                    "stage": "tile_year_processing",
                    "tile": entry.tile,
                    "year": entry.year,
                    "campaign": entry.campaign,
                    "error": str(exc),
                }
                _append_error(reports_dir_local / "errors.jsonl", err)
                failures_local.append(err)
                logger_local.exception("[tile=%s year=%s] FAILED", entry.tile, entry.year)

                manifest_rows_local.append(
                    {
                        "tile": entry.tile,
                        "year": entry.year,
                        "campaign": entry.campaign,
                        "laz_path": str(entry.laz_path),
                        "baseline_chm": str(entry.baseline_chm) if entry.baseline_chm else "",
                        "label_src": str(entry.label_csv) if entry.label_csv else "",
                        "label_copy": "",
                        "grid_source": "",
                        "raw_chm": "",
                        "gauss_chm": "",
                        "smooth_backend": "",
                        "points_total": 0,
                        "points_after_return_filter": 0,
                        "points_in_grid": 0,
                        "points_with_dtm": 0,
                        "points_after_hag_filter": 0,
                        "raw_valid_count": 0,
                        "raw_min": math.nan,
                        "raw_max": math.nan,
                        "raw_mean": math.nan,
                        "gauss_valid_count": 0,
                        "gauss_min": math.nan,
                        "gauss_max": math.nan,
                        "gauss_mean": math.nan,
                        "csf_ground_ratio_pct": math.nan,
                        "harmonized_valid_pct": math.nan,
                        "elapsed_sec": 0.0,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

                if not cfg.get("continue_on_error", False):
                    # Re-raise to surface to main; main will decide how to handle.
                    raise

        logger_local.info("[tile=%s] done", tile)

    except Exception as exc:
        tile_error = {"stage": "tile_processing", "tile": tile, "error": str(exc)}
        _append_error(reports_dir_local / "errors.jsonl", tile_error)
        failures_local.append(tile_error)
        logger_local.exception("[tile=%s] tile-level FAILED", tile)
        # If not continue_on_error, re-raise to allow main to handle fail-fast
        if not cfg.get("continue_on_error", False):
            raise

    return {
        "manifest_rows": manifest_rows_local,
        "failures": failures_local,
        "processed_pairs": processed_pairs_local,
    }



def main() -> int:
    parser = argparse.ArgumentParser(description="Build harmonized raw+gauss CHM dataset from LAZ files")
    parser.add_argument("--laz-dir", type=Path, default=Path("data/lamapuit/laz"))
    parser.add_argument("--labels-dir", type=Path, default=Path("output/onboarding_labels_v2_drop13"))
    parser.add_argument("--baseline-chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss"))

    parser.add_argument("--dem-resolution", type=float, default=0.8)
    parser.add_argument("--fallback-grid-resolution", type=float, default=0.2)
    parser.add_argument("--hag-max", type=float, default=1.3)
    parser.add_argument("--chm-clip-min", type=float, default=0.0)
    parser.add_argument("--hag-upper-mode", choices=["drop", "clip"], default="drop")
    parser.add_argument("--gaussian-sigma", type=float, default=0.3)

    parser.add_argument("--return-mode", choices=["last", "all"], default="last")
    parser.add_argument("--point-sample-rate", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=800_000)
    parser.add_argument("--epsg", type=int, default=3301)
    parser.add_argument("--mad-factor", type=float, default=2.5)
    parser.add_argument("--mad-floor", type=float, default=0.15)

    parser.add_argument("--gpu-mode", choices=["off", "auto", "force"], default="auto")
    parser.add_argument("--reuse-csf", action="store_true")
    parser.add_argument(
        "--csf-cache-dir",
        type=Path,
        default=None,
        help="Optional external cache root with <tile>/<stem>_csf_ground.laz files.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--tiles", default="", help="Optional comma-separated tile list")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (tiles processed concurrently). Default=1 (serial).")
    parser.add_argument(
        "--max-safe-workers",
        type=int,
        default=2,
        help="Safety cap for CPU workers to reduce OOM/crash risk. Set <=0 to disable.",
    )

    args = parser.parse_args()

    if not (0.0 < args.point_sample_rate <= 1.0):
        raise ValueError("--point-sample-rate must be in (0,1].")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    out_dir = args.out_dir
    raw_dir = out_dir / "chm_raw"
    gauss_dir = out_dir / "chm_gauss"
    labels_dir = out_dir / "labels"
    reports_dir = out_dir / "reports"
    work_dir = out_dir / "_work"

    for d in (raw_dir, gauss_dir, labels_dir, reports_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = _setup_logging(reports_dir)
    errors_jsonl = reports_dir / "errors.jsonl"
    if errors_jsonl.exists():
        errors_jsonl.unlink()

    t0 = time.time()
    run_params = {}
    for key, value in vars(args).items():
        run_params[key] = str(value) if isinstance(value, Path) else value
    _write_json(reports_dir / "run_parameters.json", run_params)

    grouped, discovery = _discover_entries(args.laz_dir, args.labels_dir, args.baseline_chm_dir)

    selected_tiles = {t.strip() for t in args.tiles.split(",") if t.strip()}
    if selected_tiles:
        grouped = {tile: rows for tile, rows in grouped.items() if tile in selected_tiles}

    if not grouped:
        raise RuntimeError("No LAZ entries found to process.")

    logger.info("Discovery summary: %s", json.dumps(discovery))
    logger.info("Tiles selected: %d", len(grouped))

    smoothing_backend = _choose_smoothing_backend(args.gpu_mode, args.gaussian_sigma, logger)

    manifest_rows: List[dict] = []
    failures: List[dict] = []
    processed_pairs = 0
    # Prepare serializable tile payloads for worker processes
    tile_payloads = []
    for tile, entries in sorted(grouped.items()):
        serialized = [
            {
                "tile": e.tile,
                "year": int(e.year),
                "campaign": e.campaign,
                "laz_path": str(e.laz_path),
                "baseline_chm": str(e.baseline_chm) if e.baseline_chm else "",
                "label_csv": str(e.label_csv) if e.label_csv else "",
            }
            for e in entries
        ]
        tile_payloads.append((tile, serialized))

    # Worker config: simple values shared by worker tasks.
    worker_config = {
        "dem_resolution": args.dem_resolution,
        "fallback_grid_resolution": args.fallback_grid_resolution,
        "hag_max": args.hag_max,
        "chm_clip_min": args.chm_clip_min,
        "hag_upper_mode": args.hag_upper_mode,
        "gaussian_sigma": args.gaussian_sigma,
        "return_mode": args.return_mode,
        "point_sample_rate": args.point_sample_rate,
        "chunk_size": args.chunk_size,
        "epsg": args.epsg,
        "mad_factor": args.mad_factor,
        "mad_floor": args.mad_floor,
        "gpu_mode": args.gpu_mode,
        "reuse_csf": args.reuse_csf,
        "csf_cache_dir": str(args.csf_cache_dir) if args.csf_cache_dir is not None else None,
        "continue_on_error": args.continue_on_error,
        "keep_work": args.keep_work,
    }

    out_dir_str = str(out_dir)
    requested_workers = min(args.workers, len(tile_payloads))
    effective_workers = requested_workers

    if effective_workers > 1 and not smoothing_backend.startswith("cuda"):
        effective_workers = _apply_cpu_worker_safety_cap(
            workers=effective_workers,
            max_safe_workers=args.max_safe_workers,
            logger=logger,
        )

    if effective_workers > 1 and smoothing_backend.startswith("cuda"):
        logger.warning(
            "CUDA smoothing backend selected (%s). For stability, forcing workers=1. "
            "Use --gpu-mode off if you want CPU multi-worker parallelism.",
            smoothing_backend,
        )
        effective_workers = 1

    logger.info(
        "Worker plan: requested=%d, tile_limited=%d, selected=%d",
        args.workers,
        requested_workers,
        effective_workers,
    )

    if effective_workers > 1:
        logger.info(
            "Parallel mode: starting up to %d workers (cpu_count=%s)",
            effective_workers,
            os.cpu_count(),
        )
        futures = {}
        # Thread pool avoids heavy multi-process fork memory duplication (which crashes WSL/Docker via OOM)
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            for tile, serialized in tile_payloads:
                fut = executor.submit(_process_tile_job, tile, serialized, worker_config, out_dir_str, smoothing_backend)
                futures[fut] = tile

            for fut in concurrent.futures.as_completed(futures):
                tile = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    logger.exception("Tile worker crashed for %s: %s", tile, exc)
                    failures.append({"stage": "tile_processing", "tile": tile, "error": str(exc)})
                    if not args.continue_on_error:
                        logger.error("Fail-fast active: stopping due to worker crash.")
                        break
                    continue

                manifest_rows.extend(res.get("manifest_rows", []))
                failures.extend(res.get("failures", []))
                processed_pairs += int(res.get("processed_pairs", 0))
                logger.info("[tile=%s] worker finished processed_pairs=%d failures=%d", tile, res.get("processed_pairs", 0), len(res.get("failures", [])))

    else:
        # Serial fallback: run jobs in-process to preserve original behavior
        for tile, serialized in tile_payloads:
            try:
                res = _process_tile_job(tile, serialized, worker_config, out_dir_str, smoothing_backend)
            except Exception as exc:
                logger.exception("Tile processing failed for %s: %s", tile, exc)
                failures.append({"stage": "tile_processing", "tile": tile, "error": str(exc)})
                if not args.continue_on_error:
                    logger.error("Fail-fast is active. Stopping after first failure.")
                    break
                continue

            manifest_rows.extend(res.get("manifest_rows", []))
            failures.extend(res.get("failures", []))
            processed_pairs += int(res.get("processed_pairs", 0))

    manifest_csv = reports_dir / "dataset_manifest.csv"
    if manifest_rows:
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    else:
        manifest_csv.write_text("", encoding="utf-8")

    summary = {
        "discovery": discovery,
        "processed_tile_year_pairs": processed_pairs,
        "failed_count": len(failures),
        "failures": failures,
        "smoothing_backend_selected": smoothing_backend,
        "manifest_csv": str(manifest_csv),
        "out_dir": str(out_dir),
        "elapsed_seconds": float(time.time() - t0),
    }
    _write_json(reports_dir / "dataset_summary.json", summary)

    logger.info("processed_tile_year_pairs=%d", processed_pairs)
    logger.info("failed_count=%d", len(failures))
    logger.info("elapsed_seconds=%.2f", summary["elapsed_seconds"])

    if len(failures) == 0 and not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Cleaned temporary work directory: %s", work_dir)
    elif len(failures) > 0:
        logger.info("Temporary work directory kept for debugging: %s", work_dir)

    return 0 if len(failures) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
