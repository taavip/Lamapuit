#!/usr/bin/env python3
"""Compare old vs improved LAZ->CHM methods and downstream normalization semantics.

Produces JSON report with:
- per-raster stats
- pixel-level overlap/diff stats
- tile-level normalization comparison (fixed 1.3 scale vs p2-p98)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def _read_valid(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        valid = arr.compressed().astype(np.float32)
        meta = {
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "transform": tuple(src.transform),
            "tags": src.tags() or {},
        }
    return valid, meta


def _raster_stats(valid: np.ndarray) -> dict:
    if valid.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(valid.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
    }


def _read_window(path: Path, row_off: int, col_off: int, chunk: int) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(
            1,
            window=Window(col_off, row_off, chunk, chunk),
            boundless=True,
            fill_value=src.nodata if src.nodata is not None else np.nan,
        ).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == float(src.nodata), np.nan, arr)
        return arr


def _norm_fixed_1p3(tile: np.ndarray, clip_max: float) -> np.ndarray:
    t = np.clip(tile, 0.0, clip_max) / clip_max
    t[~np.isfinite(t)] = 0.0
    return t


def _norm_p2p98(tile: np.ndarray) -> np.ndarray:
    t = tile.astype(np.float32).copy()
    valid = np.isfinite(t)
    if not np.any(valid):
        return np.zeros_like(t, dtype=np.float32)
    p2, p98 = np.nanpercentile(t, 2), np.nanpercentile(t, 98)
    rng = float(p98 - p2) if float(p98 - p2) > 1e-6 else 1.0
    n = np.clip((t - p2) / rng, 0.0, 1.0)
    n[~valid] = 0.0
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare CHM generation methods")
    ap.add_argument("--old", type=Path, required=True, help="Old method CHM tif")
    ap.add_argument("--new", type=Path, required=True, help="Improved method CHM tif")
    ap.add_argument("--clip-max", type=float, default=1.3)
    ap.add_argument("--row-off", type=int, default=0)
    ap.add_argument("--col-off", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--out-json", type=Path, default=Path("tmp/method_compare_report.json"))
    args = ap.parse_args()

    old_valid, old_meta = _read_valid(args.old)
    new_valid, new_meta = _read_valid(args.new)

    # Pixel-level aligned diff only if same shape and transform
    aligned = (
        old_meta["width"] == new_meta["width"]
        and old_meta["height"] == new_meta["height"]
        and old_meta["transform"] == new_meta["transform"]
    )

    pix = {}
    if aligned:
        with rasterio.open(args.old) as so, rasterio.open(args.new) as sn:
            a = so.read(1, masked=True).astype(np.float32)
            b = sn.read(1, masked=True).astype(np.float32)
            valid = (~a.mask) & (~b.mask)
            av = np.asarray(a.data[valid], dtype=np.float32)
            bv = np.asarray(b.data[valid], dtype=np.float32)
            d = bv - av
            pix = {
                "overlap_pixels": int(av.size),
                "mean_abs_diff": float(np.mean(np.abs(d))) if d.size else None,
                "max_abs_diff": float(np.max(np.abs(d))) if d.size else None,
                "corr": float(np.corrcoef(av, bv)[0, 1]) if av.size > 10 else None,
            }

    old_tile = _read_window(args.old, args.row_off, args.col_off, args.chunk_size)
    new_tile = _read_window(args.new, args.row_off, args.col_off, args.chunk_size)

    old_fixed = _norm_fixed_1p3(old_tile, args.clip_max)
    new_fixed = _norm_fixed_1p3(new_tile, args.clip_max)
    old_p2p98 = _norm_p2p98(old_tile)
    new_p2p98 = _norm_p2p98(new_tile)

    norm_comp = {
        "tile_offsets": {"row_off": args.row_off, "col_off": args.col_off, "chunk_size": args.chunk_size},
        "old_fixed_mean": float(np.mean(old_fixed)),
        "new_fixed_mean": float(np.mean(new_fixed)),
        "old_p2p98_mean": float(np.mean(old_p2p98)),
        "new_p2p98_mean": float(np.mean(new_p2p98)),
        "old_fixed_vs_p2p98_mae": float(np.mean(np.abs(old_fixed - old_p2p98))),
        "new_fixed_vs_p2p98_mae": float(np.mean(np.abs(new_fixed - new_p2p98))),
    }

    report = {
        "old_path": str(args.old),
        "new_path": str(args.new),
        "aligned_same_grid": aligned,
        "old_stats": _raster_stats(old_valid),
        "new_stats": _raster_stats(new_valid),
        "old_tags": old_meta["tags"],
        "new_tags": new_meta["tags"],
        "pixel_comparison": pix,
        "normalization_comparison": norm_comp,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"report={args.out_json}")
    print(f"aligned_same_grid={aligned}")
    print(f"old_max={report['old_stats']['max']}")
    print(f"new_max={report['new_stats']['max']}")
    print(f"new_fixed_vs_p2p98_mae={norm_comp['new_fixed_vs_p2p98_mae']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
