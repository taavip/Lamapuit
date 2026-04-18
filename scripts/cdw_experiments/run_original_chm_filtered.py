#!/usr/bin/env python3
"""Run the original CHM generator (compute_hag_raster_streamed) with class/return filters.

This script calls the canonical streamed HAG raster generator but applies
optional class exclusion and return filtering (last / last2) so we can
reproduce the "original method" while only using specific returns.

Example:
  python scripts/cdw_experiments/run_original_chm_filtered.py \
    --laz data/lamapuit/laz/436646_2018_madal.laz \
    --out-dir data/lamapuit/chm_original_filtered \
    --exclude-classes 6,9 --return-mode last
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import rasterio

import sys
from pathlib import Path as _P
# Ensure we can import sibling module in scripts/
_scripts_dir = _P(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from process_laz_to_chm_improved import compute_hag_raster_streamed


def summarize_tif(path: Path) -> dict:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        vals = arr.compressed().astype(np.float32)
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "valid_px": int(vals.size),
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "mean": float(vals.mean()) if vals.size else None,
            "std": float(vals.std()) if vals.size else None,
        }


def parse_classes(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Run original CHM method with optional class/return filters")
    p.add_argument("--laz", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/lamapuit/chm_original_filtered"))
    p.add_argument("--exclude-classes", default="6,9")
    p.add_argument("--return-mode", choices=["all", "last", "last2"], default="all")
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--hag-max", type=float, default=1.3)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--baseline", type=Path, default=None, help="Optional baseline CHM to compare against")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    res_cm = int(args.resolution * 100)
    stem = args.laz.stem
    out_tif = out_dir / f"{stem}_chm_max_hag_{res_cm}cm_{args.return_mode}.tif"

    if out_tif.exists() and not args.overwrite:
        print(f"Output exists: {out_tif}. Use --overwrite to replace.")
    else:
        exclude = parse_classes(args.exclude_classes)
        compute_hag_raster_streamed(
            laz_path=args.laz,
            out_tif=out_tif,
            resolution=args.resolution,
            hag_max=args.hag_max,
            nodata=-9999.0,
            chunk_size=args.chunk_size,
            cog=False,
            drop_above_hag_max=True,
            exclude_classes=exclude,
            return_mode=args.return_mode,
        )

    summary = summarize_tif(out_tif)
    print("CHM generated:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.baseline:
        if args.baseline.exists():
            with rasterio.open(args.baseline) as bsrc:
                base = bsrc.read(1).astype(np.float32)
                base_nodata = bsrc.nodata
                with rasterio.open(out_tif) as ssrc:
                    exp = ssrc.read(1).astype(np.float32)

                valid_base = np.isfinite(base)
                if base_nodata is not None:
                    valid_base &= base != float(base_nodata)
                valid_exp = np.isfinite(exp)
                valid = valid_base & valid_exp
                if valid.any():
                    delta = exp[valid] - base[valid]
                    print("Comparison to baseline:")
                    print(f"  baseline: {args.baseline}")
                    print(f"  mean delta: {float(delta.mean()):.6f}")
                    print(f"  std delta: {float(delta.std()):.6f}")
                    print(f"  valid pixels compared: {int(delta.size)}")
                else:
                    print("No overlapping valid pixels between experiment and baseline")
        else:
            print(f"Baseline not found: {args.baseline}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
