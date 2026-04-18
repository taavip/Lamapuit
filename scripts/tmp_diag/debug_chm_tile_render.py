#!/usr/bin/env python3
"""Export tile debug images to inspect clipping and rendering behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


def _norm_to_u8(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    v = arr[valid]
    lo, hi = float(np.min(v)), float(np.max(v))
    rng = hi - lo
    if rng < 1e-9:
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[valid] = 127
        return out
    n = (arr - lo) / rng
    n[~valid] = 0.0
    return np.clip(n * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description="Debug CHM tile rendering / clipping")
    p.add_argument("--chm", type=Path, required=True)
    p.add_argument("--row-off", type=int, required=True)
    p.add_argument("--col-off", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--clip-max", type=float, default=1.3)
    p.add_argument("--out-dir", type=Path, default=Path("tmp/chm_tile_debug"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.chm) as src:
        tile = src.read(
            1,
            window=Window(args.col_off, args.row_off, args.chunk_size, args.chunk_size),
            boundless=True,
            fill_value=src.nodata if src.nodata is not None else np.nan,
        ).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        tile = np.where(tile == float(nodata), np.nan, tile)

    clipped = np.clip(tile, 0.0, args.clip_max)
    mask_above = np.isfinite(tile) & (tile > args.clip_max)

    raw_u8 = _norm_to_u8(tile)
    clip_u8 = _norm_to_u8(clipped)
    mask_u8 = (mask_above.astype(np.uint8) * 255)

    cv2.imwrite(str(args.out_dir / "raw_float_normalized.png"), raw_u8)
    cv2.imwrite(str(args.out_dir / "clipped_0_1p3_normalized.png"), clip_u8)
    cv2.imwrite(str(args.out_dir / "above_1p3_mask.png"), mask_u8)

    valid = tile[np.isfinite(tile)]
    max_raw = float(np.max(valid)) if valid.size else None
    above_cnt = int(np.sum(mask_above))

    print(f"tile_shape={tile.shape}")
    print(f"tile_max_raw={max_raw}")
    print(f"pixels_above_clip={above_cnt}")
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
