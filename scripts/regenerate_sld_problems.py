#!/usr/bin/env python3
"""Regenerate SLD-colored problem tiles in tmp/sep using label_tiles._apply_sld.

Finds PROB_*.png files in tmp/sep, parses raster stem and row/col from the
filename, reads the CHM window (default chunk 128), applies the project's
SLD styling, and saves a sibling file with suffix `_sld.png`.
"""

from pathlib import Path
import sys
import re

try:
    import rasterio
    from rasterio.windows import Window
    import numpy as np
except Exception as e:
    print("rasterio + numpy required", file=sys.stderr)
    raise

try:
    from label_tiles import _apply_sld
except Exception as e:
    print("Could not import _apply_sld from label_tiles:", e, file=sys.stderr)
    raise

OUT = Path("tmp/sep")
CHM_DIR = Path("chm_max_hag")


def parse_filename(p: Path):
    # expected stem: PROB_<n>_<raster_stem>_<row>_<col>
    stem = p.stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[0] != "PROB":
        return None
    row = int(parts[-2])
    col = int(parts[-1])
    raster_stem = "_".join(parts[2:-2])
    return raster_stem, row, col


def find_chm(raster_stem: str):
    matches = list(CHM_DIR.glob(f"{raster_stem}*.tif"))
    return matches[0] if matches else None


def read_window(chm_path: Path, row: int, col: int, cs: int = 128):
    with rasterio.open(chm_path) as src:
        arr = src.read(1, window=Window(col, row, cs, cs), boundless=True, fill_value=0).astype(
            "float32"
        )
    return arr


def save_sld(tile, out_path: Path):
    rgb = _apply_sld(tile)
    # _apply_sld returns uint8 RGB; ensure shape
    import imageio

    imageio.imwrite(str(out_path), rgb[:, :, ::-1])


def main():
    files = sorted(OUT.glob("PROB_*.png"))
    if not files:
        print("No PROB_*.png files in tmp/sep")
        return
    for p in files:
        parsed = parse_filename(p)
        if not parsed:
            print("Skipping", p)
            continue
        raster_stem, row, col = parsed
        chm = find_chm(raster_stem)
        if not chm:
            print("CHM not found for", raster_stem)
            continue
        try:
            tile = read_window(chm, row, col, cs=128)
        except Exception as e:
            print("Failed reading", chm, row, col, e)
            continue
        out = p.with_name(p.stem + "_sld.png")
        try:
            save_sld(tile, out)
            print("Wrote", out)
        except Exception as e:
            print("Failed saving SLD for", p, e)


if __name__ == "__main__":
    main()
