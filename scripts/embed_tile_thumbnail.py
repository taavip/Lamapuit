#!/usr/bin/env python3
"""Create a PNG thumbnail from a GeoTIFF around given coordinates and embed it
as a base64 image into a markdown file section.

Designed to run inside the project's Conda environment (`cwd-detect`).
Defaults assume coordinates are in L-EST97 (EPSG:3301).
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
import importlib
import importlib.util

try:
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover - runtime dependency
    rasterio = None

from PIL import Image
from pyproj import Transformer
import numpy as np


def build_parser():
    p = argparse.ArgumentParser(description="Crop GeoTIFF around a coord and embed image into MD")
    p.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    p.add_argument("--x", type=float, help="X coordinate (L-EST97 by default)")
    p.add_argument("--y", type=float, help="Y coordinate (L-EST97 by default)")
    p.add_argument("--coord-crs", default="EPSG:3301", help="CRS of provided coordinates (default L-EST97 EPSG:3301)")
    p.add_argument("--size", type=int, default=512, help="Output square size in pixels (default: 512)")
    p.add_argument("--out", default=None, help="Output PNG path (optional)")
    p.add_argument("--md", default="examples/annotation_examples.md", help="Markdown file to edit")
    p.add_argument("--section", default="Ideal example", help="Section header to replace (e.g. 'Ideal example')")
    p.add_argument("--caption", default=None, help="Caption text to add under image")
    return p


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="float32")
    # handle constant arrays
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype="uint8")
    a_min = float(np.nanmin(np.where(finite, arr, np.nan)))
    a_max = float(np.nanmax(np.where(finite, arr, np.nan)))
    if a_max <= a_min:
        out = np.zeros(arr.shape, dtype="uint8")
    else:
        out = ((arr - a_min) / (a_max - a_min) * 255.0).clip(0, 255).astype("uint8")
    return out


def crop_thumbnail(dataset, x_proj, y_proj, size_px: int) -> Image.Image:
    # rasterio dataset: convert x,y -> row/col
    try:
        row, col = dataset.index(x_proj, y_proj)
    except Exception:
        # fallback to centre
        bounds = dataset.bounds
        x_proj = (bounds.left + bounds.right) / 2.0
        y_proj = (bounds.bottom + bounds.top) / 2.0
        row, col = dataset.index(x_proj, y_proj)

    half = size_px // 2
    col_off = int(col - half)
    row_off = int(row - half)
    if col_off < 0:
        col_off = 0
    if row_off < 0:
        row_off = 0

    width = min(size_px, dataset.width - col_off)
    height = min(size_px, dataset.height - row_off)

    window = Window(col_off, row_off, width, height)
    arr = dataset.read(1, window=window).astype("float32")
    # Resample to requested size using PIL's float mode if needed
    if width != size_px or height != size_px:
        im_f = Image.fromarray(arr, mode="F")
        im_res = im_f.resize((size_px, size_px), resample=Image.BILINEAR)
        arr = np.array(im_res, dtype="float32")
    return arr


def embed_base64_png_into_md(md_path: str, section: str, png_bytes: bytes, caption: str | None):
    b64 = base64.b64encode(png_bytes).decode("ascii")
    img_md = f"![{section}](data:image/png;base64,{b64})\n\n"
    if caption:
        img_md += f"_{caption}_\n\n"

    with open(md_path, "r", encoding="utf-8") as fh:
        txt = fh.read()

    # Replace content under the target header (keep header, replace until next '### ')
    pattern = re.compile(r"(###\s*" + re.escape(section) + r"\s*\n)(.*?)(\n###\s|$)", re.S)
    m = pattern.search(txt)
    if not m:
        # If header not found, append at end
        txt = txt.rstrip() + "\n\n### " + section + "\n\n" + img_md
    else:
        new_section = m.group(1) + "\n" + img_md
        rest_marker = m.group(3)
        start, end = m.span()
        # replace exact matched span
        txt = txt[: m.start()] + new_section + rest_marker + txt[m.end() :]

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(txt)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if rasterio is None:
        print("Error: rasterio is required. Run inside the Conda env with rasterio installed.")
        sys.exit(2)

    tif = args.tif
    if not os.path.exists(tif):
        print(f"Error: TIFF not found: {tif}")
        sys.exit(2)

    with rasterio.open(tif) as ds:
        ds_crs = ds.crs
        if args.x is None or args.y is None:
            # use centre
            bounds = ds.bounds
            x_proj = (bounds.left + bounds.right) / 2.0
            y_proj = (bounds.bottom + bounds.top) / 2.0
        else:
            # transform coordinates if needed
            coord_crs = args.coord_crs
            if ds_crs is not None and coord_crs is not None:
                try:
                    transformer = Transformer.from_crs(coord_crs, ds_crs.to_string(), always_xy=True)
                    x_proj, y_proj = transformer.transform(args.x, args.y)
                except Exception:
                    # try simple assignment
                    x_proj, y_proj = args.x, args.y
            else:
                x_proj, y_proj = args.x, args.y

        arr = crop_thumbnail(ds, x_proj, y_proj, args.size)

    # Try to reuse the SLD coloring used by the labeling tool (scripts/label_tiles.py)
    apply_sld = None
    try:
        # First try a normal import if package path allows
        try:
            import label_tiles as _lt
            apply_sld = getattr(_lt, "_apply_sld", None)
        except Exception:
            # Fallback: load the script directly from the scripts directory
            mod_path = os.path.join(os.path.dirname(__file__), "label_tiles.py")
            if os.path.exists(mod_path):
                spec = importlib.util.spec_from_file_location("label_tiles_embed", mod_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                apply_sld = getattr(mod, "_apply_sld", None)
    except Exception:
        apply_sld = None

    # Apply SLD coloring if available, otherwise fall back to grayscale
    if apply_sld is not None:
        try:
            rgb = apply_sld(arr)
            img = Image.fromarray(rgb)
        except Exception:
            img_arr = normalize_to_uint8(arr)
            img = Image.fromarray(img_arr).convert("L")
    else:
        img_arr = normalize_to_uint8(arr)
        img = Image.fromarray(img_arr).convert("L")

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(tif))[0]
        out_dir = os.path.join("examples", "images")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_thumb.png")

    img.save(out_path, format="PNG")

    # read bytes
    with open(out_path, "rb") as fh:
        png_bytes = fh.read()

    embed_base64_png_into_md(args.md, args.section, png_bytes, args.caption)

    print(f"Wrote thumbnail {out_path} and embedded into {args.md} under section '{args.section}'.")


if __name__ == "__main__":
    main()
