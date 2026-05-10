#!/usr/bin/env python3
"""Phase I V10: Area-Masked Mask Synthesis — No Ensemble.

Converts sparse polygon labels + validated area polygon into a clean 3-band
supervision raster used by all downstream training phases:
    Band 1 (target):       1=CWD, 0=background
    Band 2 (valid_mask):   1=use pixel in loss, 0=ignore (outside area or nodata)
    Band 3 (ensemble_prob): 0.0 everywhere (no ensemble; stub for compatibility)

V10 improvement: completely removes the Phase 0 tile-classifier ensemble (broken
on composite CHM) and replaces it with an explicit area polygon that defines the
validation boundary. Only pixels inside the area are used for training.

Area masking logic:
    Positive (target=1, valid=1):  pixel inside area AND inside GPKG polygon
    Negative (target=0, valid=1):  pixel inside area AND outside GPKG
    Ignored (valid=0):             pixel outside area OR CHM nodata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize


def rasterize_gpkg(
    gpkg_path: Path,
    reference_tif: Path,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize polygon GeoPackage to match reference raster grid.

    Returns float32 (H, W) with 1.0=inside polygon, 0.0=outside, nan=nodata.
    """
    import geopandas as gpd

    with rasterio.open(reference_tif) as src:
        transform = src.transform
        height, width = src.height, src.width

    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError(f"GeoPackage has no CRS: {gpkg_path}")

    # Use 2D EPSG:3301 for rasterization
    target_epsg = 3301
    gdf = gdf.to_crs(epsg=target_epsg)

    shapes = [
        (geom, 1.0)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        raise ValueError(f"No valid geometries in {gpkg_path}")

    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=all_touched,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase I V10: Area-masked mask synthesis (no ensemble)"
    )
    parser.add_argument(
        "--baseline-chm",
        type=Path,
        default=Path(__file__).parents[2] / "seg_pipeline" / "input" / "baseline_chm.tif",
    )
    parser.add_argument(
        "--area-gpkg",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "labels" / "valid_area.gpkg",
        help="Validated area boundary polygon (EPSG:3301)",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "labels" / "cdw_labels_MP.gpkg",
        help="CWD label polygons (EPSG:3301)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "seg_pipeline" / "output" / "phase1_masks",
    )
    parser.add_argument("--smoke-test", action="store_true", help="512x512 crop only")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Rasterizing area polygon…")
    area_mask = rasterize_gpkg(args.area_gpkg, args.baseline_chm)
    n_area = int((area_mask > 0.5).sum())
    print(f"  Area polygon: {n_area:,} pixels ({100*n_area/area_mask.size:.2f}%)")

    print("\n[2/5] Rasterizing CWD label polygons…")
    gpkg_mask = rasterize_gpkg(args.gpkg, args.baseline_chm)
    n_cwd = int((gpkg_mask > 0.5).sum())
    print(f"  CWD labels: {n_cwd:,} pixels ({100*n_cwd/gpkg_mask.size:.2f}%)")

    print("\n[3/5] Reading CHM coverage…")
    with rasterio.open(args.baseline_chm) as src:
        chm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    coverage = np.isfinite(chm)
    n_valid_chm = int(coverage.sum())
    print(f"  Valid CHM pixels: {n_valid_chm:,} ({100*n_valid_chm/coverage.size:.2f}%)")

    print("\n[4/5] Synthesizing 3-band mask…")
    inside_area = area_mask > 0.5
    inside_gpkg = gpkg_mask > 0.5

    # Crop to smoke region if requested
    if args.smoke_test:
        inside_area = inside_area[:512, :512]
        inside_gpkg = inside_gpkg[:512, :512]
        coverage = coverage[:512, :512]

    target = np.where(inside_area & inside_gpkg, 1.0, 0.0).astype(np.float32)
    valid = np.where(inside_area & coverage, 1.0, 0.0).astype(np.float32)
    ensemble_stub = np.zeros_like(target, dtype=np.float32)

    n_pos = int((target * valid).sum())
    n_neg = int(((1.0 - target) * valid).sum())
    n_ignored = int((~inside_area | ~coverage).sum())

    print(f"  Target pixels:  {n_pos:,}")
    print(f"  Negative pixels: {n_neg:,}")
    print(f"  Ignored pixels: {n_ignored:,}")

    print("\n[5/5] Writing 3-band TIF…")
    bands = np.stack([target, valid, ensemble_stub], axis=0)

    suffix = "_smoke" if args.smoke_test else ""
    out_path = args.output_dir / f"406455_2021_tava_truemask{suffix}.tif"

    profile.update(
        count=3,
        dtype="float32",
        nodata=None,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        height=bands.shape[1],
        width=bands.shape[2],
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands)

    meta_out = {
        "source_chm": str(args.baseline_chm),
        "source_area_gpkg": str(args.area_gpkg),
        "source_gpkg": str(args.gpkg),
        "method": "area-masked (no ensemble)",
        "n_valid": int(valid.sum()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_ignored": n_ignored,
        "smoke_test": args.smoke_test,
    }
    meta_path = args.output_dir / f"406455_2021_tava_truemask{suffix}_meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2))

    print(f"\n✓ Mask: {out_path}")
    print(f"✓ Meta: {meta_path}")
    print(f"\n=== Complete ===")


if __name__ == "__main__":
    sys.exit(main() or 0)
