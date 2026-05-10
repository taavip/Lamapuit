#!/usr/bin/env python3
"""Create a GeoPackage with test/train/val polygons for V3 spatial CV splits.

Outputs:
  - layer `cv_areas`: test + per-fold train/val polygons
  - layer `stripes`: each stripe polygon with role hints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seg_pipeline.scripts.phase2_dataset_v3 import (
    N_STRIPES,
    STRIPE_WIDTH,
    TEST_STRIPE,
    SpatialCVSplitterV3,
)


def stripe_polygon(src: rasterio.io.DatasetReader, stripe_id: int):
    col0 = stripe_id * STRIPE_WIDTH
    col1 = src.width if stripe_id == (N_STRIPES - 1) else min(src.width, (stripe_id + 1) * STRIPE_WIDTH)
    w = Window(col_off=col0, row_off=0, width=col1 - col0, height=src.height)
    left, bottom, right, top = window_bounds(w, src.transform)
    return box(left, bottom, right, top)


def main() -> None:
    p = argparse.ArgumentParser(description="Export V3 CV split polygons to GeoPackage.")
    p.add_argument(
        "--raster",
        type=Path,
        default=Path("seg_pipeline/input/composite_4band.tif"),
        help="Reference raster for extent/CRS and stripe geometry.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_output/cv_split_areas_v3.gpkg"),
        help="Output GeoPackage path.",
    )
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    splitter = SpatialCVSplitterV3()
    with rasterio.open(args.raster) as src:
        stripe_geoms = {s: stripe_polygon(src, s) for s in range(N_STRIPES)}
        crs = src.crs

    # Layer 1: stripe polygons
    stripe_rows = []
    val_stripes = [s for s in range(N_STRIPES) if s != TEST_STRIPE]
    for s in range(N_STRIPES):
        role = "test" if s == TEST_STRIPE else "train_or_val"
        fold_id = -1 if s == TEST_STRIPE else val_stripes.index(s)
        stripe_rows.append(
            {
                "stripe_id": s,
                "role": role,
                "val_fold_id": fold_id,
                "geometry": stripe_geoms[s],
            }
        )
    gdf_stripes = gpd.GeoDataFrame(stripe_rows, crs=crs)

    # Layer 2: per-split polygons
    area_rows = [
        {
            "split": "test",
            "fold_id": -1,
            "stripe_ids": str(TEST_STRIPE),
            "geometry": stripe_geoms[TEST_STRIPE],
        }
    ]
    n_folds = N_STRIPES - 1
    for fold in range(n_folds):
        val_stripe = splitter._val_stripes[fold]
        train_stripes = [s for s in range(N_STRIPES) if s not in (TEST_STRIPE, val_stripe)]

        train_geom = unary_union([stripe_geoms[s] for s in train_stripes])
        val_geom = stripe_geoms[val_stripe]

        area_rows.append(
            {
                "split": "train",
                "fold_id": fold,
                "stripe_ids": ",".join(str(s) for s in train_stripes),
                "geometry": train_geom,
            }
        )
        area_rows.append(
            {
                "split": "val",
                "fold_id": fold,
                "stripe_ids": str(val_stripe),
                "geometry": val_geom,
            }
        )

    gdf_areas = gpd.GeoDataFrame(area_rows, crs=crs)

    # Write GPKG with two layers
    gdf_areas.to_file(args.out, layer="cv_areas", driver="GPKG")
    gdf_stripes.to_file(args.out, layer="stripes", driver="GPKG")

    print(f"Wrote: {args.out}")
    print("Layers: cv_areas, stripes")
    print(f"Features: cv_areas={len(gdf_areas)}, stripes={len(gdf_stripes)}")


if __name__ == "__main__":
    main()
