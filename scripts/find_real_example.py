#!/usr/bin/env python3
"""Find a clear, isolated fallen wood object for visualization."""

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import laspy

# Files
GPKG_PATH = Path("lamapuit.gpkg")
LAZ_PATH = Path("source/406455_2021_tava/laz_input/406455_2021_tava.laz")
CHM_PATH = Path("chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")

# Load labels
gdf = gpd.read_file(GPKG_PATH)
gdf = gdf[gdf.geometry.type == "LineString"]

print(f"Total CWD objects: {len(gdf)}")

# Load CHM to get geotransform
with rasterio.open(CHM_PATH) as src:
    chm = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs

# Load LAZ
with laspy.open(LAZ_PATH) as lf:
    las = lf.read()
    laz_points = np.column_stack((las.x, las.y, las.z))
    laz_classification = las.classification

print(f"LAZ points: {len(laz_points)}")

# Find candidates: objects that:
# 1. Have length 5-20m (reasonable fallen wood)
# 2. Are not too fragmented
# 3. Have nearby points in LAZ
candidates = []
for idx, row in gdf.iterrows():
    geom = row.geometry
    length = geom.length

    # Filter by length
    if not (5 < length < 25):
        continue

    # Get bounds in geospatial coords
    bounds = geom.bounds
    minx, miny, maxx, maxy = bounds

    # Count points nearby (within 2m buffer)
    buf_geom = geom.buffer(2.0)
    nearby_mask = buf_geom.contains(
        gpd.points_from_xy(laz_points[:, 0], laz_points[:, 1])
    )
    nearby_count = nearby_mask.sum()

    if nearby_count > 100:  # Need decent point density
        candidates.append({
            "idx": idx,
            "length": length,
            "bounds": bounds,
            "nearby_count": nearby_count,
            "geom": geom,
        })

# Sort by nearby point count and pick top one
candidates.sort(key=lambda x: -x["nearby_count"])
if not candidates:
    print("No good candidates found!")
    exit(1)

best = candidates[0]
print(f"\nBest candidate (index {best['idx']}):")
print(f"  Length: {best['length']:.1f}m")
print(f"  Nearby points: {best['nearby_count']}")
print(f"  Bounds (geo): {best['bounds']}")

# Get pixel coords in CHM
geom = best["geom"]
bounds = best["bounds"]
minx, miny, maxx, maxy = bounds

# Convert to CHM pixel coords
row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)

print(f"  CHM pixel bounds: rows [{row_min}, {row_max}], cols [{col_min}, {col_max}]")

# Create 128x128 box centered on the geometry centroid
centroid = geom.centroid
row_center, col_center = rasterio.transform.rowcol(transform, centroid.x, centroid.y)

# Extend to 128x128 (at 20cm = 25.6m on ground)
pad = 64
row_start = max(0, row_center - pad)
row_end = min(chm.shape[0], row_center + pad)
col_start = max(0, col_center - pad)
col_end = min(chm.shape[1], col_center + pad)

print(f"  128x128 box: rows [{row_start}, {row_end}], cols [{col_start}, {col_end}]")

# Extract CHM tile
chm_tile = chm[row_start:row_end, col_start:col_end]

# Convert pixel box back to geospatial coords
coords_min = rasterio.transform.xy(transform, row_end, col_start)
coords_max = rasterio.transform.xy(transform, row_start, col_end)

print(f"  Geospatial bounds: x=[{coords_min[0]:.1f}, {coords_max[0]:.1f}], y=[{coords_min[1]:.1f}, {coords_max[1]:.1f}]")

# Extract LAZ points in that box
laz_mask = (
    (laz_points[:, 0] >= coords_min[0]) & (laz_points[:, 0] <= coords_max[0]) &
    (laz_points[:, 1] >= coords_min[1]) & (laz_points[:, 1] <= coords_max[1])
)
laz_subset = laz_points[laz_mask]

print(f"  LAZ points in box: {laz_subset.shape[0]}")

# Save metadata for visualization
output = {
    "laz_path": str(LAZ_PATH),
    "chm_path": str(CHM_PATH),
    "transform": transform,
    "chm_tile_bounds": {
        "row_start": int(row_start),
        "row_end": int(row_end),
        "col_start": int(col_start),
        "col_end": int(col_end),
    },
    "laz_bbox": {
        "x_min": float(coords_min[0]),
        "x_max": float(coords_max[0]),
        "y_min": float(coords_min[1]),
        "y_max": float(coords_max[1]),
    },
}

import json
with open("_example_metadata.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nMetadata saved to _example_metadata.json")
