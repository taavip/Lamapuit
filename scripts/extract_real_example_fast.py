#!/usr/bin/env python3
"""Extract real example for visualization (fast version)."""

from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import geometry_mask
import laspy

GPKG_PATH = Path("lamapuit.gpkg")
LAZ_PATH = Path("source/406455_2021_tava/laz_input/406455_2021_tava.laz")
CHM_PATH = Path("chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")

# Load labels
gdf = gpd.read_file(GPKG_PATH)
gdf = gdf[gdf.geometry.type == "LineString"]

print(f"Total objects in gpkg: {len(gdf)}")
print(f"Sample bounds: {gdf.bounds.head()}")

# Load CHM and its bounds
with rasterio.open(CHM_PATH) as src:
    chm = src.read(1).astype(np.float32)
    transform = src.transform
    bounds = src.bounds

print(f"CHM bounds: {bounds}")
print(f"CHM shape: {chm.shape}")

# Filter objects that are within CHM bounds
chm_bounds_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
gdf_in_chm = gdf.clip(
    gpd.GeoDataFrame(geometry=[chm_bounds_geom], crs=gdf.crs)
)

print(f"Objects in CHM tile: {len(gdf_in_chm)}")

# Find first well-formed object (5-20m long)
best = None
for idx, row in gdf_in_chm.iterrows():
    geom = row.geometry
    length = geom.length
    if 5 < length < 20:
        best = {
            "geom": geom,
            "length": length,
            "bounds": geom.bounds,
        }
        break

if best is None:
    print("No suitable objects found in CHM tile")
    exit(1)

geom = best["geom"]
bounds = best["bounds"]
minx, miny, maxx, maxy = bounds

# Convert to pixel coords
row_min, col_min = rasterio.transform.rowcol(transform, minx, maxy)
row_max, col_max = rasterio.transform.rowcol(transform, maxx, miny)

# Create 128x128 box
centroid = geom.centroid
row_center, col_center = rasterio.transform.rowcol(transform, centroid.x, centroid.y)

pad = 64
row_start = max(0, row_center - pad)
row_end = min(chm.shape[0], row_center + pad)
col_start = max(0, col_center - pad)
col_end = min(chm.shape[1], col_center + pad)

# Ensure it's 128x128
if row_end - row_start < 128:
    row_end = min(chm.shape[0], row_start + 128)
if col_end - col_start < 128:
    col_end = min(chm.shape[1], col_start + 128)

chm_tile = chm[row_start:row_end, col_start:col_end]

# Get geospatial bounds (ensure proper order)
x1, y1 = rasterio.transform.xy(transform, row_end, col_start)
x2, y2 = rasterio.transform.xy(transform, row_start, col_end)
x_min, x_max = min(x1, x2), max(x1, x2)
y_min, y_max = min(y1, y2), max(y1, y2)

print(f"Object length: {best['length']:.1f}m")
print(f"CHM tile shape: {chm_tile.shape}")
print(f"Geospatial box: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")

# Extract LAZ points (lazy, iterate in chunks)
laz_subset = []
with laspy.open(LAZ_PATH) as lf:
    for chunk in lf.chunk_iterator(points_per_iteration=1000000):
        mask = (
            (chunk.x >= x_min) & (chunk.x <= x_max) &
            (chunk.y >= y_min) & (chunk.y <= y_max)
        )
        if mask.any():
            laz_subset.append(np.column_stack((chunk.x[mask], chunk.y[mask], chunk.z[mask])))

if laz_subset:
    laz_subset = np.vstack(laz_subset)
    print(f"LAZ points extracted: {laz_subset.shape[0]}")
else:
    print("No LAZ points found in area!")
    laz_subset = np.array([]).reshape(0, 3)

# Save output
import pickle
data = {
    "chm_tile": chm_tile,
    "laz_subset": laz_subset,
    "bbox": {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    },
    "transform": transform,
}

with open("_real_example_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nData saved to _real_example_data.pkl")
