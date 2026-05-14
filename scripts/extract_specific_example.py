#!/usr/bin/env python3
"""Extract example using specific coordinates from gpkg."""

from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import laspy

GPKG_PATH = Path("lamapuit.gpkg")
LAZ_PATH = Path("source/406455_2021_tava/laz_input/406455_2021_tava.laz")
CHM_PATH = Path("chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")
DEM_PATH = Path(
    "source/406455_2021_tava/chm_variants_reconstructed_original_20260510/"
    "harmonized_raw_0p2m/406455_2021_tava_harmonized_dem_last_raw_chm.tif"
)

# Target coordinates
target_x, target_y = 455111.333, 6406118.386
target_point = Point(target_x, target_y)

# Load labels
gdf = gpd.read_file(GPKG_PATH)
gdf = gdf[gdf.geometry.type == "LineString"]

print(f"Total objects: {len(gdf)}")
print(f"Looking for object near: ({target_x}, {target_y})")

# Find nearest object
gdf["distance"] = gdf.geometry.distance(target_point)
nearest = gdf.nsmallest(1, "distance").iloc[0]

print(f"Nearest object distance: {nearest['distance']:.2f}m")
print(f"Object length: {nearest.geometry.length:.2f}m")
print(f"Object bounds: {nearest.geometry.bounds}")

# Load CHM
with rasterio.open(CHM_PATH) as src:
    chm = src.read(1).astype(np.float32)
    transform = src.transform
    bounds = src.bounds

print(f"CHM bounds: {bounds}")

# Center on target point and create 128x128 box
row_center, col_center = rasterio.transform.rowcol(transform, target_x, target_y)
print(f"Target point in CHM: row={row_center}, col={col_center}")

pad = 64
row_start = max(0, row_center - pad)
row_end = min(chm.shape[0], row_center + pad)
col_start = max(0, col_center - pad)
col_end = min(chm.shape[1], col_center + pad)

# Adjust if too close to edge
if row_end - row_start < 128:
    if row_end == chm.shape[0]:
        row_start = max(0, row_end - 128)
    else:
        row_end = min(chm.shape[0], row_start + 128)

if col_end - col_start < 128:
    if col_end == chm.shape[1]:
        col_start = max(0, col_end - 128)
    else:
        col_end = min(chm.shape[1], col_start + 128)

chm_tile = chm[row_start:row_end, col_start:col_end]
print(f"Extracted CHM tile: shape={chm_tile.shape}")

# Get geospatial bounds
x1, y1 = rasterio.transform.xy(transform, row_end, col_start)
x2, y2 = rasterio.transform.xy(transform, row_start, col_end)
x_min, x_max = min(x1, x2), max(x1, x2)
y_min, y_max = min(y1, y2), max(y1, y2)

print(f"Geospatial bounds: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")

# Extract LAZ points
laz_subset = []
with laspy.open(LAZ_PATH) as lf:
    for chunk in lf.chunk_iterator(points_per_iteration=1000000):
        mask = (
            (chunk.x >= x_min) & (chunk.x <= x_max) &
            (chunk.y >= y_min) & (chunk.y <= y_max)
        )
        if mask.any():
            laz_subset.append(
                np.column_stack((chunk.x[mask], chunk.y[mask], chunk.z[mask]))
            )

if laz_subset:
    laz_subset = np.vstack(laz_subset)
    print(f"LAZ points (absolute): {laz_subset.shape[0]}, z-range: {laz_subset[:, 2].min():.2f} - {laz_subset[:, 2].max():.2f}m")

    # Load DEM to compute HAG
    try:
        with rasterio.open(DEM_PATH) as src:
            dem = src.read(1).astype(np.float32)
            dem_transform = src.transform

        # Interpolate DEM values at LAZ point locations
        dem_z = []
        for x, y, z in laz_subset:
            row, col = rasterio.transform.rowcol(dem_transform, x, y)
            row = int(np.clip(row, 0, dem.shape[0] - 1))
            col = int(np.clip(col, 0, dem.shape[1] - 1))
            dem_z.append(dem[row, col])

        dem_z = np.array(dem_z)

        # Compute HAG
        laz_subset[:, 2] = laz_subset[:, 2] - dem_z
        print(f"LAZ points (HAG): z-range: {laz_subset[:, 2].min():.2f} - {laz_subset[:, 2].max():.2f}m")
    except Exception as e:
        print(f"Warning: Could not compute HAG: {e}")
        print("Using absolute heights")
else:
    print("No LAZ points found!")
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
    "object_info": {
        "target_x": float(target_x),
        "target_y": float(target_y),
        "nearest_distance": float(nearest["distance"]),
        "object_length": float(nearest.geometry.length),
    },
}

with open("_real_example_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nData saved to _real_example_data.pkl")
