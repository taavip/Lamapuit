#!/usr/bin/env python3
"""Extract example using IDW HAG computation (same as baseline CHM)."""

from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import laspy
from scipy.spatial import cKDTree

GPKG_PATH = Path("lamapuit.gpkg")
LAZ_PATH = Path("source/406455_2021_tava/laz_input/406455_2021_tava.laz")
CHM_PATH = Path("chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")

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

# Load CHM
with rasterio.open(CHM_PATH) as src:
    chm = src.read(1).astype(np.float32)
    transform = src.transform

# Center on target point and create 128x128 box
row_center, col_center = rasterio.transform.rowcol(transform, target_x, target_y)

pad = 64
row_start = max(0, row_center - pad)
row_end = min(chm.shape[0], row_center + pad)
col_start = max(0, col_center - pad)
col_end = min(chm.shape[1], col_center + pad)

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

# Extract ALL LAZ points (including ground classification)
print("Extracting LAZ points...")
laz_points_all = []
laz_classification_all = []

with laspy.open(LAZ_PATH) as lf:
    for chunk in lf.chunk_iterator(points_per_iteration=1000000):
        mask = (
            (chunk.x >= x_min) & (chunk.x <= x_max) &
            (chunk.y >= y_min) & (chunk.y <= y_max)
        )
        if mask.any():
            laz_points_all.append(
                np.column_stack((chunk.x[mask], chunk.y[mask], chunk.z[mask]))
            )
            try:
                laz_classification_all.append(chunk.classification[mask])
            except:
                laz_classification_all.append(np.full(mask.sum(), 0))

if not laz_points_all:
    print("No LAZ points found!")
    exit(1)

laz_points = np.vstack(laz_points_all)
laz_classification = np.concatenate(laz_classification_all)

print(f"LAZ points extracted: {laz_points.shape[0]}")
print(f"Absolute Z-range: {laz_points[:, 2].min():.2f} - {laz_points[:, 2].max():.2f}m")

# Identify ground points (classification == 2)
ground_mask = laz_classification == 2
n_ground = np.count_nonzero(ground_mask)
print(f"Ground points found: {n_ground}")

if n_ground < 3:
    print("Warning: Less than 3 ground points, using all points as ground")
    ground_mask = np.ones(len(laz_points), dtype=bool)

# Build cKDTree from ground points
ground_pts = laz_points[ground_mask, :2]  # XY only
ground_z = laz_points[ground_mask, 2]    # Z values

print(f"Building spatial index from {len(ground_pts)} ground points...")
tree = cKDTree(ground_pts)

# Query 3 nearest ground points for each point and compute IDW
print("Computing HAG using IDW from 3 nearest ground points...")
pts = laz_points[:, :2]  # XY
try:
    dists, idx = tree.query(pts, k=3, workers=-1)
except TypeError:
    dists, idx = tree.query(pts, k=3)

# Inverse distance weighted interpolation
weights = 1.0 / (dists + 1e-8)
ground_z_interp = (weights * ground_z[idx]).sum(axis=1) / weights.sum(axis=1)

# Compute HAG
hag = laz_points[:, 2] - ground_z_interp
laz_points[:, 2] = hag  # Replace Z with HAG

print(f"HAG Z-range: {hag.min():.2f} - {hag.max():.2f}m")

# Save output
import pickle

data = {
    "chm_tile": chm_tile,
    "laz_subset": laz_points,
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
