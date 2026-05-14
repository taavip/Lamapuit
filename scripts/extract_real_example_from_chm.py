#!/usr/bin/env python3
"""Extract real example from 406455 CHM and LAZ (find interesting region)."""

from pathlib import Path
import numpy as np
import rasterio
import laspy

LAZ_PATH = Path("source/406455_2021_tava/laz_input/406455_2021_tava.laz")
CHM_PATH = Path("chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif")

# Load CHM
with rasterio.open(CHM_PATH) as src:
    chm = src.read(1).astype(np.float32)
    transform = src.transform

print(f"CHM shape: {chm.shape}, range: {chm[chm > 0].min():.2f} - {chm.max():.2f}m")

# Find interesting 128x128 region: one with good height variation
kernel_size = 128
best_var = 0
best_pos = (0, 0)

# Sample grid positions
for r in range(0, chm.shape[0] - kernel_size, 256):
    for c in range(0, chm.shape[1] - kernel_size, 256):
        tile = chm[r:r+kernel_size, c:c+kernel_size]
        nonzero = tile[tile > 0.05]  # Exclude nodata/ground
        if len(nonzero) > 20:
            var = np.var(nonzero)
            if var > best_var:
                best_var = var
                best_pos = (r, c)

row_start, col_start = best_pos
row_end = min(row_start + 128, chm.shape[0])
col_end = min(col_start + 128, chm.shape[1])

chm_tile = chm[row_start:row_end, col_start:col_end]
print(f"Selected tile at ({row_start}, {col_start}): variance={best_var:.4f}, shape={chm_tile.shape}")
print(f"Tile CHM range: {chm_tile[chm_tile > 0].min():.2f} - {chm_tile.max():.2f}m")

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
            laz_subset.append(np.column_stack((chunk.x[mask], chunk.y[mask], chunk.z[mask])))

if laz_subset:
    laz_subset = np.vstack(laz_subset)
    print(f"LAZ points extracted: {laz_subset.shape[0]}, z-range: {laz_subset[:, 2].min():.2f} - {laz_subset[:, 2].max():.2f}m")
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
}

with open("_real_example_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nData saved to _real_example_data.pkl")
