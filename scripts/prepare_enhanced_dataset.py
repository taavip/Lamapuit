#!/usr/bin/env python
"""
Prepare enhanced YOLO training data from multiple CHM files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cdw_detect.prepare import YOLODataPreparer, augment_with_nodata
import geopandas as gpd
import rasterio

# Find all overlapping CHM files
labels = gpd.read_file("examples/data/lamapuit_labels.gpkg")
label_bounds = labels.total_bounds

chm_files = list(Path("data/chm_max_hag").glob("*.tif"))
overlapping_chms = []

print("Finding overlapping CHM files...")
for chm_file in sorted(chm_files):
    try:
        with rasterio.open(chm_file) as src:
            bounds = src.bounds
            overlaps = not (
                bounds.right < label_bounds[0]
                or bounds.left > label_bounds[2]
                or bounds.top < label_bounds[1]
                or bounds.bottom > label_bounds[3]
            )
            if overlaps:
                overlapping_chms.append(chm_file)
    except Exception:
        pass

print(f"Found {len(overlapping_chms)} overlapping CHM files\n")

# Prepare dataset from all overlapping CHMs
output_dir = "data/dataset_enhanced"
preparer = YOLODataPreparer(
    output_dir=output_dir,
    tile_size=640,
    buffer_width=0.5,
    overlap=0.3,  # Increased overlap for more augmentation
    min_log_pixels=30,  # Lower threshold for more samples
    val_split=0.2,
)

total_stats = {"total": 0, "with_cdw": 0, "empty": 0, "skipped": 0}

for i, chm_file in enumerate(overlapping_chms, 1):
    print(f"\n[{i}/{len(overlapping_chms)}] Processing {chm_file.name}...")

    stats = preparer.prepare(
        chm_path=str(chm_file),
        labels_path="examples/data/lamapuit_labels.gpkg",
    )

    print(
        f"  Tiles: {stats['total']}, With CDW: {stats['with_cdw']}, Empty: {stats['empty']}, Skipped: {stats['skipped']}"
    )

    total_stats["total"] += stats["total"]
    total_stats["with_cdw"] += stats["with_cdw"]
    total_stats["empty"] += stats["empty"]
    total_stats["skipped"] += stats["skipped"]

print(f"\n{'='*60}")
print(f"TOTAL DATASET STATISTICS:")
print(f"  Total tiles: {total_stats['total']}")
print(f"  With CDW: {total_stats['with_cdw']}")
print(f"  Empty: {total_stats['empty']}")
print(f"  Skipped: {total_stats['skipped']}")
print(f"{'='*60}")

# Apply nodata augmentation
print("\nApplying nodata augmentation for robustness...")
output_robust = output_dir + "_robust"
augment_with_nodata(output_dir, output_robust)
print(f"Robust dataset created: {output_robust}")
