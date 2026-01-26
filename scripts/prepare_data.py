#!/usr/bin/env python
"""
Prepare YOLO training data from CHM and vector labels.

Usage:
    python scripts/prepare_data.py --chm path/to/chm.tif --labels path/to/labels.gpkg --output data/dataset
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cdw_detect import YOLODataPreparer
from cdw_detect.prepare import augment_with_nodata


def main():
    parser = argparse.ArgumentParser(description='Prepare YOLO training data')
    parser.add_argument('--chm', required=True, help='Path to CHM GeoTIFF')
    parser.add_argument('--labels', required=True, help='Path to labels GeoPackage (LineStrings)')
    parser.add_argument('--output', required=True, help='Output dataset directory')
    parser.add_argument('--buffer', type=float, default=0.5, help='Buffer width in meters')
    parser.add_argument('--tile-size', type=int, default=640, help='Tile size in pixels')
    parser.add_argument('--augment-nodata', action='store_true', help='Add nodata augmentation')
    
    args = parser.parse_args()
    
    print(f"Preparing YOLO dataset")
    print(f"  CHM: {args.chm}")
    print(f"  Labels: {args.labels}")
    print(f"  Output: {args.output}")
    print(f"  Buffer: {args.buffer}m")
    print()
    
    preparer = YOLODataPreparer(
        output_dir=args.output,
        tile_size=args.tile_size,
        buffer_width=args.buffer,
    )
    
    stats = preparer.prepare(
        chm_path=args.chm,
        labels_path=args.labels,
    )
    
    print(f"\nDataset created:")
    print(f"  Total tiles: {stats['total']}")
    print(f"  With CDW: {stats['with_cdw']}")
    print(f"  Empty: {stats['empty']}")
    print(f"  Skipped (nodata): {stats['skipped']}")
    
    if args.augment_nodata:
        print("\nApplying nodata augmentation...")
        output_robust = args.output + '_robust'
        augment_with_nodata(args.output, output_robust)
        print(f"Augmented dataset: {output_robust}")


if __name__ == '__main__':
    main()
