#!/usr/bin/env python
"""
CDW Detection CLI - Run detection on CHM rasters.

Usage:
    python scripts/run_detection.py --chm path/to/chm.tif --model path/to/best.pt --output detections.gpkg
"""

import argparse
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cdw_detect import CDWDetector


def main():
    parser = argparse.ArgumentParser(description='Detect CDW in CHM rasters')
    parser.add_argument('--chm', required=True, help='Path to CHM GeoTIFF')
    parser.add_argument('--model', required=True, help='Path to trained YOLO model (.pt)')
    parser.add_argument('--output', required=True, help='Output GeoPackage path')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--min-area', type=float, default=0.5, help='Minimum area in m²')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    print(f"CDW Detection")
    print(f"  CHM: {args.chm}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Confidence: {args.confidence}")
    print()
    
    detector = CDWDetector(
        model_path=args.model,
        confidence=args.confidence,
        min_area_m2=args.min_area,
        device=args.device,
    )
    
    detections = detector.detect(
        chm_path=args.chm,
        output_path=args.output,
    )
    
    print(f"\nResults:")
    print(f"  Total detections: {len(detections)}")
    if len(detections) > 0:
        print(f"  Confidence range: {detections['confidence'].min():.3f} - {detections['confidence'].max():.3f}")
        print(f"  Area range: {detections['area_m2'].min():.1f} - {detections['area_m2'].max():.1f} m²")


if __name__ == '__main__':
    main()
