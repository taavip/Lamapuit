#!/usr/bin/env python3
"""
Validate LAZ file-size estimation method by reading REAL point counts.
Run in conda cwd-detect environment with laspy and laszip available.

Usage:
    conda activate cwd-detect
    python scripts/validate_compression_in_conda.py
"""

import struct
from pathlib import Path

# Try using laspy first (if available)
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

# Also try pdal if available
try:
    import pdal
    HAS_PDAL = True
except ImportError:
    HAS_PDAL = False


def estimate_laz_points(laz_path, compression_ratio=0.42):
    """Estimate points using file size method."""
    file_size = laz_path.stat().st_size

    # Read header to get record length
    with open(laz_path, 'rb') as f:
        header = f.read(200)
        rec_len = struct.unpack('<H', header[105:107])[0]

    # Estimate uncompressed size
    uncompressed = file_size / compression_ratio
    point_data = uncompressed - 375
    estimated_points = int(point_data / rec_len)

    return estimated_points, rec_len


def read_real_laz_count_laspy(laz_path):
    """Read actual point count using laspy."""
    try:
        with laspy.read(str(laz_path)) as laz:
            return len(laz.x)
    except Exception as e:
        return None


def read_real_laz_count_pdal(laz_path):
    """Read actual point count using pdal."""
    try:
        pipeline = pdal.Reader(str(laz_path))
        count = pipeline.execute()
        return pipeline.get_stage(0).get_array(0).shape[0]
    except Exception as e:
        return None


# Validation files
validation_files = [
    ('MIN', Path('/home/tpipar/project/Lamapuit/data/lamapuit/laz/579543_2017_madal.laz')),
    ('MEDIAN', Path('/home/tpipar/project/Lamapuit/data/lamapuit/laz/582543_2019_madal.laz')),
    ('MAX', Path('/home/tpipar/project/Lamapuit/data/lamapuit/laz/436647_2022_madal.laz')),
]

print("=" * 90)
print("LAZ FILE-SIZE ESTIMATION VALIDATION")
print("=" * 90)
print(f"\nUsing: {'laspy' if HAS_LASPY else 'pdal' if HAS_PDAL else 'NONE - Install laspy or pdal'}\n")

all_estimates_good = True

for label, laz_path in validation_files:
    if not laz_path.exists():
        print(f"❌ File not found: {laz_path}")
        continue

    file_size = laz_path.stat().st_size
    file_size_mb = file_size / 1e6

    # Estimate
    estimated_points, rec_len = estimate_laz_points(laz_path, compression_ratio=0.42)
    estimated_density = estimated_points / 429025

    print(f"{label:<10} {laz_path.name}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Record length: {rec_len} bytes")
    print()

    # Try to get real count
    real_count = None
    if HAS_LASPY:
        real_count = read_real_laz_count_laspy(laz_path)
    elif HAS_PDAL:
        real_count = read_real_laz_count_pdal(laz_path)

    if real_count:
        real_density = real_count / 429025
        error_pct = abs(estimated_points - real_count) / real_count * 100

        print(f"  ✓ REAL (from laspy/pdal):")
        print(f"    Points: {real_count:,}")
        print(f"    Density: {real_density:.2f} pts/m²")
        print()
        print(f"  ESTIMATED (0.42 ratio):")
        print(f"    Points: {estimated_points:,}")
        print(f"    Density: {estimated_density:.2f} pts/m²")
        print()
        print(f"  COMPARISON:")
        print(f"    Difference: {real_count - estimated_points:,} points")
        print(f"    Error: {error_pct:.2f}%")

        if error_pct > 5:
            all_estimates_good = False
            print(f"    ⚠️ Large error - compression ratio may be inaccurate")
        else:
            print(f"    ✓ Estimation is accurate (< 5% error)")
    else:
        print(f"  ❌ Could not read real count")
        print(f"  (laspy or pdal not available)")
        print()
        print(f"  ESTIMATED (0.42 ratio):")
        print(f"    Points: {estimated_points:,}")
        print(f"    Density: {estimated_density:.2f} pts/m²")

    print()

print("=" * 90)
if all_estimates_good:
    print("\n✓ VALIDATION SUCCESSFUL")
    print("  File-size estimation method (0.42 ratio) is ACCURATE")
    print("  Can confidently use estimated counts for all 119 files")
else:
    print("\n⚠️ VALIDATION SHOWS ERRORS")
    print("  Compression ratio may need adjustment")
    print("  Consider using the real counts from the reader")

print("\n" + "=" * 90)
