#!/usr/bin/env python3
"""
Validate LAZ file-size estimation by decompressing actual files.
Tests min/max/median sized LAZ files to verify compression ratio.
"""

import struct
import zlib
from pathlib import Path

def read_laz_header(laz_path):
    """Read LAS header from LAZ file."""
    with open(laz_path, 'rb') as f:
        header = f.read(375)

        # Extract key fields
        rec_len = struct.unpack('<H', header[105:107])[0]

        # Point count fields (LAS 1.0-1.2 vs 1.4)
        point_count_std = struct.unpack('<I', header[107:111])[0]
        point_count_ext = struct.unpack('<Q', header[130:138])[0]

        # Start of first point record
        offset_to_points = struct.unpack('<H', header[94:96])[0]

        return {
            'rec_len': rec_len,
            'point_count_std': point_count_std,
            'point_count_ext': point_count_ext,
            'offset_to_points': offset_to_points
        }

def decompress_laz_simple(laz_path):
    """
    Attempt simple LAZ decompression.
    LAZ stores header + compressed point records.
    """
    with open(laz_path, 'rb') as f:
        # Read header
        header = f.read(375)
        rec_len = struct.unpack('<H', header[105:107])[0]

        # Offset to point data
        offset_to_points = struct.unpack('<H', header[94:96])[0]

        # Skip to point data
        f.seek(offset_to_points)

        # Read all remaining compressed data
        compressed_data = f.read()

    try:
        # Try simple zlib decompression (some LAZ files use this)
        decompressed = zlib.decompress(compressed_data)
        point_count = len(decompressed) // rec_len
        return point_count, len(decompressed)
    except:
        return None, None

def estimate_laz_points(laz_path, compression_ratio=0.42):
    """Estimate points using file size method."""
    file_size = laz_path.stat().st_size
    header = read_laz_header(laz_path)
    rec_len = header['rec_len']

    # Estimate uncompressed size
    uncompressed = file_size / compression_ratio
    point_data = uncompressed - 375
    estimated_points = int(point_data / rec_len)

    return estimated_points

# Validation files
validation_files = [
    Path('/home/tpipar/project/Lamapuit/data/lamapuit/laz/579543_2017_madal.laz'),  # MIN
    Path('/home/tpipar/project/Lamapuit/data/lamapuit/laz/582543_2019_madal.laz'),  # MEDIAN
    Path('/home/tpipar/project/Lamapuit/data/lamapuit/laz/436647_2022_madal.laz'),  # MAX
]

print("="*80)
print("LAZ FILE-SIZE ESTIMATION VALIDATION")
print("="*80)
print()

for laz_path in validation_files:
    if not laz_path.exists():
        print(f"❌ File not found: {laz_path}")
        continue

    file_size = laz_path.stat().st_size
    file_size_mb = file_size / 1e6

    # Get header info
    header = read_laz_header(laz_path)
    rec_len = header['rec_len']

    # Estimate using file-size method
    estimated_points = estimate_laz_points(laz_path, compression_ratio=0.42)
    estimated_density = estimated_points / 429025

    # Try decompression
    print(f"\n{laz_path.name}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Record length: {rec_len} bytes")
    print()
    print(f"  ESTIMATED (0.42 ratio):")
    print(f"    Points: {estimated_points:,}")
    print(f"    Density: {estimated_density:.2f} pts/m²")
    print()

    # Try decompression
    print(f"  ATTEMPTING DECOMPRESSION...")
    real_points, decompressed_size = decompress_laz_simple(laz_path)

    if real_points:
        real_density = real_points / 429025
        print(f"    ✓ SUCCESS")
        print(f"    Real points: {real_points:,}")
        print(f"    Real density: {real_density:.2f} pts/m²")
        print(f"    Decompressed size: {decompressed_size:,} bytes")

        # Calculate actual compression ratio
        actual_ratio = file_size / (decompressed_size + 375)
        print(f"    Actual compression ratio: {actual_ratio:.2f}")

        # Error in estimation
        error = abs(estimated_points - real_points) / real_points * 100
        print(f"    Estimation error: {error:.1f}%")
    else:
        print(f"    ❌ Simple decompression failed")
        print(f"    (LAZ uses chunked compression - need specialized library)")

    print()

print("="*80)
print("\nNOTE: If decompression fails, use 'pdal info' or install lazrs-python")
print("Command: pdal info file.laz | grep -i count")
print("="*80)
