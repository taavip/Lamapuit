#!/usr/bin/env python
"""
Create CDW labeling GeoPackage with comprehensive schema.

Filename pattern: {tile_id}_{year}_{source}_chm_max_hag_{resolution}.tif
Example: 465663_2023_madal_chm_max_hag_20cm.tif

Schema designed for:
- Instance segmentation of CDW (fallen logs, root plates, log piles)
- Linking labels to source CHM rasters
- Tracking overlapping objects
- Quality control and annotation metadata
"""

import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import argparse


def get_unique_values_from_filenames(chm_dir: str) -> dict:
    """
    Extract unique values from CHM filenames for validation.

    Pattern: {tile_id}_{year}_{source}_chm_max_hag_{resolution}.tif
    """
    tif_files = [f for f in os.listdir(chm_dir) if f.endswith(".tif")]

    tile_ids = set()
    years = set()
    sources = set()
    resolutions = set()

    for f in tif_files:
        parts = f.replace(".tif", "").split("_")
        if len(parts) >= 6:
            tile_ids.add(parts[0])
            years.add(parts[1])
            sources.add(parts[2])
            resolutions.add(parts[-1])

    return {
        "tile_ids": sorted(tile_ids),
        "years": sorted(years),
        "sources": sorted(sources),
        "resolutions": sorted(resolutions),
        "total_files": len(tif_files),
    }


def create_cdw_labels_gpkg(
    output_path: str,
    crs: str = "EPSG:3301",
    chm_dir: str = None,
) -> gpd.GeoDataFrame:
    """
    Create empty GeoPackage with CDW labeling schema.

    Schema fields:

    IDENTIFICATION:
    - instance_id: Unique ID for each CDW object
    - source_raster: Filename of the CHM raster this label belongs to
    - tile_id: Map tile ID (e.g., "465663")
    - acquisition_year: Year of LiDAR acquisition
    - acquisition_source: Data source (e.g., "mets", "tava", "madal")

    CDW PROPERTIES:
    - has_root_plate: Boolean - does this CDW include a root plate?
    - is_log_pile: Boolean - is this multiple logs grouped together?
    - is_partial: Boolean - does this CDW extend beyond tile boundary?

    OVERLAP HANDLING:
    - z_order: Stacking order for overlapping CDW (0=bottom, higher=top)
    - overlaps_with: Comma-separated instance_ids of overlapping CDW

    MEASUREMENTS (optional, estimated):
    - length_m: Estimated length in meters
    - diameter_m: Estimated diameter in meters
    - height_above_ground_m: Height above ground from CHM

    QUALITY & ANNOTATION:
    - certainty: 1=certain, 2=probable, 3=uncertain
    - annotator: Name/ID of person who created this label
    - annotation_date: Date of annotation
    - notes: Free text notes

    Args:
        output_path: Path for output GeoPackage
        crs: Coordinate reference system (default: Estonian EPSG:3301)
        chm_dir: Optional path to CHM directory for metadata extraction

    Returns:
        Empty GeoDataFrame with schema
    """
    # Define schema with appropriate types
    schema = {
        # Identification
        "instance_id": pd.Series(dtype="Int64"),
        "source_raster": pd.Series(dtype="object"),
        "tile_id": pd.Series(dtype="object"),
        "acquisition_year": pd.Series(dtype="Int64"),
        "acquisition_source": pd.Series(dtype="object"),
        # CDW properties (boolean)
        "has_root_plate": pd.Series(dtype="bool"),
        "is_log_pile": pd.Series(dtype="bool"),
        "is_partial": pd.Series(dtype="bool"),
        # Overlap handling
        "z_order": pd.Series(dtype="Int64"),
        "overlaps_with": pd.Series(dtype="object"),
        # Measurements
        "length_m": pd.Series(dtype="float64"),
        "diameter_m": pd.Series(dtype="float64"),
        "height_above_ground_m": pd.Series(dtype="float64"),
        # Quality & annotation
        "certainty": pd.Series(dtype="Int64"),
        "annotator": pd.Series(dtype="object"),
        "annotation_date": pd.Series(dtype="object"),
        "notes": pd.Series(dtype="object"),
    }

    # Create empty GeoDataFrame
    gdf = gpd.GeoDataFrame(
        schema,
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )

    # Save to GeoPackage
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(output_path, driver="GPKG", layer="cdw_labels")

    # Print info
    print(f"Created: {output_path}")
    print(f"Layer: cdw_labels")
    print(f"CRS: {crs}")
    print(f"\nSchema ({len(schema)} fields):")
    print("-" * 50)

    field_groups = {
        "Identification": [
            "instance_id",
            "source_raster",
            "tile_id",
            "acquisition_year",
            "acquisition_source",
        ],
        "CDW Properties": ["has_root_plate", "is_log_pile", "is_partial"],
        "Overlap Handling": ["z_order", "overlaps_with"],
        "Measurements": ["length_m", "diameter_m", "height_above_ground_m"],
        "Quality & Annotation": ["certainty", "annotator", "annotation_date", "notes"],
    }

    for group_name, fields in field_groups.items():
        print(f"\n{group_name}:")
        for field in fields:
            dtype = str(gdf[field].dtype)
            print(f"  - {field}: {dtype}")

    # Print CHM metadata if available
    if chm_dir and os.path.exists(chm_dir):
        print("\n" + "=" * 50)
        print("CHM Raster Metadata:")
        print("=" * 50)
        meta = get_unique_values_from_filenames(chm_dir)
        print(f"Total CHM files: {meta['total_files']}")
        print(f"Tile IDs: {len(meta['tile_ids'])} unique")
        print(f"Years: {', '.join(meta['years'])}")
        print(f"Sources: {', '.join(meta['sources'])}")
        print(f"Resolutions: {', '.join(meta['resolutions'])}")

    return gdf


def create_example_labels(gpkg_path: str, n_examples: int = 3) -> None:
    """
    Add example labels to demonstrate the schema.
    These should be removed before actual labeling.
    """
    # Load existing
    gdf = gpd.read_file(gpkg_path, layer="cdw_labels")

    # Create example polygons (simple rectangles for demonstration)
    examples = [
        {
            "instance_id": 1,
            "source_raster": "465663_2023_madal_chm_max_hag_20cm.tif",
            "tile_id": "465663",
            "acquisition_year": 2023,
            "acquisition_source": "madal",
            "has_root_plate": False,
            "is_log_pile": False,
            "is_partial": False,
            "z_order": 0,
            "overlaps_with": None,
            "length_m": 12.5,
            "diameter_m": 0.35,
            "height_above_ground_m": 0.4,
            "certainty": 1,
            "annotator": "example",
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "notes": "Example: single fallen log",
            "geometry": Polygon(
                [(656500, 6465500), (656512, 6465500), (656512, 6465500.7), (656500, 6465500.7)]
            ),
        },
        {
            "instance_id": 2,
            "source_raster": "465663_2023_madal_chm_max_hag_20cm.tif",
            "tile_id": "465663",
            "acquisition_year": 2023,
            "acquisition_source": "madal",
            "has_root_plate": True,
            "is_log_pile": False,
            "is_partial": False,
            "z_order": 0,
            "overlaps_with": None,
            "length_m": 8.0,
            "diameter_m": 0.5,
            "height_above_ground_m": 0.6,
            "certainty": 1,
            "annotator": "example",
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "notes": "Example: log with root plate",
            "geometry": Polygon(
                [(656520, 6465510), (656528, 6465510), (656528, 6465511), (656520, 6465511)]
            ),
        },
        {
            "instance_id": 3,
            "source_raster": "465663_2023_madal_chm_max_hag_20cm.tif",
            "tile_id": "465663",
            "acquisition_year": 2023,
            "acquisition_source": "madal",
            "has_root_plate": False,
            "is_log_pile": True,
            "is_partial": False,
            "z_order": 0,
            "overlaps_with": None,
            "length_m": None,
            "diameter_m": None,
            "height_above_ground_m": 0.8,
            "certainty": 2,
            "annotator": "example",
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "notes": "Example: log pile (multiple logs)",
            "geometry": Polygon(
                [(656540, 6465520), (656548, 6465520), (656548, 6465524), (656540, 6465524)]
            ),
        },
    ]

    example_gdf = gpd.GeoDataFrame(examples, crs=gdf.crs)

    # Save to separate layer
    example_gdf.to_file(gpkg_path, driver="GPKG", layer="cdw_labels_examples")

    print(f"\nAdded {len(examples)} example labels to layer 'cdw_labels_examples'")
    print("NOTE: Remove this layer before actual labeling!")


def main():
    parser = argparse.ArgumentParser(
        description="Create CDW labeling GeoPackage with comprehensive schema"
    )
    parser.add_argument(
        "--output",
        default="data/labels/cdw_labels.gpkg",
        help="Output GeoPackage path (default: data/labels/cdw_labels.gpkg)",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:3301",
        help="Coordinate reference system (default: EPSG:3301 Estonian)",
    )
    parser.add_argument(
        "--chm-dir",
        default="chm_max_hag",
        help="Path to CHM directory for metadata extraction",
    )
    parser.add_argument(
        "--add-examples",
        action="store_true",
        help="Add example labels to demonstrate schema",
    )

    args = parser.parse_args()

    # Create the GeoPackage
    gdf = create_cdw_labels_gpkg(
        output_path=args.output,
        crs=args.crs,
        chm_dir=args.chm_dir,
    )

    # Add examples if requested
    if args.add_examples:
        create_example_labels(args.output)

    print(f"\n✓ GeoPackage ready for labeling in QGIS!")
    print(f"  Open: {args.output}")


if __name__ == "__main__":
    main()
