"""GeoJSON polygon map of split assignments.

Generates a FeatureCollection where each feature is a 1 km × 1 km tile polygon
in EPSG:3301 (L-EST97), coloured by split role (train / val / test / buffer).

Coordinate convention (confirmed by reading actual CHM rasters):
    easting_west   = grid_y * 1 000
    northing_south = 6 000 000 + grid_x * 1 000
    tile size      = 1 000 m × 1 000 m
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Role → fill colour (hex, for QGIS / Mapbox style hints stored as a property)
ROLE_COLOUR: dict[str, str] = {
    "train": "#2196F3",   # blue
    "val": "#FF9800",     # orange
    "test": "#4CAF50",    # green
    "buffer": "#9E9E9E",  # grey
}


def _tile_polygon(grid_x: int, grid_y: int) -> list[list[float]]:
    """Return GeoJSON exterior ring coords for a 1 km tile (closed, CCW)."""
    west = float(grid_y * 1000)
    south = float(6_000_000 + grid_x * 1000)
    east = west + 1000.0
    north = south + 1000.0
    # Counter-clockwise (GeoJSON convention for exterior ring)
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def write_split_geojson(
    by_place: dict[str, dict[str, Any]],
    role_of: dict[str, str],
    region_of: dict[str, int],
    meta: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a GeoJSON FeatureCollection for all labeled places.

    Each place becomes one polygon (the physical tile extent).
    Multi-year places have their label counts aggregated.

    Feature properties
    ------------------
    place_key        : str    — year-agnostic identifier (e.g. "436648_madal")
    tile_id          : str    — 6-digit tile code (e.g. "436648")
    site             : str
    grid_x           : int    — northing index
    grid_y           : int    — easting index
    easting_west     : int    — L-EST97 easting of western tile edge (m)
    northing_south   : int    — L-EST97 northing of southern tile edge (m)
    split_role       : str    — "train" | "val" | "test" | "buffer"
    region_id        : int    — geographic cluster id
    n_rows_total     : int    — total labeled rows for this place (all years)
    n_manual         : int    — manually reviewed label rows
    n_auto           : int    — auto-threshold-gate label rows
    n_skipped        : int    — skipped / unlabeled rows (reserved, always 0 here)
    n_cdw            : int    — CWD-positive rows
    n_no_cdw         : int    — CWD-negative rows
    n_years          : int    — number of flight years with labels
    years            : str    — comma-separated sorted year list
    colour           : str    — suggested fill colour (hex)
    config_seed      : int    — random seed used for this split
    config_buffer_tiles : int — Chebyshev buffer distance in grid units
    config_test_fraction : float
    config_val_fraction  : float
    config_stratify      : bool
    """
    seed = int(meta.get("seed", 0))
    buffer_tiles = int(meta.get("buffer_tiles", 1))
    test_frac = float(meta.get("test_fraction_target", 0.2))
    val_frac = float(meta.get("val_fraction_target", 0.0))
    stratify = bool(meta.get("stratify_regions", False))

    features = []
    for place_key, info in sorted(by_place.items()):
        gx = info["grid_x"]
        gy = info["grid_y"]
        role = role_of.get(place_key, "unknown")
        region = region_of.get(place_key, -1)
        n_manual = info.get("n_manual", 0)
        n_threshold = info.get("n_threshold_gate", 0)
        # n_auto = everything that is not manual and not skipped
        n_auto = len(info["keys"]) - n_manual
        years_list = sorted(info.get("years", set()))

        feature: dict[str, Any] = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [_tile_polygon(gx, gy)],
            },
            "properties": {
                "place_key": place_key,
                "tile_id": info.get("tile_id", ""),
                "site": info.get("site", ""),
                "grid_x": int(gx),
                "grid_y": int(gy),
                "easting_west": int(gy * 1000),
                "northing_south": int(6_000_000 + gx * 1000),
                "split_role": role,
                "region_id": int(region),
                "n_rows_total": int(len(info["keys"])),
                "n_manual": int(n_manual),
                "n_auto": int(n_auto),
                "n_skipped": 0,
                "n_cdw": int(info.get("n_cdw", 0)),
                "n_no_cdw": int(info.get("n_no_cdw", 0)),
                "n_years": int(len(years_list)),
                "years": ",".join(years_list),
                "colour": ROLE_COLOUR.get(role, "#FF00FF"),
                "config_seed": seed,
                "config_buffer_tiles": buffer_tiles,
                "config_test_fraction": test_frac,
                "config_val_fraction": val_frac,
                "config_stratify": stratify,
            },
        }
        features.append(feature)

    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::3301"},
        },
        "metadata": {
            "description": "CWD detection dataset spatial split",
            "split_version": meta.get("split_version", "v2_distance"),
            "created_at": meta.get("created_at", ""),
            "total_places": int(len(by_place)),
            "n_test": int(meta.get("n_places_test", 0)),
            "n_val": int(meta.get("n_places_val", 0)),
            "n_train": int(meta.get("n_places_train", 0)),
            "n_buffer": int(meta.get("n_places_buffer", 0)),
            "buffer_tiles": buffer_tiles,
            "seed": seed,
        },
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fc, indent=2, ensure_ascii=False))
