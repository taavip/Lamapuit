"""Distance-based spatial train/val/test split with regional stratification.

Improvements over model_search_v4/_splits.py (V1):

1. **1-tile Chebyshev buffer** (1 000 m) instead of the coarse 2-tile-block + 1-block-ring
   (~4 km effective) that wastes 65 % of rows as buffer.
2. **Global buffer check**: operates on real grid coordinates so tiles from different
   geographic clusters are correctly buffered if they happen to be within buffer_tiles km.
3. **Three-way split** (train / val / test): validation set for hyperparameter tuning,
   test set held out for final reporting.
4. **Regional stratification**: proportional sampling from geographic clusters (K-means on
   grid_x / grid_y) ensures test and val sets span the geographic diversity of the dataset.
5. **GeoJSON output**: per-place polygon map coloured by split role, with label statistics.

Output JSON is a superset of the V1 format and accepted by model_search_v4.main as a
drop-in replacement for `cnn_test_split_v4.json`.

References
----------
Roberts et al. 2017, Ecography 40:913   — block CV for structured data
Le Rest et al. 2014, GEB 23:811         — buffered LOO-CV, buffer = variogram range
Milà et al. 2022, MEE 13:1304           — NNDM LOO-CV for small sparse datasets
Ploton et al. 2020, Nat. Commun. 11:4540 — spatial validation reveals inflated performance
"""

from __future__ import annotations

import json
import math
import random
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_tile_id(tile_id: str) -> tuple[int | None, int | None]:
    """Return (grid_x, grid_y) from a 6-digit Estonian ALS tile ID.

    Coordinate convention (EPSG:3301, L-EST97):
        easting_west   = grid_y * 1000
        northing_south = 6_000_000 + grid_x * 1000
    Falls back to (None, None) for non-standard IDs.
    """
    if len(tile_id) == 6 and tile_id.isdigit():
        return int(tile_id[:3]), int(tile_id[3:])
    return None, None


def _chebyshev(gx1: int, gy1: int, gx2: int, gy2: int) -> int:
    """Chebyshev (L-inf) distance in grid units (1 unit = 1 000 m)."""
    return max(abs(gx1 - gx2), abs(gy1 - gy2))


def _pseudo_coords(place_key: str) -> tuple[int, int]:
    """Deterministic fallback coords for non-6-digit tile IDs (CRC32-based)."""
    h = int(zlib.crc32(place_key.encode("utf-8")))
    return (10_000_000 + (h % 10_000), 10_000_000 + ((h // 10_000) % 10_000))


# ---------------------------------------------------------------------------
# K-means geographic clustering (pure Python, seeded)
# ---------------------------------------------------------------------------

def _kmeans(points: list[tuple[int, int]], k: int, seed: int, max_iter: int = 50) -> list[int]:
    """Lloyd's algorithm with seeded initialisation. Returns cluster labels (len = len(points))."""
    if k <= 1 or len(points) <= 1:
        return [0] * len(points)
    k = min(k, len(points))

    rng = random.Random(seed)
    indices = rng.sample(range(len(points)), k)
    centroids = [points[i] for i in indices]

    labels = [0] * len(points)
    for _ in range(max_iter):
        new_labels = []
        for x, y in points:
            best = min(range(k), key=lambda c: (x - centroids[c][0]) ** 2 + (y - centroids[c][1]) ** 2)
            new_labels.append(best)
        if new_labels == labels:
            break
        labels = new_labels
        # Recompute centroids
        sums = [(0, 0, 0)] * k
        for i, (x, y) in enumerate(points):
            c = labels[i]
            sx, sy, cnt = sums[c]
            sums[c] = (sx + x, sy + y, cnt + 1)
        centroids = [
            (sx // cnt, sy // cnt) if cnt > 0 else centroids[c]
            for c, (sx, sy, cnt) in enumerate(sums)
        ]

    return labels


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def _assign_regions(
    by_place: dict[str, dict[str, Any]],
    n_regions: int,
    seed: int,
) -> dict[str, int]:
    """Cluster places into geographic regions. Returns {place_key: region_id}."""
    keys = list(by_place.keys())
    coords = [(by_place[k]["grid_x"], by_place[k]["grid_y"]) for k in keys]
    labels = _kmeans(coords, n_regions, seed=seed)
    return {k: labels[i] for i, k in enumerate(keys)}


def _greedy_select(
    candidates: list[str],
    by_place: dict[str, dict[str, Any]],
    target_rows: int,
    rng: random.Random,
) -> set[str]:
    """Greedy row-count-based selection: pick places that bring total closest to target_rows.

    Ties broken by preferring the larger place (more rows), then random.
    """
    remaining = list(candidates)
    rng.shuffle(remaining)
    selected: set[str] = set()
    selected_rows = 0

    while remaining and selected_rows < target_rows:
        best = None
        best_score: int | None = None
        best_nxt: int | None = None
        for p in remaining:
            nxt = selected_rows + len(by_place[p]["keys"])
            score = abs(target_rows - nxt)
            if best_score is None or score < best_score or (
                score == best_score and (best_nxt is None or nxt > best_nxt)
            ):
                best, best_score, best_nxt = p, score, nxt
        if best is None:
            break
        selected.add(best)
        selected_rows = best_nxt  # type: ignore[assignment]
        remaining.remove(best)

    return selected


def _mark_buffer(
    pool: set[str],
    anchors: set[str],
    by_place: dict[str, dict[str, Any]],
    buffer_tiles: int,
) -> set[str]:
    """Return places in pool that are within buffer_tiles Chebyshev of any anchor place."""
    buf: set[str] = set()
    for p in pool:
        px, py = by_place[p]["grid_x"], by_place[p]["grid_y"]
        for a in anchors:
            ax, ay = by_place[a]["grid_x"], by_place[a]["grid_y"]
            if _chebyshev(px, py, ax, ay) <= buffer_tiles:
                buf.add(p)
                break
    return buf


def write_spatial_distance_split(
    all_candidates: list[dict[str, Any]],
    output_test_split: Path,
    seed: int,
    test_fraction: float,
    val_fraction: float = 0.0,
    buffer_tiles: int = 1,
    n_regions: int = 0,
    stratify_regions: bool = True,
    output_geojson: Path | None = None,
) -> dict[str, Any]:
    """Partition places into (train, val, test, buffer) using distance-based spatial buffering.

    Parameters
    ----------
    all_candidates:
        List of label rows. Each item must have keys:
        ``key`` (tuple[str, int, int]), ``raster`` (str), ``label`` (str),
        ``reason`` (str).
    output_test_split:
        Path for the JSON split manifest (compatible with V1 format).
    seed:
        Random seed. Different seeds yield different but equally valid splits.
    test_fraction:
        Target fraction of rows to assign to test.
    val_fraction:
        Target fraction of rows to assign to val (0 = no val set).
    buffer_tiles:
        Chebyshev distance in grid units (= km) to reserve as buffer around
        test (and val) tiles. Default 1 = 1 000 m separation.
    n_regions:
        Number of K-means geographic clusters for stratification.
        0 = auto (max(2, n_places // 5)).
    stratify_regions:
        If True, sample test/val proportionally from each geographic region.
    output_geojson:
        If given, write a GeoJSON FeatureCollection to this path (see _geojson.py).

    Returns
    -------
    dict with metadata (mirrors V1 meta format, extended with V2 fields).
    """
    from scripts.model_search_v4._labels import parse_raster_identity  # type: ignore

    rng = random.Random(seed)

    # --- Build place registry --------------------------------------------------
    by_place: dict[str, dict[str, Any]] = {}
    for item in all_candidates:
        key = item["key"]
        raster = str(item["raster"])
        ident = parse_raster_identity(raster)
        place_key = str(ident["place_key"])
        gx = ident["grid_x"]
        gy = ident["grid_y"]
        if gx is None or gy is None:
            gx, gy = _pseudo_coords(place_key)

        entry = by_place.get(place_key)
        if entry is None:
            entry = {
                "place_key": place_key,
                "tile_id": str(ident.get("tile", "")),
                "site": str(ident.get("site", "")),
                "grid_x": int(gx),
                "grid_y": int(gy),
                "keys": [],
                "years": set(),
                "n_cdw": 0,
                "n_no_cdw": 0,
                "n_manual": 0,
                "n_threshold_gate": 0,
            }
            by_place[place_key] = entry

        entry["keys"].append(key)
        yr = ident["year"]
        if yr != "unknown":
            entry["years"].add(str(yr))
        if str(item.get("label", "")).strip() == "cdw":
            entry["n_cdw"] += 1
        else:
            entry["n_no_cdw"] += 1
        src = str(item.get("source", ""))
        reason = str(item.get("reason", ""))
        if "manual" in src.lower() or "reviewed" in reason.lower():
            entry["n_manual"] += 1
        elif reason == "threshold_gate":
            entry["n_threshold_gate"] += 1

    total_rows = sum(len(v["keys"]) for v in by_place.values())
    meta_empty: dict[str, Any] = {
        "split_version": "v2_distance",
        "created_at": _utc_now(),
        "total_rows": 0,
        "test_rows": 0,
        "val_rows": 0,
        "train_rows": 0,
        "buffer_rows": 0,
        "buffer_pct": 0.0,
        "n_places_total": 0,
        "n_places_test": 0,
        "n_places_val": 0,
        "n_places_train": 0,
        "n_places_buffer": 0,
        "place_overlap_train_vs_test": 0,
        "seed": int(seed),
        "buffer_tiles": int(buffer_tiles),
        "test_fraction_target": float(test_fraction),
        "val_fraction_target": float(val_fraction),
        "n_regions": 0,
        "stratify_regions": bool(stratify_regions),
    }
    if not by_place:
        output_test_split.parent.mkdir(parents=True, exist_ok=True)
        output_test_split.write_text(json.dumps({"keys": [], "meta": meta_empty}, indent=2))
        return meta_empty

    # --- Geographic regions ---------------------------------------------------
    n_reg = max(2, len(by_place) // 5) if n_regions <= 0 else max(1, int(n_regions))
    region_of = _assign_regions(by_place, n_reg, seed)
    regions: dict[int, list[str]] = defaultdict(list)
    for pk, rid in region_of.items():
        regions[rid].append(pk)

    # --- Select test places ---------------------------------------------------
    target_test_rows = max(1, int(round(total_rows * float(test_fraction))))

    if stratify_regions and len(regions) > 1:
        # Allocate target test rows proportionally across regions.
        test_places: set[str] = set()
        for rid, region_keys in regions.items():
            region_rows = sum(len(by_place[k]["keys"]) for k in region_keys)
            region_target = max(0, int(round(target_test_rows * region_rows / total_rows)))
            chosen = _greedy_select(region_keys, by_place, region_target, rng)
            test_places.update(chosen)
    else:
        test_places = _greedy_select(list(by_place.keys()), by_place, target_test_rows, rng)

    # --- Buffer around test ---------------------------------------------------
    non_test = set(by_place.keys()) - test_places
    test_buffer = _mark_buffer(non_test, test_places, by_place, buffer_tiles)

    # --- Select val places (from non-test, non-buffer) -----------------------
    remaining_after_test = non_test - test_buffer
    val_places: set[str] = set()
    val_buffer: set[str] = set()

    if val_fraction > 0 and remaining_after_test:
        target_val_rows = max(1, int(round(total_rows * float(val_fraction))))
        if stratify_regions and len(regions) > 1:
            for rid, region_keys in regions.items():
                region_cands = [k for k in region_keys if k in remaining_after_test]
                if not region_cands:
                    continue
                region_rows = sum(len(by_place[k]["keys"]) for k in region_cands)
                total_remaining_rows = sum(len(by_place[k]["keys"]) for k in remaining_after_test)
                if total_remaining_rows == 0:
                    continue
                region_target = max(0, int(round(target_val_rows * region_rows / total_remaining_rows)))
                chosen = _greedy_select(region_cands, by_place, region_target, rng)
                val_places.update(chosen)
        else:
            val_places = _greedy_select(list(remaining_after_test), by_place, target_val_rows, rng)

        # Buffer around val (within remaining pool only)
        val_buf_pool = remaining_after_test - val_places
        val_buffer = _mark_buffer(val_buf_pool, val_places, by_place, buffer_tiles)

    # --- Assign roles ---------------------------------------------------------
    combined_buffer = test_buffer | val_buffer
    train_places = set(by_place.keys()) - test_places - val_places - combined_buffer

    # Sanity checks
    assert len(train_places & test_places) == 0, "train / test overlap"
    assert len(train_places & val_places) == 0, "train / val overlap"
    assert len(test_places & val_places) == 0, "test / val overlap"

    # --- Build test key list (V1-compatible) ----------------------------------
    test_keys_sorted = sorted(
        [(str(k[0]), int(k[1]), int(k[2])) for p in test_places for k in by_place[p]["keys"]],
        key=lambda x: (x[0], x[1], x[2]),
    )
    val_keys_sorted = sorted(
        [(str(k[0]), int(k[1]), int(k[2])) for p in val_places for k in by_place[p]["keys"]],
        key=lambda x: (x[0], x[1], x[2]),
    )

    # --- Count rows -----------------------------------------------------------
    test_rows = sum(len(by_place[p]["keys"]) for p in test_places)
    val_rows = sum(len(by_place[p]["keys"]) for p in val_places)
    buf_rows = sum(len(by_place[p]["keys"]) for p in combined_buffer)
    train_rows = sum(len(by_place[p]["keys"]) for p in train_places)

    n_regions_in_test = len(set(region_of[p] for p in test_places)) if test_places else 0
    n_regions_in_val = len(set(region_of[p] for p in val_places)) if val_places else 0

    meta: dict[str, Any] = {
        "split_version": "v2_distance",
        "created_at": _utc_now(),
        "total_rows": int(total_rows),
        "test_rows": int(test_rows),
        "val_rows": int(val_rows),
        "train_rows": int(train_rows),
        "buffer_rows": int(buf_rows),
        "buffer_pct": round(100.0 * buf_rows / total_rows, 2) if total_rows else 0.0,
        "test_fraction_target": float(test_fraction),
        "test_fraction_actual": round(test_rows / total_rows, 6) if total_rows else 0.0,
        "val_fraction_target": float(val_fraction),
        "val_fraction_actual": round(val_rows / total_rows, 6) if total_rows else 0.0,
        "n_places_total": int(len(by_place)),
        "n_places_test": int(len(test_places)),
        "n_places_val": int(len(val_places)),
        "n_places_train": int(len(train_places)),
        "n_places_buffer": int(len(combined_buffer)),
        "places_with_multi_year": int(sum(1 for v in by_place.values() if len(v["years"]) > 1)),
        "test_cdw_rows": int(sum(by_place[p]["n_cdw"] for p in test_places)),
        "test_no_cdw_rows": int(sum(by_place[p]["n_no_cdw"] for p in test_places)),
        "test_manual_rows": int(sum(by_place[p]["n_manual"] for p in test_places)),
        "test_threshold_gate_rows": int(sum(by_place[p]["n_threshold_gate"] for p in test_places)),
        "place_overlap_train_vs_test": 0,
        "n_regions": int(n_reg),
        "n_regions_in_test": int(n_regions_in_test),
        "n_regions_in_val": int(n_regions_in_val),
        "stratify_regions": bool(stratify_regions),
        "seed": int(seed),
        "buffer_tiles": int(buffer_tiles),
        # V1 compatibility aliases
        "split_mode": "spatial_distance_buffer_v2",
        "split_block_size_places": 1,
        "neighbor_buffer_blocks": int(buffer_tiles),
        "train_rows_estimate": int(train_rows),
    }

    # Role lookup for GeoJSON
    role_of: dict[str, str] = {}
    for p in test_places:
        role_of[p] = "test"
    for p in val_places:
        role_of[p] = "val"
    for p in combined_buffer:
        role_of[p] = "buffer"
    for p in train_places:
        role_of[p] = "train"

    payload: dict[str, Any] = {
        "keys": [[r, ro, co] for (r, ro, co) in test_keys_sorted],
        "val_keys": [[r, ro, co] for (r, ro, co) in val_keys_sorted],
        "meta": meta,
    }

    output_test_split.parent.mkdir(parents=True, exist_ok=True)
    output_test_split.write_text(json.dumps(payload, indent=2))

    if output_geojson is not None:
        from scripts.spatial_split_experiments._geojson import write_split_geojson  # type: ignore

        write_split_geojson(
            by_place=by_place,
            role_of=role_of,
            region_of=region_of,
            meta=meta,
            output_path=output_geojson,
        )

    return meta
