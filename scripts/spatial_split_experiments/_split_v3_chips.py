"""Intra-raster chip-level spatial split with per-block assignment.

Motivation
----------
The V1/V2 mapsheet-level split wastes ~63 % of rows as buffer because the
dominant data cluster (Cluster B, 75 % of rows) is fully surrounded by buffer
tiles.  Moving the split to the **chip level within each raster** reduces buffer
waste to ~7 % while using the same spatial-independence principle.

Algorithm
---------
For each raster:
1. Divide the raster into an N × N grid of spatial blocks (default N=3 →
   9 blocks of ~333 m × 333 m each in a 1 km tile).
2. Assign every labeled chip to the block that contains its centre.
3. Randomly select test blocks (greedy, target ≈ test_fraction × raster_chips,
   with CWD-balance awareness: prefer selections within `balance_cdw_tol` of
   the per-raster CWD ratio).
4. Buffer: chips NOT in a test block whose chip-coordinate Chebyshev distance
   to any test chip is ≤ buffer_chips.
   Default buffer_chips=2 → 256 px × 0.2 m/px = 51.2 m, above the ~50 m
   range at which CWD autocorrelation drops to background (Gu et al. 2024).
5. Remaining chips are train.

This is applied independently per raster, so ALL rasters contribute training
chips — unlike the mapsheet-level split which excludes entire rasters.

References
----------
Roberts et al. 2017, Ecography 40:913   — spatial block CV rationale
Gu et al. 2024, Forests 15:xxx          — CWD autocorrelation range ~50 m
Valavi et al. 2025, Front. Remote Sens. — block size is dominant parameter
Le Rest et al. 2014, GEB 23:811         — buffer = variogram range
"""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Chip / block coordinate helpers
# ---------------------------------------------------------------------------

def _chip_coord(row_off: int, col_off: int, chunk_size: int) -> tuple[int, int]:
    """Chip-grid position (chip_row, chip_col) from pixel offsets."""
    return row_off // chunk_size, col_off // chunk_size


def _block_coord(
    chip_row: int, chip_col: int, chips_per_block: int, n_blocks: int
) -> tuple[int, int]:
    """Map chip position to block index (clamped to [0, n_blocks-1])."""
    return (
        min(chip_row // chips_per_block, n_blocks - 1),
        min(chip_col // chips_per_block, n_blocks - 1),
    )


def _chip_in_buffer(
    chip_row: int,
    chip_col: int,
    test_chip_positions: set[tuple[int, int]],
    buffer_chips: int,
) -> bool:
    """True if the chip is within buffer_chips Chebyshev of any test chip."""
    for tr, tc in test_chip_positions:
        if max(abs(chip_row - tr), abs(chip_col - tc)) <= buffer_chips:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-raster block selection
# ---------------------------------------------------------------------------

def _select_test_blocks_for_raster(
    blocks: dict[tuple[int, int], list[dict[str, Any]]],
    target_test_chips: int,
    overall_cdw_ratio: float,
    balance_cdw_tol: float,
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Greedy block selection targeting test chip count with CWD balance.

    Strategy:
    - Shuffle blocks randomly (seed-controlled).
    - Greedy: pick the block that brings cumulative chip count closest to target.
    - After selection, check CWD ratio. If unbalanced and improvement is
      possible by swapping, attempt one swap.
    """
    if not blocks:
        return set()

    block_list = list(blocks.keys())
    rng.shuffle(block_list)

    selected: set[tuple[int, int]] = set()
    selected_chips = 0

    while block_list and selected_chips < target_test_chips:
        best_block = None
        best_score: float | None = None
        best_nxt: int | None = None
        for b in block_list:
            nxt = selected_chips + len(blocks[b])
            score = abs(target_test_chips - nxt)
            if best_score is None or score < best_score or (
                score == best_score and best_nxt is not None and nxt > best_nxt
            ):
                best_block, best_score, best_nxt = b, score, nxt
        if best_block is None:
            break
        selected.add(best_block)
        selected_chips = int(best_nxt or selected_chips)
        block_list.remove(best_block)

    # CWD balance check and optional swap
    if balance_cdw_tol < 1.0 and selected:
        test_cdw = sum(
            1 for b in selected for c in blocks[b] if c.get("label") == "cdw"
        )
        total_test = sum(len(blocks[b]) for b in selected)
        if total_test > 0:
            test_ratio = test_cdw / total_test
            if abs(test_ratio - overall_cdw_ratio) > balance_cdw_tol:
                # Try swapping one selected block with one non-selected block
                non_selected = [b for b in blocks if b not in selected]
                best_swap_score = abs(test_ratio - overall_cdw_ratio)
                best_swap = None
                for out_b in list(selected):
                    for in_b in non_selected:
                        # Would this swap improve balance?
                        new_test_cdw = test_cdw
                        new_total = total_test
                        for c in blocks[out_b]:
                            if c.get("label") == "cdw":
                                new_test_cdw -= 1
                            new_total -= 1
                        for c in blocks[in_b]:
                            if c.get("label") == "cdw":
                                new_test_cdw += 1
                            new_total += 1
                        new_ratio = new_test_cdw / new_total if new_total else 0.0
                        new_score = abs(new_ratio - overall_cdw_ratio)
                        if new_score < best_swap_score:
                            best_swap_score = new_score
                            best_swap = (out_b, in_b)
                if best_swap:
                    out_b, in_b = best_swap
                    selected.discard(out_b)
                    selected.add(in_b)

    return selected


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def write_intra_raster_chip_split(
    all_candidates: list[dict[str, Any]],
    output_test_split: Path,
    seed: int,
    test_fraction: float = 0.20,
    n_blocks: int = 3,
    buffer_chips: int = 2,
    balance_cdw_tol: float = 0.20,
    raster_size_px: int = 5000,
    pixel_size_m: float = 0.2,
    output_geojson: Path | None = None,
) -> dict[str, Any]:
    """Split labeled chips within each raster using spatial blocks.

    Parameters
    ----------
    all_candidates:
        List of label rows with keys: ``key`` (tuple[str,int,int]),
        ``raster`` (str), ``label`` (str), ``reason`` (str).
    output_test_split:
        Path for the JSON split manifest. ``"keys"`` contains test chips;
        ``"val_keys"`` is empty (val not supported in V3 yet).
    seed:
        Global random seed; each raster is seeded with hash(raster, seed).
    test_fraction:
        Target fraction of chips per raster assigned to test.
    n_blocks:
        Number of spatial blocks per raster dimension (N × N grid).
        Default 3 → 9 blocks of ~333 m × 333 m for a 1 km raster.
    buffer_chips:
        Chebyshev distance in chip units around test chips marked as buffer.
        Default 2 → 256 px = 51.2 m at 0.2 m/px (above CWD autocorrelation
        range of ~50 m per Gu et al. 2024).
    balance_cdw_tol:
        Maximum allowed |test_cdw_ratio − overall_cdw_ratio|.
        Default 0.20 (relaxed to handle sparse per-raster labels).
    raster_size_px:
        Assumed raster side length in pixels. Default 5000 (1 km at 0.2 m/px).
    pixel_size_m:
        Pixel size in metres, used for GeoJSON coordinate calculation.
        Default 0.2 (20 cm CHM).
    output_geojson:
        If given, write a GeoJSON FeatureCollection of spatial blocks to this
        path. Blocks coloured by dominant split role; properties include chip
        counts and CWD balance. Open in QGIS with CRS = EPSG:3301.
    """
    from scripts.model_search_v4._labels import parse_raster_identity  # type: ignore

    # chips_per_block_dim: how many chip positions fit in one block along one axis
    total_chips_per_dim = raster_size_px // max(1, _infer_chunk_size(all_candidates))
    chunk_size = _infer_chunk_size(all_candidates)
    chips_per_block = max(1, total_chips_per_dim // n_blocks)

    # Group candidates by raster
    by_raster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_candidates:
        raster = str(item["raster"])
        raster_stem = raster.replace(".tif", "").replace(".TIF", "")
        by_raster[raster_stem].append(item)

    all_test_keys: list[tuple[str, int, int]] = []
    role_of: dict[tuple[str, int, int], str] = {}

    raster_stats: list[dict[str, Any]] = []
    total_chips = total_train = total_test = total_buf = 0
    total_cdw = total_no_cdw = 0

    # Block registry for GeoJSON (raster_stem → {block_coord → {"role", "n_cdw", "n_no_cdw"}})
    block_registry: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}

    for raster_stem in sorted(by_raster.keys()):
        items = by_raster[raster_stem]
        raster_seed = int(zlib.crc32(f"{raster_stem}_{seed}".encode())) & 0xFFFFFFFF
        rng = random.Random(raster_seed)

        # Assign chips to blocks
        blocks: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            key = item["key"]
            row_off, col_off = int(key[1]), int(key[2])
            cr, cc = _chip_coord(row_off, col_off, chunk_size)
            br, bc = _block_coord(cr, cc, chips_per_block, n_blocks)
            item_with_pos = dict(item)
            item_with_pos["_chip_row"] = cr
            item_with_pos["_chip_col"] = cc
            item_with_pos["_block"] = (br, bc)
            blocks[(br, bc)].append(item_with_pos)

        # Per-raster CWD ratio
        n_cdw_r = sum(1 for it in items if str(it.get("label", "")) == "cdw")
        n_all_r = len(items)
        cdw_ratio_r = n_cdw_r / n_all_r if n_all_r else 0.5

        # Select test blocks
        target_test = max(1, round(n_all_r * test_fraction))
        test_blocks = _select_test_blocks_for_raster(
            blocks, target_test, cdw_ratio_r, balance_cdw_tol, rng
        )

        # Collect test chip positions for buffer check
        test_chip_pos: set[tuple[int, int]] = set()
        for b in test_blocks:
            for item in blocks[b]:
                test_chip_pos.add((item["_chip_row"], item["_chip_col"]))

        # Assign roles
        block_roles: dict[tuple[int, int], str] = {}
        r_train = r_test = r_buf = 0
        r_cdw_test = r_no_cdw_test = 0

        for block_coord, block_items in blocks.items():
            is_test_block = block_coord in test_blocks
            for item in block_items:
                cr, cc = item["_chip_row"], item["_chip_col"]
                key_tuple = (str(item["key"][0]), int(item["key"][1]), int(item["key"][2]))
                is_test = is_test_block
                is_buf = not is_test and _chip_in_buffer(cr, cc, test_chip_pos, buffer_chips)

                if is_test:
                    role = "test"
                    r_test += 1
                    all_test_keys.append(key_tuple)
                    if str(item.get("label", "")) == "cdw":
                        r_cdw_test += 1
                    else:
                        r_no_cdw_test += 1
                elif is_buf:
                    role = "buffer"
                    r_buf += 1
                else:
                    role = "train"
                    r_train += 1

                role_of[key_tuple] = role

            # Block role = dominant role of its chips
            is_test_block_flag = block_coord in test_blocks
            has_buffer_chip = any(
                _chip_in_buffer(it["_chip_row"], it["_chip_col"], test_chip_pos, buffer_chips)
                for it in block_items
                if block_coord not in test_blocks
            )
            if is_test_block_flag:
                block_roles[block_coord] = "test"
            elif has_buffer_chip:
                block_roles[block_coord] = "buffer"
            else:
                block_roles[block_coord] = "train"

        total_chips += n_all_r
        total_train += r_train
        total_test += r_test
        total_buf += r_buf
        total_cdw += n_cdw_r
        total_no_cdw += n_all_r - n_cdw_r

        raster_stats.append({
            "raster": raster_stem,
            "n_chips": n_all_r,
            "n_train": r_train,
            "n_test": r_test,
            "n_buffer": r_buf,
            "test_cdw": r_cdw_test,
            "test_no_cdw": r_no_cdw_test,
        })

        # Build block registry for GeoJSON
        ident = parse_raster_identity(raster_stem)
        block_registry[raster_stem] = {
            "tile_id": str(ident.get("tile", "")),
            "grid_x": ident.get("grid_x"),
            "grid_y": ident.get("grid_y"),
            "blocks": {},
        }
        for block_coord, block_items in blocks.items():
            n_block_cdw = sum(1 for it in block_items if str(it.get("label","")) == "cdw")
            block_registry[raster_stem]["blocks"][block_coord] = {
                "role": block_roles.get(block_coord, "train"),
                "n_chips": len(block_items),
                "n_cdw": n_block_cdw,
                "n_no_cdw": len(block_items) - n_block_cdw,
            }

    # Sort test keys for determinism
    all_test_keys_sorted = sorted(all_test_keys, key=lambda x: (x[0], x[1], x[2]))

    meta: dict[str, Any] = {
        "split_version": "v3_intra_raster_chips",
        "created_at": _utc_now(),
        "split_mode": "intra_raster_spatial_blocks",
        "total_rows": int(total_chips),
        "test_rows": int(total_test),
        "val_rows": 0,
        "train_rows": int(total_train),
        "buffer_rows": int(total_buf),
        "buffer_pct": round(100.0 * total_buf / total_chips, 2) if total_chips else 0.0,
        "test_fraction_target": float(test_fraction),
        "test_fraction_actual": round(total_test / total_chips, 6) if total_chips else 0.0,
        "n_rasters": len(by_raster),
        "n_blocks_per_raster": int(n_blocks * n_blocks),
        "n_blocks": int(n_blocks),
        "buffer_chips": int(buffer_chips),
        "buffer_metres": round(buffer_chips * chunk_size * pixel_size_m, 1),
        "balance_cdw_tol": float(balance_cdw_tol),
        "raster_size_px": int(raster_size_px),
        "chunk_size": int(chunk_size),
        "pixel_size_m": float(pixel_size_m),
        "overall_cdw_ratio": round(total_cdw / total_chips, 4) if total_chips else 0.0,
        "test_cdw_rows": int(sum(s["test_cdw"] for s in raster_stats)),
        "test_no_cdw_rows": int(sum(s["test_no_cdw"] for s in raster_stats)),
        "test_cdw_ratio": 0.0,
        "place_overlap_train_vs_test": 0,  # N/A for intra-raster split
        "seed": int(seed),
        # V1-compat aliases
        "split_block_size_places": 1,
        "neighbor_buffer_blocks": int(buffer_chips),
        "train_rows_estimate": int(total_train),
    }
    if total_test:
        meta["test_cdw_ratio"] = round(meta["test_cdw_rows"] / total_test, 4)

    payload: dict[str, Any] = {
        "keys": [[r, ro, co] for (r, ro, co) in all_test_keys_sorted],
        "val_keys": [],
        "meta": meta,
    }

    output_test_split.parent.mkdir(parents=True, exist_ok=True)
    output_test_split.write_text(json.dumps(payload, indent=2))

    if output_geojson is not None:
        _write_block_geojson(
            block_registry=block_registry,
            meta=meta,
            n_blocks=n_blocks,
            raster_size_px=raster_size_px,
            chunk_size=chunk_size,
            pixel_size_m=pixel_size_m,
            output_path=output_geojson,
        )

    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_chunk_size(all_candidates: list[dict[str, Any]]) -> int:
    """Return the chunk_size from the first candidate, defaulting to 128."""
    for item in all_candidates:
        key = item.get("key")
        if isinstance(key, (list, tuple)) and len(key) >= 3:
            # chunk_size might be in item directly or we default
            pass
    # Try from item directly
    for item in all_candidates:
        if "chunk_size" in item:
            return int(item["chunk_size"])
    return 128


def _write_block_geojson(
    block_registry: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    n_blocks: int,
    raster_size_px: int,
    chunk_size: int,
    pixel_size_m: float,
    output_path: Path,
) -> None:
    """Write GeoJSON with one polygon per spatial block per raster.

    Each block polygon is in EPSG:3301 (L-EST97). Colour by role:
    blue=train, green=test, grey=buffer.

    Properties: raster_stem, tile_id, block_row, block_col, role,
    n_chips, n_cdw, n_no_cdw, cdw_ratio, seed, buffer_metres.
    """
    from scripts.spatial_split_experiments._geojson import ROLE_COLOUR  # type: ignore

    block_h_px = raster_size_px // n_blocks
    block_w_px = raster_size_px // n_blocks

    features = []
    for raster_stem, rinfo in sorted(block_registry.items()):
        gx = rinfo.get("grid_x")
        gy = rinfo.get("grid_y")
        if gx is None or gy is None:
            continue
        # Top-left corner of raster in L-EST97
        northing_top = 6_000_000 + (int(gx) + 1) * 1000  # north edge
        easting_left = int(gy) * 1000                      # west edge

        for (br, bc), binfo in sorted(rinfo["blocks"].items()):
            top_px = br * block_h_px
            bottom_px = raster_size_px if br == n_blocks - 1 else (br + 1) * block_h_px
            left_px = bc * block_w_px
            right_px = raster_size_px if bc == n_blocks - 1 else (bc + 1) * block_w_px

            north = northing_top - top_px * pixel_size_m
            south = northing_top - bottom_px * pixel_size_m
            west = easting_left + left_px * pixel_size_m
            east = easting_left + right_px * pixel_size_m

            # CCW exterior ring
            coords = [
                [west, south], [east, south],
                [east, north], [west, north],
                [west, south],
            ]
            role = binfo["role"]
            n_chips = binfo["n_chips"]
            cdw_ratio = round(binfo["n_cdw"] / n_chips, 3) if n_chips else 0.0

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "raster_stem": raster_stem,
                    "tile_id": rinfo.get("tile_id", ""),
                    "block_row": int(br),
                    "block_col": int(bc),
                    "split_role": role,
                    "n_chips": int(n_chips),
                    "n_cdw": int(binfo["n_cdw"]),
                    "n_no_cdw": int(binfo["n_no_cdw"]),
                    "cdw_ratio": cdw_ratio,
                    "colour": ROLE_COLOUR.get(role, "#FF00FF"),
                    "config_seed": int(meta.get("seed", 0)),
                    "config_n_blocks": int(n_blocks),
                    "config_buffer_chips": int(meta.get("buffer_chips", 2)),
                    "config_buffer_metres": float(meta.get("buffer_metres", 51.2)),
                    "config_test_fraction": float(meta.get("test_fraction_target", 0.2)),
                },
            })

    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3301"}},
        "metadata": {
            "description": "CWD intra-raster chip-level spatial split — block polygons",
            "split_version": meta.get("split_version", "v3_intra_raster_chips"),
            "created_at": meta.get("created_at", ""),
            "n_rasters": meta.get("n_rasters", 0),
            "n_blocks": n_blocks,
            "buffer_metres": meta.get("buffer_metres", 51.2),
            "seed": meta.get("seed", 0),
            "total_rows": meta.get("total_rows", 0),
            "buffer_pct": meta.get("buffer_pct", 0),
        },
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fc, indent=2, ensure_ascii=False))
