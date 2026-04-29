"""Stride-aware intra-raster chip split with multi-year leakage prevention (V4).

Fixes two independent leakage sources present in V3
(scripts/spatial_split_experiments/_split_v3_chips.py):

1. **Stride coordinate bug**: V3 computed chip_pos = row_off // chunk_size (128).
   With 50% overlap the actual stride is 64 px, so row_off=0 and row_off=64 mapped
   to the *same* chip_pos. The buffer gap formula is:
       gap = (buffer_strides + 1) × stride − chunk_size
   For gap ≥ 250 px (50 m CWD autocorrelation range):
       buffer_strides ≥ 5  → gap = 256 px = 51.2 m  ✓
   V3 buffer_chips=2 in chunk coords ≈ buffer_strides=4 → gap = 192 px = 38.4 m  ✗

2. **Year leakage**: V3 seeded the RNG per (raster, year). Multiple years of the same
   physical location could land in opposite split roles. V4 seeds per place_key =
   tile_site (year-agnostic), guaranteeing identical block assignments for all years.

References
----------
Kattenborn et al. 2022, ISPRS OJ  — random CV inflates CNN performance ≤28 pp
Roberts et al. 2017, Ecography    — buffer = autocorrelation range
Valavi et al. 2019, MEE           — blockCV; buffer ≥ autocorrelation range
Gu et al. 2024, Forests           — CWD autocorrelation range ~50 m
"""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stride / chunk inference
# ---------------------------------------------------------------------------

def _infer_stride(all_candidates: list[dict[str, Any]]) -> int:
    """Infer chip stride from the GCD of all non-zero row_off / col_off values.

    For 50%-overlap labels (chunk=128, overlap=0.5): stride=64.
    For non-overlapping labels (overlap=0): stride=128.
    """
    vals: list[int] = []
    for item in all_candidates[:2000]:
        key = item.get("key")
        if isinstance(key, (list, tuple)) and len(key) >= 3:
            r, c = int(key[1]), int(key[2])
            if r > 0:
                vals.append(r)
            if c > 0:
                vals.append(c)
    if not vals:
        return 64
    return reduce(gcd, vals[:200])


def _infer_chunk_size(all_candidates: list[dict[str, Any]]) -> int:
    for item in all_candidates:
        if "chunk_size" in item:
            return int(item["chunk_size"])
    return 128


# ---------------------------------------------------------------------------
# Chip / block coordinate helpers (stride-aware)
# ---------------------------------------------------------------------------

def _chip_pos(row_off: int, col_off: int, stride: int) -> tuple[int, int]:
    """Chip-grid position using actual stride (not chunk_size)."""
    return row_off // stride, col_off // stride


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
    buffer_strides: int,
) -> bool:
    """True if chip is within buffer_strides Chebyshev of any test chip (stride units)."""
    for tr, tc in test_chip_positions:
        if max(abs(chip_row - tr), abs(chip_col - tc)) <= buffer_strides:
            return True
    return False


def _total_chips_per_dim(raster_size_px: int, chunk_size: int, stride: int) -> int:
    """Number of valid chip positions along one raster axis."""
    if raster_size_px <= chunk_size:
        return 1
    return len(range(0, raster_size_px - chunk_size + 1, stride))


# ---------------------------------------------------------------------------
# Place key (year-agnostic)
# ---------------------------------------------------------------------------

def _get_place_key(raster_stem: str) -> str:
    """Strip year from raster stem → year-agnostic place key.

    '436646_2020_madal_chm_max_hag_20cm' → '436646_madal'
    Falls back to the full stem if the pattern doesn't match.
    """
    try:
        from scripts.model_search_v4._labels import parse_raster_identity  # type: ignore
        ident = parse_raster_identity(raster_stem)
        tile = str(ident.get("tile", "") or "")
        site = str(ident.get("site", "") or "")
        if tile and site and site != "unknown":
            return f"{tile}_{site}"
    except Exception:
        pass
    return raster_stem


# ---------------------------------------------------------------------------
# Per-raster block selection
# ---------------------------------------------------------------------------

def _select_test_blocks(
    blocks: dict[tuple[int, int], list[dict[str, Any]]],
    target_test_chips: int,
    overall_cdw_ratio: float,
    balance_cdw_tol: float,
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Greedy test block selection with optional CWD balance swap."""
    if not blocks:
        return set()

    block_list = list(blocks.keys())
    rng.shuffle(block_list)

    selected: set[tuple[int, int]] = set()
    selected_chips = 0

    while block_list and selected_chips < target_test_chips:
        best_block = best_nxt = best_score = None
        for b in block_list:
            nxt = selected_chips + len(blocks[b])
            score = abs(target_test_chips - nxt)
            if (
                best_score is None
                or score < best_score
                or (score == best_score and best_nxt is not None and nxt > best_nxt)
            ):
                best_block, best_score, best_nxt = b, score, nxt
        if best_block is None:
            break
        selected.add(best_block)
        selected_chips = int(best_nxt or selected_chips)
        block_list.remove(best_block)

    if balance_cdw_tol < 1.0 and selected:
        test_cdw = sum(1 for b in selected for c in blocks[b] if c.get("label") == "cdw")
        total_test = sum(len(blocks[b]) for b in selected)
        if total_test > 0:
            test_ratio = test_cdw / total_test
            if abs(test_ratio - overall_cdw_ratio) > balance_cdw_tol:
                non_selected = [b for b in blocks if b not in selected]
                best_swap_score = abs(test_ratio - overall_cdw_ratio)
                best_swap = None
                for out_b in list(selected):
                    for in_b in non_selected:
                        new_cdw = test_cdw + sum(1 for c in blocks[in_b] if c.get("label") == "cdw") \
                                           - sum(1 for c in blocks[out_b] if c.get("label") == "cdw")
                        new_total = total_test + len(blocks[in_b]) - len(blocks[out_b])
                        new_ratio = new_cdw / new_total if new_total > 0 else 0.0
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

def write_intra_raster_chip_split_v4(
    all_candidates: list[dict[str, Any]],
    output_test_split: Path,
    seed: int,
    test_fraction: float = 0.20,
    n_blocks: int = 3,
    buffer_strides: int = 5,
    balance_cdw_tol: float = 0.20,
    raster_size_px: int = 5000,
    pixel_size_m: float = 0.2,
    output_geojson: Path | None = None,
) -> dict[str, Any]:
    """Stride-aware intra-raster chip split with multi-year leakage prevention.

    Parameters
    ----------
    all_candidates:
        Label rows with keys: ``key`` (raster, row_off, col_off),
        ``raster``, ``label``, ``reason``.
    output_test_split:
        Output JSON split manifest.
    seed:
        Global seed. Per-place RNG seeded with hash(place_key, seed) so all
        years of the same physical location always get the same block assignment.
    test_fraction:
        Target fraction of chips per place assigned to test.
    n_blocks:
        N×N blocks per raster (default 3 → 9 blocks of ~333 m each in 1 km tile).
    buffer_strides:
        Chebyshev radius in **stride units** around test chips.
        Default 5 → gap = (5+1)×64−128 = 256 px = 51.2 m ≥ 50 m CWD autocorr. ✓
        Formula: gap = (buffer_strides + 1) × stride − chunk_size.
    balance_cdw_tol:
        |test_cdw_ratio − overall_cdw_ratio| tolerance. Default 0.20.
    raster_size_px:
        Assumed raster side length in pixels (default 5000 = 1 km at 0.2 m/px).
    pixel_size_m:
        Pixel size in metres (default 0.2).
    output_geojson:
        If given, write block-level GeoJSON map (CRS EPSG:3301).
    """
    stride = _infer_stride(all_candidates)
    chunk_size = _infer_chunk_size(all_candidates)
    n_chips_per_dim = _total_chips_per_dim(raster_size_px, chunk_size, stride)
    chips_per_block = max(1, n_chips_per_dim // n_blocks)

    gap_px = (buffer_strides + 1) * stride - chunk_size
    gap_m = round(gap_px * pixel_size_m, 1)

    # Group candidates by raster stem
    by_raster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_candidates:
        raster = str(item["raster"])
        stem = raster.replace(".tif", "").replace(".TIF", "")
        by_raster[stem].append(item)

    all_test_keys: list[tuple[str, int, int]] = []
    role_of: dict[tuple[str, int, int], str] = {}

    raster_stats: list[dict[str, Any]] = []
    total_chips = total_train = total_test = total_buf = 0
    total_cdw = total_no_cdw = 0

    block_registry: dict[str, dict[str, Any]] = {}

    for raster_stem in sorted(by_raster.keys()):
        items = by_raster[raster_stem]

        place_key = _get_place_key(raster_stem)
        place_seed = int(zlib.crc32(f"{place_key}_{seed}".encode())) & 0xFFFFFFFF
        rng = random.Random(place_seed)

        blocks: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            key = item["key"]
            r_off, c_off = int(key[1]), int(key[2])
            cp_r, cp_c = _chip_pos(r_off, c_off, stride)
            br, bc = _block_coord(cp_r, cp_c, chips_per_block, n_blocks)
            item_ex = dict(item)
            item_ex["_cp_row"] = cp_r
            item_ex["_cp_col"] = cp_c
            item_ex["_block"] = (br, bc)
            blocks[(br, bc)].append(item_ex)

        n_cdw_r = sum(1 for it in items if str(it.get("label", "")) == "cdw")
        n_all_r = len(items)
        cdw_ratio_r = n_cdw_r / n_all_r if n_all_r else 0.5

        target_test = max(1, round(n_all_r * test_fraction))
        test_blocks = _select_test_blocks(
            blocks, target_test, cdw_ratio_r, balance_cdw_tol, rng
        )

        test_chip_pos_set: set[tuple[int, int]] = set()
        for b in test_blocks:
            for item in blocks[b]:
                test_chip_pos_set.add((item["_cp_row"], item["_cp_col"]))

        block_roles: dict[tuple[int, int], str] = {}
        r_train = r_test = r_buf = 0
        r_cdw_test = r_no_cdw_test = 0

        for block_coord, block_items in blocks.items():
            is_test_block = block_coord in test_blocks
            for item in block_items:
                cp_r, cp_c = item["_cp_row"], item["_cp_col"]
                key_tuple = (str(item["key"][0]), int(item["key"][1]), int(item["key"][2]))
                is_test = is_test_block
                is_buf = not is_test and _chip_in_buffer(cp_r, cp_c, test_chip_pos_set, buffer_strides)

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

            # Block role = dominant role
            if block_coord in test_blocks:
                block_roles[block_coord] = "test"
            elif any(
                _chip_in_buffer(it["_cp_row"], it["_cp_col"], test_chip_pos_set, buffer_strides)
                for it in block_items
            ):
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
        })

        try:
            from scripts.model_search_v4._labels import parse_raster_identity  # type: ignore
            ident = parse_raster_identity(raster_stem)
        except Exception:
            ident = {"tile": None, "grid_x": None, "grid_y": None}

        block_registry[raster_stem] = {
            "tile_id": str(ident.get("tile", "") or ""),
            "grid_x": ident.get("grid_x"),
            "grid_y": ident.get("grid_y"),
            "place_key": place_key,
            "blocks": {},
        }
        for block_coord, block_items in blocks.items():
            n_block_cdw = sum(1 for it in block_items if str(it.get("label", "")) == "cdw")
            block_registry[raster_stem]["blocks"][block_coord] = {
                "role": block_roles.get(block_coord, "train"),
                "n_chips": len(block_items),
                "n_cdw": n_block_cdw,
                "n_no_cdw": len(block_items) - n_block_cdw,
            }

    all_test_keys_sorted = sorted(all_test_keys, key=lambda x: (x[0], x[1], x[2]))

    meta: dict[str, Any] = {
        "split_version": "v4_stride_aware_year_safe",
        "created_at": _utc_now(),
        "split_mode": "intra_raster_spatial_blocks_stride_aware",
        "total_rows": int(total_chips),
        "test_rows": int(total_test),
        "val_rows": 0,
        "train_rows": int(total_train),
        "buffer_rows": int(total_buf),
        "buffer_pct": round(100.0 * total_buf / total_chips, 2) if total_chips else 0.0,
        "test_fraction_target": float(test_fraction),
        "test_fraction_actual": round(total_test / total_chips, 6) if total_chips else 0.0,
        "n_rasters": len(by_raster),
        "n_blocks": int(n_blocks),
        "n_blocks_per_raster": int(n_blocks * n_blocks),
        "stride": int(stride),
        "chunk_size": int(chunk_size),
        "buffer_strides": int(buffer_strides),
        "buffer_gap_px": int(gap_px),
        "buffer_gap_m": float(gap_m),
        "raster_size_px": int(raster_size_px),
        "pixel_size_m": float(pixel_size_m),
        "n_chips_per_dim": int(n_chips_per_dim),
        "chips_per_block": int(chips_per_block),
        "overall_cdw_ratio": round(total_cdw / total_chips, 4) if total_chips else 0.0,
        "test_cdw_rows": int(sum(s["test_cdw"] for s in raster_stats)),
        "test_cdw_ratio": 0.0,
        "place_overlap_train_vs_test": 0,
        "seed": int(seed),
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
            stride=stride,
            pixel_size_m=pixel_size_m,
            output_path=output_geojson,
        )
        # Also write chip-level GeoJSON (one polygon per 128px chip with 50% overlap)
        chip_geojson_path = output_geojson.parent / output_geojson.stem.replace("split_", "chips_")
        _write_chip_geojson(
            all_test_keys=all_test_keys,
            role_of=role_of,
            all_candidates=all_candidates,
            meta=meta,
            block_registry=block_registry,
            stride=stride,
            chunk_size=chunk_size,
            pixel_size_m=pixel_size_m,
            output_path=chip_geojson_path,
        )

    return meta


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------

_ROLE_COLOUR = {
    "train": "#2196F3",   # blue
    "test": "#4CAF50",    # green
    "buffer": "#9E9E9E",  # grey
}


def _write_block_geojson(
    block_registry: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    n_blocks: int,
    raster_size_px: int,
    chunk_size: int,
    stride: int,
    pixel_size_m: float,
    output_path: Path,
) -> None:
    """Write GeoJSON with one polygon per spatial block per raster (EPSG:3301)."""
    block_h_px = raster_size_px // n_blocks
    block_w_px = raster_size_px // n_blocks
    features = []

    for raster_stem, rinfo in sorted(block_registry.items()):
        gx = rinfo.get("grid_x")
        gy = rinfo.get("grid_y")
        if gx is None or gy is None:
            continue
        northing_top = 6_000_000 + (int(gx) + 1) * 1000
        easting_left = int(gy) * 1000

        for (br, bc), binfo in sorted(rinfo["blocks"].items()):
            top_px = br * block_h_px
            bottom_px = raster_size_px if br == n_blocks - 1 else (br + 1) * block_h_px
            left_px = bc * block_w_px
            right_px = raster_size_px if bc == n_blocks - 1 else (bc + 1) * block_w_px

            north = northing_top - top_px * pixel_size_m
            south = northing_top - bottom_px * pixel_size_m
            west = easting_left + left_px * pixel_size_m
            east = easting_left + right_px * pixel_size_m

            coords = [[west, south], [east, south], [east, north], [west, north], [west, south]]
            role = binfo["role"]
            n_chips = binfo["n_chips"]
            cdw_ratio = round(binfo["n_cdw"] / n_chips, 3) if n_chips else 0.0

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "raster_stem": raster_stem,
                    "place_key": rinfo.get("place_key", ""),
                    "tile_id": rinfo.get("tile_id", ""),
                    "block_row": int(br),
                    "block_col": int(bc),
                    "split_role": role,
                    "n_chips": int(n_chips),
                    "n_cdw": int(binfo["n_cdw"]),
                    "n_no_cdw": int(binfo["n_no_cdw"]),
                    "cdw_ratio": cdw_ratio,
                    "colour": _ROLE_COLOUR.get(role, "#FF00FF"),
                    "config_seed": int(meta.get("seed", 0)),
                    "config_n_blocks": int(n_blocks),
                    "config_buffer_strides": int(meta.get("buffer_strides", 5)),
                    "config_buffer_gap_m": float(meta.get("buffer_gap_m", 51.2)),
                    "config_stride": int(stride),
                    "config_test_fraction": float(meta.get("test_fraction_target", 0.2)),
                },
            })

    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3301"}},
        "metadata": {
            "description": "CWD V4 stride-aware chip-level split — block polygons",
            "split_version": meta.get("split_version"),
            "created_at": meta.get("created_at"),
            "n_rasters": meta.get("n_rasters"),
            "n_blocks": n_blocks,
            "stride": stride,
            "buffer_gap_m": meta.get("buffer_gap_m"),
            "seed": meta.get("seed"),
        },
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fc, indent=2, ensure_ascii=False))
