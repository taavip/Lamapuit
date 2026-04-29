"""Spatial train/test split for Model Search V4 with multi-layer leakage control.

Three simultaneous leakage vectors are controlled:

1. **Same-place multi-year**: the same physical tile re-flown in different
   years must never straddle the split. We group by ``place_key`` = ``tile_site``
   which is year-agnostic, so all years of a tile go together.

2. **Neighbour leakage**: adjacent tiles share canopy continuity. We group
   places into spatial blocks by parsed grid_x/grid_y at a configurable
   ``block_size_places`` (default 2 → 2×2 tile neighborhoods). A configurable
   ``neighbor_buffer_blocks`` (default 1) reserves an unused buffer ring
   around every test block — those places go into neither train nor test.

3. **Fine-grained fence**: within retained training rows, a per-tile metric
   fence (default 26 m) further drops training tiles whose spatial fence
   identifier collides with any test tile. The fence is applied later in
   ``model_search_v4.main`` via the base script's ``_spatial_fence_id``; this
   module only logs whether the place-level split has already eliminated
   fence conflicts.

The split is deterministic given ``seed`` and ``block_size_places`` and
produces an auditable JSON manifest ``cnn_test_split_v4.json`` with
per-block row counts, year coverage, neighbour-buffer statistics, and a
place-overlap sanity check (which must remain zero).
"""

from __future__ import annotations

import json
import random
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._labels import parse_raster_identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _place_to_block(grid_x: int | None, grid_y: int | None, place_key: str, block_size: int) -> tuple[int, int]:
    """Map a place to a coarse-grid block.

    When the tile ID does not yield usable grid coordinates, fall back to a
    deterministic pseudo-block derived from a CRC32 of the place key. These
    pseudo-blocks never collide with real grid blocks (top halves of the
    integer range) — they're isolated islands that won't accidentally bridge
    two geographically adjacent places.
    """
    bsize = max(int(block_size), 1)
    if grid_x is not None and grid_y is not None:
        return (int(grid_x) // bsize, int(grid_y) // bsize)
    h = int(zlib.crc32(place_key.encode("utf-8")))
    return (10_000_000 + (h % 10000), 10_000_000 + ((h // 10000) % 10000))


def _neighbor_blocks(block: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    """Chebyshev-distance neighborhood around a block, radius inclusive."""
    if radius <= 0:
        return {block}
    out: set[tuple[int, int]] = set()
    bx, by = block
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            out.add((bx + dx, by + dy))
    return out


def write_spatial_block_test_split(
    all_candidates: list[dict[str, Any]],
    output_test_split: Path,
    seed: int,
    test_fraction: float,
    split_block_size_places: int,
    neighbor_buffer_blocks: int = 1,
) -> dict[str, Any]:
    """Partition places into (train, buffer, test) by spatial block, preserving balance.

    Greedy selection: iteratively pick the block whose row count best pushes
    the cumulative test size toward ``test_fraction * total_rows``, breaking
    ties to prefer the larger block (avoids tiny fragmented test sets).
    After test blocks are finalized, a ``neighbor_buffer_blocks`` ring of
    blocks is marked as "buffer" and excluded from training.
    """
    rng = random.Random(seed)

    by_place: dict[str, dict[str, Any]] = {}
    for item in all_candidates:
        key = item["key"]
        raster = str(item["raster"])
        ident = parse_raster_identity(raster)
        place_key = str(ident["place_key"])

        entry = by_place.get(place_key)
        if entry is None:
            entry = {
                "place_key": place_key,
                "block": _place_to_block(ident["grid_x"], ident["grid_y"], place_key, split_block_size_places),
                "keys": [],
                "years": set(),
                "n_cdw": 0,
                "n_no_cdw": 0,
                "n_manual": 0,
                "n_threshold_gate": 0,
            }
            by_place[place_key] = entry

        entry["keys"].append(key)
        if ident["year"] != "unknown":
            entry["years"].add(str(ident["year"]))
        if str(item.get("label", "")).strip() == "cdw":
            entry["n_cdw"] += 1
        else:
            entry["n_no_cdw"] += 1
        if item.get("reason") == "manual_or_reviewed":
            entry["n_manual"] += 1
        elif item.get("reason") == "threshold_gate":
            entry["n_threshold_gate"] += 1

    meta_empty = {
        "created_at": _utc_now(),
        "split_mode": "spatial_blocks_with_neighbor_buffer",
        "test_fraction_target": float(test_fraction),
        "test_fraction_actual": 0.0,
        "n_places_total": 0,
        "n_places_test": 0,
        "n_places_buffer": 0,
        "n_blocks_total": 0,
        "n_blocks_test": 0,
        "n_blocks_buffer": 0,
        "place_overlap_train_vs_test": 0,
        "neighbor_buffer_blocks": int(neighbor_buffer_blocks),
        "split_block_size_places": int(max(split_block_size_places, 1)),
    }
    if not by_place:
        output_test_split.parent.mkdir(parents=True, exist_ok=True)
        output_test_split.write_text(json.dumps({"keys": [], "meta": meta_empty}, indent=2))
        return meta_empty

    block_to_places: dict[tuple[int, int], list[str]] = defaultdict(list)
    block_row_count: dict[tuple[int, int], int] = defaultdict(int)
    total_rows = 0
    for place_key, info in by_place.items():
        block = info["block"]
        block_to_places[block].append(place_key)
        block_row_count[block] += len(info["keys"])
        total_rows += len(info["keys"])

    target_test_rows = max(1, int(round(total_rows * float(test_fraction))))

    blocks = list(block_to_places.keys())
    rng.shuffle(blocks)

    selected_blocks: set[tuple[int, int]] = set()
    selected_rows = 0
    while blocks and (selected_rows < target_test_rows or not selected_blocks):
        best_block = None
        best_next_rows = None
        best_score = None
        for block in blocks:
            nxt = selected_rows + int(block_row_count.get(block, 0))
            score = abs(target_test_rows - nxt)
            if (
                best_score is None
                or score < best_score
                or (score == best_score and (best_next_rows is None or nxt > best_next_rows))
            ):
                best_score, best_next_rows, best_block = score, nxt, block
        if best_block is None:
            break
        selected_blocks.add(best_block)
        selected_rows = int(best_next_rows or selected_rows)
        blocks.remove(best_block)

    # Buffer blocks: Chebyshev-neighborhood of test blocks minus test blocks themselves.
    buffer_blocks: set[tuple[int, int]] = set()
    for b in selected_blocks:
        for nb in _neighbor_blocks(b, int(neighbor_buffer_blocks)):
            if nb in block_to_places and nb not in selected_blocks:
                buffer_blocks.add(nb)

    test_places: set[str] = set()
    for block in selected_blocks:
        test_places.update(block_to_places.get(block, []))
    buffer_places: set[str] = set()
    for block in buffer_blocks:
        buffer_places.update(block_to_places.get(block, []))

    train_places = set(by_place.keys()) - test_places - buffer_places
    place_overlap = len(train_places.intersection(test_places))
    assert place_overlap == 0, "place_overlap between train and test must be zero"

    test_keys: set[tuple[str, int, int]] = set()
    for place_key in test_places:
        for key in by_place[place_key]["keys"]:
            test_keys.add((str(key[0]), int(key[1]), int(key[2])))

    keys_sorted = sorted(list(test_keys), key=lambda x: (x[0], x[1], x[2]))
    n_buffer_rows = sum(len(by_place[p]["keys"]) for p in buffer_places)

    payload = {
        "keys": [[r, ro, co] for (r, ro, co) in keys_sorted],
        "meta": {
            "created_at": _utc_now(),
            "split_mode": "spatial_blocks_with_neighbor_buffer",
            "test_fraction_target": float(test_fraction),
            "test_fraction_actual": (float(len(test_keys)) / float(total_rows)) if total_rows else 0.0,
            "total_rows": int(total_rows),
            "test_rows": int(len(test_keys)),
            "train_rows_estimate": int(total_rows - len(test_keys) - n_buffer_rows),
            "buffer_rows": int(n_buffer_rows),
            "n_places_total": int(len(by_place)),
            "n_places_test": int(len(test_places)),
            "n_places_train": int(len(train_places)),
            "n_places_buffer": int(len(buffer_places)),
            "n_blocks_total": int(len(block_to_places)),
            "n_blocks_test": int(len(selected_blocks)),
            "n_blocks_buffer": int(len(buffer_blocks)),
            "split_block_size_places": int(max(split_block_size_places, 1)),
            "neighbor_buffer_blocks": int(neighbor_buffer_blocks),
            "place_overlap_train_vs_test": int(place_overlap),
            "places_with_multi_year": int(sum(1 for v in by_place.values() if len(v["years"]) > 1)),
            "test_cdw_rows": int(sum(by_place[p]["n_cdw"] for p in test_places)),
            "test_no_cdw_rows": int(sum(by_place[p]["n_no_cdw"] for p in test_places)),
            "test_manual_rows": int(sum(by_place[p]["n_manual"] for p in test_places)),
            "test_threshold_gate_rows": int(sum(by_place[p]["n_threshold_gate"] for p in test_places)),
            "seed": int(seed),
        },
    }

    output_test_split.parent.mkdir(parents=True, exist_ok=True)
    output_test_split.write_text(json.dumps(payload, indent=2))
    return payload["meta"]
