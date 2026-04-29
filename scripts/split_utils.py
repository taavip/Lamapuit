"""Utilities for pseudo-label selection and spatial cluster splits.

Provides:
- load_curated_rows(curated_dir)
- select_manual_and_pseudo(rows, pseudo_low, pseudo_high)
- spatial_cluster_splits(rows, test_fraction, val_fraction, cluster_size, guardband, seed)

This module keeps to the standard library and reuses the existing
write_spatial_block_test_split for spatial-block grouping.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.model_search_v4._labels import parse_raster_identity, is_manual_source, safe_float
from scripts.model_search_v4._splits import write_spatial_block_test_split, _place_to_block, _neighbor_blocks


Row = Dict[str, Any]
Key = Tuple[str, int, int]


def load_curated_rows(curated_dir: Path) -> List[Row]:
    """Read all *_labels.csv files from `curated_dir` and return normalized rows.

    Each returned row has keys: raster, row_off (int), col_off (int), label,
    model_prob (float|None), source (str).
    """
    rows: List[Row] = []
    curated_dir = Path(curated_dir)
    if not curated_dir.exists():
        return rows

    for csv_path in sorted(curated_dir.glob("*_labels.csv")):
        try:
            with csv_path.open(newline="") as fp:
                rd = csv.DictReader(fp)
                for r in rd:
                    label = (r.get("label") or "").strip()
                    if label not in ("cdw", "no_cdw"):
                        continue
                    try:
                        ro = int(r.get("row_off", 0))
                        co = int(r.get("col_off", 0))
                    except Exception:
                        continue
                    mp = safe_float(r.get("model_prob"))
                    rows.append(
                        {
                            "raster": r.get("raster", ""),
                            "row_off": ro,
                            "col_off": co,
                            "label": label,
                            "model_prob": mp,
                            "source": r.get("source", ""),
                        }
                    )
        except Exception:
            continue
    return rows


def build_candidates_from_rows(rows: List[Row]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({"key": (r["raster"], int(r["row_off"]), int(r["col_off"])), "raster": r["raster"], "label": r["label"], "model_prob": r.get("model_prob"), "source": r.get("source", "")})
    return out


def select_manual_and_pseudo(rows: List[Row], pseudo_low: float = 0.05, pseudo_high: float = 0.95) -> Tuple[List[Row], List[Row]]:
    """Return (manual_rows, pseudo_pool_rows).

    - manual_rows: rows whose `source` marks them as manual/auto_reviewed
    - pseudo_pool_rows: auto rows with confidence >= pseudo_high (cdw)
      or <= pseudo_low (no_cdw)
    """
    manual: List[Row] = [r for r in rows if is_manual_source(r.get("source", ""))]
    pseudo: List[Row] = []
    for r in rows:
        mp = r.get("model_prob")
        if mp is None:
            continue
        if r["label"] == "cdw" and mp >= float(pseudo_high):
            pseudo.append(r)
        elif r["label"] == "no_cdw" and mp <= float(pseudo_low):
            pseudo.append(r)
    return manual, pseudo


def _place_and_block_for_row(row: Row, block_size: int) -> Tuple[str, Tuple[int, int]]:
    ident = parse_raster_identity(row["raster"])
    place_key = str(ident.get("place_key"))
    block = _place_to_block(ident.get("grid_x"), ident.get("grid_y"), place_key, block_size)
    return place_key, block


def spatial_cluster_splits(
    rows: List[Row],
    out_dir: Path,
    test_fraction: float = 0.20,
    val_fraction: float = 0.10,
    cluster_size: int = 3,
    guardband: int = 1,
    seed: int = 2026,
    strat_tolerance: float = 0.05,
    max_tries: int = 10,
) -> Dict[str, Any]:
    """Create test/val/train splits by randomized spatial clusters.

    Returns a dict with keys: test_keys, val_keys, train_keys, pseudo_pool_keys,
    manual_keys and metadata.

    Implementation notes:
    - Uses write_spatial_block_test_split to choose spatial blocks (cluster_size)
      with a neighbour buffer of `guardband`.
    - Attempts several random seeds up to `max_tries` to keep test/val class
      balance within `strat_tolerance` of the global class ratio.
    - The final training set is: manual rows + "hard" examples with model_prob
      in [0.05,0.30] U [0.70,0.95], excluding val/test and their guardbands.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = build_candidates_from_rows(rows)
    total_rows = len(all_candidates)
    if total_rows == 0:
        raise ValueError("No curated rows available to split")

    global_pos_ratio = sum(1 for r in rows if r["label"] == "cdw") / max(1, len(rows))

    # Helper to compute place/block maps
    place_to_rows: Dict[str, List[Row]] = {}
    block_for_place: Dict[str, Tuple[int, int]] = {}
    for r in rows:
        ident = parse_raster_identity(r["raster"])
        place_key = str(ident.get("place_key"))
        place_to_rows.setdefault(place_key, []).append(r)
        if place_key not in block_for_place:
            block_for_place[place_key] = _place_to_block(ident.get("grid_x"), ident.get("grid_y"), place_key, cluster_size)

    # Try multiple seeds to get acceptable stratification
    chosen_test_keys: List[Key] = []
    chosen_val_keys: List[Key] = []
    chosen_meta: Dict[str, Any] = {}

    for attempt in range(max_tries):
        try_seed = int(seed) + attempt

        # write_spatial selects test clusters (and writes a JSON with keys)
        tmp_test = out_dir / f"split_tmp_test_{try_seed}.json"
        try:
            write_spatial_block_test_split(all_candidates, tmp_test, try_seed, test_fraction, cluster_size, neighbor_buffer_blocks=guardband)
            payload = json.loads(tmp_test.read_text())
            test_keys = [tuple(k) for k in payload.get("keys", [])]
        except Exception:
            test_keys = []

        if not test_keys:
            continue

        # Map test keys -> places and compute buffer places around test
        test_places: Set[str] = set()
        selected_blocks: Set[Tuple[int, int]] = set()
        for (r, ro, co) in test_keys:
            ident = parse_raster_identity(r)
            place_key = str(ident.get("place_key"))
            test_places.add(place_key)
            selected_blocks.add(_place_to_block(ident.get("grid_x"), ident.get("grid_y"), place_key, cluster_size))

        buffer_blocks: Set[Tuple[int, int]] = set()
        for b in selected_blocks:
            for nb in _neighbor_blocks(b, guardband):
                if nb not in selected_blocks:
                    buffer_blocks.add(nb)

        buffer_places: Set[str] = set()
        for p, b in block_for_place.items():
            if b in buffer_blocks:
                buffer_places.add(p)

        # Build candidates remaining after removing test + buffer places
        remaining_candidates = [c for c in all_candidates if parse_raster_identity(c["raster"]).get("place_key") not in (test_places | buffer_places)]
        if not remaining_candidates:
            continue

        # Scale val fraction relative to remaining rows
        actual_test_rows = len(test_keys)
        remaining_rows = len(remaining_candidates)
        remaining_fraction = remaining_rows / max(1, total_rows)
        if remaining_fraction <= 0:
            continue
        val_fraction_scaled = float(val_fraction) / float(max(1e-9, 1.0 - float(actual_test_rows) / float(total_rows)))
        val_fraction_scaled = max(0.0, min(0.9, val_fraction_scaled))

        tmp_val = out_dir / f"split_tmp_val_{try_seed}.json"
        try:
            write_spatial_block_test_split(remaining_candidates, tmp_val, try_seed + 1, val_fraction_scaled, cluster_size, neighbor_buffer_blocks=guardband)
            payload_v = json.loads(tmp_val.read_text())
            val_keys = [tuple(k) for k in payload_v.get("keys", [])]
        except Exception:
            val_keys = []

        if not val_keys:
            continue

        # Check stratification: positive ratio in test & val similar to global
        test_key_set = set(test_keys)
        val_key_set = set(val_keys)
        test_pos = sum(1 for r in rows if (r["raster"], r["row_off"], r["col_off"]) in test_key_set and r["label"] == "cdw")
        val_pos = sum(1 for r in rows if (r["raster"], r["row_off"], r["col_off"]) in val_key_set and r["label"] == "cdw")
        test_pos_ratio = test_pos / max(1, len(test_keys))
        val_pos_ratio = val_pos / max(1, len(val_keys))

        if abs(test_pos_ratio - global_pos_ratio) <= strat_tolerance and abs(val_pos_ratio - global_pos_ratio) <= strat_tolerance:
            chosen_test_keys = test_keys
            chosen_val_keys = val_keys
            chosen_meta = {"attempt": attempt, "seed": try_seed, "test_pos_ratio": test_pos_ratio, "val_pos_ratio": val_pos_ratio}
            break

    if not chosen_test_keys or not chosen_val_keys:
        # fall back to deterministic single-run split (last attempt's results)
        if not chosen_test_keys:
            try:
                tmp_test = out_dir / f"split_tmp_test_fallback.json"
                write_spatial_block_test_split(all_candidates, tmp_test, seed, test_fraction, cluster_size, neighbor_buffer_blocks=guardband)
                payload = json.loads(tmp_test.read_text())
                chosen_test_keys = [tuple(k) for k in payload.get("keys", [])]
            except Exception:
                chosen_test_keys = []
        if not chosen_val_keys:
            remaining_candidates = [c for c in all_candidates if parse_raster_identity(c["raster"]).get("place_key") not in {parse_raster_identity(k[0]).get("place_key") for k in chosen_test_keys}]
            try:
                tmp_val = out_dir / f"split_tmp_val_fallback.json"
                write_spatial_block_test_split(remaining_candidates, tmp_val, seed + 1, val_fraction, cluster_size, neighbor_buffer_blocks=guardband)
                payload_v = json.loads(tmp_val.read_text())
                chosen_val_keys = [tuple(k) for k in payload_v.get("keys", [])]
            except Exception:
                chosen_val_keys = []

    # Final exclusion: union of guardband around test and val
    selected_places: Set[str] = set()
    for (r, ro, co) in chosen_test_keys + chosen_val_keys:
        ident = parse_raster_identity(r)
        selected_places.add(str(ident.get("place_key")))

    excluded_places: Set[str] = set()
    for p in selected_places:
        b = block_for_place.get(p)
        if b is None:
            continue
        for nb in _neighbor_blocks(b, guardband):
            # find places that map to this neighbor block
            for place, block in block_for_place.items():
                if block == nb:
                    excluded_places.add(place)

    # Also exclude the selected places themselves (test + val)
    excluded_places.update(selected_places)

    # Build final sets of keys
    test_set = set(chosen_test_keys)
    val_set = set(chosen_val_keys)

    train_rows = [r for r in rows if parse_raster_identity(r["raster"]).get("place_key") not in excluded_places]

    # Manual and pseudo pools (from full rows)
    manual_rows_all, pseudo_rows = select_manual_and_pseudo(rows)

    # Hard example mining for train: keep manual + mid-confidence hard samples in ranges
    hard_ranges = [(0.05, 0.30), (0.70, 0.95)]
    hard_examples: List[Row] = []
    for r in train_rows:
        if is_manual_source(r.get("source", "")):
            hard_examples.append(r)
            continue
        mp = r.get("model_prob")
        if mp is None:
            continue
        for lo, hi in hard_ranges:
            if mp >= lo and mp <= hi:
                hard_examples.append(r)
                break

    # Compose final key lists
    def to_key(r: Row) -> Key:
        return (r["raster"], int(r["row_off"]), int(r["col_off"]))

    train_keys = list({to_key(r) for r in hard_examples})
    # Only include manual rows that are not in excluded places
    manual_rows = [r for r in manual_rows_all if parse_raster_identity(r["raster"]).get("place_key") not in excluded_places]
    manual_keys = list({to_key(r) for r in manual_rows})
    pseudo_keys = list({to_key(r) for r in pseudo_rows})

    # Ensure manual keys included (but only those allowed by exclusion)
    train_keys = list(set(train_keys) | set(manual_keys))

    out = {
        "meta": {
            "total_rows": total_rows,
            "test_count": len(chosen_test_keys),
            "val_count": len(chosen_val_keys),
            "train_hard_count": len(train_keys),
            **chosen_meta,
        },
        "test_keys": [list(k) for k in sorted(list(test_set))],
        "val_keys": [list(k) for k in sorted(list(val_set))],
        "train_keys": [list(k) for k in sorted(train_keys)],
        "manual_keys": [list(k) for k in sorted(manual_keys)],
        "pseudo_keys": [list(k) for k in sorted(pseudo_keys)],
    }

    out_path = out_dir / "splits_by_cluster.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print("Run via scripts/run_split_utils.py")
