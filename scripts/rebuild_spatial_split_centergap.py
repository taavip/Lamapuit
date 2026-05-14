#!/usr/bin/env python3
"""Rebuild a strict spatial split with explicit test/buffer/train roles.

Rules:
1. Same map_sheet + center coordinates across years are aggregated into one place.
2. Eligible places are rows from manual / auto_skip / high-confidence auto labels.
3. A random 2% of eligible places per map_sheet are selected as test seeds.
4. Around each seed:
   - Chebyshev distance <= 1  -> test
   - Chebyshev distance 2..3  -> buffer
   - Chebyshev distance >= 4  -> train
5. If a place is claimed both as test and buffer by different seeds, test wins.
   This keeps train outside the test footprint instead of leaking it into the
   test neighborhood.
6. Ineligible places remain none.

The script preserves the original `split` column and writes the new result to a
separate split column by default: `split_center_gap`.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PlaceInfo:
    map_sheet: str
    row_off: int
    col_off: int
    row_s: int
    col_s: int
    row_indices: list[int]
    eligible_row_indices: list[int]
    eligible: bool


def _snap_to_stride(value: int, stride: int) -> int:
    return int(np.floor((float(value) + (stride / 2.0)) / float(stride)))


def _is_eligible_row(row: pd.Series, low: float, high: float) -> bool:
    source = str(row.get("source", "")).strip().lower()
    if "manual" in source:
        return True
    if source in {"auto_skip", "auto_reviewed"}:
        return True

    try:
        mp = float(row.get("model_prob"))
    except Exception:
        return False
    if np.isnan(mp):
        return False
    return mp <= low or mp >= high


def _sheet_rng(seed: int, map_sheet: str) -> np.random.Generator:
    sheet_seed = int(zlib.crc32(f"{seed}:{map_sheet}".encode("utf-8"))) & 0xFFFFFFFF
    return np.random.default_rng(sheet_seed)


def build_places(
    df: pd.DataFrame,
    stride: int,
    low: float,
    high: float,
) -> tuple[
    dict[tuple[str, int, int], PlaceInfo],
    dict[str, list[tuple[str, int, int]]],
    dict[str, list[tuple[str, int, int]]],
]:
    places: dict[tuple[str, int, int], PlaceInfo] = {}
    by_sheet: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    eligible_places_by_sheet: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

    grouped = df.groupby(["map_sheet", "row_off", "col_off"], sort=False)
    for (map_sheet, row_off, col_off), g in grouped:
        row_off_i = int(row_off)
        col_off_i = int(col_off)
        key = (str(map_sheet), row_off_i, col_off_i)
        row_s = _snap_to_stride(row_off_i, stride)
        col_s = _snap_to_stride(col_off_i, stride)
        eligible_mask = g.apply(lambda r: _is_eligible_row(r, low=low, high=high), axis=1)
        eligible_row_indices = g.index[eligible_mask].to_list()
        eligible = bool(eligible_row_indices)
        info = PlaceInfo(
            map_sheet=str(map_sheet),
            row_off=row_off_i,
            col_off=col_off_i,
            row_s=int(row_s),
            col_s=int(col_s),
            row_indices=g.index.to_list(),
            eligible_row_indices=eligible_row_indices,
            eligible=eligible,
        )
        places[key] = info
        by_sheet[str(map_sheet)].append(key)
        if eligible:
            eligible_places_by_sheet[str(map_sheet)].append(key)

    return places, by_sheet, eligible_places_by_sheet


def assign_center_gap_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.02,
    stride: int = 64,
    test_radius: int = 1,
    buffer_radius: int = 3,
    low: float = 0.05,
    high: float = 0.95,
    split_column: str = "split_center_gap",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if buffer_radius < test_radius:
        raise ValueError("buffer_radius must be >= test_radius")

    out = df.copy()
    out[split_column] = "none"

    places, by_sheet, eligible_places_by_sheet = build_places(out, stride=stride, low=low, high=high)
    place_by_coord: dict[tuple[str, int, int], tuple[str, int, int]] = {
        (info.map_sheet, info.row_s, info.col_s): key for key, info in places.items()
    }

    offsets: list[tuple[int, int, str]] = []
    for dr in range(-buffer_radius, buffer_radius + 1):
        for dc in range(-buffer_radius, buffer_radius + 1):
            d = max(abs(dr), abs(dc))
            if d <= test_radius:
                role = "test"
            elif d <= buffer_radius:
                role = "buffer"
            else:
                continue
            offsets.append((dr, dc, role))

    claims: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    sheet_stats: dict[str, dict[str, Any]] = {}

    for map_sheet, place_keys in sorted(by_sheet.items()):
        sheet_rng = _sheet_rng(seed, map_sheet)

        eligible_place_keys = eligible_places_by_sheet.get(map_sheet, [])
        n_test = max(1, int(len(eligible_place_keys) * test_frac)) if eligible_place_keys else 0
        n_test = min(n_test, len(eligible_place_keys))
        test_seed_keys: list[tuple[str, int, int]] = []
        if n_test > 0:
            chosen = sheet_rng.choice(len(eligible_place_keys), size=n_test, replace=False)
            test_seed_keys = sorted({eligible_place_keys[int(i)] for i in np.atleast_1d(chosen)})

        for seed_key in test_seed_keys:
            seed_info = places[seed_key]
            for dr, dc, role in offsets:
                target = (
                    map_sheet,
                    seed_info.row_s + dr,
                    seed_info.col_s + dc,
                )
                place_key = place_by_coord.get(target)
                if place_key is not None:
                    claims[place_key].add(role)

        # Assign roles for all places on this sheet.
        n_test_places = n_buffer_places = n_train_places = n_none_places = 0
        overlap_places = 0
        for place_key in place_keys:
            info = places[place_key]
            role = "none"
            place_claims = claims.get(place_key, set())
            if info.eligible:
                if "test" in place_claims:
                    role = "test"
                elif "buffer" in place_claims:
                    role = "buffer"
                else:
                    role = "train"

                if "test" in place_claims and "buffer" in place_claims:
                    overlap_places += 1

            out.loc[info.row_indices, split_column] = role

            if role == "test":
                n_test_places += 1
            elif role == "buffer":
                n_buffer_places += 1
            elif role == "train":
                n_train_places += 1
            else:
                n_none_places += 1

        sheet_stats[map_sheet] = {
            "eligible_places": int(sum(1 for k in place_keys if places[k].eligible)),
            "test_seed_places": int(len(test_seed_keys)),
            "test_places": int(n_test_places),
            "buffer_places": int(n_buffer_places),
            "train_places": int(n_train_places),
            "none_places": int(n_none_places),
            "test_buffer_overlap_places": int(overlap_places),
        }

    meta = {
        "seed": int(seed),
        "test_fraction": float(test_frac),
        "stride": int(stride),
        "test_radius": int(test_radius),
        "buffer_radius": int(buffer_radius),
        "eligibility": {
            "manual": True,
            "auto_skip": True,
            "auto_reviewed": True,
            "high_confidence_auto": {"low": float(low), "high": float(high)},
        },
        "sheet_stats": sheet_stats,
        "row_counts": out[split_column].value_counts().to_dict(),
    }
    return out, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a strict center-gap spatial split.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits_spatial_ensemble.csv"),
        help="Input CSV with the original split column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits_spatial_ensemble_centergap.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.02)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--test-radius", type=int, default=1)
    parser.add_argument("--buffer-radius", type=int, default=3)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.95)
    parser.add_argument("--split-column", default="split_center_gap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "split" not in df.columns:
        raise RuntimeError("Input CSV does not contain the original 'split' column.")

    out, meta = assign_center_gap_split(
        df,
        seed=args.seed,
        test_frac=args.test_frac,
        stride=args.stride,
        test_radius=args.test_radius,
        buffer_radius=args.buffer_radius,
        low=args.low_threshold,
        high=args.high_threshold,
        split_column=args.split_column,
    )

    # Validate exclusivity among the new roles on eligible places.
    new_counts = out[args.split_column].value_counts().to_dict()
    print("New split counts:", new_counts)
    print("Sheets processed:", len(meta["sheet_stats"]))
    print("Total test/buffer overlaps assigned to test:", sum(s["test_buffer_overlap_places"] for s in meta["sheet_stats"].values()))

    # Compare with the original split for context only.
    cmp = pd.crosstab(out["split"], out[args.split_column], dropna=False)
    print("\nOriginal vs new split crosstab:")
    print(cmp.to_string())

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        report_path = args.output.with_suffix(".json")
        report_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {args.output}")
        print(f"Saved: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
