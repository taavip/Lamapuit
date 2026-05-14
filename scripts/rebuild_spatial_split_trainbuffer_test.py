#!/usr/bin/env python3
"""Finalize a train-buffer-test split from a previous spatial split.

This second pass keeps the previously assigned train places fixed, builds a
two-tile buffer around those train places, and assigns every remaining eligible
place to test. Ineligible places stay none. The same place coordinates across
all years on a map_sheet inherit the same final role.
"""

from __future__ import annotations

import argparse
import json
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
    source_role: str


def _snap_to_stride(value: int, stride: int) -> int:
    return int(np.floor((float(value) + (stride / 2.0)) / float(stride)))


def _priority(role: str) -> int:
    order = {"train": 3, "buffer": 2, "test": 1, "none": 0}
    return order.get(str(role).strip().lower(), -1)


def build_places(
    df: pd.DataFrame,
    source_split_column: str,
    stride: int,
) -> tuple[dict[tuple[str, int, int], PlaceInfo], dict[str, list[tuple[str, int, int]]]]:
    places: dict[tuple[str, int, int], PlaceInfo] = {}
    by_sheet: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

    grouped = df.groupby(["map_sheet", "row_off", "col_off"], sort=False)
    for (map_sheet, row_off, col_off), g in grouped:
        row_off_i = int(row_off)
        col_off_i = int(col_off)
        key = (str(map_sheet), row_off_i, col_off_i)
        row_s = _snap_to_stride(row_off_i, stride)
        col_s = _snap_to_stride(col_off_i, stride)
        roles = [str(v).strip().lower() for v in g[source_split_column].fillna("none").tolist()]
        source_role = max(roles, key=_priority) if roles else "none"
        info = PlaceInfo(
            map_sheet=str(map_sheet),
            row_off=row_off_i,
            col_off=col_off_i,
            row_s=int(row_s),
            col_s=int(col_s),
            row_indices=g.index.to_list(),
            source_role=source_role,
        )
        places[key] = info
        by_sheet[str(map_sheet)].append(key)

    return places, by_sheet


def assign_train_buffer_test_split(
    df: pd.DataFrame,
    source_split_column: str = "split_center_gap",
    split_column: str = "split_train_buffer_test",
    stride: int = 64,
    buffer_radius: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if buffer_radius < 1:
        raise ValueError("buffer_radius must be at least 1")

    out = df.copy()
    out[split_column] = "none"

    places, by_sheet = build_places(out, source_split_column=source_split_column, stride=stride)
    place_by_coord: dict[tuple[str, int, int], tuple[str, int, int]] = {
        (info.map_sheet, info.row_s, info.col_s): key for key, info in places.items()
    }

    offsets: list[tuple[int, int]] = []
    for dr in range(-buffer_radius, buffer_radius + 1):
        for dc in range(-buffer_radius, buffer_radius + 1):
            if max(abs(dr), abs(dc)) == 0:
                continue
            if max(abs(dr), abs(dc)) <= buffer_radius:
                offsets.append((dr, dc))

    sheet_stats: dict[str, dict[str, Any]] = {}

    for map_sheet, place_keys in sorted(by_sheet.items()):
        train_core_keys = [k for k in place_keys if places[k].source_role == "train"]
        buffer_claims: set[tuple[str, int, int]] = set()

        for seed_key in train_core_keys:
            seed_info = places[seed_key]
            for dr, dc in offsets:
                target = (map_sheet, seed_info.row_s + dr, seed_info.col_s + dc)
                place_key = place_by_coord.get(target)
                if place_key is not None and places[place_key].source_role != "none":
                    buffer_claims.add(place_key)

        n_train = n_buffer = n_test = n_none = 0
        for place_key in place_keys:
            info = places[place_key]
            role = "none"
            if info.source_role != "none":
                if info.source_role == "train":
                    role = "train"
                elif place_key in buffer_claims:
                    role = "buffer"
                else:
                    role = "test"

            out.loc[info.row_indices, split_column] = role

            if role == "train":
                n_train += 1
            elif role == "buffer":
                n_buffer += 1
            elif role == "test":
                n_test += 1
            else:
                n_none += 1

        sheet_stats[map_sheet] = {
            "train_core_places": int(len(train_core_keys)),
            "buffer_places": int(n_buffer),
            "test_places": int(n_test),
            "none_places": int(n_none),
        }

    meta = {
        "source_split_column": source_split_column,
        "seed_roles": "train",
        "buffer_radius": int(buffer_radius),
        "stride": int(stride),
        "sheet_stats": sheet_stats,
        "row_counts": out[split_column].value_counts().to_dict(),
    }
    return out, meta


def _validate_no_train_test_overlap(df: pd.DataFrame, split_column: str) -> dict[str, Any]:
    df2 = df[df[split_column].isin(["train", "test"])].copy()
    df2 = df2.drop_duplicates(["map_sheet", "row_off", "col_off", "chunk_size"])
    bad_sheets = []
    total_overlap = 0
    for map_sheet, s in df2.groupby("map_sheet"):
        train = s[s[split_column] == "train"][["row_off", "col_off", "chunk_size"]].to_records(index=False)
        test = s[s[split_column] == "test"][["row_off", "col_off", "chunk_size"]].to_records(index=False)
        sheet_overlap = 0
        for a in train:
            ax1, ax2 = float(a.col_off), float(a.col_off + a.chunk_size)
            ay1, ay2 = float(a.row_off), float(a.row_off + a.chunk_size)
            for b in test:
                bx1, bx2 = float(b.col_off), float(b.col_off + b.chunk_size)
                by1, by2 = float(b.row_off), float(b.row_off + b.chunk_size)
                if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                    sheet_overlap += 1
                    break
        if sheet_overlap:
            bad_sheets.append((int(map_sheet), int(sheet_overlap), int(len(train)), int(len(test))))
            total_overlap += sheet_overlap
    return {
        "sheets_with_overlap": int(len(bad_sheets)),
        "total_overlapping_train_tiles": int(total_overlap),
        "bad_sheets": bad_sheets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a train-buffer-test split from a previous spatial split.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits_centergap.csv"),
        help="Input CSV with the prior split column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits_train_buffer_test.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--source-split-column", default="split_center_gap")
    parser.add_argument("--split-column", default="split_train_buffer_test")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--buffer-radius", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.source_split_column not in df.columns:
        raise RuntimeError(f"Input CSV does not contain source split column '{args.source_split_column}'.")

    out, meta = assign_train_buffer_test_split(
        df,
        source_split_column=args.source_split_column,
        split_column=args.split_column,
        stride=args.stride,
        buffer_radius=args.buffer_radius,
    )

    validation = _validate_no_train_test_overlap(out, args.split_column)
    meta["validation"] = validation

    print("New split counts:", meta["row_counts"])
    print("Sheets processed:", len(meta["sheet_stats"]))
    print("Train/test overlap check:", validation["sheets_with_overlap"], "sheets,", validation["total_overlapping_train_tiles"], "overlapping train tiles")
    print("\nSource vs new split crosstab:")
    print(pd.crosstab(out[args.source_split_column], out[args.split_column], dropna=False).to_string())

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
