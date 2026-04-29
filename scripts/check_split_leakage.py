#!/usr/bin/env python3
"""Check train/val/test leakage from tile metadata.

Detects two leakage classes:
1) Origin-group leakage: same (raster,row_off,col_off) appears in multiple splits.
2) Overlap leakage: tiles from different splits overlap spatially within the same raster.

Input is expected to be the `tile_metadata.csv` produced by prepare_instance.

Usage:
  python scripts/check_split_leakage.py \
    --metadata output/cdw_training_v2/dataset/tile_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path


@dataclass
class TileRow:
    tile: str
    split: str
    raster: str
    row_off: int
    col_off: int
    minx: float | None
    miny: float | None
    maxx: float | None
    maxy: float | None


def _to_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    vv = str(v).strip()
    if vv == "":
        return None
    try:
        return float(vv)
    except Exception:
        return None


def _read_rows(csv_path: Path) -> list[TileRow]:
    rows: list[TileRow] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                TileRow(
                    tile=str(r.get("tile", "")),
                    split=str(r.get("split", "")).strip(),
                    raster=str(r.get("raster", "")).strip(),
                    row_off=_to_int(str(r.get("row_off", "0"))),
                    col_off=_to_int(str(r.get("col_off", "0"))),
                    minx=_to_float(r.get("minx")),
                    miny=_to_float(r.get("miny")),
                    maxx=_to_float(r.get("maxx")),
                    maxy=_to_float(r.get("maxy")),
                )
            )
    return rows


def _rect_overlap(a: TileRow, b: TileRow) -> bool:
    if None in (a.minx, a.miny, a.maxx, a.maxy, b.minx, b.miny, b.maxx, b.maxy):
        return False
    # Strict overlap area > 0 (touching edges is not considered leakage)
    return (a.minx < b.maxx and a.maxx > b.minx and a.miny < b.maxy and a.maxy > b.miny)


def _origin_group_leaks(rows: list[TileRow]) -> list[dict]:
    by_origin: dict[tuple[str, int, int], set[str]] = {}
    examples: dict[tuple[str, int, int], list[str]] = {}
    for r in rows:
        k = (r.raster, r.row_off, r.col_off)
        by_origin.setdefault(k, set()).add(r.split)
        examples.setdefault(k, []).append(r.tile)

    leaks: list[dict] = []
    for (raster, row_off, col_off), splits in by_origin.items():
        clean_splits = sorted(s for s in splits if s)
        if len(clean_splits) > 1:
            leaks.append(
                {
                    "raster": raster,
                    "row_off": row_off,
                    "col_off": col_off,
                    "splits": clean_splits,
                    "tiles": examples[(raster, row_off, col_off)][:10],
                }
            )
    return leaks


def _overlap_leaks(rows: list[TileRow], max_pairs_per_raster: int) -> list[dict]:
    by_raster: dict[str, list[TileRow]] = {}
    for r in rows:
        by_raster.setdefault(r.raster, []).append(r)

    leaks: list[dict] = []
    for raster, rr in by_raster.items():
        # Small acceleration: sort by minx then only check forward windows.
        rr2 = [x for x in rr if None not in (x.minx, x.miny, x.maxx, x.maxy) and x.split]
        rr2.sort(key=lambda x: (x.minx, x.miny))
        n = len(rr2)
        hit = 0
        for i in range(n):
            a = rr2[i]
            for j in range(i + 1, n):
                b = rr2[j]
                if b.minx is not None and a.maxx is not None and b.minx >= a.maxx:
                    break
                if a.split == b.split:
                    continue
                if _rect_overlap(a, b):
                    leaks.append(
                        {
                            "raster": raster,
                            "split_a": a.split,
                            "tile_a": a.tile,
                            "row_off_a": a.row_off,
                            "col_off_a": a.col_off,
                            "split_b": b.split,
                            "tile_b": b.tile,
                            "row_off_b": b.row_off,
                            "col_off_b": b.col_off,
                        }
                    )
                    hit += 1
                    if hit >= max_pairs_per_raster:
                        break
            if hit >= max_pairs_per_raster:
                break
    return leaks


def main() -> int:
    p = argparse.ArgumentParser(description="Leakage checker for split tile metadata")
    p.add_argument("--metadata", required=True, type=Path, help="Path to tile_metadata.csv")
    p.add_argument(
        "--max-overlap-pairs-per-raster",
        type=int,
        default=200,
        help="Limit number of reported overlap conflicts per raster for large datasets",
    )
    p.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional report output path (default: sibling leakage_report.json)",
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional overlap-cases CSV path (default: sibling leakage_overlap_cases.csv)",
    )
    args = p.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    rows = _read_rows(args.metadata)
    if not rows:
        print("No metadata rows found.")
        return 2

    origin_leaks = _origin_group_leaks(rows)
    overlap_leaks = _overlap_leaks(rows, max_pairs_per_raster=args.max_overlap_pairs_per_raster)

    splits = sorted({r.split for r in rows if r.split})
    split_counts: dict[str, int] = {s: 0 for s in splits}
    for r in rows:
        if r.split:
            split_counts[r.split] = split_counts.get(r.split, 0) + 1

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metadata": str(args.metadata),
        "total_rows": len(rows),
        "split_counts": split_counts,
        "origin_group_leak_count": len(origin_leaks),
        "overlap_leak_count": len(overlap_leaks),
        "origin_group_leaks": origin_leaks,
        "overlap_leaks_sample": overlap_leaks,
        "status": "pass" if (not origin_leaks and not overlap_leaks) else "fail",
    }

    report_json = args.report_json or args.metadata.with_name("leakage_report.json")
    report_csv = args.report_csv or args.metadata.with_name("leakage_overlap_cases.csv")
    report_json.write_text(json.dumps(report, indent=2))

    with open(report_csv, "w", newline="") as f:
        fields = [
            "raster",
            "split_a",
            "tile_a",
            "row_off_a",
            "col_off_a",
            "split_b",
            "tile_b",
            "row_off_b",
            "col_off_b",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in overlap_leaks:
            w.writerow(row)

    print(f"Leakage check: {report['status'].upper()}")
    print(f"  rows={len(rows)} splits={split_counts}")
    print(f"  origin_group_leaks={len(origin_leaks)}")
    print(f"  overlap_leaks={len(overlap_leaks)}")
    print(f"  report_json={report_json}")
    print(f"  report_csv={report_csv}")

    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
