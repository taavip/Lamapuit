#!/usr/bin/env python3
"""Auto-remediate split leakage in tile metadata.

Reads tile metadata CSV and rewrites split assignments so overlapping tiles
(within the same raster) belong to a single split per overlap-connected group.

This is intended as a post-processing safeguard before retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Row:
    idx: int
    raw: dict
    split: str
    raster: str
    minx: float | None
    miny: float | None
    maxx: float | None
    maxy: float | None


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def _to_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _read_rows(path: Path) -> list[Row]:
    out: list[Row] = []
    with open(path, newline="") as f:
        for i, raw in enumerate(csv.DictReader(f)):
            out.append(
                Row(
                    idx=i,
                    raw=raw,
                    split=str(raw.get("split", "")).strip(),
                    raster=str(raw.get("raster", "")).strip(),
                    minx=_to_float(raw.get("minx")),
                    miny=_to_float(raw.get("miny")),
                    maxx=_to_float(raw.get("maxx")),
                    maxy=_to_float(raw.get("maxy")),
                )
            )
    return out


def _overlap(a: Row, b: Row) -> bool:
    if None in (a.minx, a.miny, a.maxx, a.maxy, b.minx, b.miny, b.maxx, b.maxy):
        return False
    return (a.minx < b.maxx and a.maxx > b.minx and a.miny < b.maxy and a.maxy > b.miny)


def _choose_split(splits: list[str], strategy: str) -> str:
    c = Counter(splits)
    if strategy == "prefer_test":
        if c.get("test", 0) > 0:
            return "test"
        if c.get("val", 0) > 0:
            return "val"
        if c.get("train", 0) > 0:
            return "train"
    elif strategy == "prefer_val_test":
        if c.get("val", 0) > 0:
            return "val"
        if c.get("test", 0) > 0:
            return "test"
        if c.get("train", 0) > 0:
            return "train"

    # majority fallback (or explicit majority strategy)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return ranked[0][0] if ranked else "train"


def remediate(path: Path, out_csv: Path, report_json: Path, strategy: str, limit_checks: int) -> int:
    rows = _read_rows(path)
    if not rows:
        raise ValueError("No rows in metadata")

    dsu = DSU(len(rows))
    by_raster: dict[str, list[int]] = {}
    for r in rows:
        by_raster.setdefault(r.raster, []).append(r.idx)

    # Build overlap-connected components (same raster only).
    for _, idxs in by_raster.items():
        local = [rows[i] for i in idxs if None not in (rows[i].minx, rows[i].miny, rows[i].maxx, rows[i].maxy)]
        local.sort(key=lambda x: (x.minx, x.miny))
        n = len(local)
        checks = 0
        for i in range(n):
            a = local[i]
            for j in range(i + 1, n):
                b = local[j]
                if b.minx is not None and a.maxx is not None and b.minx >= a.maxx:
                    break
                checks += 1
                if checks > limit_checks and limit_checks > 0:
                    break
                if _overlap(a, b):
                    dsu.union(a.idx, b.idx)
            if checks > limit_checks and limit_checks > 0:
                break

    comps: dict[int, list[int]] = {}
    for i in range(len(rows)):
        root = dsu.find(i)
        comps.setdefault(root, []).append(i)

    changed = 0
    conflict_components = 0
    split_before = Counter(r.split for r in rows)

    for _, members in comps.items():
        splits = [rows[i].split for i in members if rows[i].split]
        uniq = sorted(set(splits))
        if len(uniq) <= 1:
            continue
        conflict_components += 1
        target = _choose_split(splits, strategy)
        for i in members:
            if rows[i].raw.get("split", "") != target:
                rows[i].raw["split"] = target
                changed += 1

    # write remediated CSV preserving original field order
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        fieldnames = rdr.fieldnames or []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.raw)

    split_after = Counter(r.raw.get("split", "") for r in rows)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_csv": str(path),
        "output_csv": str(out_csv),
        "strategy": strategy,
        "rows_total": len(rows),
        "conflict_components": conflict_components,
        "rows_reassigned": changed,
        "split_counts_before": dict(split_before),
        "split_counts_after": dict(split_after),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2))

    print(f"Remediation complete: components_fixed={conflict_components} rows_reassigned={changed}")
    print(f"  out_csv={out_csv}")
    print(f"  report={report_json}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Auto-remediate split leakage in tile metadata")
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output metadata CSV path (default: sibling tile_metadata.remediated.csv)",
    )
    p.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Report path (default: sibling leakage_remediation_report.json)",
    )
    p.add_argument(
        "--strategy",
        choices=["prefer_test", "prefer_val_test", "majority"],
        default="prefer_test",
        help="How to choose target split for conflicting overlap components",
    )
    p.add_argument(
        "--max-overlap-checks-per-raster",
        type=int,
        default=0,
        help="Optional cap for overlap pair checks per raster (0 = unlimited)",
    )
    args = p.parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    out_csv = args.out_csv or args.metadata.with_name("tile_metadata.remediated.csv")
    report_json = args.report_json or args.metadata.with_name("leakage_remediation_report.json")

    return remediate(
        path=args.metadata,
        out_csv=out_csv,
        report_json=report_json,
        strategy=args.strategy,
        limit_checks=args.max_overlap_checks_per_raster,
    )


if __name__ == "__main__":
    raise SystemExit(main())
