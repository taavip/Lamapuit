#!/usr/bin/env python3
"""Recalculate manual review queue from auto-labeled tiles.

Builds a deterministic queue combining:
1) Low-confidence auto labels in [low_min, low_max]
2) A fixed fraction sample from remaining auto labels

Queue rows include last-label provenance fields so review can audit what happened:
- reason (low_confidence | spotcheck)
- last_source (manual/auto/auto_skip/auto_reviewed)
- last_model_name
- last_model_prob
- last_timestamp

Usage:
  python scripts/recalculate_manual_review_queue.py \
    --labels-dir output/onboarding_labels_v1 \
    --out output/onboarding_labels_v1/manual_review_queue_pre_split.csv \
    --low-min 0.05 --low-max 0.95 --spotcheck-frac 0.05 --seed 2026
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TileRow:
    raster: str
    row_off: int
    col_off: int
    source: str
    model_name: str
    model_prob: float
    timestamp: str


def _read_last_rows(labels_dir: Path) -> dict[tuple[str, int, int], TileRow]:
    """Load latest row per tile key across all *_labels.csv files."""
    latest: dict[tuple[str, int, int], TileRow] = {}

    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    raster = str(row.get("raster", "")).strip()
                    row_off = int(float(row.get("row_off", "0") or 0))
                    col_off = int(float(row.get("col_off", "0") or 0))
                    source = str(row.get("source", "")).strip()
                    model_name = str(row.get("model_name", "")).strip()
                    prob_raw = str(row.get("model_prob", "")).strip()
                    if not raster or not prob_raw:
                        continue
                    model_prob = float(prob_raw)
                    timestamp = str(row.get("timestamp", "")).strip()
                except Exception:
                    continue

                latest[(raster, row_off, col_off)] = TileRow(
                    raster=raster,
                    row_off=row_off,
                    col_off=col_off,
                    source=source,
                    model_name=model_name,
                    model_prob=model_prob,
                    timestamp=timestamp,
                )

    return latest


def main() -> int:
    p = argparse.ArgumentParser(description="Recalculate manual review queue from auto labels")
    p.add_argument("--labels-dir", type=Path, required=True, help="Directory with *_labels.csv")
    p.add_argument("--out", type=Path, required=True, help="Output review queue CSV")
    p.add_argument(
        "--low-min",
        type=float,
        default=0.05,
        help="Lower bound of low-confidence bucket (inclusive)",
    )
    p.add_argument(
        "--low-max",
        type=float,
        default=0.95,
        help="Upper bound of low-confidence bucket (inclusive)",
    )
    p.add_argument(
        "--spotcheck-frac",
        type=float,
        default=0.05,
        help="Fraction sampled from non-low-confidence auto tiles",
    )
    p.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic sample")
    args = p.parse_args()

    if not args.labels_dir.exists():
        raise FileNotFoundError(f"labels-dir not found: {args.labels_dir}")
    if not (0.0 <= args.low_min <= 1.0 and 0.0 <= args.low_max <= 1.0):
        raise ValueError("low-min and low-max must be in [0,1]")
    if args.low_min > args.low_max:
        raise ValueError("low-min cannot be greater than low-max")
    if not (0.0 <= args.spotcheck_frac <= 1.0):
        raise ValueError("spotcheck-frac must be in [0,1]")

    latest = _read_last_rows(args.labels_dir)
    autos = [r for r in latest.values() if r.source == "auto"]

    low = [r for r in autos if args.low_min <= r.model_prob <= args.low_max]
    low_keys = {(r.raster, r.row_off, r.col_off) for r in low}
    remainder = [r for r in autos if (r.raster, r.row_off, r.col_off) not in low_keys]

    rng = random.Random(args.seed)
    n_spot = round(len(remainder) * args.spotcheck_frac)
    spot = rng.sample(remainder, n_spot) if n_spot > 0 else []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "raster",
                "row_off",
                "col_off",
                "reason",
                "last_source",
                "last_model_name",
                "last_model_prob",
                "last_timestamp",
            ],
        )
        w.writeheader()
        for r in low:
            w.writerow(
                {
                    "raster": r.raster,
                    "row_off": r.row_off,
                    "col_off": r.col_off,
                    "reason": "low_confidence",
                    "last_source": r.source,
                    "last_model_name": r.model_name,
                    "last_model_prob": f"{r.model_prob:.6f}",
                    "last_timestamp": r.timestamp,
                }
            )
        for r in spot:
            w.writerow(
                {
                    "raster": r.raster,
                    "row_off": r.row_off,
                    "col_off": r.col_off,
                    "reason": "spotcheck",
                    "last_source": r.source,
                    "last_model_name": r.model_name,
                    "last_model_prob": f"{r.model_prob:.6f}",
                    "last_timestamp": r.timestamp,
                }
            )

    print(f"auto_total={len(autos)}")
    print(f"low_bucket=[{args.low_min:.2f},{args.low_max:.2f}] -> {len(low)}")
    print(f"spotcheck_frac={args.spotcheck_frac:.4f} -> {len(spot)}")
    print(f"queue_total={len(low) + len(spot)}")
    print(f"queue_path={args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
