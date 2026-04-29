#!/usr/bin/env python3
"""
Create a permanent, stratified 10% test split for the CNN-Deep-Attn classifier.

Reads all *_labels.csv files, samples 10% of labeled tiles **within every
raster + class stratum**, and writes the reserved keys to:

    output/tile_labels/cnn_test_split.json

The split is deterministic (fixed seed) so that:
  - train_ensemble.py and fine_tune_cnn.py can load and exclude these tiles.
  - The test set is NEVER in the train or val sets.
  - Re-running the script with the same --seed produces identical output.

The file format is a JSON object:
  {
    "keys": [[raster, row_off, col_off], ...],    ← test set tile identifiers
    "meta": {...}    ← summary statistics
  }

Usage
-----
python scripts/create_cnn_test_split.py \\
    --labels output/tile_labels \\
    --output output/tile_labels/cnn_test_split.json \\
    --test-pct 10.0 \\
    --seed 2026

# Preview without writing:
python scripts/create_cnn_test_split.py ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_labels(labels_dir: Path) -> dict[str, list[dict]]:
    """Return {raster: [row_dict, ...]} for all CDW/no_cdw rows."""
    raster_rows: dict[str, list[dict]] = {}
    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        rows: list[dict] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label", "") in ("cdw", "no_cdw"):
                    rows.append(dict(row))
        if rows:
            raster_name = rows[0]["raster"]
            raster_rows[raster_name] = rows
    return raster_rows


def create_split(
    raster_rows: dict[str, list[dict]],
    test_pct: float = 10.0,
    seed: int = 2026,
) -> tuple[list[tuple], dict]:
    """Return (test_keys, meta) where test_keys = [(raster, row_off, col_off), ...].

    Stratification: sample test_pct% from EACH (raster, class) stratum so that
    both CDW and no_CDW tiles are represented from every raster.
    """
    rng = random.Random(seed)

    # Group by (raster, class)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for raster, rows in raster_rows.items():
        for row in rows:
            strata[(raster, row["label"])].append(row)

    test_keys: list[tuple] = []
    stratum_summary: list[dict] = []

    for (raster, cls), rows in sorted(strata.items()):
        n_test = max(1, round(len(rows) * test_pct / 100.0))
        n_test = min(n_test, len(rows))

        sampled = rng.sample(rows, n_test)
        for row in sampled:
            test_keys.append((raster, int(row["row_off"]), int(row["col_off"])))

        stratum_summary.append(
            {
                "raster": raster,
                "class": cls,
                "total": len(rows),
                "n_test": n_test,
                "pct_test": round(100.0 * n_test / len(rows), 1),
            }
        )

    # Build meta
    n_cdw_test = sum(
        1
        for k in test_keys
        for r, c, lbl in [(*k,)]  # raster, row_off, col_off — we need to track class
    )
    # Count by class
    n_cdw_test = sum(s["n_test"] for s in stratum_summary if s["class"] == "cdw")
    n_no_test = sum(s["n_test"] for s in stratum_summary if s["class"] == "no_cdw")
    n_total = sum(len(v) for v in raster_rows.values())

    meta = {
        "total_labels": n_total,
        "test_tiles": len(test_keys),
        "test_cdw": n_cdw_test,
        "test_no_cdw": n_no_test,
        "test_pct_requested": test_pct,
        "test_pct_actual": round(100.0 * len(test_keys) / max(n_total, 1), 2),
        "seed": seed,
        "n_rasters": len(raster_rows),
        "strata": stratum_summary,
    }

    return test_keys, meta


def print_summary(meta: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  CNN Test Split Summary")
    print(f"{'='*65}")
    print(f"  Total labels     : {meta['total_labels']:,}")
    print(
        f"  Test tiles       : {meta['test_tiles']:,}  "
        f"({meta['test_pct_actual']:.1f}%  requested {meta['test_pct_requested']:.1f}%)"
    )
    print(f"  Test CDW         : {meta['test_cdw']:,}")
    print(f"  Test no_CDW      : {meta['test_no_cdw']:,}")
    print(f"  Rasters covered  : {meta['n_rasters']}")
    print(f"\n  {'Raster':<52} {'CDW kept':>9} {'no_CDW kept':>11}")
    print(f"  {'-'*52} {'-'*9} {'-'*11}")

    by_raster: dict[str, dict] = {}
    for s in meta["strata"]:
        by_raster.setdefault(s["raster"], {})[s["class"]] = s["n_test"]

    for raster, cls_map in sorted(by_raster.items()):
        n_cdw = cls_map.get("cdw", 0)
        n_no = cls_map.get("no_cdw", 0)
        print(f"  {raster:<52} {n_cdw:>9,} {n_no:>11,}")

    print(f"{'='*65}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create a stratified permanent test split for the CNN classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--labels",
        default="output/tile_labels",
        help="Directory containing *_labels.csv (default: output/tile_labels)",
    )
    p.add_argument(
        "--output",
        default="output/tile_labels/cnn_test_split.json",
        help="Output JSON file path (default: output/tile_labels/cnn_test_split.json)",
    )
    p.add_argument(
        "--test-pct",
        type=float,
        default=10.0,
        help="Percentage of tiles per stratum to reserve as test set (default: 10.0)",
    )
    p.add_argument(
        "--seed", type=int, default=2026, help="Random seed for reproducible split (default: 2026)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Preview splits without writing the JSON file"
    )
    p.add_argument("--force", action="store_true", help="Overwrite an existing split file")
    args = p.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"Split file already exists: {output_path}")
        print("Use --force to overwrite or --dry-run to preview.")
        import sys
        import json as _json

        existing = _json.loads(output_path.read_text())
        print_summary(existing["meta"])
        return

    labels_dir = Path(args.labels)
    print(f"Loading labels from {labels_dir} …")
    raster_rows = load_labels(labels_dir)
    if not raster_rows:
        print("No labels found.")
        import sys

        sys.exit(1)
    print(
        f"  Found {len(raster_rows)} rasters  "
        f"{sum(len(v) for v in raster_rows.values()):,} labeled tiles"
    )

    print(f"Creating {args.test_pct:.1f}% stratified test split (seed={args.seed}) …")
    test_keys, meta = create_split(raster_rows, test_pct=args.test_pct, seed=args.seed)

    print_summary(meta)

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": [[r, ro, co] for r, ro, co in test_keys],
        "meta": meta,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(test_keys):,} test keys → {output_path}")
    print("\nNext step: python scripts/train_ensemble.py --test-split", args.output)


if __name__ == "__main__":
    main()
