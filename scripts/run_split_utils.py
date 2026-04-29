"""CLI runner for `split_utils` to generate splits and save JSON outputs.

Usage:
    python3 scripts/run_split_utils.py --curated output/model_search_v4/prepared/labels_curated_v4 --out output/splits --test-frac 0.2 --val-frac 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.split_utils import load_curated_rows, spatial_cluster_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curated", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--cluster-size", type=int, default=3)
    p.add_argument("--guardband", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    curated = Path(args.curated)
    out = Path(args.out)

    rows = load_curated_rows(curated)
    if not rows:
        print("No curated rows found in", curated)
        return

    splits = spatial_cluster_splits(
        rows,
        out_dir=out,
        test_fraction=args.test_frac,
        val_fraction=args.val_frac,
        cluster_size=args.cluster_size,
        guardband=args.guardband,
        seed=args.seed,
    )
    print("Wrote splits to", out / "splits_by_cluster.json")


if __name__ == "__main__":
    main()
