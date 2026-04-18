#!/usr/bin/env python3
"""Tiny runner for the LAZ classifier module.

Uses a small sample from one LAZ file so you can sanity-check training quickly.
"""
from __future__ import annotations

from pathlib import Path
import sys


# Ensure local src/ package import when running from repository root.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cdw_detect.laz_classifier.cli import main  # noqa: E402


if __name__ == "__main__":
    # Equivalent to:
    # python -m cdw_detect.laz_classifier.cli train ...
    sys.argv = [
        "demo_laz_classifier_train.py",
        "train",
        "--laz",
        "data/lamapuit/laz/436646_2018_madal.laz",
        "--label-dim",
        "classification",
        "--exclude-labels",
        "0",
        "--max-points",
        "120000",
        "--out-dir",
        "runs/laz_classifier_demo_436646_2018",
        "--use-neighborhood-features",
        "--knn",
        "12",
        "--radius-m",
        "1.0",
    ]
    raise SystemExit(main())
