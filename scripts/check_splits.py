"""Check integrity of splits created by `split_utils.spatial_cluster_splits`.

Usage:
  PYTHONPATH=. python3 scripts/check_splits.py --splits splits/splits_by_cluster.json --curated output/model_search_v4/prepared/labels_curated_v4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from scripts.split_utils import load_curated_rows


def key_to_tuple(k) -> tuple:
    return (k[0], int(k[1]), int(k[2]))


def load_splits(path: Path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--curated", required=True)
    args = p.parse_args()

    splits = load_splits(Path(args.splits))
    rows = load_curated_rows(Path(args.curated))

    test = {tuple(k) for k in [tuple(x) for x in splits.get("test_keys", [])]}
    val = {tuple(k) for k in [tuple(x) for x in splits.get("val_keys", [])]}
    train = {tuple(k) for k in [tuple(x) for x in splits.get("train_keys", [])]}
    manual = {tuple(k) for k in [tuple(x) for x in splits.get("manual_keys", [])]}
    pseudo = {tuple(k) for k in [tuple(x) for x in splits.get("pseudo_keys", [])]}

    issues = []
    if test & val:
        issues.append(f"Overlap test & val: {len(test & val)}")
    if test & train:
        issues.append(f"Overlap test & train: {len(test & train)}")
    if val & train:
        issues.append(f"Overlap val & train: {len(val & train)}")

    # manual keys should be subset of train
    if not manual <= train:
        issues.append(f"Some manual keys not in train: {len(manual - train)}")

    # Validate pseudo pool consistency
    row_map = { (r["raster"], r["row_off"], r["col_off"]) : r for r in rows }
    bad_pseudo = []
    for k in pseudo:
        r = row_map.get(k)
        if r is None:
            bad_pseudo.append((k, "missing"))
            continue
        mp = r.get("model_prob")
        if mp is None:
            bad_pseudo.append((k, "no_prob"))
            continue
        if r["label"] == "cdw" and mp < 0.95:
            bad_pseudo.append((k, f"cdw mp={mp:.3f}"))
        if r["label"] == "no_cdw" and mp > 0.05:
            bad_pseudo.append((k, f"no_cdw mp={mp:.3f}"))

    # Check hard examples ranges for train (non-manual)
    bad_train_hard = []
    hard_ranges = ((0.05, 0.30), (0.70, 0.95))
    for k in train:
        if k in manual:
            continue
        r = row_map.get(k)
        if r is None:
            bad_train_hard.append((k, "missing"))
            continue
        mp = r.get("model_prob")
        if mp is None:
            bad_train_hard.append((k, "no_prob"))
            continue
        ok = any(lo <= mp <= hi for (lo, hi) in hard_ranges)
        if not ok:
            bad_train_hard.append((k, f"mp={mp:.3f}"))

    print("Splits summary:")
    print("  test:", len(test))
    print("  val:", len(val))
    print("  train:", len(train))
    print("  manual:", len(manual))
    print("  pseudo:", len(pseudo))

    if issues:
        print("Issues:")
        for it in issues:
            print(" -", it)
    else:
        print("No overlap issues detected.")

    if bad_pseudo:
        print(f"Pseudo pool inconsistencies: {len(bad_pseudo)} examples (showing up to 10)")
        for b in bad_pseudo[:10]:
            print(" -", b)
    else:
        print("Pseudo pool OK")

    if bad_train_hard:
        print(f"Train hard-example inconsistencies: {len(bad_train_hard)} (showing up to 10)")
        for b in bad_train_hard[:10]:
            print(" -", b)
    else:
        print("Train hard-example set OK")

    if issues or bad_pseudo or bad_train_hard:
        raise SystemExit(2)
    print("All checks passed.")


if __name__ == "__main__":
    main()
