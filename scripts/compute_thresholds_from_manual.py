#!/usr/bin/env python3
"""Compute model_prob thresholds that achieve given precision on manual `cdw` labels.

This variant uses only the Python standard library (no pandas/numpy) so it can
run in a minimal environment.

Usage:
  python scripts/compute_thresholds_from_manual.py --labels-dir path/to/labels_curated_v2
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import OrderedDict


def load_rows(labels_dir):
    pattern = os.path.join(labels_dir, "*_labels.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no label files found in {labels_dir}")

    rows = []
    for fn in files:
        with open(fn, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                label = r.get("label")
                source = r.get("source")
                prob_s = r.get("model_prob")
                try:
                    prob = float(prob_s) if prob_s not in (None, "") else None
                except Exception:
                    prob = None
                rows.append({"label": label, "source": source, "model_prob": prob})
    return rows


def find_thresholds_from_manual(manual_rows, targets=(1.0, 0.995, 0.99, 0.95)):
    probs = sorted({r["model_prob"] for r in manual_rows if r["model_prob"] is not None}, reverse=True)
    if not probs:
        return {t: None for t in targets}

    results = OrderedDict()
    for t in targets:
        found = None
        for v in probs:
            # count manual rows with prob >= v
            denom = sum(1 for r in manual_rows if (r["model_prob"] is not None and r["model_prob"] >= v))
            if denom == 0:
                continue
            tp = sum(1 for r in manual_rows if (r["model_prob"] is not None and r["model_prob"] >= v and r["label"] == "cdw"))
            prec = tp / denom
            if prec + 1e-12 >= t:
                found = float(v)
                break
        results[t] = found
    return results


def report(rows, thresholds):
    manual = [r for r in rows if r.get("source") == "manual"]
    auto = [r for r in rows if r.get("source") != "manual"]

    total_manual = len(manual)
    total_auto = len(auto)
    total_all = len(rows)

    print(f"Loaded rows: total={total_all}, manual={total_manual}, auto={total_auto}")
    print("")

    for target, thr in thresholds.items():
        print(f"Target precision: {target*100:.3f}%")
        if thr is None:
            print("  No threshold found achieving this precision on manual rows")
            print("")
            continue

        man_above = sum(1 for r in manual if (r.get("model_prob") is not None and r["model_prob"] >= thr))
        man_above_pct = 100.0 * man_above / total_manual if total_manual else float("nan")
        man_tp = sum(1 for r in manual if (r.get("model_prob") is not None and r["model_prob"] >= thr and r["label"] == "cdw"))
        man_precision = (man_tp / man_above) if man_above else float("nan")

        auto_above = sum(1 for r in auto if (r.get("model_prob") is not None and r["model_prob"] >= thr))
        auto_above_pct = 100.0 * auto_above / total_auto if total_auto else float("nan")

        print(f"  Threshold (model_prob >=): {thr:.6f}")
        print(f"  Manual: above={man_above} ({man_above_pct:.2f}%), below={total_manual-man_above} ({100-man_above_pct:.2f}%)")
        print(f"    Manual positives among above (TP) = {man_tp}")
        print(f"    Precision on manual above = {man_precision:.4f}")
        print(f"  Auto:   above={auto_above} ({auto_above_pct:.2f}%), below={total_auto-auto_above} ({100-auto_above_pct:.2f}%)")
        print("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", default="output/model_search_v3_academic_leakage26/prepared/labels_curated_v2", help="Directory with *_labels.csv files")
    p.add_argument("--targets", nargs="*", type=float, default=[1.0, 0.995, 0.99, 0.95], help="Precision targets (e.g. 1.0 0.995 0.99 0.95)")
    args = p.parse_args()

    try:
        rows = load_rows(args.labels_dir)
    except Exception as e:
        print(f"Error loading labels: {e}")
        sys.exit(2)

    manual_rows = [r for r in rows if r.get("source") == "manual"]
    thresholds = find_thresholds_from_manual(manual_rows, targets=tuple(args.targets))
    report(rows, thresholds)


if __name__ == "__main__":
    main()
