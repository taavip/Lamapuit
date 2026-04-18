#!/usr/bin/env python3
"""Compute precision/coverage statistics for a range of model_prob thresholds.

Usage:
  python3 scripts/threshold_coverage_table.py --labels-dir path/to/labels_curated_v2

Outputs a table of thresholds with:
- precision estimated on manual rows (TP / (TP+FP) among manual rows above threshold)
- manual and auto counts above threshold and percents
- manual recall (TP / total_manual_cd)
- manual negatives predicted positive (FP count)

This helps selecting an autolabel cutoff balancing precision vs number of autolabels.
"""

import argparse
import csv
import glob
import os
from math import nan


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
                prob_s = r.get("model_prob")
                try:
                    prob = float(prob_s) if prob_s not in (None, "") else None
                except Exception:
                    prob = None
                rows.append({
                    "label": (r.get("label") or "").strip().lower(),
                    "source": (r.get("source") or "").strip().lower(),
                    "model_prob": prob,
                })
    return rows


def compute_table(rows, thresholds):
    manual = [r for r in rows if r['source'] == 'manual']
    auto = [r for r in rows if r['source'] != 'manual']
    total_manual = len(manual)
    total_auto = len(auto)
    total_manual_cd = sum(1 for r in manual if r['label'] == 'cdw')

    table = []
    for thr in thresholds:
        man_above = [r for r in manual if (r['model_prob'] is not None and r['model_prob'] >= thr)]
        man_above_n = len(man_above)
        man_tp = sum(1 for r in man_above if r['label'] == 'cdw')
        man_fp = man_above_n - man_tp
        man_prec = (man_tp / man_above_n) if man_above_n else nan
        man_recall = (man_tp / total_manual_cd) if total_manual_cd else nan
        auto_above_n = sum(1 for r in auto if (r['model_prob'] is not None and r['model_prob'] >= thr))

        table.append({
            'thr': thr,
            'manual_above_n': man_above_n,
            'manual_above_pct': (100.0 * man_above_n / total_manual) if total_manual else nan,
            'manual_tp': man_tp,
            'manual_fp': man_fp,
            'manual_precision': man_prec,
            'manual_recall': man_recall,
            'auto_above_n': auto_above_n,
            'auto_above_pct': (100.0 * auto_above_n / total_auto) if total_auto else nan,
        })
    return table


def print_table(table):
    print("thr\tman_above\tman_%\tman_tp\tman_fp\tman_prec\tman_rec\tauto_above\tauto_%")
    for r in table:
        print(f"{r['thr']:.6f}\t{r['manual_above_n']}\t{r['manual_above_pct']:.2f}\t{r['manual_tp']}\t{r['manual_fp']}\t{(r['manual_precision'] if r['manual_precision']==r['manual_precision'] else 'nan'):.4f}\t{(r['manual_recall'] if r['manual_recall']==r['manual_recall'] else 'nan'):.4f}\t{r['auto_above_n']}\t{r['auto_above_pct']:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", default="output/model_search_v3_academic_leakage26/prepared/labels_curated_v2")
    args = p.parse_args()

    rows = load_rows(args.labels_dir)
    thresholds = [1.0, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    table = compute_table(rows, thresholds)
    print_table(table)


if __name__ == '__main__':
    main()
