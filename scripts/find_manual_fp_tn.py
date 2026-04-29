#!/usr/bin/env python3
"""Find highest false-positive and lowest true-negative model_prob among manual rows.

Definitions used in this run:
- Manual rows: `source` == 'manual' (case-insensitive)
- "False positive" candidate: manual label == 'no_cdw' but model_prob is high (we report the highest model_prob among manual negatives).
- "True negative" candidate: manual label == 'no_cdw' and model_prob <= 0.5 (predicted negative). We report the lowest model_prob among these.

Outputs the top rows (raster, coords, model_prob, model_name, timestamp) for each case.
"""

import argparse
import csv
import glob
import os
import sys


def load_all_rows(labels_dir):
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
                    "raster": r.get("raster"),
                    "row_off": r.get("row_off"),
                    "col_off": r.get("col_off"),
                    "chunk_size": r.get("chunk_size"),
                    "label": (r.get("label") or "").strip().lower(),
                    "source": (r.get("source") or "").strip().lower(),
                    "model_name": r.get("model_name"),
                    "model_prob": prob,
                    "timestamp": r.get("timestamp"),
                })
    return rows


def pretty_row(r):
    return f"{r['raster']} @ ({r['row_off']},{r['col_off']}) prob={r['model_prob']:.6f} model={r.get('model_name') or ''}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", default="output/model_search_v3_academic_leakage26/prepared/labels_curated_v2")
    args = p.parse_args()

    rows = load_all_rows(args.labels_dir)
    manual = [r for r in rows if r['source'] == 'manual']
    if not manual:
        print("No manual rows found")
        return

    manual_no = [r for r in manual if r['label'] == 'no_cdw' and r['model_prob'] is not None]
    manual_cd = [r for r in manual if r['label'] == 'cdw' and r['model_prob'] is not None]

    print(f"Manual rows total={len(manual)}, manual no_cdw={len(manual_no)}, manual cdw={len(manual_cd)}")

    if manual_no:
        # highest model_prob among manual negatives (worst FP candidate)
        max_prob_no = max(r['model_prob'] for r in manual_no)
        max_rows = [r for r in manual_no if r['model_prob'] == max_prob_no]
        print('\nHighest false-positive candidate(s) (manual no_cdw, highest model_prob):')
        for r in max_rows[:10]:
            print('  -', pretty_row(r))
        # also show how many manual negatives would be predicted positive at 0.5
        fp_pred = [r for r in manual_no if r['model_prob'] >= 0.5]
        print(f"\nManual negatives with model_prob >= 0.5 (predicted positive) = {len(fp_pred)}")
        if fp_pred:
            max_fp_pred_prob = max(r['model_prob'] for r in fp_pred)
            max_fp_pred_rows = [r for r in fp_pred if r['model_prob'] == max_fp_pred_prob]
            print('Top predicted-positive manual negatives:')
            for r in max_fp_pred_rows[:10]:
                print('  -', pretty_row(r))
    else:
        print('\nNo manual no_cdw rows with numeric model_prob found')

    # true negatives (manual no_cdw with model_prob <= 0.5)
    tn_pred = [r for r in manual_no if r['model_prob'] <= 0.5]
    if tn_pred:
        min_tn_prob = min(r['model_prob'] for r in tn_pred)
        min_tn_rows = [r for r in tn_pred if r['model_prob'] == min_tn_prob]
        print('\nLowest true-negative candidate(s) (manual no_cdw, model_prob <= 0.5):')
        for r in min_tn_rows[:10]:
            print('  -', pretty_row(r))
    else:
        print('\nNo manual true-negatives (model_prob <= 0.5) found among manual negatives')


if __name__ == '__main__':
    main()
