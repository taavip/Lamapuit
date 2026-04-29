#!/usr/bin/env python3
"""Compute numeric metrics on the test split and print precision/recall/F1."""

import json
from pathlib import Path
import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

try:
    import label_tiles
except Exception as e:
    print("ERROR importing label_tiles:", e, file=sys.stderr)
    raise


def read_all_label_csvs(labels_dir: str) -> pd.DataFrame:
    import glob

    files = sorted(glob.glob(os.path.join(labels_dir, "*_labels.csv")))
    dfs = []
    for p in files:
        dfs.append(pd.read_csv(p))
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["row_off"] = all_df["row_off"].astype(int)
    all_df["col_off"] = all_df["col_off"].astype(int)
    # last-row-wins
    all_df["_order"] = range(len(all_df))
    deduped = (
        all_df.sort_values("_order")
        .drop_duplicates(subset=["raster", "row_off", "col_off"], keep="last")
        .reset_index(drop=True)
    )
    return deduped


def make_id(row):
    return f"{row['raster']}|{int(row['row_off'])}|{int(row['col_off'])}"


def parse_label_value(v):
    if pd.isna(v):
        raise ValueError("NaN")
    try:
        iv = int(v)
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "cdw", "pos"):
        return 1
    if s in ("0", "false", "no", "no_cdw", "neg"):
        return 0
    raise ValueError(f"Unrecognized label: {v}")


def read_chm_tile(raster_path, row_off, col_off, size=128):
    with rasterio.open(raster_path) as src:
        win = Window(col_off, row_off, size, size)
        arr = src.read(1, window=win, boundless=True, fill_value=np.nan)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


def main():
    labels_dir = "output/tile_labels"
    test_keys_json = "output/tile_labels/tmp_test_ids.json"
    ensemble_meta = "output/tile_labels/ensemble_meta.json"
    chm_dir = "chm_max_hag"

    labels = read_all_label_csvs(labels_dir)
    labels["id"] = labels.apply(make_id, axis=1)

    with open(test_keys_json, "r") as f:
        test_ids = set(json.load(f))
    test_df = labels[labels["id"].isin(test_ids)].copy()
    print(f"Test rows found: {len(test_df)}")
    if len(test_df) == 0:
        print("No test rows; aborting")
        return

    # resolve raster paths relative to chm_dir
    def resolve(r):
        if os.path.isabs(r) and os.path.exists(r):
            return r
        c = os.path.join(chm_dir, os.path.basename(r))
        if os.path.exists(c):
            return c
        if os.path.exists(r):
            return r
        return r

    test_df["_raster_path"] = test_df["raster"].apply(resolve)

    predictor = label_tiles.CNNPredictor()
    predictor.load_ensemble_meta(Path(ensemble_meta))
    ensemble_thresh = getattr(predictor, "ensemble_thresh", None)
    if ensemble_thresh is None:
        em = getattr(predictor, "ensemble_meta", {})
        if isinstance(em, dict) and "ensemble_thresh" in em:
            ensemble_thresh = em["ensemble_thresh"]
    if ensemble_thresh is None:
        ensemble_thresh = 0.5
    print("Using ensemble_thresh =", ensemble_thresh)

    y = []
    p = []
    for _, row in test_df.iterrows():
        try:
            yval = parse_label_value(row["label"])
        except Exception:
            continue
        tile = read_chm_tile(row["_raster_path"], int(row["row_off"]), int(row["col_off"]))
        prob = predictor.predict_proba_cdw(tile)
        if isinstance(prob, (list, tuple, np.ndarray)):
            prob = float(np.asarray(prob).ravel()[0])
        else:
            prob = float(prob)
        y.append(int(yval))
        p.append(float(prob))

    import sklearn.metrics as skm

    y = np.array(y)
    p = np.array(p)
    preds = (p >= ensemble_thresh).astype(int)
    roc_auc = skm.roc_auc_score(y, p)
    pr_auc = skm.average_precision_score(y, p)
    precision, recall, f1, _ = skm.precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    cm = skm.confusion_matrix(y, preds)
    print("Test metrics: n=", len(y))
    print(f" ROC AUC: {roc_auc:.6f}")
    print(f" PR  AUC: {pr_auc:.6f}")
    print(f" Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(" Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
