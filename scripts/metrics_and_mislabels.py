#!/usr/bin/env python3
"""
scripts/metrics_and_mislabels.py

Compute metrics, plots and mislabel reports for a CDW ensemble.

Usage:
  python scripts/metrics_and_mislabels.py \
    --ensemble-meta path/to/ensemble_meta.pkl \
    --chm-dir path/to/chm_dir \
    [--labels-dir output/tile_labels] \
    [--test-split test_split.json] \
    [--output-dir output] \
    [--top-heatmaps 100] \
    [--top-mispreds 200]

The script:
- Reads all *_labels.csv from `labels_dir` (default: output/tile_labels), with last-row-wins dedup by raster,row_off,col_off
- Optionally splits train/test by a test-split JSON
- Loads ensemble with label_tiles.CNNPredictor().load_ensemble_meta(ensemble_meta_path)
- Predicts probabilities for every tile by reading 128x128 CHM tile (via rasterio Window)
- Computes metrics (ROC AUC, PR AUC, precision/recall/f1 at ensemble_thresh, confusion matrix, calibration curve (5 bins), class counts)
- Saves plots in `output_dir/metrics`
- Writes CSVs for training mismatches and top mispreds
- For top N training mispreds, computes heatmaps using label_tiles._compute_heatmap for methods IntGrad, HiResCAM, GradCAM+, RISE and saves individual images plus a montage per tile
- Prints a short summary and locations
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import os
import sys
import glob
from typing import List, Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from PIL import Image

# Import the project's label_tiles module (assumed present in src/ or installed)
try:
    import label_tiles
except Exception as e:
    print("ERROR: could not import label_tiles module:", e, file=sys.stderr)
    raise


# -------------------------
# Helpers
# -------------------------
def read_all_label_csvs(labels_dir: str) -> pd.DataFrame:
    patterns = os.path.join(labels_dir, "*_labels.csv")
    files = sorted(glob.glob(patterns))
    if not files:
        raise FileNotFoundError(f"No *_labels.csv files found in {labels_dir}")
    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
            df["_source_file"] = os.path.basename(p)
            dfs.append(df)
            print(f"Loaded {len(df):d} rows from {p}")
        except Exception:
            warnings.warn(f"Failed to read {p}; skipping", UserWarning)
    if not dfs:
        raise RuntimeError("No label CSVs could be read.")
    all_df = pd.concat(dfs, ignore_index=True)
    # Expect columns: raster, row_off, col_off, label (label might be 0/1)
    # Ensure needed columns exist
    for c in ("raster", "row_off", "col_off", "label"):
        if c not in all_df.columns:
            raise RuntimeError(f"Expected column '{c}' in label CSVs")
    # Ensure numeric offsets
    all_df["row_off"] = all_df["row_off"].astype(int)
    all_df["col_off"] = all_df["col_off"].astype(int)
    # Last-row-wins dedup: keep last occurrence per (raster,row_off,col_off)
    all_df["_order"] = np.arange(len(all_df))
    deduped = (
        all_df.sort_values("_order")
        .drop_duplicates(subset=["raster", "row_off", "col_off"], keep="last")
        .reset_index(drop=True)
    )
    return deduped


def parse_label_value(v) -> int:
    """Parse label to binary 0/1. Accepts ints, floats, and common strings."""
    if pd.isna(v):
        raise ValueError("NaN label")
    # already numeric
    try:
        iv = int(v)
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "cdw", "pos", "positive", "has_cdw"):
        return 1
    if s in ("0", "false", "f", "no", "n", "no_cdw", "neg", "negative", "none"):
        return 0
    # try to extract digit
    for token in s.replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            return int(token)
    raise ValueError(f"Unrecognized label value: {v}")


def make_id(row: pd.Series) -> str:
    return f"{row['raster']}|{int(row['row_off'])}|{int(row['col_off'])}"


def load_test_split(path: str) -> set:
    with open(path, "r") as fh:
        data = json.load(fh)
    # Accept many formats: list of id strings, dict with 'test' key, list of dicts with raster/row_off/col_off
    ids = set()
    if isinstance(data, list):
        # Determine element shape
        if all(isinstance(x, str) for x in data):
            ids = set(data)
        elif all(isinstance(x, dict) for x in data):
            for d in data:
                if {"raster", "row_off", "col_off"} <= set(d.keys()):
                    ids.add(f"{d['raster']}|{int(d['row_off'])}|{int(d['col_off'])}")
    elif isinstance(data, dict):
        # look for common keys
        for key in ("test", "test_tiles", "test_ids"):
            if key in data:
                return load_test_split_obj(data[key])
        # fallback: iterate values if list-like
        for v in data.values():
            if isinstance(v, list):
                return load_test_split_obj(v)
    return ids


def load_test_split_obj(obj) -> set:
    ids = set()
    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            ids.update(obj)
        elif all(isinstance(x, dict) for x in obj):
            for d in obj:
                if {"raster", "row_off", "col_off"} <= set(d.keys()):
                    ids.add(f"{d['raster']}|{int(d['row_off'])}|{int(d['col_off'])}")
    return ids


def read_chm_tile(raster_path: str, row_off: int, col_off: int, size: int = 128) -> np.ndarray:
    if not os.path.exists(raster_path):
        raise FileNotFoundError(raster_path)
    with rasterio.open(raster_path) as src:
        # Ensure reading single band
        # window: col_off, row_off, width, height
        win = Window(col_off, row_off, size, size)
        arr = src.read(1, window=win, boundless=True, fill_value=np.nan)
    # If arr is float with nans, fill with 0 for safety
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_plot(fig, path: str):
    ensure_dir(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confidence_from_prob(p: float) -> float:
    return abs(p - 0.5)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Metrics and mislabels for ensemble predictions.")
    parser.add_argument(
        "--labels-dir",
        default="output/tile_labels",
        help="Directory with *_labels.csv (default: output/tile_labels)",
    )
    parser.add_argument(
        "--chm-dir",
        required=True,
        help="Directory containing CHM rasters (path prefixes used in labels should resolve here)",
    )
    parser.add_argument(
        "--ensemble-meta",
        required=True,
        help="Path to ensemble_meta file to load via label_tiles.CNNPredictor().load_ensemble_meta()",
    )
    parser.add_argument(
        "--test-split", default=None, help="Optional JSON file specifying test split"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Base output dir (plots -> output_dir/metrics)"
    )
    parser.add_argument(
        "--top-heatmaps",
        type=int,
        default=100,
        help="Top-N mispredictions to compute heatmaps for (default 100)",
    )
    parser.add_argument(
        "--top-mispreds",
        type=int,
        default=200,
        help="Top-k training mispredictions to list in CSV (default 200)",
    )
    parser.add_argument(
        "--heatmap-methods",
        nargs="+",
        default=["IntGrad", "HiResCAM", "GradCAM+", "RISE"],
        help="Heatmap methods",
    )
    parser.add_argument("--tile-size", type=int, default=128, help="Tile size (default 128)")
    args = parser.parse_args()

    labels_dir = args.labels_dir
    chm_dir = args.chm_dir
    ensemble_meta_path = args.ensemble_meta
    test_split_path = args.test_split
    output_dir = args.output_dir
    metrics_dir = os.path.join(output_dir, "metrics")
    top_heatmaps = args.top_heatmaps
    top_mispreds_csv_k = args.top_mispreds
    heatmap_methods = args.heatmap_methods
    tile_size = args.tile_size

    ensure_dir(metrics_dir)

    print("Reading label CSVs...")
    labels_df = read_all_label_csvs(labels_dir)
    labels_df["id"] = labels_df.apply(make_id, axis=1)

    # Resolve raster paths: labels may contain basenames; try to resolve relative to chm_dir
    def resolve_raster_path(raster_field: str) -> str:
        if os.path.isabs(raster_field) and os.path.exists(raster_field):
            return raster_field
        # try relative to chm_dir
        candidate = os.path.join(chm_dir, os.path.basename(raster_field))
        if os.path.exists(candidate):
            return candidate
        # try as-is relative path
        if os.path.exists(raster_field):
            return raster_field
        # fallback: original value (might still work)
        return raster_field

    labels_df["_raster_path"] = labels_df["raster"].apply(resolve_raster_path)

    # Split train/test
    if test_split_path:
        print(f"Loading test split from {test_split_path} ...")
        test_ids = load_test_split(test_split_path)
        if not test_ids:
            warnings.warn(
                "Test split JSON parsed to empty set; proceeding with all data as train",
                UserWarning,
            )
            labels_df["split"] = "train"
        else:
            labels_df["split"] = labels_df["id"].apply(
                lambda i: "test" if i in test_ids else "train"
            )
    else:
        labels_df["split"] = "train"

    n_train = (labels_df["split"] == "train").sum()
    n_test = (labels_df["split"] == "test").sum()
    print(f"Tiles: total={len(labels_df)}, train={n_train}, test={n_test}")

    # Load ensemble
    print("Loading ensemble meta into CNNPredictor...")
    predictor = label_tiles.CNNPredictor()
    try:
        predictor.load_ensemble_meta(Path(ensemble_meta_path))
    except Exception as e:
        print("ERROR loading ensemble_meta:", e, file=sys.stderr)
        raise
    # Try to find ensemble_thresh
    ensemble_thresh = None
    if hasattr(predictor, "ensemble_thresh"):
        ensemble_thresh = getattr(predictor, "ensemble_thresh")
    elif getattr(predictor, "ensemble_meta", None):
        em = predictor.ensemble_meta
        if isinstance(em, dict) and "ensemble_thresh" in em:
            ensemble_thresh = em["ensemble_thresh"]
    if ensemble_thresh is None:
        print("Warning: could not find ensemble_thresh in predictor; defaulting to 0.5")
        ensemble_thresh = 0.5
    print(f"Using ensemble_thresh = {ensemble_thresh}")

    # For every tile, read raw CHM tile and predict probability
    probs = []
    labels = []
    ids = []
    splits = []
    raster_cols = []
    row_offs = []
    col_offs = []

    total = len(labels_df)
    print(f"Predicting probabilities for {total} tiles (this may take a while)...")
    for idx, row in labels_df.iterrows():
        sid = row["id"]
        raster = row["_raster_path"]
        ro = int(row["row_off"])
        co = int(row["col_off"])
        try:
            label_val = parse_label_value(row["label"])
        except Exception as e:
            warnings.warn(f"Skipping tile {sid} due to label parse error: {e}", UserWarning)
            continue
        try:
            raw_tile = read_chm_tile(raster, ro, co, size=tile_size)
        except Exception as e:
            warnings.warn(f"Skipping tile {sid} due to read error: {e}", UserWarning)
            continue
        try:
            # predictor.predict_proba_cdw should yield prob of positive class
            prob = predictor.predict_proba_cdw(raw_tile)
            # If returns array, flatten to scalar
            if isinstance(prob, (list, tuple, np.ndarray)):
                prob = float(np.asarray(prob).ravel()[0])
            prob = float(prob)
        except Exception as e:
            warnings.warn(f"Prediction failed for {sid}: {e}", UserWarning)
            continue
        probs.append(prob)
        labels.append(label_val)
        ids.append(sid)
        splits.append(row["split"])
        raster_cols.append(row["raster"])
        row_offs.append(ro)
        col_offs.append(co)
        if (len(probs) % 200) == 0:
            print(f"  Predicted {len(probs)}/{total} tiles...")

    if not probs:
        raise RuntimeError("No predictions were produced.")

    df_preds = pd.DataFrame(
        {
            "id": ids,
            "raster": raster_cols,
            "row_off": row_offs,
            "col_off": col_offs,
            "label": labels,
            "model_prob": probs,
            "split": splits,
        }
    )

    # Compute metrics for train and test separately
    def compute_and_save_metrics(sub_df: pd.DataFrame, split_name: str):
        y = np.array(sub_df["label"])
        p = np.array(sub_df["model_prob"])
        if len(np.unique(y)) == 1:
            print(
                f"Warning: only one class present in {split_name}; some metrics will be undefined."
            )
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y, p)
        except Exception:
            roc_auc = float("nan")
        # PR AUC (average precision)
        try:
            pr_auc = average_precision_score(y, p)
        except Exception:
            pr_auc = float("nan")
        # Precision/Recall/F1 at ensemble_thresh
        preds = (p >= ensemble_thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )
        cm = confusion_matrix(y, preds)
        # Calibration (reliability diagram)
        try:
            prob_true, prob_pred = calibration_curve(y, p, n_bins=5, strategy="uniform")
        except Exception:
            prob_true, prob_pred = np.array([]), np.array([])

        # Class counts
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(map(int, unique.tolist()), counts.tolist()))

        # Plots
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y, p)
            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve ({split_name})")
            plt.legend()
            roc_path = os.path.join(metrics_dir, f"roc_{split_name}.png")
            save_plot(fig, roc_path)
        except Exception as e:
            roc_path = None
            warnings.warn(f"Could not create ROC plot for {split_name}: {e}", UserWarning)
        # PR
        try:
            prec, rec, _ = precision_recall_curve(y, p)
            fig = plt.figure()
            plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve ({split_name})")
            plt.legend()
            pr_path = os.path.join(metrics_dir, f"pr_{split_name}.png")
            save_plot(fig, pr_path)
        except Exception as e:
            pr_path = None
            warnings.warn(f"Could not create PR plot for {split_name}: {e}", UserWarning)
        # Confusion matrix heatmap
        try:
            fig = plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix ({split_name})")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["neg", "pos"])
            plt.yticks(tick_marks, ["neg", "pos"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            cm_path = os.path.join(metrics_dir, f"confusion_{split_name}.png")
            save_plot(fig, cm_path)
        except Exception as e:
            cm_path = None
            warnings.warn(
                f"Could not create confusion matrix plot for {split_name}: {e}", UserWarning
            )
        # Reliability diagram
        try:
            fig = plt.figure()
            plt.plot(prob_pred, prob_true, marker="o", label="Reliability")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed frequency")
            plt.title(f"Reliability diagram ({split_name})")
            plt.legend()
            rel_path = os.path.join(metrics_dir, f"reliability_{split_name}.png")
            save_plot(fig, rel_path)
        except Exception as e:
            rel_path = None
            warnings.warn(
                f"Could not create reliability diagram for {split_name}: {e}", UserWarning
            )
        # Histogram of model_prob by true label
        try:
            fig = plt.figure()
            plt.hist(p[y == 0], bins=30, alpha=0.6, label="true=0")
            plt.hist(p[y == 1], bins=30, alpha=0.6, label="true=1")
            plt.xlabel("Model probability")
            plt.ylabel("Count")
            plt.legend()
            plt.title(f"Probability histogram by true label ({split_name})")
            hist_path = os.path.join(metrics_dir, f"hist_probs_{split_name}.png")
            save_plot(fig, hist_path)
        except Exception as e:
            hist_path = None
            warnings.warn(
                f"Could not create probability histogram for {split_name}: {e}", UserWarning
            )

        summary = {
            "split": split_name,
            "n": int(len(y)),
            "roc_auc": float(roc_auc) if not math.isnan(roc_auc) else None,
            "pr_auc": float(pr_auc) if not math.isnan(pr_auc) else None,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "class_counts": class_counts,
            "roc_path": roc_path,
            "pr_path": pr_path,
            "confusion_path": cm_path,
            "reliability_path": rel_path,
            "hist_path": hist_path,
        }
        return summary

    summaries = {}
    for split in ("train", "test"):
        sub = df_preds[df_preds["split"] == split]
        if len(sub) == 0:
            print(f"No {split} data present; skipping metrics for it.")
            continue
        print(f"Computing metrics and plots for {split} (n={len(sub)})...")
        summaries[split] = compute_and_save_metrics(sub, split)

    # Training mismatches and top mispreds
    train_df = df_preds[df_preds["split"] == "train"].copy()
    if len(train_df) == 0:
        print("No training data present; skipping misprediction reports.")
    else:
        train_df["predicted"] = (train_df["model_prob"] >= ensemble_thresh).astype(int)
        train_df["confidence"] = train_df["model_prob"].apply(confidence_from_prob)
        mismatches = train_df[train_df["predicted"] != train_df["label"]].copy()
        mismatches_csv = os.path.join(output_dir, "relabel_train_mismatches.csv")
        mismatches[
            ["raster", "row_off", "col_off", "label", "model_prob", "predicted", "confidence"]
        ].to_csv(mismatches_csv, index=False)
        print(f"Wrote training mismatches CSV: {mismatches_csv} ({len(mismatches)} rows)")

        # Top mispreds by confidence (descending), limit top_mispreds_csv_k
        top_mispreds_df = mismatches.sort_values("confidence", ascending=False).head(
            top_mispreds_csv_k
        )
        top_mispreds_csv = os.path.join(output_dir, "top_mispreds.csv")
        top_mispreds_df[
            ["raster", "row_off", "col_off", "label", "model_prob", "predicted", "confidence"]
        ].to_csv(top_mispreds_csv, index=False)
        print(f"Wrote top mispredictions CSV: {top_mispreds_csv} ({len(top_mispreds_df)} rows)")

        # For top N mispreds compute heatmaps
        heatmap_dir = os.path.join(metrics_dir, "heatmaps")
        ensure_dir(heatmap_dir)
        topN = min(top_heatmaps, len(top_mispreds_df))
        print(
            f"Computing heatmaps for top {topN} mispreds (methods: {', '.join(heatmap_methods)}) ..."
        )
        # iterate with index to create consistent filenames
        for i, (_, row) in enumerate(top_mispreds_df.head(topN).iterrows(), start=1):
            raster_path = resolve_raster_path(row["raster"])
            ro = int(row["row_off"])
            co = int(row["col_off"])
            idshort = f"{i:03d}"
            try:
                raw_tile = read_chm_tile(raster_path, ro, co, size=tile_size)
            except Exception as e:
                warnings.warn(
                    f"Skipping heatmaps for {row['id']} due to read error: {e}", UserWarning
                )
                continue
            method_imgs = []
            for method in heatmap_methods:
                try:
                    # label_tiles._compute_heatmap(predictor, raw_tile, method=method) expected
                    hm = label_tiles._compute_heatmap(predictor, raw_tile, method=method)
                    # hm should be a 2D array normalized to [0,1] or 0-255; convert to image
                    hma = np.asarray(hm, dtype=np.float32)
                    # normalize to 0-255
                    if hma.max() > 1.0:
                        # assume already 0-255
                        arr8 = np.clip(hma, 0, 255).astype(np.uint8)
                    else:
                        arr8 = (np.clip(hma, 0.0, 1.0) * 255).astype(np.uint8)
                    # create heatmap RGBA using matplotlib colormap
                    cmap = plt.get_cmap("jet")
                    colored = cmap(arr8 / 255.0)  # returns RGBA floats
                    img = (colored * 255).astype(np.uint8)
                    pil = Image.fromarray(img)
                    fname = os.path.join(
                        heatmap_dir, f"heatmap_{idshort}_{method.replace('+','plus')}.png"
                    )
                    pil.save(fname)
                    method_imgs.append((method, fname))
                except Exception as e:
                    warnings.warn(
                        f"Heatmap method {method} failed for {row['id']}: {e}", UserWarning
                    )
            # Create montage horizontally
            if method_imgs:
                imgs = [Image.open(p) for _, p in method_imgs]
                widths, heights = zip(*(im.size for im in imgs))
                total_w = sum(widths)
                max_h = max(heights)
                montage = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))
                x = 0
                for im in imgs:
                    montage.paste(im, (x, 0))
                    x += im.size[0]
                montage_path = os.path.join(heatmap_dir, f"heatmap_{idshort}_montage.png")
                montage.save(montage_path)
        print(f"Saved heatmaps and montages into {heatmap_dir}")

    # Print short summary and file locations
    print("\nSummary:")
    for split, s in summaries.items():
        print(f" - Metrics ({split}): n={s['n']}, ROC AUC={s['roc_auc']}, PR AUC={s['pr_auc']}")
        for key in ("roc_path", "pr_path", "confusion_path", "reliability_path", "hist_path"):
            if s.get(key):
                print(f"    {os.path.relpath(s[key])}")
    if len(train_df) > 0:
        print(f" - Training mismatches CSV: {os.path.relpath(mismatches_csv)}")
        print(f" - Top mispreds CSV: {os.path.relpath(top_mispreds_csv)}")
        print(f" - Heatmaps folder: {os.path.relpath(os.path.join(metrics_dir, 'heatmaps'))}")
    print(f" - All metric plots: {os.path.relpath(metrics_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()
