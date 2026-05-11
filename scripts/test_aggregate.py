#!/usr/bin/env python
"""Test script to verify the aggregate_results function works with existing data."""

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_results_csv(run_path):
    """Parse results.csv from a training run."""
    results_file = Path(run_path) / "results.csv"
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)
    df.columns = df.columns.str.strip()
    return df


def get_best_metrics(df):
    """Extract best validation metrics from results dataframe."""
    if df is None or df.empty:
        return None

    best_epoch_idx = df["metrics/mAP50(B)"].idxmax()
    best_epoch = df.loc[best_epoch_idx, "epoch"]

    metrics = {
        "epoch": int(best_epoch),
        "mAP50": float(df.loc[best_epoch_idx, "metrics/mAP50(B)"]),
        "mAP50-95": float(df.loc[best_epoch_idx, "metrics/mAP50-95(B)"]),
        "precision": float(df.loc[best_epoch_idx, "metrics/precision(B)"]),
        "recall": float(df.loc[best_epoch_idx, "metrics/recall(B)"]),
    }

    final_idx = df.index[-1]
    metrics["final_mAP50"] = float(df.loc[final_idx, "metrics/mAP50(B)"])
    metrics["overfitting"] = (metrics["mAP50"] - metrics["final_mAP50"]) / metrics["mAP50"] * 100

    return metrics


# Test with the existing runs
run_paths = [
    Path("runs/segment/runs/multirun/cdw_multirun_run1_seed7271"),
    Path("runs/segment/runs/multirun/cdw_multirun_run2_seed861"),
    Path("runs/segment/runs/multirun/cdw_multirun_run3_seed5391"),
]

print("Testing aggregate_results with existing data...\n")

all_metrics = []
for i, run_path in enumerate(run_paths):
    print(f"Processing run {i+1}: {run_path}")
    if not run_path.exists():
        print(f"  ⚠ Path does not exist: {run_path}")
        continue
    df = parse_results_csv(run_path)
    if df is None:
        print(f"  ⚠ No results.csv found")
        continue
    metrics = get_best_metrics(df)
    if metrics:
        metrics["run"] = i + 1
        metrics["run_name"] = run_path.name
        all_metrics.append(metrics)
        print(f"  ✓ mAP50={metrics['mAP50']:.4f}, Overfitting={metrics['overfitting']:.1f}%")

if all_metrics:
    results_df = pd.DataFrame(all_metrics)
    print(f"\n✓ Successfully processed {len(all_metrics)} runs!")
    print(f"\nResults Summary:")
    print(results_df[["run", "mAP50", "mAP50-95", "precision", "recall", "overfitting"]])

    print(f"\nMean Performance:")
    print(f"  mAP50:      {results_df['mAP50'].mean():.4f} ± {results_df['mAP50'].std():.4f}")
    print(
        f"  Overfitting: {results_df['overfitting'].mean():.1f}% ± {results_df['overfitting'].std():.1f}%"
    )
else:
    print("\n✗ No valid results found!")
