#!/usr/bin/env python
"""
Compare multiple training runs and generate comprehensive analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np

sns.set_style("whitegrid")


def load_results(run_path):
    """Load results.csv from a training run"""
    results_file = Path(run_path) / "results.csv"
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)
    df = df.rename(columns=lambda x: x.strip())  # Remove whitespace
    return df


def analyze_run(df, name):
    """Extract key metrics from a run"""
    if df is None or len(df) == 0:
        return None

    # Find best epoch
    if "metrics/mAP50(B)" in df.columns:
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_map50_box = df.loc[best_idx, "metrics/mAP50(B)"]
        best_map50_mask = df.loc[best_idx, "metrics/mAP50(M)"]
        best_map50_95_box = df.loc[best_idx, "metrics/mAP50-95(B)"]
        best_map50_95_mask = df.loc[best_idx, "metrics/mAP50-95(M)"]
    else:
        return None

    # Final metrics
    final_epoch = int(df["epoch"].iloc[-1])
    final_map50_box = df["metrics/mAP50(B)"].iloc[-1]
    final_map50_mask = df["metrics/mAP50(M)"].iloc[-1]

    # Overfitting metric (difference between best and final)
    overfitting_drop = best_map50_box - final_map50_box
    overfitting_pct = (overfitting_drop / best_map50_box * 100) if best_map50_box > 0 else 0

    return {
        "name": name,
        "total_epochs": final_epoch,
        "best_epoch": best_epoch,
        "best_box_mAP50": best_map50_box,
        "best_mask_mAP50": best_map50_mask,
        "best_box_mAP50-95": best_map50_95_box,
        "best_mask_mAP50-95": best_map50_95_mask,
        "final_box_mAP50": final_map50_box,
        "final_mask_mAP50": final_map50_mask,
        "overfitting_drop": overfitting_drop,
        "overfitting_pct": overfitting_pct,
    }


def plot_comparison(runs_data, output_dir):
    """Generate comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Best mAP50 comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["name"] for r in runs_data]
    box_map = [r["best_box_mAP50"] for r in runs_data]
    mask_map = [r["best_mask_mAP50"] for r in runs_data]

    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width / 2, box_map, width, label="Box", color="skyblue", edgecolor="black")
    ax1.bar(x + width / 2, mask_map, width, label="Mask", color="lightcoral", edgecolor="black")
    ax1.set_xlabel("Model/Run", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Best mAP50", fontsize=12, fontweight="bold")
    ax1.set_title("Best Performance Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Overfitting analysis
    overfitting = [r["overfitting_pct"] for r in runs_data]
    colors = ["green" if x < 20 else "orange" if x < 40 else "red" for x in overfitting]

    ax2.bar(x, overfitting, color=colors, edgecolor="black", alpha=0.7)
    ax2.axhline(y=20, color="orange", linestyle="--", label="Warning (20%)")
    ax2.axhline(y=40, color="red", linestyle="--", label="Severe (40%)")
    ax2.set_xlabel("Model/Run", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Overfitting Drop (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Overfitting Analysis (Best → Final)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_summary.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'comparison_summary.png'}")
    plt.close()

    # 3. Training curves comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    for run_info in runs_data:
        df = load_results(f"runs/segment/runs/cdw_detect/{run_info['name']}")
        if df is None:
            continue

        label = run_info["name"]

        # Box loss
        if "train/box_loss" in df.columns:
            ax1.plot(df["epoch"], df["train/box_loss"], label=f"{label} (train)", alpha=0.7)
            if "val/box_loss" in df.columns:
                ax1.plot(
                    df["epoch"],
                    df["val/box_loss"],
                    label=f"{label} (val)",
                    linestyle="--",
                    alpha=0.7,
                )

        # Box mAP50
        if "metrics/mAP50(B)" in df.columns:
            ax2.plot(df["epoch"], df["metrics/mAP50(B)"], label=label, linewidth=2)

        # Mask mAP50
        if "metrics/mAP50(M)" in df.columns:
            ax3.plot(df["epoch"], df["metrics/mAP50(M)"], label=label, linewidth=2)

        # Learning rate
        if "lr/pg0" in df.columns:
            ax4.plot(df["epoch"], df["lr/pg0"], label=label, alpha=0.7)

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Box Loss", fontweight="bold")
    ax1.set_title("Box Loss Over Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("mAP50 (Box)", fontweight="bold")
    ax2.set_title("Box Detection Performance", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.set_xlabel("Epoch", fontweight="bold")
    ax3.set_ylabel("mAP50 (Mask)", fontweight="bold")
    ax3.set_title("Segmentation Performance", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4.set_xlabel("Epoch", fontweight="bold")
    ax4.set_ylabel("Learning Rate", fontweight="bold")
    ax4.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'training_curves_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple training runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run names to compare (e.g., cdw_ultimate cdw_conservative)",
    )
    parser.add_argument("--output", default="analysis", help="Output directory for plots")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"TRAINING RUNS COMPARISON")
    print(f"{'='*70}\n")

    runs_data = []
    for run_name in args.runs:
        run_path = Path(f"runs/segment/runs/cdw_detect/{run_name}")
        if not run_path.exists():
            print(f"⚠ Warning: {run_path} not found, skipping...")
            continue

        df = load_results(run_path)
        analysis = analyze_run(df, run_name)

        if analysis:
            runs_data.append(analysis)
            print(f"✓ Loaded: {run_name}")

    if not runs_data:
        print("❌ No valid runs found!")
        return

    # Create summary table
    summary_df = pd.DataFrame(runs_data)
    summary_df = summary_df.sort_values("best_box_mAP50", ascending=False)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}\n")
    print(summary_df.to_string(index=False))

    # Find best run
    best_run = summary_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_run['name']}")
    print(f"{'='*70}")
    print(
        f"Best Box mAP50:    {best_run['best_box_mAP50']:.4f} @ epoch {int(best_run['best_epoch'])}"
    )
    print(f"Best Mask mAP50:   {best_run['best_mask_mAP50']:.4f}")
    print(f"Overfitting drop:  {best_run['overfitting_pct']:.1f}%")
    print(f"Total epochs:      {int(best_run['total_epochs'])}")
    print(f"{'='*70}\n")

    # Generate plots
    plot_comparison(runs_data, args.output)

    # Save summary to CSV
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    print(f"✓ Saved: {output_dir / 'comparison_summary.csv'}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"\nRECOMMENDATIONS:")

    # Generate recommendations
    if best_run["overfitting_pct"] > 40:
        print("  🔴 HIGH OVERFITTING: Need more training data or stronger regularization")
    elif best_run["overfitting_pct"] > 20:
        print("  🟡 MODERATE OVERFITTING: Consider data augmentation or early stopping")
    else:
        print("  🟢 GOOD GENERALIZATION: Model is well-regularized")

    if best_run["best_box_mAP50"] < 0.3:
        print("  📊 LOW PERFORMANCE: Dataset may be too small or task too difficult")
        print("     - Collect 5-10x more labeled data")
        print("     - Verify label quality and alignment")
        print("     - Consider simpler task definition")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
