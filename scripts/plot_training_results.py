#!/usr/bin/env python
"""
Plot training results from YOLO training runs.

Usage:
    python scripts/plot_training_results.py --run runs/cdw_detect/cdw_4hour_enhanced
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10


def plot_training_results(run_path: str, output_dir: str = None):
    """
    Plot training results from a YOLO run directory.

    Args:
        run_path: Path to the training run directory
        output_dir: Optional output directory for plots (default: run_path/plots)
    """
    run_path = Path(run_path)
    results_file = run_path / "results.csv"

    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Make sure training has started and generated results.")
        return

    # Load results
    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names

    # Create output directory
    if output_dir is None:
        output_dir = run_path / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Available metrics: {list(df.columns)}")
    print(f"\nTotal epochs completed: {len(df)}")

    # Plot 1: Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Loss Curves", fontsize=16, fontweight="bold")

    loss_metrics = ["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"]
    titles = ["Box Loss", "Segmentation Loss", "Classification Loss", "DFL Loss"]

    for ax, metric, title in zip(axes.flat, loss_metrics, titles):
        if metric in df.columns:
            ax.plot(df["epoch"], df[metric], linewidth=2, label="Train")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'loss_curves.png'}")
    plt.close()

    # Plot 2: Validation Metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Validation Metrics", fontsize=16, fontweight="bold")

    val_metrics = [
        ("metrics/precision(B)", "Box Precision"),
        ("metrics/recall(B)", "Box Recall"),
        ("metrics/mAP50(B)", "Box mAP50"),
        ("metrics/mAP50-95(B)", "Box mAP50-95"),
    ]

    for ax, (metric, title) in zip(axes.flat, val_metrics):
        if metric in df.columns:
            ax.plot(
                df["epoch"],
                df[metric],
                linewidth=2,
                color="green",
                marker="o",
                markersize=3,
                label="Validation",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "validation_metrics.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'validation_metrics.png'}")
    plt.close()

    # Plot 3: Segmentation Metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Segmentation Metrics", fontsize=16, fontweight="bold")

    seg_metrics = [
        ("metrics/precision(M)", "Mask Precision"),
        ("metrics/recall(M)", "Mask Recall"),
        ("metrics/mAP50(M)", "Mask mAP50"),
        ("metrics/mAP50-95(M)", "Mask mAP50-95"),
    ]

    for ax, (metric, title) in zip(axes.flat, seg_metrics):
        if metric in df.columns:
            ax.plot(
                df["epoch"],
                df[metric],
                linewidth=2,
                color="purple",
                marker="s",
                markersize=3,
                label="Segmentation",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_metrics.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'segmentation_metrics.png'}")
    plt.close()

    # Plot 4: Learning Rate
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if "lr/pg0" in df.columns:
        ax.plot(df["epoch"], df["lr/pg0"], linewidth=2, color="red", label="Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(output_dir / "learning_rate.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_dir / 'learning_rate.png'}")
        plt.close()

    # Plot 5: Combined Overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Overview", fontsize=16, fontweight="bold")

    # Total loss
    if "train/box_loss" in df.columns and "train/seg_loss" in df.columns:
        total_loss = (
            df["train/box_loss"]
            + df["train/seg_loss"]
            + df["train/cls_loss"]
            + df["train/dfl_loss"]
        )
        axes[0, 0].plot(df["epoch"], total_loss, linewidth=2, color="blue")
        axes[0, 0].set_title("Total Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)

    # Box mAP50
    if "metrics/mAP50(B)" in df.columns:
        axes[0, 1].plot(
            df["epoch"],
            df["metrics/mAP50(B)"],
            linewidth=2,
            color="green",
            marker="o",
            markersize=3,
        )
        axes[0, 1].set_title("Box Detection mAP50")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mAP50")
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)

    # Mask mAP50
    if "metrics/mAP50(M)" in df.columns:
        axes[0, 2].plot(
            df["epoch"],
            df["metrics/mAP50(M)"],
            linewidth=2,
            color="purple",
            marker="s",
            markersize=3,
        )
        axes[0, 2].set_title("Segmentation mAP50")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("mAP50")
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(True, alpha=0.3)

    # Precision/Recall (Box)
    if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
        axes[1, 0].plot(
            df["epoch"],
            df["metrics/precision(B)"],
            linewidth=2,
            label="Precision",
            marker="o",
            markersize=3,
        )
        axes[1, 0].plot(
            df["epoch"],
            df["metrics/recall(B)"],
            linewidth=2,
            label="Recall",
            marker="s",
            markersize=3,
        )
        axes[1, 0].set_title("Box Precision & Recall")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Precision/Recall (Mask)
    if "metrics/precision(M)" in df.columns and "metrics/recall(M)" in df.columns:
        axes[1, 1].plot(
            df["epoch"],
            df["metrics/precision(M)"],
            linewidth=2,
            label="Precision",
            marker="o",
            markersize=3,
        )
        axes[1, 1].plot(
            df["epoch"],
            df["metrics/recall(M)"],
            linewidth=2,
            label="Recall",
            marker="s",
            markersize=3,
        )
        axes[1, 1].set_title("Mask Precision & Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Best metrics summary
    axes[1, 2].axis("off")
    summary_text = "Best Metrics:\n\n"

    if "metrics/mAP50(B)" in df.columns:
        best_map50_b = df["metrics/mAP50(B)"].max()
        best_epoch_b = df["metrics/mAP50(B)"].idxmax()
        summary_text += f"Box mAP50: {best_map50_b:.4f} @ epoch {best_epoch_b}\n"

    if "metrics/mAP50-95(B)" in df.columns:
        best_map_b = df["metrics/mAP50-95(B)"].max()
        summary_text += f"Box mAP50-95: {best_map_b:.4f}\n\n"

    if "metrics/mAP50(M)" in df.columns:
        best_map50_m = df["metrics/mAP50(M)"].max()
        best_epoch_m = df["metrics/mAP50(M)"].idxmax()
        summary_text += f"Mask mAP50: {best_map50_m:.4f} @ epoch {best_epoch_m}\n"

    if "metrics/mAP50-95(M)" in df.columns:
        best_map_m = df["metrics/mAP50-95(M)"].max()
        summary_text += f"Mask mAP50-95: {best_map_m:.4f}\n\n"

    summary_text += f"Total epochs: {len(df)}"

    axes[1, 2].text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "training_overview.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'training_overview.png'}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total epochs completed: {len(df)}")

    if "metrics/mAP50(B)" in df.columns:
        print(f"\nBox Detection:")
        print(
            f"  Best mAP50: {df['metrics/mAP50(B)'].max():.4f} @ epoch {df['metrics/mAP50(B)'].idxmax()}"
        )
        print(f"  Final mAP50: {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
        if "metrics/mAP50-95(B)" in df.columns:
            print(f"  Best mAP50-95: {df['metrics/mAP50-95(B)'].max():.4f}")

    if "metrics/mAP50(M)" in df.columns:
        print(f"\nSegmentation:")
        print(
            f"  Best mAP50: {df['metrics/mAP50(M)'].max():.4f} @ epoch {df['metrics/mAP50(M)'].idxmax()}"
        )
        print(f"  Final mAP50: {df['metrics/mAP50(M)'].iloc[-1]:.4f}")
        if "metrics/mAP50-95(M)" in df.columns:
            print(f"  Best mAP50-95: {df['metrics/mAP50-95(M)'].max():.4f}")

    print("=" * 70)
    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot YOLO training results")
    parser.add_argument("--run", required=True, help="Path to training run directory")
    parser.add_argument("--output", help="Output directory for plots (default: run_path/plots)")

    args = parser.parse_args()

    plot_training_results(args.run, args.output)


if __name__ == "__main__":
    main()
