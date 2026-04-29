#!/usr/bin/env python3
"""
Analyze results from comprehensive experiments.

Compares all experiments and identifies the best configuration.
"""

import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(results_dir="runs/segment/runs/experiments"):
    """Load results from all experiment runs."""
    results = []

    results_path = Path(results_dir)
    for exp_dir in results_path.glob("exp*"):
        # Load results.csv if it exists
        results_csv = exp_dir / "results.csv"
        if not results_csv.exists():
            continue

        df = pd.read_csv(results_csv)
        df = df.rename(columns=lambda x: x.strip())

        # Get best epoch
        if "metrics/mAP50(B)" in df.columns:
            best_idx = df["metrics/mAP50(B)"].idxmax()

            results.append(
                {
                    "name": exp_dir.name,
                    "best_epoch": int(df.loc[best_idx, "epoch"]),
                    "train_loss": float(df.loc[best_idx, "train/seg_loss"]),
                    "val_loss": float(df.loc[best_idx, "val/seg_loss"]),
                    "mAP50": float(df.loc[best_idx, "metrics/mAP50(B)"]),
                    "mAP50_95": float(df.loc[best_idx, "metrics/mAP50-95(B)"]),
                    "precision": float(df.loc[best_idx, "metrics/precision(B)"]),
                    "recall": float(df.loc[best_idx, "metrics/recall(B)"]),
                    "overfitting": (
                        float(df.loc[best_idx, "val/seg_loss"])
                        - float(df.loc[best_idx, "train/seg_loss"])
                    )
                    / float(df.loc[best_idx, "train/seg_loss"])
                    * 100,
                }
            )

    return pd.DataFrame(results)


def analyze_experiments():
    """Analyze and visualize experiment results."""

    print("\n" + "=" * 80)
    print("EXPERIMENT ANALYSIS")
    print("=" * 80)

    # Load results
    df = load_experiment_results()

    if len(df) == 0:
        print("❌ No experiment results found in runs/experiments/")
        print("Run: python scripts/run_experiments.py first")
        return

    print(f"\nLoaded {len(df)} experiments")

    # Sort by mAP50
    df = df.sort_values("mAP50", ascending=False)

    # Print summary table
    print("\n" + "-" * 80)
    print("RESULTS SUMMARY (sorted by mAP50)")
    print("-" * 80)
    print(df.to_string(index=False))

    # Best model
    best = df.iloc[0]
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL")
    print("=" * 80)
    print(f"Experiment: {best['name']}")
    print(f"Best Epoch: {best['best_epoch']}")
    print(f"mAP50: {best['mAP50']:.4f}")
    print(f"mAP50-95: {best['mAP50_95']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"Overfitting: {best['overfitting']:.1f}%")
    print("=" * 80)

    # Save summary
    df.to_csv("experiment_analysis.csv", index=False)
    print(f"\n✓ Saved detailed analysis to: experiment_analysis.csv")

    # Create visualizations
    _create_visualizations(df)

    # Recommendations
    _print_recommendations(df)


def _create_visualizations(df):
    """Create comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: mAP50 comparison
    ax = axes[0, 0]
    df_sorted = df.sort_values("mAP50")
    ax.barh(df_sorted["name"], df_sorted["mAP50"])
    ax.set_xlabel("mAP50")
    ax.set_title("Model Performance (mAP50)")
    ax.grid(axis="x", alpha=0.3)

    # Plot 2: Precision vs Recall
    ax = axes[0, 1]
    ax.scatter(df["recall"], df["precision"], s=100, alpha=0.6)
    for _, row in df.iterrows():
        ax.annotate(row["name"], (row["recall"], row["precision"]), fontsize=8, alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.grid(alpha=0.3)

    # Plot 3: Overfitting analysis
    ax = axes[1, 0]
    df_sorted = df.sort_values("overfitting")
    colors = [
        "green" if x < 20 else "orange" if x < 50 else "red" for x in df_sorted["overfitting"]
    ]
    ax.barh(df_sorted["name"], df_sorted["overfitting"], color=colors)
    ax.set_xlabel("Overfitting %")
    ax.set_title("Overfitting Analysis")
    ax.axvline(x=20, color="green", linestyle="--", alpha=0.5, label="Good (<20%)")
    ax.axvline(x=50, color="orange", linestyle="--", alpha=0.5, label="Moderate (20-50%)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Plot 4: mAP50-95 comparison
    ax = axes[1, 1]
    df_sorted = df.sort_values("mAP50_95")
    ax.barh(df_sorted["name"], df_sorted["mAP50_95"])
    ax.set_xlabel("mAP50-95")
    ax.set_title("Strict Performance (mAP50-95)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved visualization to: experiment_comparison.png")
    plt.close()


def _print_recommendations(df):
    """Print recommendations based on analysis."""

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best = df.iloc[0]

    # Check if overfitting is still an issue
    if best["overfitting"] > 50:
        print("\n⚠️  HIGH OVERFITTING DETECTED (>50%)")
        print("Recommendations:")
        print("  1. Collect more training data (critical!)")
        print("  2. Increase regularization (weight_decay, dropout)")
        print("  3. Use stronger augmentation")
        print("  4. Consider smaller model (if using large)")
    elif best["overfitting"] > 20:
        print("\n⚠️  MODERATE OVERFITTING (20-50%)")
        print("Recommendations:")
        print("  1. Slightly increase regularization")
        print("  2. Add more data if possible")
    else:
        print("\n✓ GOOD GENERALIZATION (<20% overfitting)")

    # Check if mAP50 is acceptable
    if best["mAP50"] < 0.3:
        print("\n⚠️  LOW mAP50 (<0.3)")
        print("Recommendations:")
        print("  1. Verify data quality and labels")
        print("  2. Try larger model")
        print("  3. Increase training epochs")
        print("  4. Tune learning rate")
    elif best["mAP50"] < 0.5:
        print("\n📊 MODERATE mAP50 (0.3-0.5)")
        print("Room for improvement - consider:")
        print("  1. Multi-scale detection")
        print("  2. Ensemble methods")
        print("  3. Fine-tuning hyperparameters")
    else:
        print("\n✓ GOOD mAP50 (>0.5)")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(f"1. Run best config ({best['name']}) with multiple seeds:")
    print(f"   python scripts/train_multirun.py --name {best['name']}_multirun")
    print("\n2. Evaluate on test set:")
    print("   python scripts/evaluate_testset.py")
    print("\n3. If satisfied, create production model:")
    print("   python scripts/export_production_model.py")
    print("=" * 80)


if __name__ == "__main__":
    analyze_experiments()
