#!/usr/bin/env python
"""Compare 3-run vs 9-run results and create summary."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_3 = pd.read_csv("analysis/multirun_3runs/aggregate_stats.csv")
results_9 = pd.read_csv("analysis/multirun9/aggregate_stats.csv")

# Create comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("3-Run vs 9-Run Training Comparison", fontsize=16, fontweight="bold")

metrics = ["mAP50", "mAP50-95", "precision", "recall"]
titles = ["mAP@50", "mAP@50-95", "Precision", "Recall"]

for ax, metric, title in zip(axes.flat, metrics, titles):
    metric_3 = results_3[results_3["metric"] == metric]
    metric_9 = results_9[results_9["metric"] == metric]

    mean_3 = metric_3["mean"].values[0]
    std_3 = metric_3["std"].values[0]
    mean_9 = metric_9["mean"].values[0]
    std_9 = metric_9["std"].values[0]

    x = [1, 2]
    means = [mean_3, mean_9]
    stds = [std_3, std_9]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=10,
        alpha=0.7,
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_ylabel(title, fontsize=12, fontweight="bold")
    ax.set_title(f"{title}", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["3 runs", "9 runs"])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

plt.tight_layout()
output_dir = Path("analysis")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "comparison_3runs_vs_9runs.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_dir / 'comparison_3runs_vs_9runs.png'}")

# Print summary
print("\n" + "=" * 70)
print("COMPARISON: 3-RUN vs 9-RUN TRAINING")
print("=" * 70)

print("\nmAP50:")
print(
    f"  3 runs:  {results_3[results_3['metric'] == 'mAP50']['mean'].values[0]:.4f} ± {results_3[results_3['metric'] == 'mAP50']['std'].values[0]:.4f}"
)
print(
    f"  9 runs:  {results_9[results_9['metric'] == 'mAP50']['mean'].values[0]:.4f} ± {results_9[results_9['metric'] == 'mAP50']['std'].values[0]:.4f}"
)

print("\nOverfitting:")
print(
    f"  3 runs:  {results_3[results_3['metric'] == 'overfitting']['mean'].values[0]:.2f}% ± {results_3[results_3['metric'] == 'overfitting']['std'].values[0]:.2f}%"
)
print(
    f"  9 runs:  {results_9[results_9['metric'] == 'overfitting']['mean'].values[0]:.2f}% ± {results_9[results_9['metric'] == 'overfitting']['std'].values[0]:.2f}%"
)

print("\nCoefficient of Variation (lower is better):")
cv_3 = (
    results_3[results_3["metric"] == "mAP50"]["std"].values[0]
    / results_3[results_3["metric"] == "mAP50"]["mean"].values[0]
    * 100
)
cv_9 = (
    results_9[results_9["metric"] == "mAP50"]["std"].values[0]
    / results_9[results_9["metric"] == "mAP50"]["mean"].values[0]
    * 100
)
print(f"  3 runs:  {cv_3:.2f}%")
print(f"  9 runs:  {cv_9:.2f}%")

print("\n" + "=" * 70)
print("CONCLUSIONS:")
print("=" * 70)
print("\n❌ High variability persists in both 3-run and 9-run experiments")
print(f"   - CV remains very high: {cv_9:.1f}% (target: <10%)")
print("\n❌ Severe overfitting in both experiments")
print(
    f"   - Mean overfitting: {results_9[results_9['metric'] == 'overfitting']['mean'].values[0]:.1f}% (target: <30%)"
)
print("\n⚠️  Enhanced regularization did not significantly improve results")
print("   - Similar performance despite dropout, weight decay, augmentation")
print("\n🎯 ROOT CAUSE: Dataset is too small (~33 train, 14 val samples)")
print("   - Model cannot learn robust features from limited data")
print("   - Random initialization dominates results (high CV)")
print("   - Easy memorization causes severe overfitting")
print("\n✅ SOLUTION: Expand dataset to 500+ samples")
print("   - This is the ONLY way to achieve:")
print("     • Stable results (CV < 10%)")
print("     • Good generalization (overfitting < 30%)")
print("     • Deployable model (mAP50 > 0.3)")
print("\n" + "=" * 70)
