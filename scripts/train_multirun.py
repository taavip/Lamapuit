#!/usr/bin/env python
"""
Multi-run training script to assess model variability and robustness.
Runs the best model configuration multiple times with different random seeds.

Best practices:
- Multiple runs (3-5) with different seeds
- Aggregate metrics (mean, std, min, max)
- Generate comparison plots and confidence intervals
- Report robust results for publication/deployment
"""

import argparse
import time
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_results_csv(run_path):
    """Parse results.csv from a training run."""
    results_file = Path(run_path) / "results.csv"
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    return df


def get_best_metrics(df):
    """Extract best validation metrics from results dataframe."""
    if df is None or df.empty:
        return None

    # Find epoch with best mAP50
    best_epoch_idx = df["metrics/mAP50(B)"].idxmax()
    best_epoch = df.loc[best_epoch_idx, "epoch"]

    metrics = {
        "epoch": int(best_epoch),
        "mAP50": float(df.loc[best_epoch_idx, "metrics/mAP50(B)"]),
        "mAP50-95": float(df.loc[best_epoch_idx, "metrics/mAP50-95(B)"]),
        "precision": float(df.loc[best_epoch_idx, "metrics/precision(B)"]),
        "recall": float(df.loc[best_epoch_idx, "metrics/recall(B)"]),
    }

    # Get final epoch metrics (for overfitting analysis)
    final_idx = df.index[-1]
    metrics["final_mAP50"] = float(df.loc[final_idx, "metrics/mAP50(B)"])
    metrics["overfitting"] = (metrics["mAP50"] - metrics["final_mAP50"]) / metrics["mAP50"] * 100

    return metrics


def train_single_run(args, run_idx, seed):
    """Train a single model with specified seed."""
    from ultralytics import YOLO
    import gc

    print(f"\n{'='*70}")
    print(f"RUN {run_idx + 1}/{args.num_runs} - SEED: {seed}")
    print(f"{'='*70}\n")

    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    yolo = YOLO(args.model)

    run_name = f"{args.name}_run{run_idx + 1}_seed{seed}"

    # Train with best configuration (from cdw_4hour_enhanced - best generalization)
    results = yolo.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        name=run_name,
        project="runs/multirun",
        imgsz=640,
        # Moderate augmentation (best balance from previous analysis)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,  # Increased for better regularization
        copy_paste=0.15,  # Increased for better regularization
        erasing=0.4,  # Added random erasing for regularization
        # Training settings
        patience=args.patience,
        save_period=20,
        cache="disk",
        workers=8,
        cos_lr=True,
        close_mosaic=15,  # Close mosaic later for better regularization
        # Enhanced Regularization
        dropout=0.1,  # Added dropout for regularization
        weight_decay=0.001,  # Increased weight decay
        # Other settings
        exist_ok=True,
        verbose=False,  # Reduce clutter for multi-run
        amp=True if args.device != "cpu" else False,
        seed=seed,  # Set YOLO seed
        deterministic=True,  # Make training deterministic where possible
        # Validation
        val=True,
        plots=True,
    )

    run_path = Path("runs/segment/multirun") / run_name
    return run_path


def aggregate_results(run_paths, output_dir):
    """Aggregate metrics from multiple runs."""
    all_metrics = []

    for i, run_path in enumerate(run_paths):
        print(f"Processing run {i+1}: {run_path}")
        if not run_path.exists():
            print(f"  ⚠ Path does not exist: {run_path}")
            continue
        df = parse_results_csv(run_path)
        if df is None:
            print(f"  ⚠ No results.csv found in: {run_path}")
            continue
        metrics = get_best_metrics(df)
        if metrics:
            metrics["run"] = i + 1
            metrics["run_name"] = run_path.name
            all_metrics.append(metrics)
            print(f"  ✓ Metrics extracted: mAP50={metrics['mAP50']:.4f}")
        else:
            print(f"  ⚠ Could not extract metrics from results")

    if not all_metrics:
        print("\n⚠ No valid results found!")
        print("\nChecking for results in alternative locations...")
        # Try alternate path
        for i, run_path in enumerate(run_paths):
            alt_path = Path("runs/segment/runs/segment/multirun") / run_path.name
            if alt_path.exists():
                print(f"  Found: {alt_path}")
                df = parse_results_csv(alt_path)
                if df is not None:
                    metrics = get_best_metrics(df)
                    if metrics:
                        metrics["run"] = i + 1
                        metrics["run_name"] = alt_path.name
                        all_metrics.append(metrics)

        if not all_metrics:
            print("\n❌ Still no valid results. Training may have failed.")
            return None

    results_df = pd.DataFrame(all_metrics)

    # Calculate aggregate statistics
    stats = {
        "metric": ["mAP50", "mAP50-95", "precision", "recall", "overfitting"],
        "mean": [
            results_df["mAP50"].mean(),
            results_df["mAP50-95"].mean(),
            results_df["precision"].mean(),
            results_df["recall"].mean(),
            results_df["overfitting"].mean(),
        ],
        "std": [
            results_df["mAP50"].std(),
            results_df["mAP50-95"].std(),
            results_df["precision"].std(),
            results_df["recall"].std(),
            results_df["overfitting"].std(),
        ],
        "min": [
            results_df["mAP50"].min(),
            results_df["mAP50-95"].min(),
            results_df["precision"].min(),
            results_df["recall"].min(),
            results_df["overfitting"].min(),
        ],
        "max": [
            results_df["mAP50"].max(),
            results_df["mAP50-95"].max(),
            results_df["precision"].max(),
            results_df["recall"].max(),
            results_df["overfitting"].max(),
        ],
    }

    stats_df = pd.DataFrame(stats)

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "individual_runs.csv", index=False)
    stats_df.to_csv(output_dir / "aggregate_stats.csv", index=False)

    return results_df, stats_df


def plot_multirun_results(results_df, stats_df, output_dir):
    """Generate visualization plots for multi-run results."""
    output_dir = Path(output_dir)

    # 1. Box plots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Run Performance Distribution", fontsize=16, fontweight="bold")

    metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    titles = ["mAP@50", "mAP@50-95", "Precision", "Recall"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        data = results_df[metric].values
        bp = ax.boxplot(
            [data],
            labels=[""],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )

        # Add individual points
        x = np.random.normal(1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=100, c="darkblue", edgecolors="black")

        # Add mean line
        mean_val = data.mean()
        ax.axhline(
            mean_val, color="green", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.4f}"
        )

        ax.set_ylabel(title, fontsize=12)
        ax.set_title(
            f"{title}\n(μ={data.mean():.4f}, σ={data.std():.4f})", fontsize=11, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "multirun_boxplots.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'multirun_boxplots.png'}")
    plt.close()

    # 2. Bar plot with error bars
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(metrics))
    means = [results_df[m].mean() for m in metrics]
    stds = [results_df[m].std() for m in metrics]

    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=10,
        alpha=0.7,
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Run Mean Performance (±1 std)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(titles)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "multirun_bars.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'multirun_bars.png'}")
    plt.close()

    # 3. Overfitting analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    runs = results_df["run"].values
    overfitting = results_df["overfitting"].values

    bars = ax.bar(runs, overfitting, alpha=0.7, color="coral", edgecolor="black", linewidth=1.5)
    ax.axhline(
        overfitting.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {overfitting.mean():.1f}%",
    )
    ax.axhline(20, color="green", linestyle=":", linewidth=2, label="Target: <20%")

    ax.set_xlabel("Run", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overfitting (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Overfitting Analysis Across Runs\n(μ={overfitting.mean():.1f}%, σ={overfitting.std():.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(runs)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11)

    # Add value labels
    for bar, val in zip(bars, overfitting):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "multirun_overfitting.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_dir / 'multirun_overfitting.png'}")
    plt.close()


def generate_report(results_df, stats_df, output_dir, config):
    """Generate a comprehensive markdown report."""
    output_dir = Path(output_dir)

    report = f"""# Multi-Run Training Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Model:** {config['model']}
- **Dataset:** {config['data']}
- **Number of Runs:** {config['num_runs']}
- **Epochs:** {config['epochs']}
- **Batch Size:** {config['batch_size']}
- **Random Seeds:** {config['seeds']}

## Aggregate Statistics

| Metric | Mean | Std Dev | Min | Max | CV (%) |
|--------|------|---------|-----|-----|--------|
"""

    for _, row in stats_df.iterrows():
        metric = row["metric"]
        mean = row["mean"]
        std = row["std"]
        min_val = row["min"]
        max_val = row["max"]
        cv = (std / mean * 100) if mean != 0 else 0

        report += f"| **{metric}** | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} | {cv:.2f}% |\n"

    report += f"""
## Individual Run Results

| Run | Seed | Best Epoch | mAP50 | mAP50-95 | Precision | Recall | Overfitting (%) |
|-----|------|------------|-------|----------|-----------|--------|-----------------|
"""

    for _, row in results_df.iterrows():
        report += f"| {row['run']} | {row['run_name'].split('seed')[1]} | {row['epoch']} | "
        report += f"{row['mAP50']:.4f} | {row['mAP50-95']:.4f} | {row['precision']:.4f} | "
        report += f"{row['recall']:.4f} | {row['overfitting']:.2f}% |\n"

    # Interpretation
    mAP50_cv = (
        stats_df[stats_df["metric"] == "mAP50"]["std"].values[0]
        / stats_df[stats_df["metric"] == "mAP50"]["mean"].values[0]
        * 100
    )

    report += f"""
## Interpretation

### Variability Analysis

- **mAP50 Coefficient of Variation:** {mAP50_cv:.2f}%
  - < 5%: Excellent consistency ✅
  - 5-10%: Good consistency ✅
  - 10-20%: Moderate variability ⚠️
  - > 20%: High variability, consider more runs or data ❌

"""

    if mAP50_cv < 5:
        report += "**Verdict:** ✅ Excellent - Results are highly reproducible and robust.\n"
    elif mAP50_cv < 10:
        report += "**Verdict:** ✅ Good - Results are reliable with acceptable variation.\n"
    elif mAP50_cv < 20:
        report += "**Verdict:** ⚠️ Moderate - Some variability present, consider additional runs.\n"
    else:
        report += "**Verdict:** ❌ High Variability - Results may not be reliable. Increase dataset size or runs.\n"

    report += f"""
### Overfitting Analysis

- **Mean Overfitting:** {stats_df[stats_df['metric'] == 'overfitting']['mean'].values[0]:.2f}%
- **Overfitting Std Dev:** {stats_df[stats_df['metric'] == 'overfitting']['std'].values[0]:.2f}%

"""

    mean_overfit = stats_df[stats_df["metric"] == "overfitting"]["mean"].values[0]
    if mean_overfit < 20:
        report += "**Verdict:** ✅ Good generalization across all runs.\n"
    elif mean_overfit < 40:
        report += (
            "**Verdict:** ⚠️ Moderate overfitting - consider data augmentation or more data.\n"
        )
    else:
        report += "**Verdict:** ❌ Severe overfitting - urgent need for more training data.\n"

    report += """
## Recommendations

### For Publication/Reporting

Report results as: **Mean ± Std Dev**

Example:
- mAP@50: {:.4f} ± {:.4f}
- mAP@50-95: {:.4f} ± {:.4f}
- Precision: {:.4f} ± {:.4f}
- Recall: {:.4f} ± {:.4f}

### For Model Deployment

""".format(
        stats_df[stats_df["metric"] == "mAP50"]["mean"].values[0],
        stats_df[stats_df["metric"] == "mAP50"]["std"].values[0],
        stats_df[stats_df["metric"] == "mAP50-95"]["mean"].values[0],
        stats_df[stats_df["metric"] == "mAP50-95"]["std"].values[0],
        stats_df[stats_df["metric"] == "precision"]["mean"].values[0],
        stats_df[stats_df["metric"] == "precision"]["std"].values[0],
        stats_df[stats_df["metric"] == "recall"]["mean"].values[0],
        stats_df[stats_df["metric"] == "recall"]["std"].values[0],
    )

    # Find best run
    best_run_idx = results_df["mAP50"].idxmax()
    best_run = results_df.loc[best_run_idx]

    report += f"""
Select the run with best validation performance for deployment:

- **Best Run:** Run {best_run['run']} ({best_run['run_name']})
- **mAP50:** {best_run['mAP50']:.4f}
- **Model Path:** `runs/segment/multirun/{best_run['run_name']}/weights/best.pt`

### Next Steps

"""

    if mean_overfit > 30:
        report += "1. **Priority:** Collect more training data (target: 500+ samples)\n"
        report += "2. Apply stronger data augmentation\n"
        report += "3. Consider model regularization techniques\n"
    else:
        report += "1. Deploy best model for production use\n"
        report += "2. Monitor performance on real-world data\n"
        report += "3. Continue collecting data to improve robustness\n"

    report += """
## Files Generated

- `individual_runs.csv` - Detailed metrics for each run
- `aggregate_stats.csv` - Statistical summary
- `multirun_boxplots.png` - Distribution visualization
- `multirun_bars.png` - Mean performance with error bars
- `multirun_overfitting.png` - Overfitting consistency
- `MULTIRUN_REPORT.md` - This report

---
*Generated by train_multirun.py*
"""

    report_path = output_dir / "MULTIRUN_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"✓ Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run training to assess model variability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 times with auto-detected settings (recommended)
  python scripts/train_multirun.py --num-runs 3
  
  # Run 5 times with custom configuration
  python scripts/train_multirun.py --num-runs 5 --epochs 200 --batch 16
  
  # Use specific seeds for reproducibility
  python scripts/train_multirun.py --num-runs 3 --seeds 42 123 456
        """,
    )

    parser.add_argument(
        "--data",
        default="data/dataset_enhanced_robust/dataset_filtered.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--model", default="yolo11n-seg.pt", help="Model (yolo11n-seg.pt for best generalization)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=9,
        help="Number of training runs (9 recommended for robust statistics)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (auto-generated if not specified)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (auto-detect if not specified)"
    )
    parser.add_argument(
        "--patience", type=int, default=40, help="Early stopping patience (default: 40)"
    )
    parser.add_argument("--name", default="cdw_multirun", help="Base name for runs")
    parser.add_argument(
        "--output", default="analysis/multirun", help="Output directory for analysis"
    )

    args = parser.parse_args()

    # Auto-detect device and batch size
    if torch.cuda.is_available():
        args.device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")

        if args.batch_size is None:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 16:
                args.batch_size = 16
            elif gpu_memory_gb >= 8:
                args.batch_size = 12
            else:
                args.batch_size = 8
            print(f"✓ Auto-selected batch size: {args.batch_size}")
    else:
        args.device = "cpu"
        args.batch_size = args.batch_size or 4
        print("⚠ No GPU detected, using CPU")

    # Generate seeds if not provided
    if args.seeds is None:
        np.random.seed(42)  # For reproducible seed generation
        args.seeds = [int(s) for s in np.random.randint(1, 10000, args.num_runs)]
    elif len(args.seeds) != args.num_runs:
        print(f"⚠ Number of seeds ({len(args.seeds)}) != num_runs ({args.num_runs})")
        print("   Using only first seeds or generating additional ones...")
        if len(args.seeds) < args.num_runs:
            np.random.seed(args.seeds[-1])
            additional = [
                int(s) for s in np.random.randint(1, 10000, args.num_runs - len(args.seeds))
            ]
            args.seeds.extend(additional)
        else:
            args.seeds = args.seeds[: args.num_runs]

    print(f"\n{'='*70}")
    print(f"MULTI-RUN TRAINING FOR VARIABILITY ASSESSMENT")
    print(f"{'='*70}")
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.data}")
    print(f"Runs:         {args.num_runs}")
    print(f"Seeds:        {args.seeds}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Device:       {args.device}")
    print(f"Patience:     {args.patience}")
    print(f"Output:       {args.output}")
    print(f"{'='*70}\n")

    # Confirm with user
    response = input("Start multi-run training? This will take significant time. [y/N]: ")
    if response.lower() != "y":
        print("Aborted.")
        return

    start_time = time.time()
    run_paths = []

    # Run training multiple times
    for i, seed in enumerate(args.seeds):
        run_path = train_single_run(args, i, seed)
        run_paths.append(run_path)

        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (args.num_runs - i - 1)
        print(f"\n✓ Completed run {i + 1}/{args.num_runs}")
        print(f"  Elapsed: {elapsed/3600:.2f}h | ETA: {eta/3600:.2f}h\n")

    # Aggregate results
    print(f"\n{'='*70}")
    print("ANALYZING RESULTS")
    print(f"{'='*70}\n")

    result = aggregate_results(run_paths, args.output)

    if result is not None:
        results_df, stats_df = result
        # Generate plots
        plot_multirun_results(results_df, stats_df, args.output)

        # Generate report
        config = {
            "model": args.model,
            "data": args.data,
            "num_runs": args.num_runs,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        }
        generate_report(results_df, stats_df, args.output, config)

        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("MULTI-RUN ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time:   {total_time/3600:.2f} hours")
        print(f"Results:      {args.output}")
        print(f"\nKey Findings:")
        print(
            f"  mAP50:      {stats_df[stats_df['metric'] == 'mAP50']['mean'].values[0]:.4f} ± "
            f"{stats_df[stats_df['metric'] == 'mAP50']['std'].values[0]:.4f}"
        )
        print(
            f"  Overfitting: {stats_df[stats_df['metric'] == 'overfitting']['mean'].values[0]:.2f}% ± "
            f"{stats_df[stats_df['metric'] == 'overfitting']['std'].values[0]:.2f}%"
        )

        # Find best run
        best_run_idx = results_df["mAP50"].idxmax()
        best_run = results_df.loc[best_run_idx]
        print(f"\nBest Model:   runs/segment/multirun/{best_run['run_name']}/weights/best.pt")
        print(f"              (mAP50: {best_run['mAP50']:.4f})")
        print(f"\nSee detailed report: {args.output}/MULTIRUN_REPORT.md")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
