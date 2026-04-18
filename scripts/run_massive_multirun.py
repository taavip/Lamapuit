#!/usr/bin/env python3
"""
Massive multi-run training script for robust statistical analysis.
Runs top models 60 times each to eliminate variability and get reliable metrics.
"""

import argparse
import time
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Top model configurations from experiments
MODEL_CONFIGS = {
    "exp7_conservative": {
        "model": "yolo11s-seg.pt",
        "batch": 12,
        "mixup": 0.05,
        "copy_paste": 0.05,
        "erasing": 0.2,
        "dropout": 0.1,
        "weight_decay": 0.001,
    },
    "exp4_low_lr": {
        "model": "yolo11s-seg.pt",
        "batch": 12,
        "lr0": 0.005,
        "mixup": 0.15,
        "copy_paste": 0.15,
        "erasing": 0.4,
        "dropout": 0.1,
        "weight_decay": 0.001,
    },
    "exp3_small": {
        "model": "yolo11s-seg.pt",
        "batch": 12,
        "mixup": 0.15,
        "copy_paste": 0.15,
        "erasing": 0.4,
        "dropout": 0.1,
        "weight_decay": 0.001,
    },
    "exp2_medium": {
        "model": "yolo11m-seg.pt",
        "batch": 8,
        "mixup": 0.15,
        "copy_paste": 0.15,
        "erasing": 0.4,
        "dropout": 0.1,
        "weight_decay": 0.001,
    },
}


def train_single_run(config_name, config, run_idx, seed, dataset, epochs, patience, device):
    """Train a single model with specified seed."""
    try:
        from ultralytics import YOLO
        import gc

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        yolo = YOLO(config["model"])

        run_name = f"{config_name}_run{run_idx:03d}_seed{seed}"

        # Training arguments
        train_args = {
            "data": dataset,
            "epochs": epochs,
            "batch": config.get("batch", 12),
            "device": device,
            "name": run_name,
            "project": "runs/massive_multirun",
            "imgsz": 640,
            "patience": patience,
            "seed": seed,
            "deterministic": True,
            # Augmentation
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": config.get("mixup", 0.15),
            "copy_paste": config.get("copy_paste", 0.15),
            "erasing": config.get("erasing", 0.4),
            # Regularization
            "dropout": config.get("dropout", 0.1),
            "weight_decay": config.get("weight_decay", 0.001),
            "lr0": config.get("lr0", 0.01),
            # Performance
            "workers": 4,
            "cache": "disk",
            "exist_ok": True,
            "verbose": False,  # Reduce output for 60 runs
            "amp": torch.cuda.is_available(),
        }

        results = yolo.train(**train_args)

        # Parse results
        run_path = Path("runs/massive_multirun") / run_name
        results_csv = run_path / "results.csv"

        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()

            best_idx = df["metrics/mAP50(B)"].idxmax()

            metrics = {
                "run": run_idx,
                "seed": seed,
                "config": config_name,
                "best_epoch": int(df.loc[best_idx, "epoch"]),
                "mAP50": float(df.loc[best_idx, "metrics/mAP50(B)"]),
                "mAP50_95": float(df.loc[best_idx, "metrics/mAP50-95(B)"]),
                "precision": float(df.loc[best_idx, "metrics/precision(B)"]),
                "recall": float(df.loc[best_idx, "metrics/recall(B)"]),
                "train_loss": float(df.loc[best_idx, "train/seg_loss"]),
                "val_loss": float(df.loc[best_idx, "val/seg_loss"]),
                "overfitting": (
                    float(df.loc[best_idx, "val/seg_loss"])
                    - float(df.loc[best_idx, "train/seg_loss"])
                )
                / float(df.loc[best_idx, "train/seg_loss"])
                * 100,
                "total_epochs": len(df),
                "success": True,
            }
        else:
            metrics = {"run": run_idx, "seed": seed, "config": config_name, "success": False}

        return metrics

    except Exception as e:
        print(f"Error in run {run_idx} seed {seed}: {e}")
        return {
            "run": run_idx,
            "seed": seed,
            "config": config_name,
            "success": False,
            "error": str(e),
        }


def run_massive_multirun(
    config_name,
    num_runs=60,
    epochs=150,
    patience=40,
    dataset="data/dataset_final/dataset_trainval.yaml",
    device="0",
    parallel=False,
):
    """Run massive multi-run training for a configuration."""

    print(f"\n{'='*80}")
    print(f"MASSIVE MULTI-RUN: {config_name}")
    print(f"{'='*80}")
    print(f"Runs: {num_runs}")
    print(f"Epochs: {epochs} (patience: {patience})")
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Parallel: {parallel}")
    print(f"{'='*80}\n")

    config = MODEL_CONFIGS[config_name]

    # Generate random seeds
    np.random.seed(42)
    seeds = np.random.randint(1000, 9999, size=num_runs).tolist()

    results = []
    start_time = time.time()

    if parallel:
        # Parallel execution (use with caution - high GPU memory)
        print("⚠️  Running in parallel mode - ensure sufficient GPU memory!")
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    train_single_run,
                    config_name,
                    config,
                    i,
                    seeds[i],
                    dataset,
                    epochs,
                    patience,
                    device,
                ): i
                for i in range(num_runs)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if result["success"]:
                    print(
                        f"✓ Run {result['run']+1}/{num_runs} complete: "
                        f"mAP50={result['mAP50']:.4f}"
                    )
                else:
                    print(f"✗ Run {result['run']+1}/{num_runs} failed")

                # Save progress
                save_progress(config_name, results)
    else:
        # Sequential execution (safer, recommended)
        for i in range(num_runs):
            result = train_single_run(
                config_name, config, i, seeds[i], dataset, epochs, patience, device
            )
            results.append(result)

            if result["success"]:
                print(
                    f"✓ Run {i+1}/{num_runs} complete: "
                    f"mAP50={result['mAP50']:.4f} "
                    f"(epoch {result['best_epoch']}/{result['total_epochs']})"
                )
            else:
                print(f"✗ Run {i+1}/{num_runs} failed")

            # Save progress after each run
            save_progress(config_name, results)

    elapsed = time.time() - start_time

    # Final analysis
    analyze_results(config_name, results, elapsed)

    return results


def save_progress(config_name, results):
    """Save progress to file."""
    output_file = f"massive_multirun_{config_name}_progress.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def analyze_results(config_name, results, elapsed_time):
    """Comprehensive analysis of multi-run results."""

    print(f"\n{'='*80}")
    print(f"ANALYSIS: {config_name}")
    print(f"{'='*80}\n")

    # Filter successful runs
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    print(f"Successful runs: {len(successful)}/{len(results)}")
    print(f"Failed runs: {len(failed)}")
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Avg time per run: {elapsed_time/len(results)/60:.1f} minutes")

    if len(successful) == 0:
        print("\n❌ No successful runs!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(successful)

    # Compute statistics
    metrics = ["mAP50", "mAP50_95", "precision", "recall", "overfitting", "best_epoch"]

    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}\n")

    stats = pd.DataFrame(
        {
            "Mean": df[metrics].mean(),
            "Std": df[metrics].std(),
            "Min": df[metrics].min(),
            "Max": df[metrics].max(),
            "CV%": (df[metrics].std() / df[metrics].mean() * 100),
            "Median": df[metrics].median(),
            "Q1": df[metrics].quantile(0.25),
            "Q3": df[metrics].quantile(0.75),
        }
    )

    print(stats.to_string())

    # Save statistics
    stats.to_csv(f"massive_multirun_{config_name}_statistics.csv")
    df.to_csv(f"massive_multirun_{config_name}_all_runs.csv", index=False)

    # Best and worst runs
    print(f"\n{'='*80}")
    print("EXTREMES")
    print(f"{'='*80}\n")

    best_run = df.loc[df["mAP50"].idxmax()]
    worst_run = df.loc[df["mAP50"].idxmin()]

    print(f"BEST RUN (seed {best_run['seed']}):")
    print(f"  mAP50: {best_run['mAP50']:.4f}")
    print(f"  Precision: {best_run['precision']:.4f}")
    print(f"  Recall: {best_run['recall']:.4f}")
    print(f"  Overfitting: {best_run['overfitting']:.1f}%")

    print(f"\nWORST RUN (seed {worst_run['seed']}):")
    print(f"  mAP50: {worst_run['mAP50']:.4f}")
    print(f"  Precision: {worst_run['precision']:.4f}")
    print(f"  Recall: {worst_run['recall']:.4f}")
    print(f"  Overfitting: {worst_run['overfitting']:.1f}%")

    print(f"\nRANGE: {worst_run['mAP50']:.4f} - {best_run['mAP50']:.4f}")
    print(
        f"SPAN: {(best_run['mAP50'] - worst_run['mAP50']):.4f} "
        f"({(best_run['mAP50'] - worst_run['mAP50'])/stats.loc['mAP50', 'Mean']*100:.1f}% of mean)"
    )

    # Create visualizations
    create_visualizations(config_name, df)

    print(f"\n✓ Results saved:")
    print(f"  - massive_multirun_{config_name}_statistics.csv")
    print(f"  - massive_multirun_{config_name}_all_runs.csv")
    print(f"  - massive_multirun_{config_name}_plots.png")


def create_visualizations(config_name, df):
    """Create comprehensive visualization plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. mAP50 distribution
    ax = axes[0, 0]
    ax.hist(df["mAP50"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(
        df["mAP50"].mean(), color="red", linestyle="--", label=f"Mean: {df['mAP50'].mean():.4f}"
    )
    ax.axvline(
        df["mAP50"].median(),
        color="green",
        linestyle="--",
        label=f"Median: {df['mAP50'].median():.4f}",
    )
    ax.set_xlabel("mAP50")
    ax.set_ylabel("Frequency")
    ax.set_title(f"mAP50 Distribution (n={len(df)})")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Overfitting distribution
    ax = axes[0, 1]
    ax.hist(df["overfitting"], bins=20, edgecolor="black", alpha=0.7, color="orange")
    ax.axvline(
        df["overfitting"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['overfitting'].mean():.1f}%",
    )
    ax.set_xlabel("Overfitting %")
    ax.set_ylabel("Frequency")
    ax.set_title("Overfitting Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Precision vs Recall
    ax = axes[0, 2]
    ax.scatter(df["recall"], df["precision"], alpha=0.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.grid(alpha=0.3)

    # 4. mAP50 over runs (to detect trends)
    ax = axes[1, 0]
    ax.plot(df["run"], df["mAP50"], "o-", alpha=0.5)
    ax.axhline(df["mAP50"].mean(), color="red", linestyle="--", alpha=0.7)
    ax.fill_between(
        df["run"],
        df["mAP50"].mean() - df["mAP50"].std(),
        df["mAP50"].mean() + df["mAP50"].std(),
        alpha=0.2,
        color="red",
    )
    ax.set_xlabel("Run Number")
    ax.set_ylabel("mAP50")
    ax.set_title("mAP50 Stability Over Runs")
    ax.grid(alpha=0.3)

    # 5. Box plots
    ax = axes[1, 1]
    metrics_to_plot = ["mAP50", "mAP50_95", "precision", "recall"]
    bp = ax.boxplot([df[m] for m in metrics_to_plot], labels=metrics_to_plot)
    ax.set_ylabel("Value")
    ax.set_title("Metrics Box Plots")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # 6. Cumulative distribution
    ax = axes[1, 2]
    sorted_mAP50 = np.sort(df["mAP50"])
    cumulative = np.arange(1, len(sorted_mAP50) + 1) / len(sorted_mAP50)
    ax.plot(sorted_mAP50, cumulative, linewidth=2)
    ax.axvline(df["mAP50"].median(), color="green", linestyle="--", label="Median")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("mAP50")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"massive_multirun_{config_name}_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_all_configs(
    num_runs=60,
    epochs=150,
    patience=40,
    dataset="data/dataset_final/dataset_trainval.yaml",
    device="0",
    parallel=False,
):
    """Run massive multi-run for all top configurations."""

    print(f"\n{'#'*80}")
    print(f"MASSIVE MULTI-RUN SUITE")
    print(f"{'#'*80}")
    print(f"Configurations: {len(MODEL_CONFIGS)}")
    print(f"Runs per config: {num_runs}")
    print(f"Total runs: {len(MODEL_CONFIGS) * num_runs}")
    print(f"Estimated time: {len(MODEL_CONFIGS) * num_runs * 3 / 60:.1f} hours (@ 3 min/run)")
    print(f"{'#'*80}\n")

    all_results = {}

    for config_name in MODEL_CONFIGS.keys():
        results = run_massive_multirun(
            config_name, num_runs, epochs, patience, dataset, device, parallel
        )
        all_results[config_name] = results

    # Comparative analysis
    create_comparative_analysis(all_results)


def create_comparative_analysis(all_results):
    """Compare all configurations side by side."""

    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    comparison = []

    for config_name, results in all_results.items():
        successful = [r for r in results if r.get("success", False)]
        if len(successful) == 0:
            continue

        df = pd.DataFrame(successful)

        comparison.append(
            {
                "config": config_name,
                "n_runs": len(successful),
                "mAP50_mean": df["mAP50"].mean(),
                "mAP50_std": df["mAP50"].std(),
                "mAP50_cv": df["mAP50"].std() / df["mAP50"].mean() * 100,
                "mAP50_min": df["mAP50"].min(),
                "mAP50_max": df["mAP50"].max(),
                "precision_mean": df["precision"].mean(),
                "recall_mean": df["recall"].mean(),
                "overfitting_mean": df["overfitting"].mean(),
            }
        )

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values("mAP50_mean", ascending=False)

    print(comp_df.to_string(index=False))

    comp_df.to_csv("massive_multirun_comparison.csv", index=False)

    print(f"\n✓ Comparison saved to: massive_multirun_comparison.csv")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Mean ± std
    ax = axes[0]
    x = range(len(comp_df))
    ax.bar(x, comp_df["mAP50_mean"], yerr=comp_df["mAP50_std"], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["config"], rotation=45, ha="right")
    ax.set_ylabel("mAP50")
    ax.set_title("Mean mAP50 ± Std Dev")
    ax.grid(alpha=0.3)

    # Coefficient of variation
    ax = axes[1]
    ax.bar(x, comp_df["mAP50_cv"], alpha=0.7, color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["config"], rotation=45, ha="right")
    ax.set_ylabel("CV %")
    ax.set_title("Coefficient of Variation (lower = more stable)")
    ax.axhline(20, color="green", linestyle="--", alpha=0.5, label="Good (<20%)")
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="Poor (>50%)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("massive_multirun_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Comparison plot saved to: massive_multirun_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Massive multi-run training")
    parser.add_argument(
        "--config",
        type=str,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Configuration to run",
    )
    parser.add_argument("--runs", type=int, default=60, help="Number of runs")
    parser.add_argument("--epochs", type=int, default=150, help="Max epochs per run")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset_final/dataset_trainval.yaml",
        help="Dataset YAML file",
    )
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel (experimental)")

    args = parser.parse_args()

    if args.config == "all":
        run_all_configs(
            args.runs, args.epochs, args.patience, args.dataset, args.device, args.parallel
        )
    else:
        run_massive_multirun(
            args.config,
            args.runs,
            args.epochs,
            args.patience,
            args.dataset,
            args.device,
            args.parallel,
        )
