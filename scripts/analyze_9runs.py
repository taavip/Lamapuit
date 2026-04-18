#!/usr/bin/env python
"""Analyze the 9 completed runs from their actual location."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import functions from train_multirun
import importlib.util

spec = importlib.util.spec_from_file_location("train_multirun", "scripts/train_multirun.py")
multirun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multirun)

# Find all completed runs in the actual location
actual_path = Path("runs/segment/runs/segment/multirun")
run_paths = sorted(actual_path.glob("cdw_multirun9_run*"))

print(f"Found {len(run_paths)} runs in {actual_path}\n")
for p in run_paths:
    print(f"  - {p.name}")

print("\nGenerating analysis...\n")

result = multirun.aggregate_results(run_paths, "analysis/multirun9")

if result:
    results_df, stats_df = result

    # Generate plots
    multirun.plot_multirun_results(results_df, stats_df, "analysis/multirun9")

    # Generate report
    config = {
        "model": "yolo11n-seg.pt",
        "data": "data/dataset_enhanced_robust/dataset_filtered.yaml",
        "num_runs": 9,
        "seeds": [7271, 861, 5391, 5192, 5735, 6266, 467, 4427, 5579],
        "epochs": 200,
        "batch_size": 16,
    }
    multirun.generate_report(results_df, stats_df, "analysis/multirun9", config)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: analysis/multirun9/")
    print(f"\nKey Findings:")
    print(
        f"  mAP50:       {stats_df[stats_df['metric'] == 'mAP50']['mean'].values[0]:.4f} ± "
        f"{stats_df[stats_df['metric'] == 'mAP50']['std'].values[0]:.4f}"
    )
    print(
        f"  Overfitting: {stats_df[stats_df['metric'] == 'overfitting']['mean'].values[0]:.2f}% ± "
        f"{stats_df[stats_df['metric'] == 'overfitting']['std'].values[0]:.2f}%"
    )

    best_run_idx = results_df["mAP50"].idxmax()
    best_run = results_df.loc[best_run_idx]
    print(f"\nBest Model:  {actual_path / best_run['run_name']}/weights/best.pt")
    print(f"             (mAP50: {best_run['mAP50']:.4f})")
    print(f"\nSee full report: analysis/multirun9/MULTIRUN_REPORT.md")
    print("=" * 70)
else:
    print("✗ Failed to generate analysis")
