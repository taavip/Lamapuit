#!/usr/bin/env python
"""Generate full analysis report for the completed 3-run training."""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import functions from train_multirun
import importlib.util

spec = importlib.util.spec_from_file_location("train_multirun", "scripts/train_multirun.py")
multirun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multirun)

# Process the 3 completed runs
run_paths = [
    Path("runs/segment/runs/multirun/cdw_multirun_run1_seed7271"),
    Path("runs/segment/runs/multirun/cdw_multirun_run2_seed861"),
    Path("runs/segment/runs/multirun/cdw_multirun_run3_seed5391"),
]

print("Generating analysis for 3 completed runs...\n")

result = multirun.aggregate_results(run_paths, "analysis/multirun_3runs")

if result:
    results_df, stats_df = result

    # Generate plots
    multirun.plot_multirun_results(results_df, stats_df, "analysis/multirun_3runs")

    # Generate report
    config = {
        "model": "yolo11n-seg.pt",
        "data": "data/dataset_enhanced_robust/dataset_filtered.yaml",
        "num_runs": 3,
        "seeds": [7271, 861, 5391],
        "epochs": 200,
        "batch_size": 16,
    }
    multirun.generate_report(results_df, stats_df, "analysis/multirun_3runs", config)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: analysis/multirun_3runs/")
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
    print(f"\nBest Model:  runs/segment/runs/multirun/{best_run['run_name']}/weights/best.pt")
    print(f"             (mAP50: {best_run['mAP50']:.4f})")
    print(f"\nSee full report: analysis/multirun_3runs/MULTIRUN_REPORT.md")
    print("=" * 70)
else:
    print("✗ Failed to generate analysis")
