"""
Analyze CHM Ablation Results

Loads results.json and generates:
- Summary tables by input mode
- Best model per input
- Recommendations
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_results(results_file: Path) -> None:
    """Analyze and report CHM ablation results."""

    with open(results_file) as fh:
        results = json.load(fh)

    print("\n" + "="*80)
    print("CHM INPUT ABLATION RESULTS")
    print("="*80)

    # Group by input mode
    by_input = defaultdict(list)
    for r in results:
        by_input[r["input_mode"]].append(r)

    print("\n1. Summary by Input Mode (averaged across models & folds)")
    print("-" * 80)
    print(f"{'Input Mode':<15} | {'F1 mean':<10} | {'F1 std':<10} | {'AUC mean':<10}")
    print("-" * 80)

    input_summaries = {}
    for input_mode in sorted(by_input.keys()):
        f1_scores = [r["f1"] for r in by_input[input_mode]]
        auc_scores = [r["auc"] for r in by_input[input_mode]]

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_auc = np.mean(auc_scores)

        input_summaries[input_mode] = {
            "f1_mean": mean_f1,
            "f1_std": std_f1,
            "auc_mean": mean_auc,
        }

        print(f"{input_mode:<15} | {mean_f1:<10.4f} | {std_f1:<10.4f} | {mean_auc:<10.4f}")

    # Group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    print("\n2. Summary by Model (averaged across inputs & folds)")
    print("-" * 80)
    print(f"{'Model':<20} | {'F1 mean':<10} | {'F1 std':<10} | {'AUC mean':<10}")
    print("-" * 80)

    for model in sorted(by_model.keys()):
        f1_scores = [r["f1"] for r in by_model[model]]
        auc_scores = [r["auc"] for r in by_model[model]]

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_auc = np.mean(auc_scores)

        print(f"{model:<20} | {mean_f1:<10.4f} | {std_f1:<10.4f} | {mean_auc:<10.4f}")

    # Best combinations
    print("\n3. Top 5 Model × Input Combinations")
    print("-" * 80)
    print(f"{'Model':<20} | {'Input Mode':<15} | {'F1':<8} | {'AUC':<8} | {'Fold':<5}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda r: r["f1"], reverse=True)[:5]
    for r in sorted_results:
        print(f"{r['model']:<20} | {r['input_mode']:<15} | {r['f1']:<8.4f} | {r['auc']:<8.4f} | {r['fold']:<5}")

    # Recommendations
    print("\n4. RECOMMENDATIONS")
    print("-" * 80)

    best_input = max(input_summaries.items(), key=lambda x: x[1]["f1_mean"])[0]
    best_input_f1 = input_summaries[best_input]["f1_mean"]

    best_combo = sorted_results[0]
    print(f"✓ Best Input Mode: {best_input} (F1={best_input_f1:.4f})")
    print(f"✓ Best Model: {best_combo['model']} with {best_combo['input_mode']} (F1={best_combo['f1']:.4f})")

    print("\nFor production training, use:")
    print(f"  - Input: {best_input}")
    print(f"  - Model: {best_combo['model']}")
    print(f"  - Expected F1: {best_input_f1:.4f}±{input_summaries[best_input]['f1_std']:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    import sys
    results_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/model_search_chm_ablation_results/results.json")
    analyze_results(results_file)
