#!/usr/bin/env python3
"""
Analyze CHM Variant Benchmark V2 results.

Compares 5 CHM variants (baseline, harmonized_raw, harmonized_gauss, composite_2band, composite_4band)
across 6 architectures and 3 folds to identify the best variant and architecture.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

def load_results(results_file: Path) -> List[Dict]:
    """Load results from JSON."""
    with open(results_file) as f:
        return json.load(f)

def analyze_by_variant(results: List[Dict]) -> pd.DataFrame:
    """Group results by variant and compute summary statistics."""
    variant_stats = []

    for variant in sorted(set(r['variant'] for r in results)):
        variant_results = [r for r in results if r['variant'] == variant]
        f1_scores = [r['mean_f1'] for r in variant_results]

        variant_stats.append({
            'Variant': variant,
            'Mean F1': np.mean(f1_scores),
            'Std F1': np.std(f1_scores),
            'Min F1': np.min(f1_scores),
            'Max F1': np.max(f1_scores),
            'Count': len(variant_results),
        })

    return pd.DataFrame(variant_stats).sort_values('Mean F1', ascending=False)

def analyze_by_architecture(results: List[Dict]) -> pd.DataFrame:
    """Group results by architecture and compute summary statistics."""
    arch_stats = []

    for arch in sorted(set(r['architecture'] for r in results)):
        arch_results = [r for r in results if r['architecture'] == arch]
        f1_scores = [r['mean_f1'] for r in arch_results]

        arch_stats.append({
            'Architecture': arch,
            'Mean F1': np.mean(f1_scores),
            'Std F1': np.std(f1_scores),
            'Count': len(arch_results),
        })

    return pd.DataFrame(arch_stats).sort_values('Mean F1', ascending=False)

def analyze_variant_by_architecture(results: List[Dict]) -> pd.DataFrame:
    """Create variant × architecture matrix."""
    matrix_data = []

    for variant in sorted(set(r['variant'] for r in results)):
        row = {'Variant': variant}
        for arch in sorted(set(r['architecture'] for r in results)):
            variant_arch = [r for r in results if r['variant'] == variant and r['architecture'] == arch]
            if variant_arch:
                row[arch] = f"{variant_arch[0]['mean_f1']:.4f}"
            else:
                row[arch] = "—"
        matrix_data.append(row)

    return pd.DataFrame(matrix_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('output/chm_variant_benchmark_v2_fast/results.json'))
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Results file not found: {args.results}")
        return

    results = load_results(args.results)

    print("\n" + "="*80)
    print("CHM VARIANT BENCHMARK V2 — ANALYSIS SUMMARY")
    print("="*80)

    print("\n1. RANKING BY VARIANT (Average F1 across all architectures)")
    print("-" * 80)
    variant_ranking = analyze_by_variant(results)
    print(variant_ranking.to_string(index=False))

    print("\n2. RANKING BY ARCHITECTURE (Average F1 across all variants)")
    print("-" * 80)
    arch_ranking = analyze_by_architecture(results)
    print(arch_ranking.to_string(index=False))

    print("\n3. VARIANT × ARCHITECTURE MATRIX (F1 scores)")
    print("-" * 80)
    matrix = analyze_variant_by_architecture(results)
    print(matrix.to_string(index=False))

    print("\n4. WINNER SUMMARY")
    print("-" * 80)
    best_variant = variant_ranking.iloc[0]
    best_arch = arch_ranking.iloc[0]

    print(f"Best variant: {best_variant['Variant']} (F1 = {best_variant['Mean F1']:.4f} ± {best_variant['Std F1']:.4f})")
    print(f"Best architecture: {best_arch['Architecture']} (F1 = {best_arch['Mean F1']:.4f} ± {best_arch['Std F1']:.4f})")

    # Find best combination
    best_combo = max(results, key=lambda x: x['mean_f1'])
    print(f"Best combination: {best_combo['variant']} + {best_combo['architecture']} (F1 = {best_combo['mean_f1']:.4f})")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
