#!/usr/bin/env python3
"""CHM Variant Selection — Statistical Analysis and Report Generation

Reads results.csv from chm_variant_selection.py and produces:
- CHM_VARIANT_SELECTION_REPORT.md: ranked modes, statistical tests, decision
- Mode ranking table with 95% confidence intervals
- Welch's t-test between top-2 modes
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compute_mode_stats(df: pd.DataFrame, metric: str = "test_f1") -> pd.DataFrame:
    """Compute per-mode statistics: mean, std, CI, n_runs."""
    mode_stats = df.groupby("mode")[metric].agg(["mean", "std", "count"])
    mode_stats.columns = ["mean_f1", "std_f1", "n_runs"]

    # 95% CI via t-distribution
    for mode in mode_stats.index:
        n = int(mode_stats.loc[mode, "n_runs"])
        std = mode_stats.loc[mode, "std_f1"]
        if n > 1:
            ci_half = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
        else:
            ci_half = 0.0
        mode_stats.loc[mode, "ci95_lower"] = mode_stats.loc[mode, "mean_f1"] - ci_half
        mode_stats.loc[mode, "ci95_upper"] = mode_stats.loc[mode, "mean_f1"] + ci_half

    return mode_stats.sort_values("mean_f1", ascending=False)


def welch_t_test(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Welch's two-sided t-test. Returns (t_stat, p_value)."""
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return t_stat, p_val


def decision(
    top1_mean: float,
    top2_mean: float,
    margin_threshold: float = 0.02,
) -> tuple[str, bool]:
    """Decide winner or uncertain.

    Returns:
        (decision_text, is_winner)
    """
    margin = top1_mean - top2_mean
    if margin >= margin_threshold:
        return f"WINNER (margin {margin:.4f} ≥ threshold {margin_threshold})", True
    else:
        return f"UNCERTAIN (margin {margin:.4f} < threshold {margin_threshold})", False


def format_summary_table(mode_stats: pd.DataFrame) -> str:
    """Format ranking table as markdown."""
    lines = [
        "| Mode | Mean F1 | Std | 95% CI Lower | 95% CI Upper | n_runs |",
        "|------|---------|-----|--------------|--------------|--------|",
    ]
    for mode, row in mode_stats.iterrows():
        lines.append(
            f"| {mode} | {row['mean_f1']:.4f} | {row['std_f1']:.4f} | "
            f"{row['ci95_lower']:.4f} | {row['ci95_upper']:.4f} | {int(row['n_runs'])} |"
        )
    return "\n".join(lines)


def per_model_breakdown(df: pd.DataFrame) -> str:
    """Format per-model breakdown as markdown."""
    pivot = df.pivot_table(values="test_f1", index="mode", columns="model", aggfunc="mean")
    lines = ["| Mode |", "|------|"]
    for col in pivot.columns:
        lines[0] += f" {col} |"
        lines[1] += " --- |"

    for mode, row in pivot.iterrows():
        line = f"| {mode} |"
        for val in row:
            line += f" {val:.4f} |"
        lines.append(line)

    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    mode_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write CHM_VARIANT_SELECTION_REPORT.md."""

    top1_mode = mode_stats.index[0]
    top2_mode = mode_stats.index[1] if len(mode_stats) > 1 else None
    top1_mean = mode_stats.loc[top1_mode, "mean_f1"]
    top2_mean = mode_stats.loc[top2_mode, "mean_f1"] if top2_mode else 0.0

    # Welch's t-test
    top1_scores = df[df["mode"] == top1_mode]["test_f1"].values
    top2_scores = df[df["mode"] == top2_mode]["test_f1"].values if top2_mode else np.array([])

    if len(top2_scores) > 0:
        t_stat, p_val = welch_t_test(top1_scores, top2_scores)
    else:
        t_stat, p_val = np.nan, np.nan

    # Decision
    decision_text, is_winner = decision(top1_mean, top2_mean)

    # Report markdown
    report = f"""# CHM Variant Selection Report

Generated: {pd.Timestamp.now().isoformat()}

## 1. Experimental Design

- **Variants**: original (1-ch baseline), raw (1-ch harmonized), gauss (1-ch smoothed), composite_3band (2-ch pre-fused)
- **Models**: convnext_small, efficientnet_b2
- **Folds**: {int(df["fold"].max()) + 1} (spatial StratifiedGroupKFold, place-key grouping)
- **Test set**: 3,481 tiles (spatial V4 holdout, seed=2026)
- **Decision rule**: winner if test F1 margin ≥ 0.02; else prefer simpler single-channel variant

Reference: Gu et al. (2024) — 50 m spatial autocorrelation for CWD detection in forest LiDAR.

## 2. Mode Ranking (Test F1, all models and folds)

{format_summary_table(mode_stats)}

## 3. Statistical Test

Welch's t-test: {top1_mode} vs {top2_mode}

- t-statistic: {t_stat:.4f}
- p-value: {p_val:.4f}
- Margin: {top1_mean:.4f} − {top2_mean:.4f} = {top1_mean - top2_mean:.4f}

## 4. Decision

**{decision_text}**

Recommended variant: **{top1_mode}** (mean test F1 = {top1_mean:.4f} ± {mode_stats.loc[top1_mode, 'std_f1']:.4f})

## 5. Per-Model Breakdown

{per_model_breakdown(df)}

## 6. Notes on composite_3band

The `composite_3band` variant is a pre-computed 2-channel raster in `data/chm_variants/composite_3band/`,
stacking raw + gauss preprocessing. The original/baseline CHM band is NOT included. The first
convolutional layer is adapted from pretrained 3-ch weights by keeping the first 2 filter slices,
then wrapped with `_CHMInputNorm` to prevent training collapse.

Single-channel modes (original, raw, gauss) adapt the first conv by channel averaging,
equally preventing weight collapse.

---

Generated by `scripts/chm_variant_selection_analyze.py`.
"""

    (output_dir / "CHM_VARIANT_SELECTION_REPORT.md").write_text(report)
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to results.csv")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output)
    results_path = Path(args.results)

    # Load results
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} results from {results_path}")

    # Compute stats
    mode_stats = compute_mode_stats(df)

    # Write report
    write_report(df, mode_stats, output_dir)

    print(f"\n✓ Report written: {output_dir / 'CHM_VARIANT_SELECTION_REPORT.md'}")


if __name__ == "__main__":
    main()
