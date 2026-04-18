#!/usr/bin/env python3
"""Summarize reproduce/agile suite with nuance and negative-ground checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter


def _chm_stats(path: Path) -> Dict[str, float]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid = np.isfinite(arr) if nodata is None else (np.isfinite(arr) & (arr != nodata))
        vals = arr[valid]
        if vals.size == 0:
            return {
                "valid_pct": 0.0,
                "min": float("nan"),
                "p05": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "p95": float("nan"),
                "grad_p95": float("nan"),
                "hf_p95": float("nan"),
                "neg_pct": 0.0,
            }

        fill = np.where(valid, arr, np.nanmedian(vals))
        gy, gx = np.gradient(fill)
        grad = np.hypot(gx, gy)[valid]
        hf = np.abs(fill - gaussian_filter(fill, sigma=2.0))[valid]

        return {
            "valid_pct": float(100.0 * vals.size / arr.size),
            "min": float(np.min(vals)),
            "p05": float(np.percentile(vals, 5)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p95": float(np.percentile(vals, 95)),
            "grad_p95": float(np.percentile(grad, 95)),
            "hf_p95": float(np.percentile(hf, 95)),
            "neg_pct": float(100.0 * np.mean(vals < 0.0)),
        }


def _nuance_similarity(candidate: Dict[str, float], target: Dict[str, float]) -> float:
    keys = ["mean", "p95", "grad_p95", "hf_p95"]
    acc = 0.0
    n = 0
    for k in keys:
        cv = candidate.get(k)
        tv = target.get(k)
        if cv is None or tv is None or not np.isfinite(cv) or not np.isfinite(tv):
            continue
        denom = max(abs(tv), 1e-6)
        acc += ((cv - tv) / denom) ** 2
        n += 1
    if n == 0:
        return 0.0
    dist = float(np.sqrt(acc / n))
    return float(1.0 / (1.0 + dist))


def _negative_presence_score(neg_pct: float, target_pct: float) -> float:
    if not np.isfinite(neg_pct):
        return 0.0
    return float(min(max(neg_pct, 0.0) / max(target_pct, 1e-6), 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--research-reference", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--target-neg-pct", type=float, default=3.0)
    args = ap.parse_args()

    target = _chm_stats(args.research_reference)
    methods = ["tin_linear_gauss", "tin_linear_sadapt_gauss", "natural_neighbor_linear_sadapt_gauss"]

    rows: List[Dict[str, float]] = []
    run_dirs = sorted([p for p in args.runs_root.iterdir() if p.is_dir()])

    for run_dir in run_dirs:
        summary_csv = run_dir / "eval" / "method_summary.csv"
        if not summary_csv.exists():
            continue

        df = pd.read_csv(summary_csv)
        if "method" not in df.columns:
            continue

        for method in methods:
            row_match = df[df["method"] == method]
            if row_match.empty:
                continue
            mr = row_match.iloc[0]

            chm_2018 = run_dir / "chm" / "2018" / f"2018_{method}_chm.tif"
            if not chm_2018.exists():
                continue

            tex = _chm_stats(chm_2018)
            nuance_sim = _nuance_similarity(tex, target)

            youden = float(mr.get("youden_tile_max", mr.get("j_tile_max", np.nan)))
            auc = float(mr.get("auc_tile_max", np.nan))
            false_high_rate = float(mr.get("no_false_high_rate_15cm", np.nan))
            cdw_rate = float(mr.get("cdw_detect_rate_15cm", np.nan))

            neg_score = _negative_presence_score(tex["neg_pct"], args.target_neg_pct)
            score = (
                0.35 * (youden if np.isfinite(youden) else 0.0)
                + 0.20 * (auc if np.isfinite(auc) else 0.0)
                + 0.20 * nuance_sim
                + 0.15 * (1.0 - (false_high_rate if np.isfinite(false_high_rate) else 1.0))
                + 0.10 * neg_score
            )

            rows.append(
                {
                    "run": run_dir.name,
                    "method": method,
                    "youden_tile_max": youden,
                    "auc_tile_max": auc,
                    "cdw_detect_rate_15cm": cdw_rate,
                    "false_high_rate_15cm": false_high_rate,
                    "nuance_similarity_to_research": nuance_sim,
                    "neg_presence_score": neg_score,
                    "valid_pct_2018": tex["valid_pct"],
                    "min_2018": tex["min"],
                    "p05_2018": tex["p05"],
                    "mean_2018": tex["mean"],
                    "std_2018": tex["std"],
                    "p95_2018": tex["p95"],
                    "grad_p95_2018": tex["grad_p95"],
                    "hf_p95_2018": tex["hf_p95"],
                    "neg_pct_2018": tex["neg_pct"],
                    "reproduce_agile_score": score,
                    "chm_2018_path": str(chm_2018),
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No summary rows found. Ensure suite runs completed.")

    out_df = out_df.sort_values("reproduce_agile_score", ascending=False).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    top = out_df.head(10)
    best = out_df.iloc[0]

    lines: List[str] = []
    lines.append("# Reproduce + Agile Suite Report (2026-04-14)")
    lines.append("")
    lines.append("## Objective")
    lines.append(
        "Reproduce research-style CHM while improving steep-slope responsiveness and allowing realistic negative near-ground CHM values."
    )
    lines.append("")
    lines.append("## Reference Target (Research 2018 tin_linear_gauss)")
    lines.append(f"- mean={target['mean']:.6f}")
    lines.append(f"- p95={target['p95']:.6f}")
    lines.append(f"- grad_p95={target['grad_p95']:.6f}")
    lines.append(f"- hf_p95={target['hf_p95']:.6f}")
    lines.append("")
    lines.append("## Ranking Rule")
    lines.append(
        "Score = 0.35*Youden + 0.20*AUC + 0.20*NuanceSimilarity + 0.15*(1-FalseHighRate) + 0.10*NegativePresence"
    )
    lines.append(f"- NegativePresence target={args.target_neg_pct:.2f}% pixels below 0 m")
    lines.append("")
    lines.append("## Best Candidate")
    lines.append(f"- run={best['run']}")
    lines.append(f"- method={best['method']}")
    lines.append(f"- reproduce_agile_score={best['reproduce_agile_score']:.6f}")
    lines.append(f"- nuance_similarity_to_research={best['nuance_similarity_to_research']:.6f}")
    lines.append(f"- youden_tile_max={best['youden_tile_max']:.6f}")
    lines.append(f"- auc_tile_max={best['auc_tile_max']:.6f}")
    lines.append(f"- false_high_rate_15cm={best['false_high_rate_15cm']:.6f}")
    lines.append(f"- neg_pct_2018={best['neg_pct_2018']:.3f}")
    lines.append(f"- min_2018={best['min_2018']:.4f}")
    lines.append(f"- chm_2018_path={best['chm_2018_path']}")
    lines.append("")
    lines.append("## Top 10")
    lines.append("")
    lines.append("| rank | run | method | score | nuance | youden | auc | false_high | neg_pct | min |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in top.iterrows():
        lines.append(
            f"| {i+1} | {r['run']} | {r['method']} | {r['reproduce_agile_score']:.6f} | {r['nuance_similarity_to_research']:.6f} | {r['youden_tile_max']:.6f} | {r['auc_tile_max']:.6f} | {r['false_high_rate_15cm']:.6f} | {r['neg_pct_2018']:.3f} | {r['min_2018']:.4f} |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary_csv={args.out_csv}")
    print(f"summary_md={args.out_md}")
    print(f"best_run={best['run']}")
    print(f"best_method={best['method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
