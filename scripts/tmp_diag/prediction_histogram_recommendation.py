#!/usr/bin/env python3
"""Build prediction histogram and recommend auto/manual confidence thresholds.

Inputs:
- labels dir with *_labels.csv

Outputs:
- histogram CSV
- recommendation JSON
- short markdown summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_auto_probs(labels_dir: Path) -> np.ndarray:
    vals = []
    for p in sorted(labels_dir.glob("*_labels.csv")):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("source", "")).strip() != "auto":
                    continue
                prob = str(row.get("model_prob", "")).strip()
                if not prob:
                    continue
                try:
                    vals.append(float(prob))
                except Exception:
                    continue
    if not vals:
        return np.array([], dtype=np.float32)
    return np.asarray(vals, dtype=np.float32)


def _recommend_band(
    probs: np.ndarray,
    spotcheck_frac: float,
    target_manual_frac: float,
) -> dict:
    if probs.size == 0:
        return {}

    best = None
    for w in np.arange(0.01, 0.301, 0.01):
        lo = max(0.0, 0.5 - float(w))
        hi = min(1.0, 0.5 + float(w))
        low = int(np.sum((probs >= lo) & (probs <= hi)))
        rem = int(probs.size - low)
        spot = int(round(rem * spotcheck_frac))
        manual_total = low + spot
        manual_frac = manual_total / probs.size
        score = abs(manual_frac - target_manual_frac)
        cand = {
            "low_min": round(lo, 2),
            "low_max": round(hi, 2),
            "half_width": round(float(w), 2),
            "low_conf_count": low,
            "remainder_count": rem,
            "spotcheck_count": spot,
            "manual_total": manual_total,
            "manual_frac": manual_frac,
            "auto_total": rem,
            "auto_frac": rem / probs.size,
            "score_to_target": score,
        }
        if best is None or score < best["score_to_target"]:
            best = cand

    assert best is not None
    return best


def main() -> int:
    p = argparse.ArgumentParser(description="Recommend confidence thresholds from prediction histogram")
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--spotcheck-frac", type=float, default=0.05)
    p.add_argument("--target-manual-frac", type=float, default=0.10)
    p.add_argument(
        "--out-hist-csv",
        type=Path,
        default=Path("analysis/onboarding_new_laz/prediction_histogram.csv"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("analysis/onboarding_new_laz/prediction_confidence_recommendation.json"),
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("analysis/onboarding_new_laz/prediction_confidence_recommendation.md"),
    )
    args = p.parse_args()

    probs = _load_auto_probs(args.labels_dir)
    if probs.size == 0:
        raise RuntimeError(f"No auto model_prob values found in {args.labels_dir}")

    counts, edges = np.histogram(probs, bins=args.bins, range=(0.0, 1.0))

    args.out_hist_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_hist_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["bin_idx", "bin_left", "bin_right", "count", "pct"],
        )
        w.writeheader()
        total = probs.size
        for i, c in enumerate(counts):
            w.writerow(
                {
                    "bin_idx": i,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(c),
                    "pct": float(100.0 * c / total),
                }
            )

    rec = _recommend_band(probs, args.spotcheck_frac, args.target_manual_frac)
    out = {
        "n_auto_predictions": int(probs.size),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prob_mean": float(np.mean(probs)),
        "prob_p25": float(np.percentile(probs, 25)),
        "prob_p50": float(np.percentile(probs, 50)),
        "prob_p75": float(np.percentile(probs, 75)),
        "spotcheck_frac": args.spotcheck_frac,
        "target_manual_frac": args.target_manual_frac,
        "recommended": rec,
        "queue_command": (
            "python scripts/recalculate_manual_review_queue.py "
            f"--labels-dir {args.labels_dir} "
            "--out output/onboarding_labels_v2_drop13/manual_review_queue_pre_split.csv "
            f"--low-min {rec['low_min']:.2f} --low-max {rec['low_max']:.2f} "
            f"--spotcheck-frac {args.spotcheck_frac:.2f} --seed 2026"
        ),
    }

    args.out_json.write_text(json.dumps(out, indent=2))

    md = [
        "# Prediction Confidence Recommendation",
        "",
        f"- Auto predictions analyzed: {out['n_auto_predictions']:,}",
        f"- Probability mean: {out['prob_mean']:.4f}",
        f"- Median: {out['prob_p50']:.4f}",
        "",
        "## Recommended Confidence Band",
        "",
        f"- Low-confidence range: [{rec['low_min']:.2f}, {rec['low_max']:.2f}]",
        f"- Auto outside band: {rec['auto_total']:,} ({100.0 * rec['auto_frac']:.2f}%)",
        f"- Manual low-confidence: {rec['low_conf_count']:,}",
        f"- Manual spotcheck (5% of auto outside band): {rec['spotcheck_count']:,}",
        f"- Manual total: {rec['manual_total']:,} ({100.0 * rec['manual_frac']:.2f}%)",
        "",
        "## Queue Rebuild Command",
        "",
        "```bash",
        out["queue_command"],
        "```",
        "",
    ]
    args.out_md.write_text("\n".join(md))

    print(f"n_auto_predictions={out['n_auto_predictions']}")
    print(f"recommended_low_band=[{rec['low_min']:.2f},{rec['low_max']:.2f}]")
    print(f"manual_total={rec['manual_total']}")
    print(f"manual_frac={100.0 * rec['manual_frac']:.2f}%")
    print(f"hist_csv={args.out_hist_csv}")
    print(f"recommend_json={args.out_json}")
    print(f"recommend_md={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
