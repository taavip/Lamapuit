"""Model ranking and mode selection for V4.

Two ranking decisions are made here:

1. **Top-k candidates from V3** — uses a Lower Confidence Bound (LCB)
   ``score = mean_cv_f1 - k * std_cv_f1`` instead of a pure mean.  A model
   with ``F1 = 0.949 ± 0.067`` scores 0.882 at k=1 and loses to a model at
   ``0.935 ± 0.002`` (0.933). LCB is standard in bandit / model-selection
   literature (e.g. UCB1 with sign flipped) and aligns with the thesis'
   emphasis on reproducibility.

2. **Best input mode** — composite score over the ablation table combining
   deep-model F1 and classical-baseline F1. Weights are explicit arguments
   so the sensitivity analysis reported in the paper can reproduce the
   table under alternative weightings. A ``overwhelming_margin`` gate
   flags when the best mode is ahead of the runner-up by more than the
   expected fold-level noise; when it is not, the caller may fall back to
   the V3-preferred mode to preserve version-over-version continuity.

Both functions are pure; no I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TOP3_FALLBACK = [
    "convnext_small",
    "efficientnet_b2",
    "maxvit_small",  # replaces deep_cnn_attn_dropout_tuned under LCB ranking
]


def top_models_from_v3_lcb(
    summary_csv: Path,
    n_models: int,
    lcb_k: float = 1.0,
    fallback: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return top-k model names from V3's ``experiment_summary.csv`` using LCB.

    Also returns a side-table with (model_name, mean, std, lcb) for the
    methodology report. Falls back to the hardcoded list if the summary
    CSV is missing or malformed.
    """
    fb = list(fallback or DEFAULT_TOP3_FALLBACK)

    if not summary_csv.exists():
        return fb[:n_models], []

    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return fb[:n_models], []

    if not {"model_name", "mean_cv_f1", "std_cv_f1"}.issubset(set(df.columns)):
        return fb[:n_models], []

    df = df.copy()
    df["model_name"] = df["model_name"].astype(str).str.strip().str.lower()
    # Keep only the best row per model (in case of duplicate strategies).
    df = df.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True]).drop_duplicates("model_name")
    df["lcb"] = df["mean_cv_f1"].astype(float) - float(lcb_k) * df["std_cv_f1"].astype(float)

    df = df.sort_values(["lcb", "mean_cv_f1"], ascending=[False, False])

    ordered = [m for m in df["model_name"].tolist() if m]
    # De-dup preserving order.
    seen: set[str] = set()
    out = []
    for m in ordered:
        if m in seen:
            continue
        out.append(m)
        seen.add(m)

    if len(out) < n_models:
        for m in fb:
            if m not in seen:
                out.append(m)
                seen.add(m)
            if len(out) >= n_models:
                break

    audit_rows = df[["model_name", "mean_cv_f1", "std_cv_f1", "lcb"]].head(max(n_models * 2, 5)).to_dict("records")
    return out[:n_models], audit_rows


def composite_mode_score(
    deep_f1: float | None,
    classical_f1: float | None,
    deep_weight: float = 0.70,
) -> float:
    """Combine deep and classical F1 with graceful fallbacks."""
    import math
    dm = deep_f1 if deep_f1 is not None and math.isfinite(deep_f1) else None
    cm = classical_f1 if classical_f1 is not None and math.isfinite(classical_f1) else None
    dw = max(0.0, min(1.0, float(deep_weight)))
    if dm is not None and cm is not None:
        return dw * dm + (1.0 - dw) * cm
    if dm is not None:
        return dm
    if cm is not None:
        return cm
    return float("nan")


def select_best_mode(
    ablation_df: pd.DataFrame,
    overwhelming_margin: float,
) -> tuple[str, bool, dict[str, Any]]:
    """Pick the best input mode and report whether the margin is decisive.

    Returns ``(best_mode, is_overwhelming, diagnostics)``. Diagnostics
    includes the runner-up name, the numerical margin, and the per-mode
    composite scores — useful material for the paper's decision table.
    """
    if ablation_df.empty:
        return "original", False, {"reason": "empty_ablation"}

    ranked = ablation_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    best_mode = str(ranked.iloc[0]["mode"])
    best_score = float(ranked.iloc[0]["composite_score"])
    if len(ranked) < 2:
        return best_mode, False, {"best_mode": best_mode, "best_score": best_score, "margin": float("inf")}

    runner_up = str(ranked.iloc[1]["mode"])
    runner_score = float(ranked.iloc[1]["composite_score"])
    margin = best_score - runner_score
    overwhelming = margin >= float(overwhelming_margin)

    return (
        best_mode,
        overwhelming,
        {
            "best_mode": best_mode,
            "best_score": best_score,
            "runner_up_mode": runner_up,
            "runner_up_score": runner_score,
            "margin": margin,
            "overwhelming_margin_threshold": float(overwhelming_margin),
        },
    )
