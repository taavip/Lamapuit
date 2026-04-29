"""Post-hoc audit of test-set predictions by provenance.

The single most important number for the paper is **manual-only test F1**:
the score restricted to the ~2% of test rows whose labels were produced or
reviewed by a human. If ``manual_only_f1`` is substantially below
``combined_f1`` we have evidence that the model is learning pseudo-label
patterns rather than the underlying CWD signal.

Also computes per-place and per-year F1 to detect spatial or temporal
instability that the aggregate number would hide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ._labels import is_manual_source, parse_raster_identity


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "auc": _safe_auc(y, probs),
    }


def breakdown_test_metrics(
    y_test: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    meta_test: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute test F1 broken down by provenance, place, and year.

    ``meta_test`` is expected to have ``source`` and ``raster`` fields (as
    produced by ``_build_arrays_with_mode``).
    """
    pred = (probs >= float(threshold)).astype(int)

    manual_mask = np.array([is_manual_source(m.get("source", "")) for m in meta_test], dtype=bool)
    gate_mask = np.array(
        [str(m.get("source", "")).strip().lower() == "auto_threshold_gate_v4" for m in meta_test],
        dtype=bool,
    )

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "combined": _metrics(y_test, pred, probs),
        "manual_only": _metrics(y_test[manual_mask], pred[manual_mask], probs[manual_mask])
        if manual_mask.any()
        else None,
        "threshold_gate_only": _metrics(y_test[gate_mask], pred[gate_mask], probs[gate_mask])
        if gate_mask.any()
        else None,
    }

    # Per-year breakdown (helps detect "train 2018, test 2022" drift).
    years: dict[str, list[int]] = {}
    for i, m in enumerate(meta_test):
        ident = parse_raster_identity(m.get("raster", ""))
        years.setdefault(str(ident["year"]), []).append(i)
    by_year = {}
    for year, idxs in years.items():
        ii = np.asarray(idxs, dtype=int)
        by_year[year] = _metrics(y_test[ii], pred[ii], probs[ii])
    out["per_year"] = by_year

    # Per-place breakdown, keyed by the year-agnostic place_key.
    places: dict[str, list[int]] = {}
    for i, m in enumerate(meta_test):
        ident = parse_raster_identity(m.get("raster", ""))
        places.setdefault(str(ident["place_key"]), []).append(i)
    by_place = {}
    for pk, idxs in places.items():
        ii = np.asarray(idxs, dtype=int)
        by_place[pk] = _metrics(y_test[ii], pred[ii], probs[ii])
    # Summarize places with a numerical spread rather than dumping them all.
    f1s = [v["f1"] for v in by_place.values() if v["n"] >= 10]
    out["per_place_summary"] = {
        "n_places": len(by_place),
        "n_places_min10_rows": len(f1s),
        "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "f1_std": float(np.std(f1s)) if f1s else float("nan"),
        "f1_min": float(np.min(f1s)) if f1s else float("nan"),
        "f1_max": float(np.max(f1s)) if f1s else float("nan"),
    }

    return out
