#!/usr/bin/env python3
"""Fast evaluation over already-generated CHM outputs for tile 436646.

This script evaluates baseline and research CHM variants using a sampled subset
of labeled tiles per year to produce a quick recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import rankdata


def _threshold_metrics(feat: pd.DataFrame, feature: str) -> dict:
    y = feat["y"].to_numpy()
    s = feat[feature].to_numpy()
    if s.size == 0:
        return {}

    thresholds = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 120)))
    best = None
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tpr
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        j = tpr - fpr

        item = {
            "threshold": float(thr),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "precision": float(prec),
            "f1": float(f1),
            "youden_j": float(j),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if best is None or item["youden_j"] > best["youden_j"]:
            best = item

    return best or {}


def _auc_score(y: np.ndarray, s: np.ndarray) -> float | None:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s)
    rank_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    na = int(a.size)
    nb = int(b.size)
    pooled = ((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1)
    if pooled <= 1e-12:
        return None
    return float((float(np.mean(a)) - float(np.mean(b))) / math.sqrt(pooled))


def _eval_from_features(feats: pd.DataFrame) -> dict:
    cdw = feats[feats["label"] == "cdw"]
    no = feats[feats["label"] == "no_cdw"]
    y = feats["y"].to_numpy()

    return {
        "n_tiles": int(len(feats)),
        "n_cdw": int(len(cdw)),
        "n_no_cdw": int(len(no)),
        "cdw_tile_max_mean": float(cdw["tile_max"].mean()) if len(cdw) else None,
        "no_tile_max_mean": float(no["tile_max"].mean()) if len(no) else None,
        "cdw_detect_rate_15cm": float((cdw["tile_max"] >= 0.15).mean()) if len(cdw) else None,
        "no_false_high_rate_15cm": float((no["tile_max"] >= 0.15).mean()) if len(no) else None,
        "best_youden_tile_max": _threshold_metrics(feats, "tile_max"),
        "best_youden_frac_above_15cm": _threshold_metrics(feats, "tile_frac_above_15cm"),
        "auc_tile_max": _auc_score(y, feats["tile_max"].to_numpy()),
        "auc_frac_above_15cm": _auc_score(y, feats["tile_frac_above_15cm"].to_numpy()),
        "cohens_d_tile_max": _cohens_d(cdw["tile_max"].to_numpy(), no["tile_max"].to_numpy()),
    }


def _sampled_tile_features(
    chm_path: Path,
    labels_csv: Path,
    max_tiles: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df[df["label"].isin(["cdw", "no_cdw"])].copy()
    if df.empty:
        raise RuntimeError(f"No binary labels in {labels_csv}")

    if len(df) > max_tiles:
        df = df.sample(n=max_tiles, random_state=seed).copy()

    with rasterio.open(chm_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    feats = []
    h, w = arr.shape
    for row in df.itertuples(index=False):
        rr = int(row.row_off)
        cc = int(row.col_off)
        cs = int(row.chunk_size)
        r2 = min(rr + cs, h)
        c2 = min(cc + cs, w)
        tile = arr[rr:r2, cc:c2].astype(np.float32)
        if nodata is not None:
            tile = tile[tile != nodata]

        if tile.size == 0:
            tmax = 0.0
            tmean = 0.0
            tabove = 0.0
        else:
            tile = np.clip(tile, 0.0, None)
            tmax = float(np.max(tile))
            tmean = float(np.mean(tile))
            tabove = float(np.mean(tile >= 0.15))

        feats.append((row.label, tmax, tmean, tabove))

    out = pd.DataFrame(feats, columns=["label", "tile_max", "tile_mean", "tile_frac_above_15cm"])
    out["y"] = (out["label"] == "cdw").astype(int)
    return out


def _choose_best_method(results: Dict[str, dict]) -> str:
    def score(name: str) -> float:
        r = results[name]
        j1 = float(r.get("best_youden_tile_max", {}).get("youden_j", 0.0))
        j2 = float(r.get("best_youden_frac_above_15cm", {}).get("youden_j", 0.0))
        cdw = float(r.get("cdw_detect_rate_15cm", 0.0) or 0.0)
        no_fp = float(r.get("no_false_high_rate_15cm", 1.0) or 1.0)
        auc = float(r.get("auc_tile_max", 0.0) or 0.0)
        return 0.35 * j1 + 0.2 * j2 + 0.2 * cdw + 0.15 * (1.0 - no_fp) + 0.1 * auc

    return max(results.keys(), key=score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick evaluation for existing 436646 experiment outputs")
    ap.add_argument("--tile-id", default="436646")
    ap.add_argument("--years", default="2018,2020,2022,2024")
    ap.add_argument("--labels-dir", type=Path, default=Path("output/onboarding_labels_v2_drop13"))
    ap.add_argument("--baseline-chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    ap.add_argument("--results-dir", type=Path, default=Path("experiments/dtm_hag_436646_research/results"))
    ap.add_argument("--max-tiles-per-year", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20260411)
    args = ap.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]

    method_templates = {
        "idw_k12_raw": "{year}_idw_k12_raw_chm.tif",
        "idw_k12_gauss": "{year}_idw_k12_gauss_chm.tif",
        "tin_linear_raw": "{year}_tin_linear_raw_chm.tif",
        "tin_linear_gauss": "{year}_tin_linear_gauss_chm.tif",
        "tps_raw": "{year}_tps_raw_chm.tif",
        "tps_gauss": "{year}_tps_gauss_chm.tif",
    }

    method_to_paths: Dict[str, Dict[int, Path]] = {}

    # Baseline
    method_to_paths["baseline_idw3_drop13"] = {
        y: args.baseline_chm_dir / f"{args.tile_id}_{y}_madal_chm_max_hag_20cm.tif" for y in years
    }

    # Generated methods
    for m, template in method_templates.items():
        method_to_paths[m] = {
            y: args.results_dir / "chm" / str(y) / template.format(year=y) for y in years
        }

    # Verify files
    for method, by_year in method_to_paths.items():
        for y, p in by_year.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing CHM for {method} {y}: {p}")

    evaluation_per_year: Dict[str, Dict[str, dict]] = {}
    evaluation_aggregate: Dict[str, dict] = {}

    for method, by_year in method_to_paths.items():
        per_year = {}
        all_feats = []

        for y in years:
            labels = args.labels_dir / f"{args.tile_id}_{y}_madal_chm_max_hag_20cm_labels.csv"
            if not labels.exists():
                raise FileNotFoundError(f"Missing labels CSV: {labels}")

            feats = _sampled_tile_features(
                chm_path=by_year[y],
                labels_csv=labels,
                max_tiles=args.max_tiles_per_year,
                seed=args.seed + y,
            )
            per_year[str(y)] = _eval_from_features(feats)
            all_feats.append(feats)

        evaluation_per_year[method] = per_year
        evaluation_aggregate[method] = _eval_from_features(pd.concat(all_feats, ignore_index=True))

    best_method = _choose_best_method(evaluation_aggregate)

    rows = []
    for method, ev in evaluation_aggregate.items():
        b1 = ev.get("best_youden_tile_max", {})
        b2 = ev.get("best_youden_frac_above_15cm", {})
        rows.append(
            {
                "method": method,
                "auc_tile_max": ev.get("auc_tile_max"),
                "auc_frac_above_15cm": ev.get("auc_frac_above_15cm"),
                "j_tile_max": b1.get("youden_j"),
                "thr_tile_max": b1.get("threshold"),
                "j_frac15": b2.get("youden_j"),
                "thr_frac15": b2.get("threshold"),
                "cdw_detect_rate_15cm": ev.get("cdw_detect_rate_15cm"),
                "no_false_high_rate_15cm": ev.get("no_false_high_rate_15cm"),
                "cohens_d_tile_max": ev.get("cohens_d_tile_max"),
            }
        )

    eval_dir = args.results_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "method_summary.csv"
    pd.DataFrame(rows).sort_values("j_tile_max", ascending=False).to_csv(csv_path, index=False)

    payload = {
        "tile_id": args.tile_id,
        "years": years,
        "max_tiles_per_year": args.max_tiles_per_year,
        "evaluation_per_year": evaluation_per_year,
        "evaluation_aggregate": evaluation_aggregate,
        "best_method": best_method,
        "summary_csv": str(csv_path),
    }

    json_path = eval_dir / "quick_eval_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = eval_dir / "quick_eval_report.md"
    lines: List[str] = []
    lines.append("# Quick Evaluation Report (Existing Outputs)")
    lines.append("")
    lines.append(f"- Tile: {args.tile_id}")
    lines.append(f"- Years: {years}")
    lines.append(f"- Max sampled tiles per year: {args.max_tiles_per_year}")
    lines.append("")
    lines.append("## Aggregate Metrics")
    for method, ev in evaluation_aggregate.items():
        b1 = ev.get("best_youden_tile_max", {})
        lines.append(
            f"- {method}: AUC={ev.get('auc_tile_max')}, J={b1.get('youden_j')} @thr={b1.get('threshold')}, "
            f"CDW>=15cm={ev.get('cdw_detect_rate_15cm')}, NoCDW>=15cm={ev.get('no_false_high_rate_15cm')}"
        )
    lines.append("")
    lines.append(f"## Recommended Method\n**{best_method}**")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"best_method={best_method}")
    print(f"summary_csv={csv_path}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
