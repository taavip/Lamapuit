#!/usr/bin/env python3
"""Evaluate multiple trained tile models on manual labels for tile 436646.

Compares baseline CHM against CHMs generated from reclassified LAZ variants.
Outputs per-model/per-CHM metrics and a concise improvement summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class LabelRow:
    row_off: int
    col_off: int
    chunk_size: int
    y: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate multiple trained models on manual/reviewed tiles for "
            "baseline vs reclassified CHMs (tile 436646_2018_madal)."
        )
    )
    p.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("output/onboarding_labels_v2_drop13/436646_2018_madal_chm_max_hag_20cm_labels.csv"),
    )
    p.add_argument(
        "--baseline-chm",
        type=Path,
        default=Path("data/lamapuit/chm_max_hag_13_drop/436646_2018_madal_chm_max_hag_20cm.tif"),
    )
    p.add_argument(
        "--new-chm-root",
        type=Path,
        default=Path("data/lamapuit/chm_reclassified_drop13"),
    )
    p.add_argument(
        "--model-paths",
        default=(
            "output/tile_labels/ensemble_model.pt,"
            "output/tile_labels/cnn_seed42.pt,"
            "output/tile_labels/cnn_seed43.pt,"
            "output/tile_labels/cnn_seed44.pt,"
            "output/tile_labels/effnet_b2.pt"
        ),
        help="Comma-separated checkpoint paths.",
    )
    p.add_argument(
        "--sources",
        default="manual,reviewed",
        help="Comma-separated label sources to include.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("analysis/onboarding_new_laz/reclassified_chm_model_eval_436646.csv"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("analysis/onboarding_new_laz/reclassified_chm_model_eval_436646.json"),
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("analysis/onboarding_new_laz/reclassified_chm_model_eval_436646.md"),
    )
    return p.parse_args()


def _parse_sources(raw: str) -> set[str]:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def load_labels(labels_csv: Path, sources: set[str]) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with labels_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row.get("source", "")).strip().lower()
            if sources and src not in sources:
                continue
            label = str(row.get("label", "")).strip().lower()
            if label == "cdw":
                y = 1
            elif label == "no_cdw":
                y = 0
            else:
                continue
            try:
                rows.append(
                    LabelRow(
                        row_off=int(row["row_off"]),
                        col_off=int(row["col_off"]),
                        chunk_size=int(row.get("chunk_size") or 128),
                        y=y,
                    )
                )
            except Exception:
                continue
    return rows


def load_cnn_predictor_class() -> Any:
    mod_path = Path("scripts/label_tiles.py")
    spec = importlib.util.spec_from_file_location("label_tiles_mod", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CNNPredictor


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    # Rank-based AUC to avoid hard dependency differences.
    y = y_true.astype(np.int32)
    s = y_prob.astype(np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_pos = float(ranks[pos].sum())
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    n = int(y_true.size)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    auc = auc_score(y_true, y_prob)

    return {
        "n_tiles": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
    }


def _extract_tiles(chm_path: Path, label_rows: list[LabelRow]) -> np.ndarray:
    tiles: list[np.ndarray] = []
    with rasterio.open(chm_path) as src:
        for r in label_rows:
            arr = src.read(
                1,
                window=Window(r.col_off, r.row_off, r.chunk_size, r.chunk_size),
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
            tiles.append(arr)
    return np.stack(tiles, axis=0)


def _collect_chms(baseline_chm: Path, new_chm_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = [("baseline", baseline_chm)]
    for p in sorted(new_chm_root.rglob("*.tif")):
        rel = p.relative_to(new_chm_root)
        group = rel.parts[0] if len(rel.parts) > 1 else "new"
        out.append((group, p))
    return out


def _model_id(path: Path) -> str:
    return path.stem


def write_markdown(
    md_path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    labels_csv: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Reclassified LAS->CHM Detection Benchmark (436646_2018_madal)")
    lines.append("")
    lines.append(f"- Labels: {labels_csv}")
    lines.append("- Sources included: manual/reviewed")
    lines.append("- Metric focus: recall (CDW detection rate), plus F1/precision/AUC")
    lines.append("")
    lines.append("## Improvement vs Baseline (Per Model)")
    lines.append("")
    lines.append("| model | baseline_recall | best_new_recall | recall_delta | baseline_f1 | best_new_f1 | best_new_chm |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for s in summary_rows:
        lines.append(
            "| {model} | {baseline_recall:.4f} | {best_new_recall:.4f} | {recall_delta:+.4f} | {baseline_f1:.4f} | {best_new_f1:.4f} | {best_new_chm} |".format(
                **s
            )
        )

    lines.append("")
    lines.append("## Full Metrics")
    lines.append("")
    lines.append("| model | chm_group | chm | thr | n | acc | prec | recall | f1 | auc | tp | fp | tn | fn |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        auc_val = "" if r["auc"] is None else f"{r['auc']:.4f}"
        lines.append(
            f"| {r['model']} | {r['chm_group']} | {Path(r['chm_path']).name} | {r['threshold']:.2f} | "
            f"{r['n_tiles']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {auc_val} | "
            f"{r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |"
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    sources = _parse_sources(args.sources)
    label_rows = load_labels(args.labels_csv, sources)
    if not label_rows:
        raise RuntimeError(f"No usable manual/reviewed labels found in {args.labels_csv}")

    y_true = np.array([r.y for r in label_rows], dtype=np.int32)

    CNNPredictor = load_cnn_predictor_class()
    model_paths = [Path(x.strip()) for x in args.model_paths.split(",") if x.strip()]

    predictors: list[tuple[str, Any, float]] = []
    for mp in model_paths:
        pred = CNNPredictor()
        ok = pred.load_from_disk(mp)
        if not ok:
            print(f"[skip] failed loading model: {mp}")
            continue
        thr = float(getattr(pred, "_thresh", 0.5))
        predictors.append((_model_id(mp), pred, thr))

    if not predictors:
        raise RuntimeError("No models loaded successfully.")

    chms = _collect_chms(args.baseline_chm, args.new_chm_root)
    if len(chms) <= 1:
        raise RuntimeError(f"No new CHM files found under {args.new_chm_root}")

    rows: list[dict[str, Any]] = []

    for group, chm_path in chms:
        tiles = _extract_tiles(chm_path, label_rows)
        for model_name, pred, thr in predictors:
            probs = []
            for tile in tiles:
                p = pred.predict_proba_cdw(tile)
                probs.append(0.5 if p is None else float(p))
            y_prob = np.array(probs, dtype=np.float64)
            m = compute_metrics(y_true, y_prob, thr)
            row = {
                "model": model_name,
                "chm_group": group,
                "chm_path": str(chm_path),
                "threshold": thr,
            }
            row.update(m)
            rows.append(row)
            print(
                f"[{model_name}] {chm_path.name} "
                f"recall={m['recall']:.4f} f1={m['f1']:.4f} precision={m['precision']:.4f}"
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "chm_group",
                "chm_path",
                "threshold",
                "n_tiles",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "tp",
                "fp",
                "tn",
                "fn",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    for model, rws in sorted(by_model.items()):
        baseline = next((x for x in rws if x["chm_group"] == "baseline"), None)
        if baseline is None:
            continue
        new_rws = [x for x in rws if x["chm_group"] != "baseline"]
        if not new_rws:
            continue
        best_new = max(new_rws, key=lambda x: (x["recall"], x["f1"]))
        summary_rows.append(
            {
                "model": model,
                "baseline_recall": float(baseline["recall"]),
                "best_new_recall": float(best_new["recall"]),
                "recall_delta": float(best_new["recall"] - baseline["recall"]),
                "baseline_f1": float(baseline["f1"]),
                "best_new_f1": float(best_new["f1"]),
                "best_new_chm": Path(best_new["chm_path"]).name,
                "baseline_chm": Path(baseline["chm_path"]).name,
            }
        )

    payload = {
        "labels_csv": str(args.labels_csv),
        "sources": sorted(sources),
        "n_manual_tiles": int(len(label_rows)),
        "models_evaluated": [x[0] for x in predictors],
        "n_chm_files": int(len(chms)),
        "summary": summary_rows,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_markdown(args.out_md, rows, summary_rows, args.labels_csv)

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote JSON: {args.out_json}")
    print(f"Wrote MD: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
