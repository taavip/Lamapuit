#!/usr/bin/env python3
"""
Non-destructive scanner to find high-confidence false predictions in the
training/label CSVs using existing `model_prob` values.

Produces two outputs in the output directory:
 - `falses_full.csv`: all mismatches where a model prediction disagrees with
    the recorded label and the model confidence exceeds `--threshold`.
 - `review_queue.csv`: prioritized subset for human review (by confidence).

This scanner does not load models; it relies on `model_prob` already present
in the CSVs (backfill model probabilities first if needed).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

_CSV_HEADER = [
    "raster",
    "row_off",
    "col_off",
    "chunk_size",
    "label",
    "source",
    "annotator",
    "model_prob",
    "timestamp",
]


def load_dedup_labels(labels_dir: Path) -> Dict[Tuple[str, int, int], Dict]:
    """Read all *_labels.csv and return last-row-wins dict keyed by (raster,row,col).

    Value is the full row dict (including model_prob as string if present).
    """
    out = {}
    for p in labels_dir.glob("*_labels.csv"):
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["raster"], int(r["row_off"]), int(r["col_off"]))
                out[key] = r
    return out


def parse_prob(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Find high-confidence false train predictions")
    p.add_argument("--labels-dir", required=True, help="Directory containing *_labels.csv")
    p.add_argument("--out-dir", default="output/tile_labels", help="Where to write outputs")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Model probability threshold for high-confidence mismatch (default: 0.9)",
    )
    p.add_argument(
        "--review-pct",
        type=float,
        default=0.05,
        help="Fraction of top mismatches to include in `review_queue.csv`",
    )
    args = p.parse_args()

    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dedup_labels(labels_dir)
    mismatches = []
    for (raster, row_off, col_off), row in rows.items():
        prob = parse_prob(row.get("model_prob", ""))
        label = row.get("label", "")
        if prob is None:
            continue
        pred = "cdw" if prob >= 0.5 else "no_cdw"
        # High-confidence disagreement
        if pred != label:
            # confidence as distance from 0.5
            conf = abs(prob - 0.5)
            # require model confidence beyond threshold margin
            if (pred == "cdw" and prob >= args.threshold) or (
                pred == "no_cdw" and prob <= (1.0 - args.threshold)
            ):
                mismatches.append(
                    {
                        "raster": raster,
                        "row_off": row_off,
                        "col_off": col_off,
                        "chunk_size": row.get("chunk_size", ""),
                        "label": label,
                        "predicted": pred,
                        "model_prob": f"{prob:.4f}",
                        "confidence": f"{conf:.4f}",
                        "error_type": "FP" if pred == "cdw" and label == "no_cdw" else "FN",
                        "source": row.get("source", ""),
                        "annotator": row.get("annotator", ""),
                        "timestamp": row.get("timestamp", ""),
                    }
                )

    # Write full mismatches file
    full_path = out_dir / "falses_full.csv"
    with open(full_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "raster",
                "row_off",
                "col_off",
                "chunk_size",
                "label",
                "predicted",
                "model_prob",
                "confidence",
                "error_type",
                "source",
                "annotator",
                "timestamp",
            ]
        )
        for m in mismatches:
            w.writerow(
                [
                    m["raster"],
                    m["row_off"],
                    m["col_off"],
                    m["chunk_size"],
                    m["label"],
                    m["predicted"],
                    m["model_prob"],
                    m["confidence"],
                    m["error_type"],
                    m["source"],
                    m["annotator"],
                    m["timestamp"],
                ]
            )

    # Create prioritized review queue (top fraction by confidence)
    if mismatches:
        mismatches.sort(key=lambda x: float(x["confidence"]), reverse=True)
        k = max(1, int(len(mismatches) * args.review_pct))
        queue = mismatches[:k]
    else:
        queue = []

    queue_path = out_dir / "review_queue.csv"
    with open(queue_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "raster",
                "row_off",
                "col_off",
                "chunk_size",
                "label",
                "predicted",
                "model_prob",
                "confidence",
                "error_type",
            ]
        )
        for m in queue:
            w.writerow(
                [
                    m["raster"],
                    m["row_off"],
                    m["col_off"],
                    m["chunk_size"],
                    m["label"],
                    m["predicted"],
                    m["model_prob"],
                    m["confidence"],
                    m["error_type"],
                ]
            )

    print(
        f"Wrote {full_path} ({len(mismatches)} mismatches) and {queue_path} ({len(queue)} review items)"
    )


if __name__ == "__main__":
    main()
