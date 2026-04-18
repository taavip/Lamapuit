#!/usr/bin/env python3
"""
Deep-audit collector.

Reads every *_labels.csv in a labels directory and assembles a prioritised
review queue covering:

  1. MISMATCH-FN  – label=cdw   but model says no_cdw  (any confidence)
  2. MISMATCH-FP  – label=no_cdw but model says cdw    (any confidence)
  3. UNCERTAIN    – model_prob in [uncertain_lo, uncertain_hi]
  4. CDW-VERIFY   – label=cdw, model agrees (--include-all-cdw flag)
  5. UNKNOWN      – label=unknown

Tiles that qualify for multiple categories are deduplicated; the highest
priority reason is kept.

Priority order (lower number → shown first in GUI):
  FN > FP > UNCERTAIN > UNKNOWN > CDW-VERIFY

Output: deep_audit_queue.csv  (columns: raster, row_off, col_off, chunk_size,
                                         label, model_prob, reason, priority)

Compatible with  label_tiles.py --tile-list deep_audit_queue.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

# ──────────────────────────────────────────────────────────────────────────────
_PRIORITY = {
    "mismatch_fn": 1,  # missed CDW – worst
    "mismatch_fp": 2,  # false alarm
    "uncertain": 3,  # model unsure
    "unknown": 4,  # human skipped
    "cdw_verify": 5,  # positive class spot-check
}


def _parse_prob(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_labels(labels_dir: Path) -> Dict[Tuple[str, int, int], dict]:
    """Return last-row-wins dict keyed by (raster, row_off, col_off)."""
    out: Dict[Tuple[str, int, int], dict] = {}
    for p in sorted(labels_dir.glob("*_labels.csv")):
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    key = (row["raster"], int(row["row_off"]), int(row["col_off"]))
                except (KeyError, ValueError):
                    continue
                out[key] = row
    return out


def collect_audit_tiles(
    labels: Dict[Tuple[str, int, int], dict],
    uncertain_lo: float,
    uncertain_hi: float,
    include_all_cdw: bool,
) -> list[dict]:
    """Categorise every tile and return a deduplicated, prioritised list."""
    # key → best (priority, entry)
    best: Dict[Tuple[str, int, int], tuple[int, dict]] = {}

    def _update(key: tuple, reason: str, prob: float | None, label: str, row: dict) -> None:
        prio = _PRIORITY[reason]
        if key not in best or prio < best[key][0]:
            best[key] = (
                prio,
                {
                    "raster": key[0],
                    "row_off": key[1],
                    "col_off": key[2],
                    "chunk_size": row.get("chunk_size", "128"),
                    "label": label,
                    "model_prob": f"{prob:.4f}" if prob is not None else "",
                    "reason": reason,
                    "priority": prio,
                },
            )

    for key, row in labels.items():
        label = row.get("label", "")
        prob = _parse_prob(row.get("model_prob", ""))

        # ── unknown ──────────────────────────────────────────────────────
        if label == "unknown":
            _update(key, "unknown", prob, label, row)
            continue  # nothing more to check without a real label

        # ── need model prob for remaining checks ─────────────────────────
        if prob is None:
            continue

        pred = "cdw" if prob >= 0.5 else "no_cdw"

        # ── mismatches ───────────────────────────────────────────────────
        if pred != label:
            reason = "mismatch_fn" if label == "cdw" else "mismatch_fp"
            _update(key, reason, prob, label, row)
            continue  # already worst category; no point checking further

        # ── uncertainty zone ─────────────────────────────────────────────
        if uncertain_lo <= prob <= uncertain_hi:
            _update(key, "uncertain", prob, label, row)
            # don't continue – may also be cdw_verify

        # ── positive class spot-check ────────────────────────────────────
        if include_all_cdw and label == "cdw":
            _update(key, "cdw_verify", prob, label, row)

    # Sort: priority ASC, then confidence (distance from 0.5) DESC
    items = [v for _, v in best.values()]
    items.sort(
        key=lambda x: (
            x["priority"],
            -abs(float(x["model_prob"]) - 0.5) if x["model_prob"] else 0.0,
        )
    )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a deep-audit review queue from existing label CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--labels-dir", default="output/tile_labels", help="Directory containing *_labels.csv"
    )
    ap.add_argument(
        "--out-dir", default="output/tile_labels", help="Where to write deep_audit_queue.csv"
    )
    ap.add_argument(
        "--uncertain-lo", type=float, default=0.30, help="Lower bound of model-uncertainty zone"
    )
    ap.add_argument(
        "--uncertain-hi", type=float, default=0.70, help="Upper bound of model-uncertainty zone"
    )
    ap.add_argument(
        "--include-all-cdw",
        action="store_true",
        help="Add all CDW-labeled tiles for positive-class verification",
    )
    ap.add_argument("--max-tiles", type=int, default=0, help="Cap on queue size (0 = no cap)")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading labels from {labels_dir} …")
    labels = load_labels(labels_dir)
    print(f"  {len(labels):,} unique tiles loaded")

    items = collect_audit_tiles(
        labels,
        uncertain_lo=args.uncertain_lo,
        uncertain_hi=args.uncertain_hi,
        include_all_cdw=args.include_all_cdw,
    )

    if args.max_tiles > 0:
        items = items[: args.max_tiles]

    # ── summary ──────────────────────────────────────────────────────────────
    from collections import Counter

    reason_counts = Counter(x["reason"] for x in items)
    raster_counts = Counter(x["raster"] for x in items)
    print(f"\n{'Reason':<20} {'Count':>6}")
    print("─" * 28)
    for reason in ["mismatch_fn", "mismatch_fp", "uncertain", "unknown", "cdw_verify"]:
        if reason_counts[reason]:
            print(f"  {reason:<18} {reason_counts[reason]:>6,}")
    print("─" * 28)
    print(f"  {'TOTAL':<18} {len(items):>6,}")
    print(f"\nRasters covered: {len(raster_counts)}")
    for raster in sorted(raster_counts):
        print(f"  {raster_counts[raster]:4,}  {raster}")

    # ── write output ─────────────────────────────────────────────────────────
    out_path = out_dir / "deep_audit_queue.csv"
    fieldnames = [
        "raster",
        "row_off",
        "col_off",
        "chunk_size",
        "label",
        "model_prob",
        "reason",
        "priority",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)

    print(f"\nWrote  {out_path}  ({len(items):,} tiles)")
    print(f"\nTo re-label, run:")
    print(f"  python scripts/label_all_rasters.py --tile-list {out_path}")
    print(f"  # or target a single raster:")
    print(
        f"  python scripts/label_tiles.py --chm <path>.tif "
        f"--output {out_dir} --resume --tile-list {out_path}"
    )


if __name__ == "__main__":
    main()
