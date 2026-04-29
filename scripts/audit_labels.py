#!/usr/bin/env python3
"""
Label-quality audit tool — 5-check priority pipeline.

Runs five independent quality checks, then merges all flagged tiles into a
unified, priority-sorted review queue (audit_review_queue.csv) that feeds
directly into the label_tiles.py GUI via --tile-list.

Checks (highest to lowest priority)
--------------------------------------
1. Label–model disagreement      (CRITICAL)
   Tiles where the saved CNN model_prob strongly disagrees with the human label.
   |model_prob - label_numeric| > --disagreement-thresh  (default 0.7)
   These are the most likely mislabels.

2. Spatial outlier isolation      (HIGH)
   CDW tiles having 0 CDW neighbours within 1-stride radius.
   A solo CDW island is a strong mislabel indicator.

3. Overlap conflict detection     (MEDIUM)
   Adjacent tiles (overlapping pixel footprints) with contradicting labels.
   Uses a hash-grid for O(1) neighbour lookup (replaces the old O(n×chunk) loop).

4. Borderline / high-entropy tiles  (MEDIUM)
   Tiles where |model_prob - 0.5| < --margin.
   These are hardest to label reliably.

5. Stratified random spot-check    (LOW)
   --audit-pct % per raster+class stratum (minimum 1 per stratum).

Output files (written to --output-dir)
---------------------------------------
  audit_review_queue.csv    Unified priority-sorted queue for GUI re-review
  audit_disagree.csv        Check-1 disagreements
  audit_isolated.csv        Check-2 isolated CDW tiles
  audit_conflicts.csv       Check-3 overlapping contradictions (pairs)
  audit_borderline.csv      Check-4 high-entropy / borderline tiles
  audit_sample.csv          Check-5 random spot-check sample
  audit_meta.json           Run statistics

Usage
-----
# Standard run after backfill_model_prob.py:
python scripts/audit_labels.py \\
    --labels output/tile_labels \\
    --output-dir output/audits

# Use generated queue for GUI re-labeling:
python scripts/label_all_rasters.py \\
    --chm-dir chm_max_hag --output output/tile_labels \\
    --tile-list output/audits/audit_review_queue.csv --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

# ── Helpers ───────────────────────────────────────────────────────────────────


def _entropy(p: float) -> float:
    """Binary entropy H(p). Maximum 1.0 at p=0.5."""
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _label_num(label: str) -> float:
    """Convert label string to 0/1 float."""
    return 1.0 if label == "cdw" else 0.0


# ── Data loading ──────────────────────────────────────────────────────────────


def load_labels(labels_dir: Path) -> dict[str, list[dict]]:
    """Return {raster_name: [row_dict, ...]} for all CDW/no_cdw rows.

    Deduplicates by (row_off, col_off) using last-row-wins semantics so that
    re-labeled tiles (which have two rows in the CSV after an audit session)
    appear only once with their latest label.
    """
    raster_rows: dict[str, list[dict]] = {}
    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        # Last-row-wins deduplication
        seen: dict[tuple, dict] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label", "") in ("cdw", "no_cdw"):
                    key = (row.get("row_off", ""), row.get("col_off", ""))
                    seen[key] = dict(row)
        if seen:
            rows = list(seen.values())
            raster = rows[0]["raster"]
            raster_rows[raster] = rows
    return raster_rows


# ── Check 1: Label–model disagreement ─────────────────────────────────────────


def check_disagreements(
    raster_rows: dict[str, list[dict]],
    thresh: float = 0.70,
) -> list[dict]:
    """Find tiles where CNN model_prob strongly disagrees with the label.

    A disagreement score is |model_prob – label_numeric|.
    Tiles with score > thresh are returned, sorted by descending score.
    Tiles lacking model_prob are skipped (run backfill_model_prob.py first).
    """
    results: list[dict] = []
    for raster, rows in raster_rows.items():
        for row in rows:
            prob_str = row.get("model_prob", "")
            if not prob_str:
                continue
            try:
                prob = float(prob_str)
            except ValueError:
                continue
            lbl = row["label"]
            lnum = _label_num(lbl)
            score = abs(prob - lnum)
            if score > thresh:
                results.append(
                    {
                        "raster": raster,
                        "row_off": row["row_off"],
                        "col_off": row["col_off"],
                        "chunk_size": row.get("chunk_size", "128"),
                        "current_label": lbl,
                        "model_prob": prob_str,
                        "disagree_score": f"{score:.4f}",
                        "source": row.get("source", ""),
                        "annotator": row.get("annotator", ""),
                        "timestamp": row.get("timestamp", ""),
                        "audit_label": "",
                        "auditor": "",
                        "audit_note": "",
                    }
                )
    results.sort(key=lambda r: -float(r["disagree_score"]))
    return results


# ── Check 2: Spatial outlier isolation ────────────────────────────────────────


def check_isolated_cdw(
    raster_rows: dict[str, list[dict]],
    stride: int = 64,
    min_neighbours: int = 1,
) -> list[dict]:
    """Find CDW tiles that have zero CDW neighbours within one stride radius.

    Uses a hash-grid bucketed at (row//stride, col//stride) for O(1) lookups.
    """
    results: list[dict] = []

    for raster, rows in raster_rows.items():
        tile_label: dict[tuple[int, int], str] = {
            (int(r["row_off"]), int(r["col_off"])): r["label"] for r in rows
        }
        # grid: (bucket_r, bucket_c) → set of (row_off, col_off) keys
        grid: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for ro, co in tile_label:
            grid[(ro // stride, co // stride)].add((ro, co))

        for row in rows:
            if row["label"] != "cdw":
                continue
            ro = int(row["row_off"])
            co = int(row["col_off"])
            br = ro // stride
            bc = co // stride

            cdw_neighbours = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    for nro, nco in grid.get((br + dr, bc + dc), set()):
                        if tile_label.get((nro, nco)) == "cdw":
                            cdw_neighbours += 1

            if cdw_neighbours < min_neighbours:
                prob_str = row.get("model_prob", "")
                results.append(
                    {
                        "raster": raster,
                        "row_off": ro,
                        "col_off": co,
                        "chunk_size": row.get("chunk_size", "128"),
                        "current_label": "cdw",
                        "cdw_neighbours": cdw_neighbours,
                        "model_prob": prob_str,
                        "source": row.get("source", ""),
                        "annotator": row.get("annotator", ""),
                        "timestamp": row.get("timestamp", ""),
                        "audit_label": "",
                        "auditor": "",
                        "audit_note": "",
                    }
                )

    # Sort by ascending model_prob so most-suspect tiles come first
    results.sort(key=lambda r: float(r.get("model_prob", "0.5") or "0.5"))
    return results


# ── Check 3: Overlap conflict detection ───────────────────────────────────────


def check_conflicts(
    raster_rows: dict[str, list[dict]],
    chunk_size: int = 128,
    stride: int | None = None,
) -> list[dict]:
    """Find pairs of overlapping tiles with contradictory labels.

    Two tiles overlap when |r1−r2| < chunk AND |c1−c2| < chunk.

    Uses a hash-grid bucketed at chunk resolution for O(1) neighbour lookup
    (fixes the previous O(n×chunk) inner loop).
    """
    if stride is None:
        stride = chunk_size // 2

    conflicts: list[dict] = []

    for raster, rows in raster_rows.items():
        chunk = int(rows[0].get("chunk_size", chunk_size)) if rows else chunk_size

        tile_label: dict[tuple[int, int], str] = {
            (int(r["row_off"]), int(r["col_off"])): r["label"] for r in rows
        }
        grid: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for ro, co in tile_label:
            grid[(ro // chunk, co // chunk)].append((ro, co))

        checked: set[tuple] = set()
        for (r1, c1), lbl1 in tile_label.items():
            gr1, gc1 = r1 // chunk, c1 // chunk
            for dgr in (-1, 0, 1):
                for dgc in (-1, 0, 1):
                    for r2, c2 in grid.get((gr1 + dgr, gc1 + dgc), []):
                        if (r1, c1) >= (r2, c2):
                            continue
                        if abs(r1 - r2) >= chunk or abs(c1 - c2) >= chunk:
                            continue
                        key = ((r1, c1), (r2, c2))
                        if key in checked:
                            continue
                        checked.add(key)
                        lbl2 = tile_label.get((r2, c2))
                        if lbl2 and lbl2 != lbl1:
                            conflicts.append(
                                {
                                    "raster": raster,
                                    "row_a": r1,
                                    "col_a": c1,
                                    "label_a": lbl1,
                                    "row_b": r2,
                                    "col_b": c2,
                                    "label_b": lbl2,
                                }
                            )
    return conflicts


# ── Check 4: Borderline / high-entropy tiles ──────────────────────────────────


def check_borderline(
    raster_rows: dict[str, list[dict]],
    margin: float = 0.10,
) -> list[dict]:
    """Return tiles where CNN model_prob is within *margin* of 0.5.

    Requires the model_prob column — run backfill_model_prob.py first.
    """
    results: list[dict] = []
    for raster, rows in raster_rows.items():
        for row in rows:
            prob_str = row.get("model_prob", "")
            if not prob_str:
                continue
            try:
                prob = float(prob_str)
                ent = _entropy(prob)
            except ValueError:
                continue
            if abs(prob - 0.5) < margin:
                results.append(
                    {
                        "raster": raster,
                        "row_off": row["row_off"],
                        "col_off": row["col_off"],
                        "chunk_size": row.get("chunk_size", "128"),
                        "current_label": row["label"],
                        "model_prob": prob_str,
                        "entropy": f"{ent:.4f}",
                        "source": row.get("source", ""),
                        "annotator": row.get("annotator", ""),
                        "timestamp": row.get("timestamp", ""),
                        "audit_label": "",
                        "auditor": "",
                        "audit_note": "",
                    }
                )
    results.sort(key=lambda r: -float(r["entropy"]))
    return results


# ── Check 5: Stratified random spot-check ────────────────────────────────────


def check_random_sample(
    raster_rows: dict[str, list[dict]],
    audit_pct: float = 2.0,
    seed: int = 42,
) -> list[dict]:
    """Stratified random sample: pct% of tiles per (raster, class) stratum."""
    rng = random.Random(seed)
    sample: list[dict] = []

    for raster, rows in raster_rows.items():
        cdw_rows = [r for r in rows if r["label"] == "cdw"]
        no_rows = [r for r in rows if r["label"] == "no_cdw"]

        for class_rows, cls in ((cdw_rows, "cdw"), (no_rows, "no_cdw")):
            n = max(1, round(len(class_rows) * audit_pct / 100))
            n = min(n, len(class_rows))
            drawn = rng.sample(class_rows, n)
            for row in drawn:
                sample.append(
                    {
                        "raster": raster,
                        "row_off": row["row_off"],
                        "col_off": row["col_off"],
                        "chunk_size": row.get("chunk_size", "128"),
                        "current_label": row["label"],
                        "model_prob": row.get("model_prob", ""),
                        "source": row.get("source", ""),
                        "annotator": row.get("annotator", ""),
                        "timestamp": row.get("timestamp", ""),
                        "audit_label": "",
                        "auditor": "",
                        "audit_note": "",
                    }
                )

    rng.shuffle(sample)
    return sample


# ── Build unified review queue ────────────────────────────────────────────────

_QUEUE_FIELDS = [
    "priority",
    "check_type",
    "raster",
    "row_off",
    "col_off",
    "chunk_size",
    "current_label",
    "model_prob",
    "disagree_score",
    "entropy",
    "source",
    "annotator",
    "timestamp",
    "audit_label",
    "auditor",
    "audit_note",
]


def build_review_queue(
    disagree: list[dict],
    isolated: list[dict],
    conflicts: list[dict],
    borderline: list[dict],
    sample: list[dict],
) -> list[dict]:
    """Merge all checks into a deduplicated, priority-sorted review queue.

    Deduplication: a (raster, row_off, col_off) triple keeps only the highest-
    priority entry when it appears in multiple checks.
    """
    seen: set[tuple] = set()
    queue: list[dict] = []

    def _add(rows: list[dict], priority: str, check_type: str) -> None:
        for row in rows:
            raster = row.get("raster", "")
            row_off = str(row.get("row_off", row.get("row_a", "")))
            col_off = str(row.get("col_off", row.get("col_a", "")))
            key = (raster, row_off, col_off)
            if key in seen:
                continue
            seen.add(key)
            prob_str = row.get("model_prob", "")
            entropy = row.get("entropy", "")
            if not entropy and prob_str:
                try:
                    entropy = f"{_entropy(float(prob_str)):.4f}"
                except ValueError:
                    pass
            queue.append(
                {
                    "priority": priority,
                    "check_type": check_type,
                    "raster": raster,
                    "row_off": row_off,
                    "col_off": col_off,
                    "chunk_size": row.get("chunk_size", "128"),
                    "current_label": row.get("current_label", row.get("label_a", "")),
                    "model_prob": prob_str,
                    "disagree_score": row.get("disagree_score", ""),
                    "entropy": entropy,
                    "source": row.get("source", ""),
                    "annotator": row.get("annotator", ""),
                    "timestamp": row.get("timestamp", ""),
                    "audit_label": "",
                    "auditor": "",
                    "audit_note": "",
                }
            )

    _add(disagree, "critical", "label_model_disagreement")
    _add(isolated, "high", "isolated_cdw")
    _add(conflicts, "medium", "overlap_conflict")
    _add(borderline, "medium", "borderline")
    _add(sample, "low", "random_spot_check")

    return queue


# ── Output helpers ────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        print(f"  (empty — nothing written to {path.name})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields if fields else list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {path.name}")


def print_summary(
    raster_rows: dict[str, list[dict]],
    disagree: list[dict],
    isolated: list[dict],
    conflicts: list[dict],
    borderline: list[dict],
    sample: list[dict],
    queue: list[dict],
) -> None:
    total = sum(len(v) for v in raster_rows.values())
    n_cdw = sum(r["label"] == "cdw" for rows in raster_rows.values() for r in rows)
    n_no = sum(r["label"] == "no_cdw" for rows in raster_rows.values() for r in rows)
    n_prob = sum(bool(r.get("model_prob")) for rows in raster_rows.values() for r in rows)
    n_auto = sum(
        r.get("source", "") in ("auto", "auto_skip") for rows in raster_rows.values() for r in rows
    )

    print("\n" + "=" * 72)
    print("  Label-quality audit — 5-check pipeline")
    print("=" * 72)
    print(f"  Rasters    : {len(raster_rows)}")
    print(f"  Total tiles: {total:,}  CDW={n_cdw:,}  no_CDW={n_no:,}")
    print(
        f"  With prob  : {n_prob:,} ({100*n_prob/max(total,1):.1f}%)  " f"Auto-labeled: {n_auto:,}"
    )
    print()

    print(f"  {'Raster':<52} {'CDW':>6} {'No':>7} {'Total':>7} {'%CDW':>6}")
    print(f"  {'-'*52} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    for raster, rows in sorted(raster_rows.items()):
        nc = sum(1 for r in rows if r["label"] == "cdw")
        nn = sum(1 for r in rows if r["label"] == "no_cdw")
        nt = nc + nn
        print(f"  {raster:<52} {nc:>6,} {nn:>7,} {nt:>7,} {100*nc/max(nt,1):>5.1f}%")
    print()

    print(f"  Check 1 — Label–model disagreement : {len(disagree):6,}  (priority: CRITICAL)")
    print(f"  Check 2 — Isolated CDW tiles       : {len(isolated):6,}  (priority: HIGH)")
    print(f"  Check 3 — Overlap conflicts         : {len(conflicts):6,}  tile-pairs (MEDIUM)")
    print(f"  Check 4 — Borderline tiles          : {len(borderline):6,}  (priority: MEDIUM)")
    print(f"  Check 5 — Random spot-check         : {len(sample):6,}  (priority: LOW)")
    print()
    print(f"  Unified review queue                : {len(queue):6,}  unique tiles")
    print("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="5-check label-quality audit for CDW tile labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--labels",
        default="output/tile_labels",
        help="Directory containing *_labels.csv (default: output/tile_labels)",
    )
    p.add_argument(
        "--output-dir",
        default="output/audits",
        help="Directory for audit output files (default: output/audits)",
    )
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument(
        "--disagreement-thresh",
        type=float,
        default=0.70,
        help="|prob - label| threshold for disagreement check (default: 0.70)",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.10,
        help="|prob - 0.5| threshold for borderline check (default: 0.10)",
    )
    p.add_argument(
        "--audit-pct",
        type=float,
        default=2.0,
        help="Percent per stratum for random spot-check (default: 2.0)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-conflicts",
        action="store_true",
        help="Skip overlap conflict check (can be slow on very large datasets)",
    )
    args = p.parse_args()

    labels_dir = Path(args.labels)
    output_dir = Path(args.output_dir)

    print(f"Loading labels from {labels_dir} …")
    raster_rows = load_labels(labels_dir)
    if not raster_rows:
        print("No label CSVs found.  Run label_all_rasters.py first.")
        return

    total = sum(len(v) for v in raster_rows.values())
    n_with_prob = sum(bool(r.get("model_prob")) for rows in raster_rows.values() for r in rows)
    if n_with_prob == 0:
        print("\n  WARNING: No model_prob values found in label CSVs.")
        print("  Checks 1 and 4 will find nothing until you run:")
        print("    python scripts/backfill_model_prob.py --labels", args.labels)
        print()

    print("Check 1 — Label–model disagreement …")
    disagree = check_disagreements(raster_rows, thresh=args.disagreement_thresh)
    print(f"  Found {len(disagree):,} tiles  (|prob - label| > {args.disagreement_thresh})")

    print("Check 2 — Isolated CDW tiles …")
    isolated = check_isolated_cdw(raster_rows, stride=args.chunk_size // 2)
    print(f"  Found {len(isolated):,} solo CDW tiles with 0 CDW neighbours")

    if args.no_conflicts:
        conflicts: list[dict] = []
        print("Check 3 — Skipped (--no-conflicts)")
    else:
        print("Check 3 — Overlap conflict detection …")
        conflicts = check_conflicts(raster_rows, chunk_size=args.chunk_size)
        print(f"  Found {len(conflicts):,} conflicting tile pairs")

    print("Check 4 — Borderline / high-entropy tiles …")
    borderline = check_borderline(raster_rows, margin=args.margin)
    print(f"  Found {len(borderline):,} tiles  (|prob - 0.5| < {args.margin})")

    print(f"Check 5 — Random spot-check ({args.audit_pct:.1f}% per stratum) …")
    sample = check_random_sample(raster_rows, audit_pct=args.audit_pct, seed=args.seed)
    print(f"  Sampled {len(sample):,} tiles")

    queue = build_review_queue(disagree, isolated, conflicts, borderline, sample)

    print_summary(raster_rows, disagree, isolated, conflicts, borderline, sample, queue)

    print(f"\nWriting audit files to {output_dir} …")
    _write_csv(output_dir / "audit_review_queue.csv", queue, _QUEUE_FIELDS)
    _write_csv(output_dir / "audit_disagree.csv", disagree)
    _write_csv(output_dir / "audit_isolated.csv", isolated)
    _write_csv(output_dir / "audit_conflicts.csv", conflicts)
    _write_csv(output_dir / "audit_borderline.csv", borderline)
    _write_csv(output_dir / "audit_sample.csv", sample)

    meta = {
        "labels_dir": str(labels_dir),
        "n_rasters": len(raster_rows),
        "n_total": total,
        "n_with_model_prob": n_with_prob,
        "n_disagree": len(disagree),
        "n_isolated": len(isolated),
        "n_conflicts": len(conflicts),
        "n_borderline": len(borderline),
        "n_sample": len(sample),
        "n_queue": len(queue),
        "disagreement_thresh": args.disagreement_thresh,
        "margin": args.margin,
        "audit_pct": args.audit_pct,
        "seed": args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2))
    print("  Wrote audit_meta.json")

    print(f"\nDone.  Total tiles flagged for review: {len(queue):,}")
    print()
    print("Next steps:")
    print("  1. Re-label flagged tiles via the GUI:")
    print(f"     python scripts/label_all_rasters.py \\")
    print(f"         --chm-dir chm_max_hag --output output/tile_labels \\")
    print(f"         --tile-list {output_dir / 'audit_review_queue.csv'} --resume")
    print("  2. After correction, run: python scripts/train_ensemble.py")


if __name__ == "__main__":
    main()
