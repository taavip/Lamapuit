"""
Feed tile-level CDW labels into the YOLO instance-segmentation pipeline.

This module bridges the output of the binary tile labeling workflow
(scripts/label_tiles.py / scripts/predict_tiles.py) with the existing
YOLO dataset preparation:

  1.  Reads all *_labels.csv  (human labels)  and *_predicted.csv
      (classifier-predicted labels with high confidence) from a tile
      labels directory.

  2.  For every tile labeled **no_cdw** (human or model, conf ≥ threshold),
      finds the 640×640 YOLO dataset tile(s) that spatially overlap it and
      ensures that tile has an **empty** annotation file — making it an
      explicit hard negative rather than a heuristic one.

  3.  Outputs a report of how many YOLO tiles were updated / created.

Tile-label CSV formats:
  _labels.csv    : raster,row_off,col_off,chunk_size,label,timestamp
  _predicted.csv : raster,row_off,col_off,chunk_size,label,confidence,flag,timestamp

Usage (standalone)
------------------
python -m cdw_detect.prepare_instance_from_tiles \
    --tile-labels-dir  output/tile_labels \
    --dataset-dir      output/cdw_training_v4/dataset \
    [--conf-threshold  0.90]      # min classifier confidence for negatives
    [--split           train]     # train | val | all
    [--dry-run]                   # report without writing

Or import and call inject_tile_negatives() from prepare_instance.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import NamedTuple


# ── data structures ────────────────────────────────────────────────────────────
class TileRef(NamedTuple):
    raster: str
    row_off: int
    col_off: int
    chunk_size: int
    label: str  # "cdw" | "no_cdw"
    confidence: float  # 1.0 for human labels


# ── CSV readers ────────────────────────────────────────────────────────────────
def _read_human_labels(labels_dir: Path) -> list[TileRef]:
    refs: list[TileRef] = []
    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        raster_stem = csv_path.stem.replace("_labels", "")
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") not in ("cdw", "no_cdw"):
                    continue
                refs.append(
                    TileRef(
                        raster=raster_stem,
                        row_off=int(row["row_off"]),
                        col_off=int(row["col_off"]),
                        chunk_size=int(row.get("chunk_size", 128)),
                        label=row["label"],
                        confidence=1.0,
                    )
                )
    return refs


def _read_predicted_labels(labels_dir: Path, conf_threshold: float) -> list[TileRef]:
    refs: list[TileRef] = []
    for csv_path in sorted(labels_dir.glob("*_predicted.csv")):
        raster_stem = csv_path.stem.replace("_predicted", "")
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") not in ("cdw", "no_cdw"):
                    continue
                conf = float(row.get("confidence", 0))
                if conf < conf_threshold:
                    continue
                refs.append(
                    TileRef(
                        raster=raster_stem,
                        row_off=int(row["row_off"]),
                        col_off=int(row["col_off"]),
                        chunk_size=int(row.get("chunk_size", 128)),
                        label=row["label"],
                        confidence=conf,
                    )
                )
    return refs


# ── YOLO dataset tile discovery ────────────────────────────────────────────────
def _find_yolo_tiles(dataset_dir: Path, split: str) -> list[Path]:
    """Return all image paths in the YOLO dataset for the given split(s)."""
    imgs: list[Path] = []
    splits = ["train", "val"] if split == "all" else [split]
    for s in splits:
        img_dir = dataset_dir / "images" / s
        if img_dir.exists():
            imgs += sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    return imgs


def _yolo_tile_meta(img_path: Path) -> dict | None:
    """
    Extract raster stem, row_off, col_off (in source-raster px) from the
    image filename.  Expected naming from prepare_instance.py:
        {raster_stem}_r{row:06d}_c{col:06d}.png
    Returns None if the filename format doesn't match.
    """
    name = img_path.stem
    # Try common separators: _rNNNNNN_cNNNNNN
    import re

    m = re.search(r"_r(\d+)_c(\d+)$", name)
    if not m:
        return None
    row = int(m.group(1))
    col = int(m.group(2))
    raster_stem = name[: m.start()]
    return {"raster": raster_stem, "row_off": row, "col_off": col}


# ── spatial overlap check ──────────────────────────────────────────────────────
def _overlaps(
    tile_row: int,
    tile_col: int,
    tile_size: int,
    yolo_row: int,
    yolo_col: int,
    yolo_size: int = 640,
) -> bool:
    """Return True if two axis-aligned rectangles overlap (even by 1 px)."""
    return (
        tile_col < yolo_col + yolo_size
        and tile_col + tile_size > yolo_col
        and tile_row < yolo_row + yolo_size
        and tile_row + tile_size > yolo_row
    )


# ── inject negatives ───────────────────────────────────────────────────────────
def inject_tile_negatives(
    tile_labels_dir: Path,
    dataset_dir: Path,
    conf_threshold: float = 0.90,
    split: str = "all",
    dry_run: bool = False,
) -> dict:
    """
    For every tile labeled no_cdw, find overlapping YOLO dataset tiles and
    ensure they have empty annotation files (hard negatives).

    Returns a stats dict with keys:
        human_no_cdw, predicted_no_cdw, yolo_tiles_examined,
        already_empty, newly_emptied, skipped_has_labels
    """
    human_refs = _read_human_labels(tile_labels_dir)
    predicted_refs = _read_predicted_labels(tile_labels_dir, conf_threshold)

    no_cdw_refs = [r for r in human_refs + predicted_refs if r.label == "no_cdw"]
    stats = {
        "human_no_cdw": sum(1 for r in human_refs if r.label == "no_cdw"),
        "predicted_no_cdw": sum(1 for r in predicted_refs if r.label == "no_cdw"),
        "yolo_tiles_examined": 0,
        "already_empty": 0,
        "newly_emptied": 0,
        "skipped_has_labels": 0,
    }

    if not no_cdw_refs:
        print("No 'no_cdw' tile references found. Nothing to inject.")
        return stats

    # Group tile refs by raster for fast lookup
    from collections import defaultdict

    by_raster: dict[str, list[TileRef]] = defaultdict(list)
    for ref in no_cdw_refs:
        by_raster[ref.raster].append(ref)

    yolo_tiles = _find_yolo_tiles(dataset_dir, split)
    stats["yolo_tiles_examined"] = len(yolo_tiles)

    for img_path in yolo_tiles:
        meta = _yolo_tile_meta(img_path)
        if meta is None:
            continue
        raster_stem = meta["raster"]
        if raster_stem not in by_raster:
            continue

        yolo_row = meta["row_off"]
        yolo_col = meta["col_off"]
        yolo_size = 640

        overlapping_refs = [
            r
            for r in by_raster[raster_stem]
            if _overlaps(r.row_off, r.col_off, r.chunk_size, yolo_row, yolo_col, yolo_size)
        ]
        if not overlapping_refs:
            continue

        # Determine label file path
        label_subdir = img_path.parent.parent.parent / "labels" / img_path.parent.name
        label_path = label_subdir / img_path.with_suffix(".txt").name

        if label_path.exists() and label_path.stat().st_size > 0:
            # This tile has polygon annotations — do NOT overwrite
            stats["skipped_has_labels"] += 1
            continue

        if label_path.exists():
            stats["already_empty"] += 1
        else:
            stats["newly_emptied"] += 1
            if not dry_run:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.touch()

    return stats


# ── summary report ─────────────────────────────────────────────────────────────
def _print_report(stats: dict, dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n{'='*55}")
    print(f"  prepare_instance_from_tiles  [{mode}]")
    print(f"{'='*55}")
    print(f"  Human no_cdw tile refs    : {stats['human_no_cdw']}")
    print(f"  Predicted no_cdw refs     : {stats['predicted_no_cdw']}")
    print(f"  YOLO tiles examined       : {stats['yolo_tiles_examined']}")
    print(f"  Already empty (no change) : {stats['already_empty']}")
    print(f"  Newly emptied (hard-neg)  : {stats['newly_emptied']}")
    print(f"  Skipped (has labels)      : {stats['skipped_has_labels']}")
    print(f"{'='*55}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Inject tile-level no_cdw labels as hard negatives into YOLO dataset"
    )
    p.add_argument("--tile-labels-dir", default="output/tile_labels")
    p.add_argument("--dataset-dir", default="output/cdw_training_v4/dataset")
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.90,
        help="Min classifier confidence required to trust a predicted no_cdw",
    )
    p.add_argument("--split", default="all", choices=["train", "val", "all"])
    p.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    p.add_argument("--save-report", help="Optional JSON path to save the stats report")
    args = p.parse_args()

    stats = inject_tile_negatives(
        tile_labels_dir=Path(args.tile_labels_dir),
        dataset_dir=Path(args.dataset_dir),
        conf_threshold=args.conf_threshold,
        split=args.split,
        dry_run=args.dry_run,
    )
    _print_report(stats, args.dry_run)

    if args.save_report:
        Path(args.save_report).write_text(json.dumps(stats, indent=2))
        print(f"Report saved to {args.save_report}")


if __name__ == "__main__":
    main()
