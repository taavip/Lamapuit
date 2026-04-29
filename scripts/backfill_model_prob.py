#!/usr/bin/env python3
"""
Backfill model_prob / source / annotator columns into legacy 6-column label CSVs.

For every *_labels.csv that uses the old schema (no model_prob column), this script:
  1. Loads the current CNN-Deep-Attn checkpoint from --model.
  2. Runs inference on every labeled tile to obtain P(CDW).
  3. Rewrites the CSV file with the full 9-column schema:
       raster, row_off, col_off, chunk_size, label,
       source, annotator, model_prob, timestamp
     source is set to "manual" for all pre-existing rows (they were all hand-labeled).

NOTE: This is a one-time migration.  CSVs whose header already contains
"model_prob" are skipped unless --force is given.

Usage
-----
python scripts/backfill_model_prob.py \\
    --labels output/tile_labels \\
    --chm-dir chm_max_hag \\
    --model output/tile_labels/ensemble_model.pt

# Preview only (no file writes):
python scripts/backfill_model_prob.py ... --dry-run

# Re-run even on already-migrated CSVs:
python scripts/backfill_model_prob.py ... --force
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

# ── Shared constants (must match label_tiles.py & fine_tune_cnn.py) ──────────
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
_CANONICAL_SIZE = 128
_CLIP_MAX = 20.0  # metres
_BATCH_SIZE = 256  # tiles per inference batch


# ── Normalization (identical to fine_tune_cnn._norm_tile) ────────────────────
def _norm_tile(raw: np.ndarray) -> np.ndarray:
    return np.clip(raw, 0.0, _CLIP_MAX) / _CLIP_MAX


# ── CNN model (identical definition to fine_tune_cnn.py) ─────────────────────
def _build_deep_cnn_attn_net():
    import torch.nn as nn

    class SE(nn.Module):
        def __init__(self, c, r=8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, max(c // r, 4)),
                nn.ReLU(),
                nn.Linear(max(c // r, 4), c),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)

    class AttnBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
            self.se = SE(out_c)
            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
            self.mp = nn.MaxPool2d(2)

        def forward(self, x):
            return self.mp(self.se(self.conv(x)) + self.skip(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(
                AttnBlock(1, 32),
                AttnBlock(32, 64),
                AttnBlock(64, 128),
                AttnBlock(128, 256),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2),
            )

        def forward(self, x):
            return self.head(self.blocks(x))

    return Net()


# ── Model loader ─────────────────────────────────────────────────────────────
def _load_model(model_path: Path):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _build_deep_cnn_attn_net().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"  Loaded checkpoint v{ckpt.get('meta', {}).get('version', '?')} from {model_path}")
    return net, device


# ── Batch inference ───────────────────────────────────────────────────────────
def _infer_batch(net, device, tiles: list[np.ndarray]) -> list[float]:
    """Run CNN on a list of (H, W) float32 tiles; return P(CDW) for each."""
    import torch

    arr = np.stack(tiles, axis=0)[:, np.newaxis, :, :]  # (B, 1, H, W)
    t = torch.from_numpy(arr.astype(np.float32)).to(device)
    with torch.no_grad():
        probs = torch.softmax(net(t), dim=1)[:, 1].cpu().numpy()
    return probs.tolist()


# ── CSV schema detection ──────────────────────────────────────────────────────
def _needs_migration(csv_path: Path) -> bool:
    """Return True if the CSV is missing the model_prob column."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return "model_prob" not in header


# ── Per-raster migration ──────────────────────────────────────────────────────
def _migrate_csv(
    csv_path: Path,
    chm_dir: Path,
    net,
    device,
    dry_run: bool = False,
) -> dict:
    """Backfill one CSV file.  Returns summary statistics dict."""
    stats = {
        "csv": csv_path.name,
        "total": 0,
        "inferred": 0,
        "failed": 0,
        "skipped": 0,
        "error": None,
    }

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    if not rows:
        stats["error"] = "empty"
        return stats

    stats["total"] = len(rows)

    # Find the CHM raster for this CSV
    raster_name = rows[0].get("raster", "")
    raster_path = chm_dir / raster_name
    if not raster_path.exists():
        matches = list(chm_dir.glob(f"{Path(raster_name).stem}*"))
        if not matches:
            stats["error"] = f"raster not found: {raster_name}"
            return stats
        raster_path = matches[0]

    # Load all tiles from this raster in one pass
    tile_map: dict[tuple[int, int], np.ndarray] = {}
    try:
        with rasterio.open(raster_path) as src:
            for row in rows:
                r = int(row["row_off"])
                c = int(row["col_off"])
                cs = int(row.get("chunk_size", 128))
                raw = src.read(
                    1,
                    window=Window(c, r, cs, cs),
                    boundless=True,
                    fill_value=0,
                ).astype(np.float32)
                if raw.shape != (_CANONICAL_SIZE, _CANONICAL_SIZE):
                    import cv2

                    raw = cv2.resize(raw, (_CANONICAL_SIZE, _CANONICAL_SIZE))
                tile_map[(r, c)] = _norm_tile(raw)
    except Exception as exc:
        stats["error"] = str(exc)
        return stats

    # Run inference in batches
    keys = list(tile_map.keys())
    tiles = [tile_map[k] for k in keys]
    probs: dict[tuple[int, int], float] = {}

    for i in range(0, len(keys), _BATCH_SIZE):
        batch_keys = keys[i : i + _BATCH_SIZE]
        batch_tiles = tiles[i : i + _BATCH_SIZE]
        batch_probs = _infer_batch(net, device, batch_tiles)
        for k, p in zip(batch_keys, batch_probs):
            probs[k] = p
    stats["inferred"] = len(probs)

    # Build migrated rows
    new_rows: list[dict] = []
    for row in rows:
        r = int(row["row_off"])
        c = int(row["col_off"])
        prob = probs.get((r, c))
        new_row = {
            "raster": row.get("raster", ""),
            "row_off": row["row_off"],
            "col_off": row["col_off"],
            "chunk_size": row.get("chunk_size", _CANONICAL_SIZE),
            "label": row.get("label", ""),
            "source": row.get("source", "manual"),  # legacy = hand-labeled
            "annotator": row.get("annotator", ""),
            "model_prob": f"{prob:.5f}" if prob is not None else "",
            "timestamp": row.get("timestamp", ""),
        }
        new_rows.append(new_row)
        if prob is None:
            stats["skipped"] += 1

    if dry_run:
        print(
            f"    [dry-run] Would rewrite {csv_path.name}  "
            f"rows={len(new_rows)} inferred={stats['inferred']}"
        )
        return stats

    # Backup original, then rewrite
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        w.writeheader()
        w.writerows(new_rows)

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Backfill model_prob / source into legacy 6-column label CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--labels", default="output/tile_labels", help="Directory containing *_labels.csv files"
    )
    p.add_argument(
        "--chm-dir", default="chm_max_hag", help="Directory containing CHM GeoTIFF rasters"
    )
    p.add_argument(
        "--model",
        default="output/tile_labels/ensemble_model.pt",
        help="CNN checkpoint to use for inference",
    )
    p.add_argument(
        "--batch-size", type=int, default=256, help="Inference batch size (default: 256)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing any files"
    )
    p.add_argument(
        "--force", action="store_true", help="Migrate all CSVs even if they already have model_prob"
    )
    args = p.parse_args()

    global _BATCH_SIZE
    _BATCH_SIZE = args.batch_size

    labels_dir = Path(args.labels)
    chm_dir = Path(args.chm_dir)
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(labels_dir.glob("*_labels.csv"))
    if not csv_files:
        print(f"No *_labels.csv files found in {labels_dir}")
        sys.exit(0)

    # Filter to only those needing migration (unless --force)
    to_migrate = []
    for csv_path in csv_files:
        if args.force or _needs_migration(csv_path):
            to_migrate.append(csv_path)
        else:
            print(f"  Skip (already migrated): {csv_path.name}")

    if not to_migrate:
        print("All CSVs already have the full schema.  Use --force to re-run.")
        sys.exit(0)

    print(f"\nLoading CNN model …")
    net, device = _load_model(model_path)

    print(f"\nMigrating {len(to_migrate)} CSV file(s) …")
    all_stats: list[dict] = []

    for i, csv_path in enumerate(to_migrate, 1):
        print(f"  [{i:3d}/{len(to_migrate)}]  {csv_path.name} …", end="", flush=True)
        stats = _migrate_csv(csv_path, chm_dir, net, device, dry_run=args.dry_run)
        all_stats.append(stats)
        if stats["error"]:
            print(f" ERROR: {stats['error']}")
        else:
            print(
                f"  {stats['total']} rows  inferred={stats['inferred']}  "
                f"skipped={stats['skipped']}"
            )

    # Summary
    total_rows = sum(s["total"] for s in all_stats)
    total_inferred = sum(s["inferred"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats if s["error"])
    print(f"\n{'='*60}")
    print(f"  Files processed : {len(to_migrate)}")
    print(f"  Total rows      : {total_rows:,}")
    print(f"  Inferred probs  : {total_inferred:,}")
    print(f"  Errors          : {total_failed}")
    if args.dry_run:
        print("\n  [dry-run] No files were written.")
    else:
        print(f"\n  Originals backed up as *.csv.bak")
        print(f"  CSVs now use 9-column schema with model_prob filled.")
    print(f"{'='*60}")
    print("\nNext step:  python scripts/audit_labels.py --labels", args.labels)


if __name__ == "__main__":
    main()
