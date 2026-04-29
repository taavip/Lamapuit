#!/usr/bin/env python3
"""
Generate a per-window CSV of ensemble P(CDW) for a CHM dataset and
optionally merge existing label metadata from CSVs.

Defaults target the baseline 20cm CHMs (`data/lamapuit/chm_max_hag_13_drop`).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.windows import Window


def _iter_chunks(height: int, width: int, chunk_size: int, overlap: float) -> list[tuple[int, int]]:
    """Return list of (row_off, col_off) chunk origins.

    Mirrors the queue-building used by the labeler so grids align.
    """
    stride = max(1, int(chunk_size * (1.0 - overlap)))
    min_gap = chunk_size // 4

    def make_offsets(size: int) -> list[int]:
        if size <= chunk_size:
            return [0]
        offsets = list(range(0, size - chunk_size + 1, stride))
        last_border = size - chunk_size
        gap = size - (offsets[-1] + chunk_size)
        if gap > min_gap and last_border not in offsets:
            offsets.append(last_border)
        return offsets

    rows = make_offsets(height)
    cols = make_offsets(width)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for row_off in rows:
        for col_off in cols:
            key = (row_off, col_off)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _batch_predict_proba(predictor, tiles: list[np.ndarray]) -> list[float]:
    """Run batched soft-vote ensemble inference using a loaded predictor.

    Expects *tiles* as raw CHM float32 arrays (H,W) in metres.
    """
    try:
        import torch
    except Exception:
        raise RuntimeError("Torch is required for batched inference")

    nets = getattr(predictor, "_nets", None) or getattr(predictor, "_net", None)
    weights = getattr(predictor, "_weights", None) or [1.0]
    device = getattr(predictor, "_device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(nets, list):
        nets = [nets]

    # Normalize weights
    w = [float(x) for x in (weights or [1.0] * len(nets))]
    s = float(sum(max(0.0, vv) for vv in w))
    if s <= 0.0:
        w = [1.0 / len(nets)] * len(nets)
    else:
        w = [max(0.0, vv) / s for vv in w]

    arr = np.stack(tiles, axis=0)  # (B,H,W)
    arr = np.clip(arr, 0.0, 20.0) / 20.0
    t = torch.from_numpy(arr.astype(np.float32))[:, None].to(device)

    probs = None
    with torch.no_grad():
        for net, weight in zip(nets, w):
            if net is None:
                continue
            net.eval()
            out = torch.softmax(net(t), dim=1)[:, 1].cpu().numpy()
            if probs is None:
                probs = weight * out
            else:
                probs = probs + weight * out

    if probs is None:
        return [0.0] * len(tiles)
    return [float(x) for x in probs.tolist()]


def _gather_label_rows(label_dirs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for d in label_dirs:
        if not d:
            continue
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.glob("*_labels.csv")):
            out.append(f)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate per-window ensemble probs + merge labels")
    p.add_argument(
        "--chm-dir",
        default="data/lamapuit/chm_max_hag_13_drop",
        help="Directory with CHM GeoTIFFs (default: baseline 20cm)",
    )
    p.add_argument("--output", default="output/per_window_baseline.csv")
    p.add_argument(
        "--model-meta",
        default="output/tile_labels/ensemble_meta.json",
        help="Path to ensemble_meta.json or single .pt checkpoint",
    )
    p.add_argument(
        "--labels-dirs",
        nargs="*",
        default=["output/tile_labels", "output/model_search_v4/prepared/labels_curated_v4"],
        help="Directories to scan for existing *_labels.csv files to merge",
    )
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    chm_dir = Path(args.chm_dir)
    if not chm_dir.exists():
        print("CHM directory not found:", chm_dir, file=sys.stderr)
        sys.exit(1)

    # Load predictor
    try:
        import label_tiles as lt
    except ModuleNotFoundError:
        from scripts import label_tiles as lt

    predictor = lt.CNNPredictor()
    meta_path = Path(args.model_meta)
    ok = False
    if meta_path.exists() and meta_path.suffix.lower() == ".json":
        ok = bool(predictor.load_ensemble_meta(meta_path))
    elif meta_path.exists():
        ok = bool(predictor.load_from_disk(meta_path))
    if not ok:
        print("Failed to load model/ensemble from:", meta_path, file=sys.stderr)
        sys.exit(1)

    model_name = getattr(predictor, "_model_name", "ensemble")

    # Build full grid for all rasters and compute ensemble probs
    per_window: dict[tuple[str, int, int], dict] = {}
    tifs = sorted(chm_dir.glob("*.tif"))
    if not tifs:
        print("No TIFFs found in:", chm_dir, file=sys.stderr)
        sys.exit(1)

    for tif in tifs:
        raster_name = tif.name
        print(f"Processing {raster_name} ...", flush=True)
        try:
            with rasterio.open(tif) as src:
                h, w = src.height, src.width
                offs = _iter_chunks(h, w, args.chunk_size, args.overlap)
                # Process in batches to leverage GPU/CPU batching
                for i in range(0, len(offs), args.batch_size):
                    batch = offs[i : i + args.batch_size]
                    tiles = []
                    keys = []
                    for r_off, c_off in batch:
                        raw = src.read(1, window=Window(c_off, r_off, args.chunk_size, args.chunk_size), boundless=True, fill_value=0).astype(np.float32)
                        if raw.shape != (args.chunk_size, args.chunk_size):
                            import cv2

                            raw = cv2.resize(raw, (args.chunk_size, args.chunk_size))
                        tiles.append(raw)
                        keys.append((raster_name, int(r_off), int(c_off)))

                    probs = _batch_predict_proba(predictor, tiles)
                    for key, prob in zip(keys, probs):
                        per_window[key] = {
                            "raster": key[0],
                            "row_off": key[1],
                            "col_off": key[2],
                            "chunk_size": args.chunk_size,
                            "model_name": model_name,
                            "model_prob": f"{prob:.5f}",
                            "label": "",
                            "source": "",
                            "annotator": "",
                            "timestamp": "",
                            "source_csv": "",
                            "_priority": 99,
                        }
        except Exception as exc:
            print(f"  ERROR reading {tif}: {exc}", file=sys.stderr)

    # Merge existing label CSVs (priority: manual/auto_reviewed first)
    label_files = _gather_label_rows(args.labels_dirs)
    print(f"Merging {len(label_files)} label CSV(s) ...", flush=True)
    for csv_path in label_files:
        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        key = (str(row.get("raster", "")), int(row.get("row_off", 0)), int(row.get("col_off", 0)))
                    except Exception:
                        continue
                    if key not in per_window:
                        continue
                    src_val = str(row.get("source", "")).strip().lower()
                    if src_val in ("manual", "auto_reviewed", "manual_mask"):
                        pr = 0
                    else:
                        pr = 1

                    cur = per_window[key]
                    if pr < cur.get("_priority", 99):
                        cur["label"] = row.get("label", "")
                        cur["source"] = row.get("source", "")
                        cur["annotator"] = row.get("annotator", "")
                        cur["timestamp"] = row.get("timestamp", "")
                        cur["source_csv"] = str(csv_path)
                        cur["_priority"] = pr
                    # keep original model_prob computed above as canonical
        except Exception as exc:
            print(f"  WARN: failed to read {csv_path}: {exc}", file=sys.stderr)

    # Write consolidated CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "raster",
        "row_off",
        "col_off",
        "chunk_size",
        "model_name",
        "model_prob",
        "label",
        "source",
        "annotator",
        "timestamp",
        "source_csv",
    ]

    print(f"Writing consolidated CSV to {out_path} (rows={len(per_window)}) ...", flush=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for k, v in sorted(per_window.items()):
            row = {k: v.get(k, "") for k in header}
            # ensure numeric fields are written as ints where appropriate
            row["row_off"] = int(v["row_off"])
            row["col_off"] = int(v["col_off"])
            row["chunk_size"] = int(v["chunk_size"])
            w.writerow(row)

    print("Done.")


if __name__ == "__main__":
    main()
