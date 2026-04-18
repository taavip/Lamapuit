#!/usr/bin/env python3
"""
Auto-label CHM raster tiles using a trained ResNet-18 classifier.

For each CHM raster in --chm-dir, tiles are generated with the same
chunk/overlap parameters used during labeling, and the model predicts
cdw / no_cdw for each chunk.  Results land in per-raster
*_predicted.csv files.  Tiles predicted with confidence > --conf-accept
are added unconditionally; tiles in [--conf-review, --conf-accept) are
flagged "review" for optional manual verification; tiles below
--conf-review are skipped.

Usage
-----
python scripts/predict_tiles.py \
    --chm-dir       chm_max_hag \
    --model         output/tile_labels/classifier/best_classifier.pt \
    --output        output/tile_labels \
    [--skip-labeled]          # skip rasters that already have a _labels.csv
    [--skip-predicted]        # skip rasters that already have a _predicted.csv
    [--chunk-size   128]
    [--overlap      0.5]
    [--conf-accept  0.90]
    [--conf-review  0.70]
    [--auto-skip-threshold 0.15]
    [--batch-size   256]
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

# ── classifier helpers (mirror of train_classifier.py) ────────────────────────
IDX2LABEL = {0: "no_cdw", 1: "cdw"}


def _build_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)


def _normalize_tile(data: np.ndarray) -> np.ndarray:
    valid = data[(data > 0) & np.isfinite(data)]
    if valid.size < 10:
        return np.zeros_like(data, dtype=np.uint8)
    p2, p98 = np.percentile(valid, [2, 98])
    span = max(p98 - p2, 1e-6)
    img = np.clip((data - p2) / span, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img8)


_TFM = T.Compose(
    [
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def _tile_to_tensor(data: np.ndarray, chunk_size: int) -> torch.Tensor:
    img8 = _normalize_tile(data)
    img8 = cv2.resize(img8, (chunk_size, chunk_size), interpolation=cv2.INTER_AREA)
    return _TFM(img8)


# ── chunk generator ────────────────────────────────────────────────────────────
def _iter_chunks(height: int, width: int, chunk_size: int, overlap: float) -> list[tuple[int, int]]:
    stride = max(1, int(chunk_size * (1 - overlap)))
    chunks = []
    for r in range(0, height, stride):
        for c in range(0, width, stride):
            chunks.append((r, c))
    return chunks


# ── CSV helpers ────────────────────────────────────────────────────────────────
_PREDICTED_FIELDNAMES = [
    "raster",
    "row_off",
    "col_off",
    "chunk_size",
    "label",
    "confidence",
    "flag",
    "timestamp",
]


def _init_csv(csv_path: Path) -> csv.DictWriter:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(csv_path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=_PREDICTED_FIELDNAMES)
    w.writeheader()
    return w


# ── raster prediction ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_raster(
    chm_path: Path,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    chunk_size: int,
    overlap: float,
    conf_accept: float,
    conf_review: float,
    auto_skip_threshold: float,
    batch_size: int,
) -> dict[str, int]:
    stem = chm_path.stem
    csv_path = output_dir / f"{stem}_predicted.csv"
    writer = _init_csv(csv_path)
    counts = {"cdw": 0, "no_cdw": 0, "review": 0, "skipped_conf": 0, "auto_skip": 0}

    with rasterio.open(chm_path) as src:
        height, width = src.height, src.width
        chunks = _iter_chunks(height, width, chunk_size, overlap)
        n_chunks = len(chunks)

        # Batch processing
        batch_coords: list[tuple[int, int]] = []
        batch_tensors: list[torch.Tensor] = []

        def _flush_batch() -> None:
            nonlocal batch_coords, batch_tensors
            if not batch_tensors:
                return
            imgs = torch.stack(batch_tensors).to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            for (r, c), prob in zip(batch_coords, probs):
                conf = float(prob.max())
                pred_idx = int(prob.argmax())
                pred_lbl = IDX2LABEL[pred_idx]

                if conf < conf_review:
                    counts["skipped_conf"] += 1
                    batch_coords = []
                    batch_tensors = []
                    return  # handled below via continue outer loop

                flag = "ok" if conf >= conf_accept else "review"
                if flag == "review":
                    counts["review"] += 1
                counts[pred_lbl] = counts.get(pred_lbl, 0) + 1

                writer.writerow(
                    {
                        "raster": stem,
                        "row_off": r,
                        "col_off": c,
                        "chunk_size": chunk_size,
                        "label": pred_lbl,
                        "confidence": f"{conf:.4f}",
                        "flag": flag,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )
            batch_coords = []
            batch_tensors = []

        for i, (r, c) in enumerate(chunks):
            if i % 500 == 0:
                pct = 100 * i / n_chunks
                print(
                    f"\r  {i:6d}/{n_chunks}  ({pct:5.1f}%)  "
                    f"cdw={counts['cdw']}  no_cdw={counts['no_cdw']}  "
                    f"review={counts['review']}",
                    end="",
                    flush=True,
                )

            window = Window(col_off=c, row_off=r, width=chunk_size, height=chunk_size)
            data = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            valid = data[(data > 0) & np.isfinite(data)]

            if valid.size > 0 and float(valid.max()) < auto_skip_threshold:
                counts["auto_skip"] += 1
                writer.writerow(
                    {
                        "raster": stem,
                        "row_off": r,
                        "col_off": c,
                        "chunk_size": chunk_size,
                        "label": "no_cdw",
                        "confidence": "1.0",
                        "flag": "auto_skip",
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )
                continue

            tensor = _tile_to_tensor(data, chunk_size)
            batch_coords.append((r, c))
            batch_tensors.append(tensor)

            if len(batch_tensors) >= batch_size:
                # We need a proper flush that handles confidence filtering per-item
                imgs = torch.stack(batch_tensors).to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                for (br, bc), prob in zip(batch_coords, probs):
                    conf = float(prob.max())
                    pred_idx = int(prob.argmax())
                    pred_lbl = IDX2LABEL[pred_idx]
                    if conf < conf_review:
                        counts["skipped_conf"] += 1
                        continue
                    flag = "ok" if conf >= conf_accept else "review"
                    if flag == "review":
                        counts["review"] += 1
                    counts[pred_lbl] = counts.get(pred_lbl, 0) + 1
                    writer.writerow(
                        {
                            "raster": stem,
                            "row_off": br,
                            "col_off": bc,
                            "chunk_size": chunk_size,
                            "label": pred_lbl,
                            "confidence": f"{conf:.4f}",
                            "flag": flag,
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                        }
                    )
                batch_coords = []
                batch_tensors = []

        # flush remaining
        if batch_tensors:
            imgs = torch.stack(batch_tensors).to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            for (br, bc), prob in zip(batch_coords, probs):
                conf = float(prob.max())
                pred_idx = int(prob.argmax())
                pred_lbl = IDX2LABEL[pred_idx]
                if conf < conf_review:
                    counts["skipped_conf"] += 1
                    continue
                flag = "ok" if conf >= conf_accept else "review"
                if flag == "review":
                    counts["review"] += 1
                counts[pred_lbl] = counts.get(pred_lbl, 0) + 1
                writer.writerow(
                    {
                        "raster": stem,
                        "row_off": br,
                        "col_off": bc,
                        "chunk_size": chunk_size,
                        "label": pred_lbl,
                        "confidence": f"{conf:.4f}",
                        "flag": flag,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )

    print(
        f"\r  Done. "
        f"cdw={counts['cdw']}  no_cdw={counts['no_cdw']}  "
        f"review={counts['review']}  auto_skip={counts['auto_skip']}  "
        f"low_conf={counts['skipped_conf']}          "
    )
    return counts


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Auto-label CHM tiles with trained classifier")
    p.add_argument("--chm-dir", default="chm_max_hag")
    p.add_argument("--model", default="output/tile_labels/classifier/best_classifier.pt")
    p.add_argument("--output", default="output/tile_labels")
    p.add_argument("--pattern", default="*20cm.tif")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--conf-accept", type=float, default=0.90)
    p.add_argument("--conf-review", type=float, default=0.70)
    p.add_argument("--auto-skip-threshold", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--skip-labeled",
        action="store_true",
        help="Skip rasters that already have a *_labels.csv (human-labeled)",
    )
    p.add_argument(
        "--skip-predicted",
        action="store_true",
        help="Skip rasters that already have a *_predicted.csv",
    )
    args = p.parse_args()

    chm_dir = Path(args.chm_dir)
    output_dir = Path(args.output)
    model_path = Path(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python -m cdw_detect.train_classifier")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = _build_model(model_path, device)
    print(f"Model loaded from {model_path}\n")

    rasters = sorted(chm_dir.glob(args.pattern))
    if not rasters:
        print(f"No rasters found matching '{args.pattern}' in {chm_dir}")
        sys.exit(1)

    print(f"Found {len(rasters)} rasters")
    grand: dict[str, int] = {"cdw": 0, "no_cdw": 0, "review": 0, "auto_skip": 0, "skipped_conf": 0}

    for i, chm_path in enumerate(rasters, 1):
        stem = chm_path.stem

        if args.skip_labeled and (output_dir / f"{stem}_labels.csv").exists():
            print(f"[{i:3d}/{len(rasters)}] skip (labeled)  {stem}")
            continue
        if args.skip_predicted and (output_dir / f"{stem}_predicted.csv").exists():
            print(f"[{i:3d}/{len(rasters)}] skip (predicted) {stem}")
            continue

        print(f"\n[{i:3d}/{len(rasters)}] {stem}")
        counts = predict_raster(
            chm_path=chm_path,
            model=model,
            device=device,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            conf_accept=args.conf_accept,
            conf_review=args.conf_review,
            auto_skip_threshold=args.auto_skip_threshold,
            batch_size=args.batch_size,
        )
        for k in grand:
            grand[k] += counts.get(k, 0)

    print(f"\n{'='*60}")
    print(
        f"Grand totals  cdw={grand['cdw']}  no_cdw={grand['no_cdw']}  "
        f"review={grand['review']}  auto_skip={grand['auto_skip']}  "
        f"low_conf={grand['skipped_conf']}"
    )


if __name__ == "__main__":
    main()
