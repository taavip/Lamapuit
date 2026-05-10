#!/usr/bin/env python3
"""Phase III-a V5: YOLO11m-seg fine-tuning for CWD instance segmentation.

Fine-tunes yolo11m-seg.pt on 640×640 CHM pseudo-RGB patches using
4-fold spatial cross-validation (same stripe structure as V3).

Architecture:
    YOLO11m-seg — anchor-free instance segmentation
    Input: 640×640, 3-channel uint8 (CHM bands 1-3 normalized)
    Output: per-instance bounding boxes + polygon masks
    Pre-trained on COCO (80 classes) → fine-tuned for 1 class (CWD)

Training config:
    epochs=100, patience=20, batch=8, imgsz=640
    AdamW, lr0=1e-4, copy_paste=0.3 for small-object augmentation

Usage:
    python phase3_train_v5_yolo.py                   # all 4 folds
    python phase3_train_v5_yolo.py --fold 0          # single fold
    python phase3_train_v5_yolo.py --fold 0 --epochs 3   # smoke test
    python phase3_train_v5_yolo.py --device cpu      # CPU fallback
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

DATASET_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v5" / "yolo"
RUNS_DIR = ROOT / "seg_pipeline" / "output" / "phase3_runs_v5" / "yolo"
PRETRAINED_CKPT = ROOT / "yolo11m-seg.pt"


def train_fold(
    fold_id: int,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 8,
    device: str = "cuda",
    resume: bool = False,
) -> dict:
    """Train YOLO11m-seg on one fold."""
    from ultralytics import YOLO

    fold_dir = DATASET_DIR / f"fold{fold_id}"
    data_yaml = fold_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Data YAML not found: {data_yaml}\n"
            "Run phase2_dataset_v5.py first."
        )

    run_dir = RUNS_DIR / f"fold{fold_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing best.pt to resume or skip
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists() and not resume:
        print(f"  Fold {fold_id}: checkpoint exists at {best_pt}, skipping (use --resume to retrain)")
        return _load_metrics(run_dir)

    print(f"\n=== Fold {fold_id} ===")
    print(f"  Data: {data_yaml}")
    print(f"  Pretrained: {PRETRAINED_CKPT}")
    print(f"  Epochs: {epochs}, patience: {patience}, batch: {batch}, device: {device}")

    model = YOLO(str(PRETRAINED_CKPT))

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=640,
        device=device,
        project=str(RUNS_DIR),
        name=f"fold{fold_id}",
        exist_ok=True,
        # Optimizer
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        # Augmentation
        degrees=45.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.3,
        copy_paste=0.3,       # small-object augmentation
        hsv_h=0.0,            # no hue shift (non-RGB input)
        hsv_s=0.3,
        hsv_v=0.3,
        # DataLoader — workers=0 avoids multiprocessing shared memory issues in Docker
        workers=0,
        # Metrics
        save=True,
        save_period=-1,       # save best only
        val=True,
        plots=True,
        verbose=True,
        # Detection thresholds
        conf=0.25,
        iou=0.5,
    )

    # Extract key metrics from results (ultralytics SegmentMetrics object)
    rd = results.results_dict if hasattr(results, "results_dict") else {}
    metrics = {
        "fold_id": fold_id,
        "best_map50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "best_map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        "best_map50_seg": float(rd.get("metrics/mAP50(M)", 0.0)),
        "best_map50_95_seg": float(rd.get("metrics/mAP50-95(M)", 0.0)),
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  ✓ Fold {fold_id}: mAP@50={metrics['best_map50_seg']:.4f} (seg), saved to {run_dir}")
    return metrics


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {"fold_id": int(run_dir.name.replace("fold", "")), "skipped": True}


def main():
    parser = argparse.ArgumentParser(description="Phase III-a V5: YOLO11m-seg training")
    parser.add_argument("--fold", type=int, default=None, help="Single fold (0-3). Default: all folds")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume/retrain even if checkpoint exists")
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(4))

    print(f"=== Phase III-a V5: YOLO11m-seg Training ===")
    print(f"Device: {args.device}, folds: {folds}")
    print(f"Pretrained: {PRETRAINED_CKPT}")

    if not PRETRAINED_CKPT.exists():
        print(f"⚠ Pre-trained checkpoint not found at {PRETRAINED_CKPT}")
        print("  YOLO will download yolo11m-seg.pt automatically on first run.")

    all_metrics = []
    for fold_id in folds:
        m = train_fold(
            fold_id=fold_id,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
        )
        all_metrics.append(m)

    # Summary
    print("\n=== Training Summary ===")
    for m in all_metrics:
        if m.get("skipped"):
            print(f"  Fold {m['fold_id']}: skipped (checkpoint exists)")
        else:
            print(f"  Fold {m.get('fold_id', '?')}: "
                  f"mAP@50={m.get('best_map50_seg', 0):.4f} (seg), "
                  f"mAP@50:95={m.get('best_map50_95_seg', 0):.4f}")

    # Save combined summary
    summary_path = RUNS_DIR / "summary_v5_yolo.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n✅ Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
