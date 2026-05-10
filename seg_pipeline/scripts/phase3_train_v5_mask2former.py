#!/usr/bin/env python3
"""Phase III-b V5: Mask2Former fine-tuning for CWD instance segmentation.

Fine-tunes facebook/mask2former-swin-tiny-coco-instance on COCO-format
CHM pseudo-RGB patches using 4-fold spatial cross-validation.

Architecture:
    Mask2Former (Cheng et al., 2022) — transformer-based instance segmentation
    Backbone: Swin-Tiny (28M params), pixel decoder, transformer decoder
    Loss: bipartite matching (Hungarian algorithm) on set-prediction queries
    Input: 3-channel CHM pseudo-RGB, resized to 640×640
    Output: up to 100 instance masks per image (num_queries=100)

Training config:
    epochs=50, patience=10, batch=4
    lr=5e-5 (backbone) / 5e-4 (head), CosineAnnealingLR

Dependencies:
    pip install transformers pycocotools

Usage:
    python phase3_train_v5_mask2former.py                   # all 4 folds
    python phase3_train_v5_mask2former.py --fold 0          # single fold
    python phase3_train_v5_mask2former.py --fold 0 --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _ensure_transformers() -> None:
    """Ensure a transformers version compatible with the current PyTorch is installed."""
    try:
        import transformers
        # transformers 5.x requires PyTorch >= 2.4; downgrade if needed
        from packaging.version import Version
        if Version(transformers.__version__) >= Version("5.0"):
            torch_version = Version(torch.__version__.split("+")[0])
            if torch_version < Version("2.4"):
                print(f"⚠ transformers {transformers.__version__} requires PyTorch >= 2.4 "
                      f"(found {torch.__version__}). Downgrading to transformers<5.0...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                       "transformers>=4.35,<5.0", "--force-reinstall"])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "transformers>=4.35,<5.0"])


_ensure_transformers()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

DATASET_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v5" / "coco"
RUNS_DIR = ROOT / "seg_pipeline" / "output" / "phase3_runs_v5" / "mask2former"
MODEL_NAME = "facebook/mask2former-swin-tiny-coco-instance"
PATCH_SIZE = 640


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CocoInstanceDataset(Dataset):
    """Loads COCO-format instance segmentation patches.

    Each sample: dict with 'pixel_values' (3, H, W) float32 tensor and
    'labels' — list of instance polygon dicts in COCO format.
    """

    def __init__(self, coco_json_path: Path, image_dir: Path, processor) -> None:
        import json as _json
        self.processor = processor
        self.image_dir = image_dir

        data = _json.loads(coco_json_path.read_text())
        self.images = {img["id"]: img for img in data["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        self.id2anns: dict[int, list] = {}
        for ann in data["annotations"]:
            self.id2anns.setdefault(ann["image_id"], []).append(ann)
        self.image_ids = sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        img_id = self.image_ids[idx]
        img_meta = self.images[img_id]
        img_path = self.image_dir / img_meta["file_name"]

        image = Image.open(img_path).convert("RGB")
        anns = self.id2anns.get(img_id, [])

        # Build instance masks from COCO segmentation polygons
        masks, class_labels = [], []
        for ann in anns:
            if not ann["segmentation"]:
                continue
            mask = self._poly_to_mask(ann["segmentation"], img_meta["height"], img_meta["width"])
            if mask.sum() < 10:
                continue
            masks.append(mask)
            class_labels.append(0)  # single class: CWD (mapped to class 0 in processor)

        # Process with Mask2Former processor
        if masks:
            inputs = self.processor(
                images=image,
                segmentation_maps=None,
                instance_id_to_semantic_id=None,
                return_tensors="pt",
            )
            # Build target dicts manually
            h, w = img_meta["height"], img_meta["width"]
            target = {
                "masks": torch.stack([torch.from_numpy(m) for m in masks]),
                "labels": torch.tensor(class_labels, dtype=torch.long),
            }
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            target = {
                "masks": torch.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=torch.bool),
                "labels": torch.zeros(0, dtype=torch.long),
            }

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "pixel_mask": inputs.get("pixel_mask", torch.ones(PATCH_SIZE, PATCH_SIZE)).squeeze(0)
            if "pixel_mask" in inputs else torch.ones(PATCH_SIZE, PATCH_SIZE, dtype=torch.long),
            "target": target,
        }

    @staticmethod
    def _poly_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
        """Convert COCO polygon segmentation to binary mask."""
        try:
            from pycocotools import mask as coco_mask
            import itertools
            rle = coco_mask.frPyObjects(segmentation, height, width)
            m = coco_mask.decode(coco_mask.merge(rle))
            return m.astype(bool)
        except Exception:
            mask = np.zeros((height, width), dtype=bool)
            if segmentation:
                from PIL import ImageDraw, Image as PILImage
                poly_img = PILImage.new("L", (width, height), 0)
                draw = ImageDraw.Draw(poly_img)
                for seg in segmentation:
                    if len(seg) >= 6:
                        coords = list(zip(seg[::2], seg[1::2]))
                        draw.polygon(coords, fill=1)
                mask = np.array(poly_img, dtype=bool)
            return mask


def collate_fn(batch: list[dict]) -> dict:
    """Stack pixel_values; keep targets as list (variable number of instances)."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    targets = [b["target"] for b in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "targets": targets}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_fold(
    fold_id: int,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 4,
    device_str: str = "cuda",
    resume: bool = False,
) -> dict:
    from transformers import (
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )

    fold_coco_dir = DATASET_DIR / f"fold{fold_id}"
    run_dir = RUNS_DIR / f"fold{fold_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = run_dir / "checkpoint" / "model.pt"
    if best_ckpt.exists() and not resume:
        print(f"  Fold {fold_id}: checkpoint exists at {best_ckpt}, skipping")
        return _load_metrics(run_dir)

    print(f"\n=== Fold {fold_id} ===")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load processor and model
    print(f"  Loading {MODEL_NAME}...")
    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_NAME,
        do_resize=True,
        size={"height": PATCH_SIZE, "width": PATCH_SIZE},
        do_rescale=True,
        do_normalize=True,
        ignore_index=255,
        reduce_labels=False,
        num_labels=1,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Datasets
    img_dir = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v5" / "yolo" / f"fold{fold_id}"
    train_ds = CocoInstanceDataset(fold_coco_dir / "train.json", img_dir, processor)
    val_ds = CocoInstanceDataset(fold_coco_dir / "val.json", img_dir, processor)
    print(f"  Train: {len(train_ds)} patches, Val: {len(val_ds)} patches")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # Optimizer: lower LR for backbone, higher for head
    backbone_params = [p for n, p in model.named_parameters() if "model.pixel_level_module.encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "model.pixel_level_module.encoder" not in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 5e-5},
        {"params": head_params, "lr": 5e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    metrics_history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            targets = batch["targets"]

            # Format targets for Mask2Former
            formatted_targets = []
            for t in targets:
                formatted_targets.append({
                    "masks": t["masks"].to(device).bool(),
                    "labels": t["labels"].to(device),
                })

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                mask_labels=[t["masks"].float() for t in formatted_targets],
                class_labels=[t["labels"] for t in formatted_targets],
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                targets = batch["targets"]
                formatted_targets = [
                    {"masks": t["masks"].to(device).bool(), "labels": t["labels"].to(device)}
                    for t in targets
                ]
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=[t["masks"].float() for t in formatted_targets],
                    class_labels=[t["labels"] for t in formatted_targets],
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        metrics_history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Early stopping + checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            ckpt_dir = run_dir / "checkpoint"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            processor.save_pretrained(str(ckpt_dir))
            print(f"    ✓ New best checkpoint (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    final_metrics = {
        "fold_id": fold_id,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epoch + 1,
        "history": metrics_history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print(f"  ✓ Fold {fold_id}: best_val_loss={best_val_loss:.4f} @ epoch {best_epoch}")
    return final_metrics


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {"fold_id": int(run_dir.name.replace("fold", "")), "skipped": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase III-b V5: Mask2Former training")
    parser.add_argument("--fold", type=int, default=None, help="Single fold (0-3). Default: all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(4))

    print(f"=== Phase III-b V5: Mask2Former Training ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {args.device}, folds: {folds}")

    all_metrics = []
    for fold_id in folds:
        m = train_fold(
            fold_id=fold_id,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device_str=args.device,
            resume=args.resume,
        )
        all_metrics.append(m)

    print("\n=== Training Summary ===")
    for m in all_metrics:
        if m.get("skipped"):
            print(f"  Fold {m['fold_id']}: skipped")
        else:
            print(f"  Fold {m.get('fold_id', '?')}: best_val_loss={m.get('best_val_loss', -1):.4f}")

    summary_path = RUNS_DIR / "summary_v5_mask2former.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n✅ Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
