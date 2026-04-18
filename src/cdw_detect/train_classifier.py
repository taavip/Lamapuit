"""
Binary tile classifier training for CDW detection.

Trains a ResNet-18 (pretrained) binary/3-class classifier
(cdw / no_cdw / unknown) from per-raster label CSVs.

Usage
-----
python -m cdw_detect.train_classifier \
    --labels-dir   output/tile_labels \
    --chm-dir      chm_max_hag \
    --output       output/tile_labels/classifier \
    --epochs       30 \
    --batch-size   32
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

# ── label mapping ─────────────────────────────────────────────────────────────
LABEL2IDX: dict[str, int] = {"no_cdw": 0, "cdw": 1, "unknown": 2}
IDX2LABEL: dict[int, str] = {v: k for k, v in LABEL2IDX.items()}
NUM_CLASSES = 3


# ── dataset ───────────────────────────────────────────────────────────────────
def _normalize_tile(data: np.ndarray) -> np.ndarray:
    """p2-p98 stretch + CLAHE identical to training pipeline."""
    valid = data[(data > 0) & np.isfinite(data)]
    if valid.size < 10:
        return np.zeros_like(data, dtype=np.uint8)
    p2, p98 = np.percentile(valid, [2, 98])
    span = max(p98 - p2, 1e-6)
    img = np.clip((data - p2) / span, 0, 1)
    img8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img8)


class TileClassificationDataset(Dataset):
    """Reads CHM rasters on-the-fly and extracts labelled chunks."""

    def __init__(
        self,
        samples: list[dict],
        chm_dir: Path,
        chunk_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.chm_dir = chm_dir
        self.chunk_size = chunk_size
        self.augment = augment

        base = T.Compose(
            [
                T.ToTensor(),  # H×W uint8 → (1,H,W) float [0,1]
                T.Lambda(lambda x: x.repeat(3, 1, 1)),  # grey→ 3ch
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        aug = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(90),
            ]
        )
        self.tfm_base = base
        self.tfm_aug = aug

    # rasterio handle cache (one per chm stem)
    _raster_cache: dict[str, rasterio.DatasetReader] = {}

    def _get_raster(self, raster_stem: str) -> rasterio.DatasetReader:
        if raster_stem not in self._raster_cache:
            # find the file
            matches = sorted(self.chm_dir.glob(f"{raster_stem}*.tif"))
            if not matches:
                raise FileNotFoundError(f"No raster for stem '{raster_stem}' in {self.chm_dir}")
            # open without closing (Python will GC when dataset is freed)
            self._raster_cache[raster_stem] = rasterio.open(matches[0])
        return self._raster_cache[raster_stem]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        src = self._get_raster(s["raster"])
        cs = self.chunk_size

        window = rasterio.windows.Window(
            col_off=int(s["col_off"]),
            row_off=int(s["row_off"]),
            width=cs,
            height=cs,
        )
        data = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
        img8 = _normalize_tile(data)  # (H,W) uint8
        img8 = cv2.resize(img8, (cs, cs), interpolation=cv2.INTER_AREA)

        tensor = self.tfm_base(img8)
        if self.augment:
            tensor = self.tfm_aug(tensor)

        return tensor, LABEL2IDX[s["label"]]


# ── model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, frozen_backbone: bool = False) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if frozen_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    return model


# ── data loading helpers ───────────────────────────────────────────────────────
def _read_label_csvs(labels_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        raster_stem = csv_path.stem.replace("_labels", "")
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                lbl = row.get("label", "")
                if lbl not in LABEL2IDX:
                    continue
                samples.append(
                    {
                        "raster": raster_stem,
                        "row_off": int(row["row_off"]),
                        "col_off": int(row["col_off"]),
                        "label": lbl,
                    }
                )
    return samples


def _stratified_split(
    samples: list[dict], val_frac: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for lbl, group in by_label.items():
        rng.shuffle(group)
        split = max(1, int(len(group) * (1 - val_frac)))
        train += group[:split]
        val += group[split:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── training loop ─────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = _read_label_csvs(Path(args.labels_dir))

    if not samples:
        print("No labelled samples found — run label_tiles.py first.")
        return

    # filter out 'unknown' for training, keep for stats
    unknowns = [s for s in samples if s["label"] == "unknown"]
    training_samples = [s for s in samples if s["label"] != "unknown"]

    print(
        f"\nSamples  cdw={sum(1 for s in samples if s['label']=='cdw')}  "
        f"no_cdw={sum(1 for s in samples if s['label']=='no_cdw')}  "
        f"unknown={len(unknowns)} (excluded from training)"
    )

    train_s, val_s = _stratified_split(training_samples, val_frac=0.2)
    print(f"Train: {len(train_s)}   Val: {len(val_s)}")

    chm_dir = Path(args.chm_dir)
    train_ds = TileClassificationDataset(train_s, chm_dir, augment=True)
    val_ds = TileClassificationDataset(val_s, chm_dir, augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = build_model(num_classes=2).to(device)  # 2-class: no_cdw vs cdw
    # Override: train binary (cdw vs no_cdw), ignore 'unknown'
    # Remap: no_cdw→0, cdw→1 (unknown filtered beforehand)
    LABEL2IDX["no_cdw"] = 0
    LABEL2IDX["cdw"] = 1

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    nsteps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=nsteps)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = output_dir / "best_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = correct = total = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optim.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            sched.step()
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        train_loss /= total
        train_acc = 100 * correct / total

        # ── val ──
        model.eval()
        val_loss = vcorrect = vtotal = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += loss_fn(logits, labels).item() * imgs.size(0)
                vcorrect += (logits.argmax(1) == labels).sum().item()
                vtotal += imgs.size(0)
        val_loss /= vtotal
        val_acc = 100 * vcorrect / vtotal

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  acc={train_acc:.1f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  ✓ saved best  val_acc={val_acc:.1f}%")

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print(f"Weights saved to:  {best_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Train ResNet-18 binary CDW tile classifier")
    p.add_argument("--labels-dir", default="output/tile_labels")
    p.add_argument("--chm-dir", default="chm_max_hag")
    p.add_argument("--output", default="output/tile_labels/classifier")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
