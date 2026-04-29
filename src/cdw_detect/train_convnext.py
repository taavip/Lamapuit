"""
Training loop for ConvNeXt V2 + Mask R-CNN CDW detection.

Features
--------
- AdamW optimiser + cosine-annealing LR with linear warm-up
- Mixed-precision (torch.cuda.amp)
- Best-model tracking by validation loss
- Gradient clipping
- Detailed per-epoch logging
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from cdw_detect.dataset_tiles import TileDataset, collate_fn
from cdw_detect.model_convnext import build_convnext_maskrcnn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _cosine_lr(optimizer, step: int, total_steps: int, warmup_steps: int, base_lr: float):
    """Apply cosine LR with linear warm-up in-place."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _compute_val_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Run forward in *training* mode on val set to get loss dict (Mask R-CNN
    only returns losses in train mode).  Use no_grad to save memory."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Skip batches with no annotations (would cause division-by-zero in heads)
            if all(t["boxes"].numel() == 0 for t in targets):
                continue
            try:
                loss_dict = model(images, targets)
                total_loss += sum(loss_dict.values()).item()
                n_batches += 1
            except Exception:
                pass
    return total_loss / max(1, n_batches)


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────


def train_convnext(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "convnextv2_tiny",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: float = 3.0,
    grad_clip: float = 5.0,
    device_id: int | str = 0,
    num_workers: int = 4,
    pretrained_backbone: bool = True,
    tile_size: int = 640,
) -> Path:
    """Train ConvNeXt Mask R-CNN and return path to best weights.

    Parameters
    ----------
    dataset_dir : root of the dataset produced by prepare_instance.py
                  (expects images/{train,val} and labels/{train,val})
    output_dir  : where to save checkpoints and training log
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f"cuda:{device_id}"
        if isinstance(device_id, int)
        else device_id if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    # ── datasets / loaders ──────────────────────────────────────────────────
    train_ds = TileDataset(
        dataset_dir / "images" / "train",
        dataset_dir / "labels" / "train",
        tile_size=tile_size,
    )
    val_ds = TileDataset(
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "val",
        tile_size=tile_size,
    )
    logger.info("Train tiles: %d  |  Val tiles: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ── model ────────────────────────────────────────────────────────────────
    logger.info("Building ConvNeXt V2 Mask R-CNN: %s", model_name)
    model = build_convnext_maskrcnn(
        model_name=model_name,
        num_classes=2,
        pretrained_backbone=pretrained_backbone,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %.1fM total, %.1fM trainable", total_params / 1e6, trainable / 1e6)

    # ── optimiser ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_epochs * len(train_loader))

    # ── training ─────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_weights = output_dir / "best.pt"
    history: list[dict] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip batches that are all-negative (Mask R-CNN chokes on zero-box batches)
            if all(t["boxes"].numel() == 0 for t in targets):
                continue

            current_lr = _cosine_lr(optimizer, global_step, total_steps, warmup_steps, lr)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_train_loss = epoch_loss / max(1, n_batches)
        avg_val_loss = _compute_val_loss(model, val_loader, device)
        elapsed = time.time() - epoch_start

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | %.0fs",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            current_lr,
            elapsed,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(avg_val_loss, 5),
            "lr": round(current_lr, 8),
        }
        history.append(epoch_record)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best_val_loss},
                best_weights,
            )
            logger.info("  ↑ New best val_loss=%.4f — saved %s", best_val_loss, best_weights)

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_dir / "last.pt",
        )

    # ── save training history ────────────────────────────────────────────────
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val_loss=%.4f at %s", best_val_loss, best_weights)
    return best_weights
