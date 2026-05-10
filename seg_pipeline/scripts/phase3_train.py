#!/usr/bin/env python3
"""Phase III: SOTA Architecture Benchmarking — 5-fold spatial CV.

Trains three segmentation architectures via segmentation_models_pytorch (SMP):
    unetpp_effb2      U-Net++ with EfficientNet-B2 encoder
    deeplabv3plus_r50 DeepLabV3+ with ResNet50 encoder
    segformer_b2      SegFormer (U-Net decoder) with Mix Transformer B2 encoder

All architectures:
    - 4-channel input (composite CHM: gauss, raw, baseline, validity_mask)
    - PositiveWeightedDiceFocalLoss (Dice + Focal, pos_weight=3)
    - AdamW, CosineAnnealingLR
    - AMP (CUDA only), gradient clipping 1.0
    - 50 epochs, patience=10 (early stopping on val Dice)
    - TensorBoard logging per fold

Training loop adapted from scripts/train_deeplabv3plus_manual_masks.py:967-1031.

Usage:
    python phase3_train.py --arch unetpp_effb2 --fold 0 --epochs 3 --device cuda
    python phase3_train.py --arch all --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Install SMP + timm inside Docker at startup (idempotent)
# ---------------------------------------------------------------------------

_REQUIRED_PKGS = [
    "segmentation-models-pytorch>=0.5.0,<0.6",
    "timm>=0.9.16",
    "tensorboard",
]

try:
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401
except ImportError:
    print("[phase3] Installing required packages…", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + _REQUIRED_PKGS)
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.losses import PositiveWeightedDiceFocalLoss
from common.metrics import accumulate_pixel_metrics, threshold_sweep
from common.sliding_window import sliding_window_predict
from phase2_dataset import (
    PATCH_SIZE,
    STRIDE,
    BINARY_BANDS,
    SpatialCVSplitter,
    CWDSegDataset,
    load_patch_index,
    make_weighted_sampler,
    N_STRIPES,
    TEST_STRIPE,
)

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCH_CONFIGS = {
    "unetpp_effb2": {
        "cls": "UnetPlusPlus",
        "encoder_candidates": ["tu-efficientnet_b2", "efficientnet-b2"],
        "decoder_channels": (256, 128, 64, 32, 16),
    },
    "deeplabv3plus_r50": {
        "cls": "DeepLabV3Plus",
        "encoder_candidates": ["resnet50"],
        "decoder_atrous_rates": (6, 12, 18),
    },
    "segformer_b2": {
        "cls": "Unet",
        "encoder_candidates": ["tu-mit_b2", "mit_b2"],
        "decoder_channels": (256, 128, 64, 32, 16),
    },
}

ALL_ARCHS = list(_ARCH_CONFIGS.keys())


def build_model(arch: str, in_channels: int = 4, pretrained: bool = True) -> nn.Module:
    """Instantiate an SMP model with encoder-name fallback chain."""
    cfg = _ARCH_CONFIGS[arch]
    cls_name = cfg["cls"]
    encoder_weights = "imagenet" if pretrained else None

    for enc_name in cfg["encoder_candidates"]:
        try:
            kwargs: dict = dict(
                encoder_name=enc_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
            if cls_name == "UnetPlusPlus":
                kwargs["decoder_channels"] = cfg["decoder_channels"]
            elif cls_name == "DeepLabV3Plus":
                kwargs["decoder_atrous_rates"] = cfg["decoder_atrous_rates"]
            elif cls_name == "Unet":
                kwargs["decoder_channels"] = cfg["decoder_channels"]

            model = getattr(smp, cls_name)(**kwargs)
            print(f"  [{arch}] Created {cls_name} with encoder '{enc_name}'")

            # Zero-initialize the Band 4 (validity mask) channel weights in the
            # first Conv2d so early training is dominated by the CHM bands.
            _zero_init_extra_channel(model, in_channels=in_channels, rgb_channels=3)

            return model

        except Exception as exc:
            print(f"  [{arch}] Encoder '{enc_name}' failed: {exc}")

    raise RuntimeError(f"All encoder candidates failed for arch '{arch}': {cfg['encoder_candidates']}")


def _zero_init_extra_channel(model: nn.Module, in_channels: int, rgb_channels: int) -> None:
    """Zero out weight columns for channels beyond the 3 RGB pretrained channels."""
    if in_channels <= rgb_channels:
        return
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == in_channels:
            with torch.no_grad():
                module.weight[:, rgb_channels:] = 0.0
            return


# ---------------------------------------------------------------------------
# Training loop — adapted from train_deeplabv3plus_manual_masks.py:967-1031
# ---------------------------------------------------------------------------


def train_fold(
    arch: str,
    fold_id: int,
    train_entries,
    val_entries,
    composite_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    output_dir: Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    num_workers: int = 4,
    val_stripe: int | None = None,
) -> dict:
    """Train one (arch, fold) combination and return best validation metrics."""
    fold_dir = output_dir / arch / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = output_dir / "tensorboard" / arch / f"fold{fold_id}"
    writer = SummaryWriter(log_dir=str(tb_dir))

    model = build_model(arch).to(device)

    train_ds = CWDSegDataset(
        entries=train_entries,
        composite_tif=composite_tif,
        mask_tif=mask_tif,
        band_stats=band_stats,
        augment=True,
        val_stripe=val_stripe,
    )
    val_ds = CWDSegDataset(
        entries=val_entries,
        composite_tif=composite_tif,
        mask_tif=mask_tif,
        band_stats=band_stats,
        augment=False,
    )

    sampler = make_weighted_sampler(train_entries, pos_weight=3.0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    criterion = PositiveWeightedDiceFocalLoss(
        dice_weight=1.0,
        focal_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_weight=3.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_dice = -1.0
    no_improve = 0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(image)
                loss = criterion(logits, target, valid)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach())
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(1, steps)

        # --- Validate ---
        model.eval()
        prob_list, tgt_list, val_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                logits = model(image)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
                for k in range(len(probs)):
                    prob_list.append(probs[k])
                    tgt_list.append(batch["target"][k, 0].numpy())
                    val_list.append(batch["valid"][k, 0].numpy())

        val_m = accumulate_pixel_metrics(prob_list, tgt_list, val_list, threshold=0.5)
        val_dice = float(val_m["dice"])
        val_iou = float(val_m["iou"])

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_precision": float(val_m["precision"]),
            "val_recall": float(val_m["recall"]),
            "val_f1": float(val_m["f1"]),
        }
        history.append(row)

        print(
            f"[{arch}|fold{fold_id}] epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_dice={val_dice:.4f} "
            f"val_iou={val_iou:.4f}",
            flush=True,
        )

        if val_dice > best_dice + 1e-6:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is None:
        raise RuntimeError(f"Training produced no valid state for {arch}/fold{fold_id}")

    model.load_state_dict(best_state)
    torch.save(
        {"arch": arch, "fold_id": fold_id, "state_dict": best_state},
        fold_dir / "best.pt",
    )

    # Threshold sweep on val set
    best_thr, sweep_rows = threshold_sweep(prob_list, tgt_list, val_list)
    final_metrics = {
        "arch": arch,
        "fold_id": fold_id,
        "best_val_dice": float(best_dice),
        "best_val_iou": best_thr.get("iou", 0.0),
        "best_threshold": float(best_thr["threshold"]),
        "best_f1": float(best_thr["f1"]),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "n_epochs_trained": len(history),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

    _append_history_csv(fold_dir / "history.csv", history)
    writer.close()

    return final_metrics


def _append_history_csv(path: Path, rows: list[dict]) -> None:
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# All-folds runner
# ---------------------------------------------------------------------------


def run_all_folds(
    arch: str,
    composite_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    patch_index: list,
    output_dir: Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    patience: int = 10,
    num_workers: int = 4,
) -> list[dict]:
    splitter = SpatialCVSplitter()
    n_folds = N_STRIPES - 1  # 4 folds (stripes 1-4)

    all_metrics: list[dict] = []
    for fold_id in range(n_folds):
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=fold_id)
        val_stripe = splitter._val_stripes[fold_id]
        print(
            f"\n{'='*60}\n[{arch}] Fold {fold_id}/{n_folds-1} — "
            f"train={len(train_e):,}  val={len(val_e):,}\n{'='*60}"
        )
        m = train_fold(
            arch=arch,
            fold_id=fold_id,
            train_entries=train_e,
            val_entries=val_e,
            composite_tif=composite_tif,
            mask_tif=mask_tif,
            band_stats=band_stats,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            val_stripe=val_stripe,
        )
        all_metrics.append(m)
        print(f"  Fold {fold_id} done — best_dice={m['best_val_dice']:.4f}  best_thr={m['best_threshold']:.2f}")

    # Write aggregated history CSV
    agg_path = output_dir / arch / "fold_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    mean_dice = float(np.mean([m["best_val_dice"] for m in all_metrics]))
    std_dice = float(np.std([m["best_val_dice"] for m in all_metrics]))
    print(f"\n[{arch}] 5-fold Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase III: SOTA model benchmarking")
    p.add_argument(
        "--arch",
        type=str,
        default="all",
        help=f"Architecture to train: {ALL_ARCHS} or 'all'",
    )
    p.add_argument("--fold", type=int, default=-1, help="Single fold to train (-1 = all folds)")
    p.add_argument(
        "--composite-tif",
        type=Path,
        default=ROOT / "seg_pipeline" / "input" / "composite_4band.tif",
    )
    p.add_argument(
        "--mask-tif",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Load dataset index and band stats
    patch_index = load_patch_index(args.dataset_dir / "patch_index.csv")
    band_stats = json.loads((args.dataset_dir / "band_stats.json").read_text())
    print(f"Patch index: {len(patch_index):,} patches")

    archs = ALL_ARCHS if args.arch == "all" else [args.arch]

    for arch in archs:
        if arch not in _ARCH_CONFIGS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from: {ALL_ARCHS}")

        if args.fold >= 0:
            # Single fold for debugging
            splitter = SpatialCVSplitter()
            train_e, val_e = splitter.train_val_split(patch_index, val_fold=args.fold)
            val_stripe = splitter._val_stripes[args.fold]
            train_fold(
                arch=arch,
                fold_id=args.fold,
                train_entries=train_e,
                val_entries=val_e,
                composite_tif=args.composite_tif,
                mask_tif=args.mask_tif,
                band_stats=band_stats,
                output_dir=args.output_dir,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                num_workers=args.num_workers,
                val_stripe=val_stripe,
            )
        else:
            run_all_folds(
                arch=arch,
                composite_tif=args.composite_tif,
                mask_tif=args.mask_tif,
                band_stats=band_stats,
                patch_index=patch_index,
                output_dir=args.output_dir,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                num_workers=args.num_workers,
            )


if __name__ == "__main__":
    main()
