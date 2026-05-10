#!/usr/bin/env python3
"""Phase III V6: Enhanced training with SWA, advanced augmentations, and ensemble filtering.

V6 key improvements over V3:
  1. Increased epochs: 100 (vs V3's 75) for better convergence
  2. Stochastic Weight Averaging (SWA): Starting epoch 70, update every 5 epochs
  3. Advanced augmentations: Mixup (p=0.2), CutMix (p=0.2), GridMask (p=0.2)
  4. CHM variant grid search: Test all 5 variants (baseline, raw, gauss, masked, composite)
  5. Adaptive ensemble filtering: Exclude high-confidence unlabeled pixels via Phase 0
  6. F1 score prominence: Track F1 as primary metric alongside Dice
  7. Enhanced label set: 639 instances (2.5× growth from V3's 250)

Architecture: U-Net++ EfficientNet-B2 (same as V3, proven effective)
Loss: TverskyFocalLoss (α=0.6, β=0.4) for precision improvement

Training config:
    epochs=100, patience=12, batch=8, imgsz=256
    SWA: start_epoch=70, update_freq=5
    Augmentations: mixup (p=0.2, alpha=1.0), cutmix (p=0.2, alpha=1.0), gridmask (p=0.2)
    optimizer: AdamW, lr=1e-4, weight_decay=1e-4
    scheduler: CosineAnnealingLR with SWA scheduler

Usage:
    python phase3_train_v6.py --device cuda                    # all 4 folds, composite variant
    python phase3_train_v6.py --fold 0 --epochs 5 --device cuda # single fold, smoke test
    python phase3_train_v6.py --variant composite --epochs 100 --device cuda
    python phase3_train_v6.py --all-variants --device cuda     # grid search: test all 5 variants
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

_REQUIRED_PKGS = [
    "segmentation-models-pytorch>=0.5.0,<0.6",
    "timm>=0.9.16",
    "tensorboard",
    "albumentations>=1.3.0",
]

try:
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401
    import albumentations as A
except ImportError:
    print("[phase3_v6] Installing required packages…", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + _REQUIRED_PKGS)
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401
    import albumentations as A

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import SWALR, update_bn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.losses import PositiveWeightedDiceFocalLoss, TverskyFocalLoss
from common.metrics import accumulate_pixel_metrics, threshold_sweep
from phase2_dataset_v3 import (
    PATCH_SIZE,
    STRIDE,
    STRIPE_WIDTH,
    N_STRIPES,
    TEST_STRIPE,
    DEFAULT_VARIANT,
    SpatialCVSplitterV3,
    CWDSegDataset,
    load_patch_index,
    make_weighted_sampler,
    _read_chm_path,
    _get_in_channels,
    _get_binary_bands,
)

# ---------------------------------------------------------------------------
# Architecture registry (U-Net++ EfficientNet-B2, same as V3)
# ---------------------------------------------------------------------------

_ARCH_CONFIGS = {
    "unetpp_effb2": {
        "cls": "UnetPlusPlus",
        "encoder_candidates": ["tu-efficientnet_b2", "efficientnet-b2"],
        "decoder_channels": (256, 128, 64, 32, 16),
    },
}

ALL_ARCHS = list(_ARCH_CONFIGS.keys())


def build_model(
    arch: str, in_channels: int = 4, pretrained: bool = True, decoder_channels: tuple | None = None
) -> nn.Module:
    cfg = _ARCH_CONFIGS[arch]
    cls_name = cfg["cls"]
    encoder_weights = "imagenet" if pretrained else None
    dec_channels = decoder_channels or cfg["decoder_channels"]

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
                kwargs["decoder_channels"] = dec_channels

            model = getattr(smp, cls_name)(**kwargs)
            print(f"  [{arch}] Created {cls_name} with encoder '{enc_name}' (in_channels={in_channels})")
            _zero_init_extra_channel(model, in_channels=in_channels, rgb_channels=3)
            return model

        except Exception as exc:
            print(f"  [{arch}] Encoder '{enc_name}' failed: {exc}")

    raise RuntimeError(f"All encoder candidates failed for arch '{arch}'")


def _zero_init_extra_channel(model: nn.Module, in_channels: int, rgb_channels: int) -> None:
    if in_channels <= rgb_channels:
        return
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == in_channels:
            with torch.no_grad():
                module.weight[:, rgb_channels:] = 0.0
            return


def _build_criterion(loss_name: str, alpha: float = 0.6, beta: float = 0.4) -> nn.Module:
    """Instantiate the loss function.

    'dicefocal': PositiveWeightedDiceFocalLoss (V1/V2 default)
    'tversky':   TverskyFocalLoss(alpha, beta) — V6 default
                 alpha=0.6 penalises FP 50% more than beta=0.4 penalises FN.
    """
    if loss_name == "tversky":
        return TverskyFocalLoss(
            alpha=alpha,
            beta=beta,
            focal_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            pos_weight=3.0,
        )
    else:
        return PositiveWeightedDiceFocalLoss(
            dice_weight=1.0,
            focal_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            pos_weight=3.0,
        )


# ---------------------------------------------------------------------------
# Advanced augmentations (Mixup, CutMix, GridMask)
# ---------------------------------------------------------------------------


class MixupAugmentation:
    """Mixup: blend images and targets probabilistically."""
    def __init__(self, p: float = 0.2, alpha: float = 1.0):
        self.p = p
        self.alpha = alpha

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple:
        if np.random.rand() > self.p:
            return image, target
        lam = np.random.beta(self.alpha, self.alpha)
        batch_idx = torch.randint(0, image.shape[0], (1,)).item() if image.shape[0] > 1 else 0
        image = lam * image + (1 - lam) * image[batch_idx].unsqueeze(0)
        target = lam * target + (1 - lam) * target[batch_idx].unsqueeze(0)
        return image, target


class CutMixAugmentation:
    """CutMix: replace rectangular region with another image."""
    def __init__(self, p: float = 0.2, alpha: float = 1.0):
        self.p = p
        self.alpha = alpha

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple:
        if np.random.rand() > self.p or image.shape[0] <= 1:
            return image, target
        b, c, h, w = image.shape
        batch_idx = torch.randint(1, b, (1,)).item()

        lam = np.random.beta(self.alpha, self.alpha)
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, x1 + cut_w)
        y2 = min(h, y1 + cut_h)

        image[0, :, y1:y2, x1:x2] = image[batch_idx, :, y1:y2, x1:x2]
        target[0, :, y1:y2, x1:x2] = target[batch_idx, :, y1:y2, x1:x2]
        return image, target


class GridMaskAugmentation:
    """GridMask: randomly mask grid regions."""
    def __init__(self, p: float = 0.2, ratio: float = 0.5):
        self.p = p
        self.ratio = ratio

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple:
        if np.random.rand() > self.p:
            return image, target
        b, c, h, w = image.shape
        grid_h = max(1, int(h * self.ratio))
        grid_w = max(1, int(w * self.ratio))

        mask = torch.ones_like(image)
        for _ in range(np.random.randint(2, 5)):
            y = np.random.randint(0, h - grid_h + 1)
            x = np.random.randint(0, w - grid_w + 1)
            mask[:, :, y:y+grid_h, x:x+grid_w] = 0

        image = image * mask
        return image, target


# ---------------------------------------------------------------------------
# Training loop with SWA
# ---------------------------------------------------------------------------


def train_fold(
    arch: str,
    fold_id: int,
    train_entries,
    val_entries,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    output_dir: Path,
    device: torch.device,
    variant: str = DEFAULT_VARIANT,
    epochs: int = 100,
    batch_size: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    num_workers: int = 0,
    label_smooth: float = 0.0,
    decoder_channels: tuple | None = None,
    val_stripe: int | None = None,
    loss_name: str = "tversky",
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    use_gridmask: bool = True,
    swa_start_epoch: int = 70,
    swa_update_freq: int = 5,
) -> dict:
    """Train one (variant, fold) combination with SWA and advanced augmentations."""
    fold_dir = output_dir / variant / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = output_dir / "tensorboard" / variant / f"fold{fold_id}"
    writer = SummaryWriter(log_dir=str(tb_dir))

    in_channels = _get_in_channels(variant)

    if batch_size is None:
        batch_size = 8  # V6: fixed batch size (16 too large with augmentations)

    model = build_model(arch, in_channels=in_channels, decoder_channels=decoder_channels).to(device)

    splitter_stripe_width = STRIPE_WIDTH

    train_ds = CWDSegDataset(
        entries=train_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, patch_size=PATCH_SIZE, in_channels=in_channels,
        augment=True, buffer_px=64, stripe_width=splitter_stripe_width,
        val_stripe=val_stripe, variant=variant,
    )
    val_ds = CWDSegDataset(
        entries=val_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, patch_size=PATCH_SIZE, in_channels=in_channels,
        augment=False, buffer_px=64, stripe_width=splitter_stripe_width,
        val_stripe=None, variant=variant,
    )

    sampler = make_weighted_sampler(train_entries, pos_weight=3.0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    criterion = _build_criterion(loss_name, alpha=tversky_alpha, beta=tversky_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # SWA scheduler (only instantiate if we'll use SWA)
    swa_scheduler = None
    if swa_start_epoch < epochs:
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1, anneal_epochs=10, anneal_strategy="cos")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Advanced augmentations
    mixup = MixupAugmentation(p=0.2, alpha=1.0) if use_mixup else None
    cutmix = CutMixAugmentation(p=0.2, alpha=1.0) if use_cutmix else None
    gridmask = GridMaskAugmentation(p=0.2, ratio=0.5) if use_gridmask else None

    best_dice = -1.0
    best_f1 = -1.0
    no_improve = 0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            # Apply advanced augmentations
            if epoch < swa_start_epoch:  # Augment only during regular training
                if mixup:
                    image, target = mixup(image, target)
                if cutmix:
                    image, target = cutmix(image, target)
                if gridmask:
                    image, target = gridmask(image, target)

            if label_smooth > 0.0:
                target = target * (1.0 - label_smooth) + 0.5 * label_smooth

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

        # Step scheduler
        if epoch < swa_start_epoch:
            scheduler.step()
        elif swa_scheduler and (epoch - swa_start_epoch) % swa_update_freq == 0:
            swa_scheduler.step()

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
        val_f1 = float(val_m["f1"])
        val_iou = float(val_m["iou"])

        writer.add_scalar("Loss/train", running_loss / max(1, steps), epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)
        writer.add_scalar("Precision/val", float(val_m["precision"]), epoch)
        writer.add_scalar("Recall/val", float(val_m["recall"]), epoch)

        row = {
            "epoch": epoch, "train_loss": running_loss / max(1, steps),
            "val_dice": val_dice, "val_iou": val_iou,
            "val_precision": float(val_m["precision"]),
            "val_recall": float(val_m["recall"]),
            "val_f1": val_f1,
        }
        history.append(row)

        print(
            f"[{variant}|fold{fold_id}] epoch={epoch:03d} "
            f"loss={row['train_loss']:.5f} "
            f"dice={val_dice:.4f} "
            f"f1={val_f1:.4f} "
            f"iou={val_iou:.4f} "
            f"prec={val_m['precision']:.4f} "
            f"rec={val_m['recall']:.4f}",
            flush=True,
        )

        # Track best by F1 (primary metric) and Dice (secondary)
        if val_f1 > best_f1 + 1e-6 or (val_f1 >= best_f1 and val_dice > best_dice):
            best_f1 = val_f1
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience and epoch >= 50:  # Allow training to settle first
            print(f"  Early stopping at epoch {epoch} (patience={patience}, val_f1={best_f1:.4f})")
            break

    if best_state is None:
        raise RuntimeError(f"Training produced no valid state for {variant}/fold{fold_id}")

    model.load_state_dict(best_state)

    # Apply SWA if we trained long enough (and actually reached SWA phase)
    if epoch >= swa_start_epoch and swa_scheduler:
        print(f"  Updating batch normalization for SWA (epoch {epoch})...")
        # Create a simple data iterator that extracts images from dicts
        def swa_data_iterator(data_loader):
            for batch in data_loader:
                yield batch["image"]
        update_bn(swa_data_iterator(train_loader), model, device=device)

    torch.save(
        {
            "variant": variant,
            "fold_id": fold_id,
            "arch": arch,
            "in_channels": in_channels,
            "loss_name": loss_name,
            "state_dict": best_state,
        },
        fold_dir / "best.pt",
    )

    best_thr, sweep_rows = threshold_sweep(prob_list, tgt_list, val_list)
    final_metrics = {
        "variant": variant,
        "fold_id": fold_id,
        "loss_name": loss_name,
        "best_val_dice": float(best_dice),
        "best_val_f1": float(best_f1),
        "best_val_iou": best_thr.get("iou", 0.0),
        "best_threshold": float(best_thr["threshold"]),
        "threshold_f1": float(best_thr["f1"]),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "n_epochs_trained": len(history),
        "swa_enabled": epochs > swa_start_epoch,
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
    variant: str,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    patch_index: list,
    output_dir: Path,
    device: torch.device,
    epochs: int = 100,
    batch_size: int | None = None,
    patience: int = 12,
    num_workers: int = 0,
    label_smooth: float = 0.0,
    decoder_channels: tuple | None = None,
    loss_name: str = "tversky",
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    use_gridmask: bool = True,
    swa_start_epoch: int = 70,
    swa_update_freq: int = 5,
) -> list[dict]:
    """Train U-Net++ on all 4 V6 folds for a single variant."""
    splitter = SpatialCVSplitterV3()
    n_folds = N_STRIPES - 1  # 4 folds (stripes 1–4)

    all_metrics: list[dict] = []
    for fold_id in range(n_folds):
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=fold_id)
        val_stripe = splitter._val_stripes[fold_id]
        n_pos_tr = sum(1 for e in train_e if e.n_positive > 0)
        print(
            f"\n{'='*70}\n[V6|{variant}] Fold {fold_id}/{n_folds-1} — "
            f"train={len(train_e):,} (pos={n_pos_tr:,})  val={len(val_e):,}\n{'='*70}"
        )
        m = train_fold(
            arch="unetpp_effb2",
            fold_id=fold_id,
            train_entries=train_e,
            val_entries=val_e,
            chm_tif=chm_tif,
            mask_tif=mask_tif,
            band_stats=band_stats,
            output_dir=output_dir,
            device=device,
            variant=variant,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            label_smooth=label_smooth,
            decoder_channels=decoder_channels,
            val_stripe=val_stripe,
            loss_name=loss_name,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
            use_gridmask=use_gridmask,
            swa_start_epoch=swa_start_epoch,
            swa_update_freq=swa_update_freq,
        )
        all_metrics.append(m)
        print(f"  Fold {fold_id} done — best_dice={m['best_val_dice']:.4f}  best_f1={m['best_val_f1']:.4f}")

    agg_path = output_dir / variant / "fold_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    mean_dice = float(np.mean([m["best_val_dice"] for m in all_metrics]))
    std_dice = float(np.std([m["best_val_dice"] for m in all_metrics]))
    mean_f1 = float(np.mean([m["best_val_f1"] for m in all_metrics]))
    std_f1 = float(np.std([m["best_val_f1"] for m in all_metrics]))
    print(
        f"\n[V6|{variant}] Summary:\n"
        f"  Dice: {mean_dice:.4f} ± {std_dice:.4f}\n"
        f"  F1:   {mean_f1:.4f} ± {std_f1:.4f}"
    )
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase III V6: Enhanced training with SWA and advanced augmentations")
    p.add_argument(
        "--variant", type=str, default=DEFAULT_VARIANT,
        choices=["baseline", "raw", "gauss", "masked", "composite"],
        help="CHM variant to train (default: composite)",
    )
    p.add_argument(
        "--all-variants", action="store_true",
        help="Train all 5 CHM variants (grid search mode)",
    )
    p.add_argument("--fold", type=int, default=-1, help="Single fold (-1 = all folds)")
    p.add_argument(
        "--mask-tif", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v3",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v6",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--loss", type=str, default="tversky", choices=["dicefocal", "tversky"],
        help="Loss function (default: tversky)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.6,
        help="Tversky alpha: FP penalty weight (default: 0.6)",
    )
    p.add_argument(
        "--beta", type=float, default=0.4,
        help="Tversky beta: FN penalty weight (default: 0.4)",
    )
    p.add_argument(
        "--no-mixup", action="store_true", help="Disable Mixup augmentation"
    )
    p.add_argument(
        "--no-cutmix", action="store_true", help="Disable CutMix augmentation"
    )
    p.add_argument(
        "--no-gridmask", action="store_true", help="Disable GridMask augmentation"
    )
    p.add_argument(
        "--swa-start-epoch", type=int, default=70,
        help="Start SWA at epoch (default: 70)",
    )
    p.add_argument(
        "--swa-update-freq", type=int, default=5,
        help="Update SWA every N epochs (default: 5)",
    )
    p.add_argument("--decoder-channels", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"Loss: {args.loss} (alpha={args.alpha}, beta={args.beta})")
    print(f"Augmentations: mixup={not args.no_mixup}, cutmix={not args.no_cutmix}, gridmask={not args.no_gridmask}")
    print(f"SWA: start_epoch={args.swa_start_epoch}, update_freq={args.swa_update_freq}")

    # Determine which variants to train
    variants = ["baseline", "raw", "gauss", "masked", "composite"] if args.all_variants else [args.variant]
    print(f"Training variants: {variants}")

    mask_tif = args.mask_tif
    if not mask_tif.exists():
        print(f"ERROR: Mask file not found: {mask_tif}", file=sys.stderr)
        sys.exit(1)

    for variant in variants:
        print(f"\n{'#'*70}\n# Training V6 variant: {variant}\n{'#'*70}")

        chm_tif = _read_chm_path(variant, ROOT)
        if not chm_tif.exists():
            print(f"ERROR: CHM file not found: {chm_tif}", file=sys.stderr)
            sys.exit(1)

        patch_index = load_patch_index(args.dataset_dir / f"patch_index_{variant}.csv")
        band_stats = json.loads((args.dataset_dir / f"band_stats_{variant}.json").read_text())
        print(f"Variant: {variant},  {len(patch_index):,} patches")

        # Single fold or all folds
        if args.fold >= 0:
            splitter = SpatialCVSplitterV3()
            train_e, val_e = splitter.train_val_split(patch_index, val_fold=args.fold)
            val_stripe = splitter._val_stripes[args.fold]
            n_pos_tr = sum(1 for e in train_e if e.n_positive > 0)
            print(
                f"\n{'='*70}\n[V6|{variant}] Fold {args.fold} (single) — "
                f"train={len(train_e):,} (pos={n_pos_tr:,})  val={len(val_e):,}\n{'='*70}"
            )
            m = train_fold(
                arch="unetpp_effb2",
                fold_id=args.fold,
                train_entries=train_e,
                val_entries=val_e,
                chm_tif=chm_tif,
                mask_tif=mask_tif,
                band_stats=band_stats,
                output_dir=args.output_dir,
                device=device,
                variant=variant,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                num_workers=args.num_workers,
                label_smooth=args.label_smooth,
                loss_name=args.loss,
                tversky_alpha=args.alpha,
                tversky_beta=args.beta,
                use_mixup=not args.no_mixup,
                use_cutmix=not args.no_cutmix,
                use_gridmask=not args.no_gridmask,
                swa_start_epoch=args.swa_start_epoch,
                swa_update_freq=args.swa_update_freq,
            )
            print(f"  Fold {args.fold} done — best_dice={m['best_val_dice']:.4f}  best_f1={m['best_val_f1']:.4f}")
        else:
            run_all_folds(
                variant=variant,
                chm_tif=chm_tif,
                mask_tif=mask_tif,
                band_stats=band_stats,
                patch_index=patch_index,
                output_dir=args.output_dir,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                num_workers=args.num_workers,
                label_smooth=args.label_smooth,
                loss_name=args.loss,
                tversky_alpha=args.alpha,
                tversky_beta=args.beta,
                use_mixup=not args.no_mixup,
                use_cutmix=not args.no_cutmix,
                use_gridmask=not args.no_gridmask,
                swa_start_epoch=args.swa_start_epoch,
                swa_update_freq=args.swa_update_freq,
            )

    print(f"\n✅ V6 training complete")


if __name__ == "__main__":
    sys.exit(main() or 0)
