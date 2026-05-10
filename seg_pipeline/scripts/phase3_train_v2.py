#!/usr/bin/env python3
"""Phase III V2: Model Training with CHM Variant Support & 3-fold CV.

Trains U-Net++ architecture across CHM variants (baseline, raw, gauss, masked, composite)
with 3-fold vertical-stripe spatial cross-validation, conflict zone masking, and
nodata dropout augmentation.

Key improvements over V1:
  - CHM variant support via --chm-variant flag
  - 3-fold CV (larger train sets, only 2 folds per variant)
  - Conflict zone masking from Phase II
  - Nodata dropout augmentation
  - Extended training (75 epochs default, patience=12)
  - Optional label smoothing
  - Adaptive batch size and in_channels per variant

Usage:
    python phase3_train_v2.py --chm-variant baseline --fold 0 --epochs 3 --device cuda
    python phase3_train_v2.py --chm-variant baseline --epochs 75 --device cuda
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
    print("[phase3_v2] Installing required packages…", flush=True)
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
from phase2_dataset_v2 import (
    PATCH_SIZE,
    STRIDE,
    STRIPE_WIDTH,
    N_STRIPES,
    TEST_STRIPE,
    SpatialCVSplitterV2,
    CWDSegDataset,
    load_patch_index,
    make_weighted_sampler,
    _read_chm_path,
    _get_in_channels,
    _get_binary_bands,
)

# ---------------------------------------------------------------------------
# Architecture registry (V1 + V2 compatible, U-Net++ only for V2)
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
    """Instantiate an SMP model with encoder-name fallback chain."""
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

            # Zero-initialize channels beyond RGB if in_channels > 3
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
# Training loop with label smoothing support
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
    variant: str = "baseline",
    epochs: int = 75,
    batch_size: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    num_workers: int = 0,
    label_smooth: float = 0.0,
    decoder_channels: tuple | None = None,
    val_stripe: int | None = None,
) -> dict:
    """Train one (variant, fold) combination and return best validation metrics."""
    fold_dir = output_dir / variant / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = output_dir / "tensorboard" / variant / f"fold{fold_id}"
    writer = SummaryWriter(log_dir=str(tb_dir))

    in_channels = _get_in_channels(variant)
    binary_bands = _get_binary_bands(variant)

    # Adaptive batch size: single-band variants can use larger batch size
    if batch_size is None:
        batch_size = 16 if in_channels == 1 else 8

    model = build_model(arch, in_channels=in_channels, decoder_channels=decoder_channels).to(device)

    train_ds = CWDSegDataset(
        entries=train_entries,
        chm_tif=chm_tif,
        mask_tif=mask_tif,
        band_stats=band_stats,
        patch_size=PATCH_SIZE,
        in_channels=in_channels,
        augment=True,
        buffer_px=64,
        stripe_width=STRIPE_WIDTH,
        val_stripe=val_stripe,
        variant=variant,
    )
    val_ds = CWDSegDataset(
        entries=val_entries,
        chm_tif=chm_tif,
        mask_tif=mask_tif,
        band_stats=band_stats,
        patch_size=PATCH_SIZE,
        in_channels=in_channels,
        augment=False,
        buffer_px=64,
        stripe_width=STRIPE_WIDTH,
        val_stripe=None,
        variant=variant,
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

            # Optional label smoothing
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
            f"[{variant}|fold{fold_id}] epoch={epoch:03d} "
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
        raise RuntimeError(f"Training produced no valid state for {variant}/fold{fold_id}")

    model.load_state_dict(best_state)
    torch.save(
        {"variant": variant, "fold_id": fold_id, "state_dict": best_state},
        fold_dir / "best.pt",
    )

    # Threshold sweep on val set
    best_thr, sweep_rows = threshold_sweep(prob_list, tgt_list, val_list)
    final_metrics = {
        "variant": variant,
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
# All-folds runner (3-fold version)
# ---------------------------------------------------------------------------


def run_all_folds(
    variant: str,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    patch_index: list,
    output_dir: Path,
    device: torch.device,
    epochs: int = 75,
    batch_size: int | None = None,
    patience: int = 12,
    num_workers: int = 0,
    label_smooth: float = 0.0,
    decoder_channels: tuple | None = None,
) -> list[dict]:
    """Train U-Net++ on all 3-fold CV folds for a single variant."""
    splitter = SpatialCVSplitterV2()
    n_folds = N_STRIPES - 1  # 2 folds (stripes 1-2)

    all_metrics: list[dict] = []
    for fold_id in range(n_folds):
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=fold_id)
        val_stripe = splitter._val_stripes[fold_id]
        print(
            f"\n{'='*60}\n[{variant}] Fold {fold_id}/{n_folds-1} — "
            f"train={len(train_e):,}  val={len(val_e):,}\n{'='*60}"
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
        )
        all_metrics.append(m)
        print(f"  Fold {fold_id} done — best_dice={m['best_val_dice']:.4f}  best_thr={m['best_threshold']:.2f}")

    # Write aggregated summary CSV
    agg_path = output_dir / variant / "fold_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    mean_dice = float(np.mean([m["best_val_dice"] for m in all_metrics]))
    std_dice = float(np.std([m["best_val_dice"] for m in all_metrics]))
    print(f"\n[{variant}] 3-fold Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase III V2: Model training with CHM variant support")
    p.add_argument(
        "--chm-variant",
        type=str,
        required=True,
        choices=["baseline", "raw", "gauss", "masked", "composite"],
        help="CHM variant to train on",
    )
    p.add_argument("--fold", type=int, default=-1, help="Single fold to train (-1 = all folds)")
    p.add_argument(
        "--mask-tif",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v2",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v2",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--epochs", type=int, default=75, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size (auto-adjusted per variant if None)")
    p.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for stability)")
    p.add_argument("--label-smooth", type=float, default=0.0, help="Label smoothing factor [0, 1]")
    p.add_argument(
        "--decoder-channels",
        type=str,
        default=None,
        help="Comma-separated decoder channels (e.g., '512,256,128,64,32')",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Resolve CHM path
    chm_tif = _read_chm_path(args.chm_variant, ROOT)
    if not chm_tif.exists():
        print(f"ERROR: CHM variant file does not exist: {chm_tif}", file=sys.stderr)
        sys.exit(1)

    # Load dataset index and band stats for this variant
    patch_index = load_patch_index(args.dataset_dir / f"patch_index_{args.chm_variant}.csv")
    band_stats = json.loads((args.dataset_dir / f"band_stats_{args.chm_variant}.json").read_text())
    print(f"Variant: {args.chm_variant}")
    print(f"Patch index: {len(patch_index):,} patches")
    print(f"Band stats loaded for {len(band_stats)} band(s)")

    # Parse decoder channels if provided
    decoder_channels = None
    if args.decoder_channels:
        decoder_channels = tuple(int(x.strip()) for x in args.decoder_channels.split(","))
        print(f"Decoder channels: {decoder_channels}")

    if args.fold >= 0:
        # Single fold for debugging
        splitter = SpatialCVSplitterV2()
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=args.fold)
        val_stripe = splitter._val_stripes[args.fold]
        train_fold(
            arch="unetpp_effb2",
            fold_id=args.fold,
            train_entries=train_e,
            val_entries=val_e,
            chm_tif=chm_tif,
            mask_tif=args.mask_tif,
            band_stats=band_stats,
            output_dir=args.output_dir,
            device=device,
            variant=args.chm_variant,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            num_workers=args.num_workers,
            label_smooth=args.label_smooth,
            decoder_channels=decoder_channels,
            val_stripe=val_stripe,
        )
    else:
        run_all_folds(
            variant=args.chm_variant,
            chm_tif=chm_tif,
            mask_tif=args.mask_tif,
            band_stats=band_stats,
            patch_index=patch_index,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            num_workers=args.num_workers,
            label_smooth=args.label_smooth,
            decoder_channels=decoder_channels,
        )


if __name__ == "__main__":
    main()
