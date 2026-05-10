#!/usr/bin/env python3
"""Phase III V3: Model Training — 5-fold CV + Tversky loss + V2 improvements.

V3 key changes over V2:
  1. 5-fold CV restored (N_STRIPES=5, STRIPE_WIDTH=1000) — recovers training data
     volume from V1 (~390 patches, ~80–95 positives per fold vs. V2's 29–45).
  2. TverskyFocalLoss option (--loss tversky) — penalises FP more than FN to
     address V2's precision=0.076, recall=0.555 imbalance.
  3. Composite CHM default — V2 ablation showed composite wins; no need to train all 5 variants.
  4. All V2 improvements retained: conflict zone masking, NodataDropout, 75 epochs, patience=12.

Design rationale:
  V2's best-threshold test Dice was already 0.1919 (composite fold 0, thr=0.75) which
  exceeds V1's 0.1686, suggesting the underlying signal is present but the model is
  poorly calibrated.  V3 addresses calibration via: (a) TverskyLoss that directly
  controls the FP/FN trade-off, and (b) larger training sets that allow the model to
  learn a more balanced decision boundary.

Usage:
    python phase3_train_v3.py --device cuda                    # all 4 folds, composite
    python phase3_train_v3.py --fold 0 --epochs 3 --device cuda # single fold, quick test
    python phase3_train_v3.py --loss tversky --alpha 0.6 --beta 0.4 --device cuda
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
]

try:
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401
except ImportError:
    print("[phase3_v3] Installing required packages…", flush=True)
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
# Architecture registry (U-Net++ EfficientNet-B2 only, same as V1/V2 best)
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
    'tversky':   TverskyFocalLoss(alpha, beta) — V3 default for precision improvement
                 alpha=0.6 penalises FP 50% more than beta=0.4 penalises FN.
                 Empirical starting point from Salehi et al. (2017) for medical imaging;
                 adapted for CWD which has similar class imbalance characteristics.
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
# Training loop
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
    epochs: int = 75,
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
) -> dict:
    """Train one (variant, fold) combination."""
    fold_dir = output_dir / variant / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = output_dir / "tensorboard" / variant / f"fold{fold_id}"
    writer = SummaryWriter(log_dir=str(tb_dir))

    in_channels = _get_in_channels(variant)

    if batch_size is None:
        batch_size = 16 if in_channels == 1 else 8

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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_dice = -1.0
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
        writer.add_scalar("Precision/val", float(val_m["precision"]), epoch)
        writer.add_scalar("Recall/val", float(val_m["recall"]), epoch)

        row = {
            "epoch": epoch, "train_loss": float(train_loss),
            "val_dice": val_dice, "val_iou": val_iou,
            "val_precision": float(val_m["precision"]),
            "val_recall": float(val_m["recall"]),
            "val_f1": float(val_m["f1"]),
        }
        history.append(row)

        print(
            f"[{variant}|fold{fold_id}] epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_dice={val_dice:.4f} "
            f"val_iou={val_iou:.4f} "
            f"val_prec={val_m['precision']:.4f} "
            f"val_rec={val_m['recall']:.4f}",
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
# All-folds runner (4 folds for 5-stripe setup)
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
    loss_name: str = "tversky",
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
) -> list[dict]:
    """Train U-Net++ on all 4 V3 folds for a single variant."""
    splitter = SpatialCVSplitterV3()
    n_folds = N_STRIPES - 1  # 4 folds (stripes 1–4)

    all_metrics: list[dict] = []
    for fold_id in range(n_folds):
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=fold_id)
        val_stripe = splitter._val_stripes[fold_id]
        n_pos_tr = sum(1 for e in train_e if e.n_positive > 0)
        print(
            f"\n{'='*60}\n[{variant}] Fold {fold_id}/{n_folds-1} — "
            f"train={len(train_e):,} (pos={n_pos_tr:,})  val={len(val_e):,}\n{'='*60}"
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
        )
        all_metrics.append(m)
        print(f"  Fold {fold_id} done — best_dice={m['best_val_dice']:.4f}  best_thr={m['best_threshold']:.2f}")

    agg_path = output_dir / variant / "fold_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    mean_dice = float(np.mean([m["best_val_dice"] for m in all_metrics]))
    std_dice = float(np.std([m["best_val_dice"] for m in all_metrics]))
    print(f"\n[{variant}] 5-fold Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase III V3: 5-fold training with Tversky loss")
    p.add_argument(
        "--chm-variant", type=str, default=DEFAULT_VARIANT,
        choices=["baseline", "raw", "gauss", "masked", "composite"],
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
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v3",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--label-smooth", type=float, default=0.0)
    p.add_argument(
        "--loss", type=str, default="tversky", choices=["dicefocal", "tversky"],
        help="Loss function: 'tversky' (V3 default, improves precision) or 'dicefocal' (V1/V2 default)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.6,
        help="Tversky alpha: FP penalty weight (higher → more precision)",
    )
    p.add_argument(
        "--beta", type=float, default=0.4,
        help="Tversky beta: FN penalty weight (higher → more recall)",
    )
    p.add_argument("--decoder-channels", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Loss: {args.loss}" + (f" (alpha={args.alpha}, beta={args.beta})" if args.loss == "tversky" else ""))

    chm_tif = _read_chm_path(args.chm_variant, ROOT)
    if not chm_tif.exists():
        print(f"ERROR: CHM file not found: {chm_tif}", file=sys.stderr)
        sys.exit(1)

    patch_index = load_patch_index(args.dataset_dir / f"patch_index_{args.chm_variant}.csv")
    band_stats = json.loads((args.dataset_dir / f"band_stats_{args.chm_variant}.json").read_text())
    print(f"Variant: {args.chm_variant},  {len(patch_index):,} patches")

    decoder_channels = None
    if args.decoder_channels:
        decoder_channels = tuple(int(x.strip()) for x in args.decoder_channels.split(","))

    if args.fold >= 0:
        splitter = SpatialCVSplitterV3()
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
            loss_name=args.loss,
            tversky_alpha=args.alpha,
            tversky_beta=args.beta,
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
            loss_name=args.loss,
            tversky_alpha=args.alpha,
            tversky_beta=args.beta,
        )


if __name__ == "__main__":
    main()
