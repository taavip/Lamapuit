#!/usr/bin/env python3
"""Phase III V9: Stability fixes — ReduceLROnPlateau, earlier SWA, α=0.4/β=0.6.

V9 targets the V8 instability (epoch-26 recall crash, bad-weight SWA averaging)
with four targeted fixes:

  1. ReduceLROnPlateau replaces CosineAnnealingLR: 5-epoch linear warmup then
     ReduceLROnPlateau (mode=max, factor=0.5, patience=5, min_lr=1e-6). Prevents
     the LR from remaining high when near a good minimum (Goyal et al. 2017).

  2. Earlier SWA (epoch 25 vs V8's 50): V8 peak at epoch 21; starting SWA at 25
     captures the post-peak stable region before ReduceLROnPlateau has decayed LR
     too far. SWA model gated: only saved if its val F1 exceeds best.pt F1.

  3. Tversky α=0.4, β=0.6 (V8: α=0.3, β=0.7): FN/FP ratio 1.5× vs 2.3×.
     Reduces the probability of catastrophic recall collapse while still biasing
     toward recall (Salehi et al. 2017).

  4. Phase 1 bypass: run phase1 with --neg-threshold 0.0 --noisy-threshold 1.0
     to disable the broken Phase 0 ensemble conflict filter (prob_mean=0 on
     new CHM). Dataset rebuilt to phase2_dataset_v9.

Usage:
    python phase3_train_v9.py --device cuda                    # v7c, all folds
    python phase3_train_v9.py --fold 0 --epochs 5 --device cuda # smoke test
    python phase3_train_v9.py --no-swa --device cuda            # disable SWA
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
    "scipy",
]

try:
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401
    import albumentations as A
except ImportError:
    print("[phase3_v7] Installing required packages…", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + _REQUIRED_PKGS)
    import segmentation_models_pytorch as smp
    import timm  # noqa: F401
    import albumentations as A

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.losses import TverskyFocalLoss, SoftCLDiceLoss
from common.metrics import accumulate_pixel_metrics, threshold_sweep
from phase2_dataset_v3 import (
    PATCH_SIZE, STRIDE, STRIPE_WIDTH, N_STRIPES, TEST_STRIPE,
    DEFAULT_VARIANT, SpatialCVSplitterV3, CWDSegDataset, load_patch_index,
    make_weighted_sampler, _read_chm_path, _get_in_channels, _get_binary_bands,
)

# ---------------------------------------------------------------------------
# Architecture (same as V6)
# ---------------------------------------------------------------------------

_ARCH_CONFIGS = {
    "unetpp_effb2": {
        "cls": "UnetPlusPlus",
        "encoder_candidates": ["tu-efficientnet_b2", "efficientnet-b2"],
        "decoder_channels": (256, 128, 64, 32, 16),
    },
}


def build_model(arch: str, in_channels: int = 4, pretrained: bool = True,
                decoder_channels: tuple | None = None) -> nn.Module:
    cfg = _ARCH_CONFIGS[arch]
    cls_name = cfg["cls"]
    encoder_weights = "imagenet" if pretrained else None
    dec_channels = decoder_channels or cfg["decoder_channels"]
    for enc_name in cfg["encoder_candidates"]:
        try:
            kwargs: dict = dict(encoder_name=enc_name, encoder_weights=encoder_weights,
                                in_channels=in_channels, classes=1, activation=None)
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


# ---------------------------------------------------------------------------
# V7 combined loss factory
# ---------------------------------------------------------------------------


class V7CombinedLoss(nn.Module):
    """TverskyFocal + λ·SoftCLDice.

    V9 default: α=0.4, β=0.6 (FN/FP ratio 1.5× vs V8's 2.3×).
    CLDice (λ=0.5) preserves thin-structure connectivity.
    """

    def __init__(self, tversky_alpha: float = 0.4, tversky_beta: float = 0.6,
                 cldice_weight: float = 0.5) -> None:
        super().__init__()
        self.tversky = TverskyFocalLoss(
            alpha=tversky_alpha, beta=tversky_beta,
            focal_weight=1.0, focal_alpha=0.25, focal_gamma=2.0, pos_weight=3.0,
        )
        self.cldice_weight = cldice_weight
        self.cldice = SoftCLDiceLoss(iter_=3) if cldice_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        loss = self.tversky(logits, targets, valid)
        if self.cldice is not None and self.cldice_weight > 0:
            loss = loss + self.cldice_weight * self.cldice(logits, targets, valid)
        return loss


# ---------------------------------------------------------------------------
# Augmentations (same as V6)
# ---------------------------------------------------------------------------


class MixupAugmentation:
    def __init__(self, p: float = 0.2, alpha: float = 1.0):
        self.p = p
        self.alpha = alpha

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple:
        if np.random.rand() > self.p:
            return image, target
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randint(0, image.shape[0], (1,)).item() if image.shape[0] > 1 else 0
        image = lam * image + (1 - lam) * image[idx].unsqueeze(0)
        target = lam * target + (1 - lam) * target[idx].unsqueeze(0)
        return image, target


class CutMixAugmentation:
    def __init__(self, p: float = 0.2, alpha: float = 1.0):
        self.p = p
        self.alpha = alpha

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple:
        if np.random.rand() > self.p or image.shape[0] <= 1:
            return image, target
        b, c, h, w = image.shape
        idx = torch.randint(1, b, (1,)).item()
        lam = np.random.beta(self.alpha, self.alpha)
        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(w, x1 + cut_w), min(h, y1 + cut_h)
        image[0, :, y1:y2, x1:x2] = image[idx, :, y1:y2, x1:x2]
        target[0, :, y1:y2, x1:x2] = target[idx, :, y1:y2, x1:x2]
        return image, target


class GridMaskAugmentation:
    def __init__(self, p: float = 0.2, ratio: float = 0.5):
        self.p = p
        self.ratio = ratio

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple:
        if np.random.rand() > self.p:
            return image, target
        b, c, h, w = image.shape
        mask = torch.ones_like(image)
        for _ in range(np.random.randint(2, 5)):
            gh = max(1, int(h * self.ratio))
            gw = max(1, int(w * self.ratio))
            y = np.random.randint(0, h - gh + 1)
            x = np.random.randint(0, w - gw + 1)
            mask[:, :, y:y+gh, x:x+gw] = 0
        return image * mask, target


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
    epochs: int = 100,
    batch_size: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    num_workers: int = 0,
    label_smooth: float = 0.0,
    decoder_channels: tuple | None = None,
    val_stripe: int | None = None,
    tversky_alpha: float = 0.4,
    tversky_beta: float = 0.6,
    cldice_weight: float = 0.5,
    soft_targets: bool = False,
    soft_sigma: float = 2.0,
    swa_start_epoch: int = 25,
    swa_update_freq: int = 5,
    warmup_epochs: int = 5,
) -> dict:
    fold_dir = output_dir / variant / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = output_dir / "tensorboard" / variant / f"fold{fold_id}"
    writer = SummaryWriter(log_dir=str(tb_dir))
    history_jsonl = fold_dir / "fold_history.jsonl"

    in_channels = _get_in_channels(variant)
    if batch_size is None:
        batch_size = 8

    model = build_model(arch, in_channels=in_channels, decoder_channels=decoder_channels).to(device)

    train_ds = CWDSegDataset(
        entries=train_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, patch_size=PATCH_SIZE, in_channels=in_channels,
        augment=True, buffer_px=64, stripe_width=STRIPE_WIDTH,
        val_stripe=val_stripe, variant=variant,
    )
    val_ds = CWDSegDataset(
        entries=val_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, patch_size=PATCH_SIZE, in_channels=in_channels,
        augment=False, buffer_px=64, stripe_width=STRIPE_WIDTH,
        val_stripe=None, variant=variant,
    )

    sampler = make_weighted_sampler(train_entries, pos_weight=3.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))

    criterion = V7CombinedLoss(tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                                cldice_weight=cldice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # V9: warmup (epochs 1–warmup_epochs) then ReduceLROnPlateau on val F1.
    # Prevents the high-LR bouncing observed at V8 epoch 26.
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    swa_model = AveragedModel(model) if swa_start_epoch > 0 else None
    swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1, anneal_epochs=10,
                          anneal_strategy="cos") if swa_start_epoch > 0 and swa_start_epoch < epochs else None
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    mixup = MixupAugmentation(p=0.2)
    cutmix = CutMixAugmentation(p=0.2)
    gridmask = GridMaskAugmentation(p=0.2)

    # Soft-target transform (applied per-batch on GPU)
    soft_fn = None
    if soft_targets:
        from common.distance_transform import binary_to_soft_target
        def soft_fn(tgt_np: np.ndarray) -> np.ndarray:
            return binary_to_soft_target(tgt_np, sigma=soft_sigma)

    best_f1 = -1.0
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

            # Soft targets: convert hard binary → Gaussian heatmap
            if soft_fn is not None:
                tgt_np = target.cpu().numpy()
                soft_np = np.stack([soft_fn(tgt_np[i, 0]) for i in range(len(tgt_np))])
                target = torch.from_numpy(soft_np[:, np.newaxis]).float().to(device)

            # Advanced augmentations (only during regular training phase)
            if epoch < swa_start_epoch:
                image, target = mixup(image, target)
                image, target = cutmix(image, target)
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

        # Warmup step (no metric needed — done before validation)
        if epoch <= warmup_epochs:
            warmup_sched.step()
        elif epoch >= swa_start_epoch and swa_scheduler and (epoch - swa_start_epoch) % swa_update_freq == 0:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validation
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
        train_loss = running_loss / max(1, steps)

        # ReduceLROnPlateau requires the monitored metric — run after validation
        if warmup_epochs < epoch < swa_start_epoch:
            plateau_sched.step(val_f1)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("IoU/val", float(val_m["iou"]), epoch)
        writer.add_scalar("Precision/val", float(val_m["precision"]), epoch)
        writer.add_scalar("Recall/val", float(val_m["recall"]), epoch)

        row = {
            "epoch": epoch, "train_loss": train_loss,
            "val_dice": val_dice, "val_f1": val_f1,
            "val_iou": float(val_m["iou"]),
            "val_precision": float(val_m["precision"]),
            "val_recall": float(val_m["recall"]),
        }
        history.append(row)

        # Append to JSONL for crash recovery (flush each epoch)
        with open(history_jsonl, "a") as jf:
            jf.write(json.dumps(row) + "\n")

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[V9|{variant}|fold{fold_id}] epoch={epoch:03d} "
            f"loss={train_loss:.5f} dice={val_dice:.4f} f1={val_f1:.4f} "
            f"iou={val_m['iou']:.4f} prec={val_m['precision']:.4f} rec={val_m['recall']:.4f} "
            f"lr={cur_lr:.2e}",
            flush=True,
        )

        if val_f1 > best_f1 + 1e-6 or (val_f1 >= best_f1 and val_dice > best_dice):
            best_f1 = val_f1
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience and epoch > warmup_epochs:
            print(f"  Early stopping at epoch {epoch} (patience={patience}, best_f1={best_f1:.4f})")
            break

    if best_state is None:
        raise RuntimeError(f"No valid state for {variant}/fold{fold_id}")

    model.load_state_dict(best_state)

    # Save best.pt (standard checkpoint from best validation epoch)
    torch.save({"variant": variant, "fold_id": fold_id, "arch": arch,
                "in_channels": in_channels, "tversky_alpha": tversky_alpha,
                "tversky_beta": tversky_beta, "cldice_weight": cldice_weight,
                "soft_targets": soft_targets, "state_dict": best_state},
               fold_dir / "best.pt")

    # Save SWA-averaged model — only if it outperforms best.pt on validation (V9 gate)
    swa_ran = swa_model is not None and epoch >= swa_start_epoch
    swa_val_f1 = 0.0
    swa_optimal_threshold = 0.5
    use_swa_for_inference = False
    if swa_ran:
        print(f"  Updating batch norm for SWA (epoch {epoch}, {epoch - swa_start_epoch} SWA epochs)...")
        def _swa_iter(loader):
            for b in loader:
                yield b["image"]
        update_bn(_swa_iter(train_loader), swa_model, device=device)

        # Evaluate SWA model on val set
        swa_model.eval()
        swa_prob_list, swa_tgt_list, swa_val_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                logits = swa_model(image)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
                for k in range(len(probs)):
                    swa_prob_list.append(probs[k])
                    swa_tgt_list.append(batch["target"][k, 0].numpy())
                    swa_val_list.append(batch["valid"][k, 0].numpy())
        swa_m = accumulate_pixel_metrics(swa_prob_list, swa_tgt_list, swa_val_list, threshold=0.5)
        swa_val_f1 = float(swa_m["f1"])
        swa_best_thr, _ = threshold_sweep(swa_prob_list, swa_tgt_list, swa_val_list)
        swa_optimal_threshold = float(swa_best_thr["threshold"])
        print(f"  SWA val_f1={swa_val_f1:.4f} (best.pt val_f1={best_f1:.4f}, "
              f"optimal_thr={swa_optimal_threshold:.2f})")

        if swa_val_f1 > best_f1:
            swa_state = swa_model.module.state_dict()
            torch.save({"variant": variant, "fold_id": fold_id, "arch": arch,
                        "in_channels": in_channels, "tversky_alpha": tversky_alpha,
                        "tversky_beta": tversky_beta, "cldice_weight": cldice_weight,
                        "soft_targets": soft_targets, "state_dict": swa_state},
                       fold_dir / "swa_model.pt")
            use_swa_for_inference = True
            print(f"  SWA BETTER than best.pt → saved swa_model.pt (use for inference)")
        else:
            print(f"  SWA WORSE than best.pt → skipping swa_model.pt")

    best_thr, _ = threshold_sweep(prob_list, tgt_list, val_list)
    final_metrics = {
        "variant": variant, "fold_id": fold_id,
        "tversky_alpha": tversky_alpha, "tversky_beta": tversky_beta,
        "cldice_weight": cldice_weight, "soft_targets": soft_targets,
        "best_val_f1": float(best_f1), "best_val_dice": float(best_dice),
        "best_threshold": float(best_thr["threshold"]),
        "threshold_f1": float(best_thr["f1"]),
        "n_train": len(train_entries), "n_val": len(val_entries),
        "n_epochs_trained": len(history),
        "swa_enabled": swa_ran,
        "swa_epochs": max(0, epoch - swa_start_epoch) if swa_ran else 0,
        "swa_val_f1": swa_val_f1,
        "swa_optimal_threshold": swa_optimal_threshold,
        "use_swa_for_inference": use_swa_for_inference,
    }
    (fold_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))
    _append_history_csv(fold_dir / "history.csv", history)
    writer.close()
    return final_metrics


def _append_history_csv(path: Path, rows: list[dict]) -> None:
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# All-folds runner
# ---------------------------------------------------------------------------


def run_all_folds(variant, chm_tif, mask_tif, band_stats, patch_index,
                  output_dir, device, epochs=100, batch_size=None, patience=15,
                  num_workers=0, label_smooth=0.0, decoder_channels=None,
                  tversky_alpha=0.4, tversky_beta=0.6, cldice_weight=0.5,
                  soft_targets=False, soft_sigma=2.0,
                  swa_start_epoch=25, swa_update_freq=5, warmup_epochs=5) -> list[dict]:
    splitter = SpatialCVSplitterV3()
    n_folds = N_STRIPES - 1
    all_metrics: list[dict] = []
    for fold_id in range(n_folds):
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=fold_id)
        val_stripe = splitter._val_stripes[fold_id]
        n_pos = sum(1 for e in train_e if e.n_positive > 0)
        print(f"\n{'='*70}\n[V9|{variant}] Fold {fold_id}/{n_folds-1} — "
              f"train={len(train_e):,} (pos={n_pos:,})  val={len(val_e):,}\n{'='*70}")
        m = train_fold(
            arch="unetpp_effb2", fold_id=fold_id,
            train_entries=train_e, val_entries=val_e,
            chm_tif=chm_tif, mask_tif=mask_tif, band_stats=band_stats,
            output_dir=output_dir, device=device, variant=variant,
            epochs=epochs, batch_size=batch_size, patience=patience,
            num_workers=num_workers, label_smooth=label_smooth,
            decoder_channels=decoder_channels, val_stripe=val_stripe,
            tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
            cldice_weight=cldice_weight, soft_targets=soft_targets,
            soft_sigma=soft_sigma, swa_start_epoch=swa_start_epoch,
            swa_update_freq=swa_update_freq, warmup_epochs=warmup_epochs,
        )
        all_metrics.append(m)
        print(f"  Fold {fold_id} done — f1={m['best_val_f1']:.4f}  dice={m['best_val_dice']:.4f}")

    mean_f1 = float(np.mean([m["best_val_f1"] for m in all_metrics]))
    std_f1 = float(np.std([m["best_val_f1"] for m in all_metrics]))
    mean_dice = float(np.mean([m["best_val_dice"] for m in all_metrics]))
    std_dice = float(np.std([m["best_val_dice"] for m in all_metrics]))
    print(f"\n[V9|{variant}] Summary:\n"
          f"  F1:   {mean_f1:.4f} ± {std_f1:.4f}\n"
          f"  Dice: {mean_dice:.4f} ± {std_dice:.4f}")

    agg = output_dir / variant / "fold_summary.csv"
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader()
        w.writerows(all_metrics)
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase III V9: ReduceLROnPlateau + SWA@25 + α=0.4/β=0.6")
    p.add_argument("--variant", default=DEFAULT_VARIANT,
                   choices=["baseline", "raw", "gauss", "masked", "composite"])
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--config", default="v7c", choices=["v7a", "v7b", "v7c"],
                   help="Ablation config: v7a=flip-Tversky only, v7b=+CLDice, v7c=+soft targets (default)")
    p.add_argument("--mask-tif", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif")
    p.add_argument("--dataset-dir", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v9")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v9")
    p.add_argument("--device", default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--tversky-alpha", type=float, default=0.4)
    p.add_argument("--tversky-beta", type=float, default=0.6)
    p.add_argument("--cldice-weight", type=float, default=0.5,
                   help="CLDice loss weight λ (0 = disable, 0.5 = default)")
    p.add_argument("--soft-targets", action="store_true", default=True,
                   help="Use Gaussian distance-transform soft targets (default=True)")
    p.add_argument("--no-swa", action="store_true",
                   help="Disable SWA (for quick testing)")
    p.add_argument("--soft-sigma", type=float, default=2.0)
    p.add_argument("--swa-start-epoch", type=int, default=25)
    p.add_argument("--swa-update-freq", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear LR warmup epochs before ReduceLROnPlateau takes over")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Apply ablation config shortcuts (for backward compat with explicit --config args)
    if args.config == "v7a":
        args.cldice_weight = 0.0
        args.soft_targets = False
    elif args.config == "v7b":
        args.cldice_weight = 0.5
        args.soft_targets = False
    elif args.config == "v7c":
        args.cldice_weight = 0.5
        args.soft_targets = True

    # Handle --no-swa flag
    swa_start_epoch = -1 if args.no_swa else args.swa_start_epoch

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Config: {args.config}  (α={args.tversky_alpha}, β={args.tversky_beta}, "
          f"λ_cldice={args.cldice_weight}, soft_targets={args.soft_targets})")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}, Warmup: {args.warmup_epochs}ep, "
          f"SWA: start={swa_start_epoch} ({'disabled' if args.no_swa else 'enabled'})")
    print(f"LR schedule: warmup {args.warmup_epochs}ep → ReduceLROnPlateau(factor=0.5, patience=5)")

    mask_tif = args.mask_tif
    if not mask_tif.exists():
        print(f"ERROR: mask not found: {mask_tif}", file=sys.stderr)
        sys.exit(1)

    variant = args.variant
    chm_tif = _read_chm_path(variant, ROOT)
    if not chm_tif.exists():
        print(f"ERROR: CHM not found: {chm_tif}", file=sys.stderr)
        sys.exit(1)

    patch_index = load_patch_index(args.dataset_dir / f"patch_index_{variant}.csv")
    band_stats = json.loads((args.dataset_dir / f"band_stats_{variant}.json").read_text())
    print(f"Variant: {variant},  {len(patch_index):,} patches")

    common_kw = dict(
        chm_tif=chm_tif, mask_tif=mask_tif, band_stats=band_stats,
        output_dir=args.output_dir, device=device, variant=variant,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        num_workers=args.num_workers,
        tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
        cldice_weight=args.cldice_weight, soft_targets=args.soft_targets,
        soft_sigma=args.soft_sigma, swa_start_epoch=swa_start_epoch,
        swa_update_freq=args.swa_update_freq, warmup_epochs=args.warmup_epochs,
    )

    if args.fold >= 0:
        splitter = SpatialCVSplitterV3()
        train_e, val_e = splitter.train_val_split(patch_index, val_fold=args.fold)
        val_stripe = splitter._val_stripes[args.fold]
        n_pos = sum(1 for e in train_e if e.n_positive > 0)
        print(f"\n[V9|{variant}] Fold {args.fold} — "
              f"train={len(train_e):,} (pos={n_pos:,})  val={len(val_e):,}")
        m = train_fold(arch="unetpp_effb2", fold_id=args.fold,
                       train_entries=train_e, val_entries=val_e,
                       val_stripe=val_stripe, **common_kw)
        print(f"  Fold {args.fold} done — f1={m['best_val_f1']:.4f}  dice={m['best_val_dice']:.4f}")
    else:
        run_all_folds(patch_index=patch_index, **common_kw)

    print("\n✅ V9 training complete")


if __name__ == "__main__":
    sys.exit(main() or 0)
