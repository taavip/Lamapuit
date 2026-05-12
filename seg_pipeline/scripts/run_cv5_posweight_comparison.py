#!/usr/bin/env python3
"""Standalone CVv5 held-out test: fixed vs dynamic pos_weight comparison.

Uses the 3x3-block dataset split (cv-version=5) for baseline variant:
  - train entries: fold_id != -1
  - test entries:  fold_id == -1

Trains the best-known model combination twice:
  1) fixed pos_weight = 3.0
  2) dynamic pos_weight = neg_pixels / pos_pixels (from training entries)

Reports whether held-out test clDice improves.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.distance_transform import binary_to_soft_target
from common.extended_metrics import cldice_metric
from common.metrics import accumulate_pixel_metrics, threshold_sweep
from phase2_dataset_v3 import CWDSegDataset, load_patch_index, _get_in_channels
from phase3_train_v10 import (
    build_model,
    MixupAugmentation,
    CutMixAugmentation,
    GridMaskAugmentation,
)
from common.losses import TverskyFocalLoss, SoftCLDiceLoss


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, cldice_weight: float, pos_weight: float) -> None:
        super().__init__()
        self.tversky = TverskyFocalLoss(
            alpha=alpha,
            beta=beta,
            focal_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            pos_weight=pos_weight,
        )
        self.cldice_weight = cldice_weight
        self.cldice = SoftCLDiceLoss(iter_=3) if cldice_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        loss = self.tversky(logits, targets, valid)
        if self.cldice is not None and self.cldice_weight > 0:
            loss = loss + self.cldice_weight * self.cldice(logits, targets, valid)
        return loss


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_dynamic_pos_weight(train_entries) -> float:
    pos = float(sum(e.n_positive for e in train_entries))
    valid = float(sum(e.n_valid for e in train_entries))
    neg = max(0.0, valid - pos)
    if pos <= 0:
        return 3.0
    return max(1.0, neg / pos)


@torch.no_grad()
def evaluate_entries(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    prob_list, tgt_list, val_list = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        logits = model(image)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        for k in range(len(probs)):
            prob_list.append(probs[k])
            tgt_list.append(batch["target"][k, 0].numpy())
            val_list.append(batch["valid"][k, 0].numpy())

    best_thr, _ = threshold_sweep(prob_list, tgt_list, val_list)
    thr = float(best_thr["threshold"])
    px = accumulate_pixel_metrics(prob_list, tgt_list, val_list, threshold=thr)

    cl_vals = []
    for p, t, v in zip(prob_list, tgt_list, val_list):
        pred = ((p >= thr) & (v > 0.5)).astype(np.uint8)
        gt = (((t > 0.5) & (v > 0.5))).astype(np.uint8)
        cl_vals.append(float(cldice_metric(pred, gt)))

    return {
        "threshold": thr,
        "precision": float(px["precision"]),
        "recall": float(px["recall"]),
        "f1": float(px["f1"]),
        "dice": float(px["dice"]),
        "iou": float(px["iou"]),
        "accuracy": float(px["accuracy"]),
        "cldice_mean_patch": float(np.mean(cl_vals)) if cl_vals else 0.0,
    }


def make_sampler(entries, pos_weight: float) -> WeightedRandomSampler:
    weights = [pos_weight if e.n_positive > 0 else 1.0 for e in entries]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_once(
    run_name: str,
    pos_weight: float,
    arch: str,
    pretrained: bool,
    train_entries,
    test_entries,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    output_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    variant: str,
    disable_swa: bool,
) -> dict:
    set_seeds(seed)
    in_channels = _get_in_channels(variant)
    model = build_model(arch, in_channels=in_channels, pretrained=pretrained).to(device)

    # Final-protocol style: train on all non-test entries; use train proxy val for scheduling only.
    train_ds = CWDSegDataset(
        entries=train_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, in_channels=in_channels,
        augment=True, aug_mode="full", variant=variant,
    )
    val_proxy_ds = CWDSegDataset(
        entries=train_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, in_channels=in_channels,
        augment=False, variant=variant,
    )
    test_ds = CWDSegDataset(
        entries=test_entries, chm_tif=chm_tif, mask_tif=mask_tif,
        band_stats=band_stats, in_channels=in_channels,
        augment=False, variant=variant,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_sampler(train_entries, pos_weight=pos_weight),
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_proxy_loader = DataLoader(
        val_proxy_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    criterion = CombinedLoss(alpha=0.6, beta=0.4, cldice_weight=0.3, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    warmup_epochs = 25
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    swa_start = epochs + 1 if disable_swa else 35
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=10, anneal_strategy="cos")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    mixup = MixupAugmentation(p=0.2)
    cutmix = CutMixAugmentation(p=0.2)
    gridmask = GridMaskAugmentation(p=0.2)

    best_state = None
    best_val_cldice = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            # Soft targets
            tgt_np = target.cpu().numpy()
            soft_np = np.stack([binary_to_soft_target(tgt_np[i, 0], sigma=2.0) for i in range(len(tgt_np))])
            target = torch.from_numpy(soft_np[:, np.newaxis]).float().to(device)

            if epoch < swa_start:
                image, target = mixup(image, target)
                image, target = cutmix(image, target)
                image, target = gridmask(image, target)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(image)
                loss = criterion(logits, target, valid)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        # val-proxy on training set only for scheduler/model checkpointing.
        model.eval()
        v_prob, v_tgt, v_val = [], [], []
        with torch.no_grad():
            for batch in val_proxy_loader:
                image = batch["image"].to(device, non_blocking=True)
                logits = model(image)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
                for k in range(len(probs)):
                    v_prob.append(probs[k])
                    v_tgt.append(batch["target"][k, 0].numpy())
                    v_val.append(batch["valid"][k, 0].numpy())
        px = accumulate_pixel_metrics(v_prob, v_tgt, v_val, threshold=0.5)

        # Patch-level clDice mean.
        cl_vals = []
        for p, t, v in zip(v_prob, v_tgt, v_val):
            pred = ((p >= 0.5) & (v > 0.5)).astype(np.uint8)
            gt = ((t > 0.5) & (v > 0.5)).astype(np.uint8)
            cl_vals.append(float(cldice_metric(pred, gt)))
        val_cldice = float(np.mean(cl_vals)) if cl_vals else 0.0

        if epoch <= warmup_epochs:
            warmup.step()
        elif epoch < swa_start:
            plateau.step(val_cldice)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        if val_cldice > best_val_cldice:
            best_val_cldice = val_cldice
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{run_name}] epoch={epoch:03d} "
            f"val_f1_proxy={px['f1']:.4f} val_cldice_proxy={val_cldice:.4f} "
            f"best_cldice={best_val_cldice:.4f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("No best state captured.")
    model.load_state_dict(best_state)

    # SWA selection by proxy F1, consistent with training defaults.
    use_swa = False
    if (not disable_swa) and epochs >= swa_start:
        def _iter_imgs(loader):
            for b in loader:
                yield b["image"]
        update_bn(_iter_imgs(train_loader), swa_model, device=device)

        best_eval = evaluate_entries(model, val_proxy_loader, device)
        swa_eval = evaluate_entries(swa_model, val_proxy_loader, device)
        if swa_eval["f1"] > best_eval["f1"]:
            model = swa_model
            use_swa = True

    test_eval = evaluate_entries(model, test_loader, device)
    run_out = output_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "pos_weight_used": float(pos_weight),
        "best_epoch_proxy_cldice": int(best_epoch),
        "best_val_cldice_proxy": float(best_val_cldice),
        "used_swa": bool(use_swa),
        "test_metrics": test_eval,
    }
    (run_out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="CVv5 fixed vs dynamic pos_weight comparison.")
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v10_blockcv5_full_20260511_105703")
    parser.add_argument("--mask-tif", type=Path,
                        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif")
    parser.add_argument("--chm-tif", type=Path,
                        default=ROOT / "seg_pipeline" / "input" / "baseline_chm.tif")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "seg_pipeline" / "output" / "cv5_posweight_comparison")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fixed-pos-weight", type=float, default=3.0)
    parser.add_argument(
        "--dynamic-pos-weight-cap",
        type=float,
        default=0.0,
        help="If >0, cap computed dynamic pos_weight to this upper bound.",
    )
    parser.add_argument("--run-tag", default="", help="Suffix for output folder names.")
    parser.add_argument("--arch", default="unetpp_effb2")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--disable-swa", action="store_true")
    parser.add_argument(
        "--single-pos-weight",
        type=float,
        default=0.0,
        help="If >0, run a single training with this pos_weight (skip fixed/dynamic comparison).",
    )
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_dir}")

    patch_index = load_patch_index(args.dataset_dir / "patch_index_baseline.csv")
    band_stats = json.loads((args.dataset_dir / "band_stats_baseline.json").read_text())
    train_entries = [e for e in patch_index if getattr(e, "fold_id", -1) != -1]
    test_entries = [e for e in patch_index if getattr(e, "fold_id", -1) == -1]
    if not train_entries or not test_entries:
        raise RuntimeError("Train/test entries missing from CVv5 patch index.")

    dyn_pos_weight = compute_dynamic_pos_weight(train_entries)
    if args.dynamic_pos_weight_cap > 0:
        dyn_pos_weight = min(dyn_pos_weight, float(args.dynamic_pos_weight_cap))
    print(f"Train entries: {len(train_entries)}, test entries: {len(test_entries)}")
    print(f"Dynamic pos_weight (neg/pos): {dyn_pos_weight:.6f}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    if args.single_pos_weight > 0:
        single = train_once(
            run_name=f"single_pw_{str(args.single_pos_weight).replace('.', 'p')}{run_suffix}",
            pos_weight=float(args.single_pos_weight),
            arch=args.arch,
            pretrained=bool(args.pretrained),
            train_entries=train_entries,
            test_entries=test_entries,
            chm_tif=args.chm_tif,
            mask_tif=args.mask_tif,
            band_stats=band_stats,
            output_dir=out,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            variant=args.variant,
            disable_swa=bool(args.disable_swa),
        )
        single_report = {
            "mode": "single",
            "arch": args.arch,
            "pretrained": bool(args.pretrained),
            "result": single,
        }
        (out / "single_report.json").write_text(json.dumps(single_report, indent=2))
        print("\n=== Single Run ===")
        print(f"Test clDice: {float(single['test_metrics']['cldice_mean_patch']):.6f}")
        print(f"Report: {out / 'single_report.json'}")
    else:
        fixed = train_once(
            run_name=f"fixed_posweight_{str(args.fixed_pos_weight).replace('.', 'p')}{run_suffix}",
            pos_weight=float(args.fixed_pos_weight),
            arch=args.arch,
            pretrained=bool(args.pretrained),
            train_entries=train_entries,
            test_entries=test_entries,
            chm_tif=args.chm_tif,
            mask_tif=args.mask_tif,
            band_stats=band_stats,
            output_dir=out,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            variant=args.variant,
            disable_swa=bool(args.disable_swa),
        )
        dynamic = train_once(
            run_name=f"dynamic_posweight{run_suffix}",
            pos_weight=dyn_pos_weight,
            arch=args.arch,
            pretrained=bool(args.pretrained),
            train_entries=train_entries,
            test_entries=test_entries,
            chm_tif=args.chm_tif,
            mask_tif=args.mask_tif,
            band_stats=band_stats,
            output_dir=out,
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            variant=args.variant,
            disable_swa=bool(args.disable_swa),
        )

        fixed_cl = float(fixed["test_metrics"]["cldice_mean_patch"])
        dyn_cl = float(dynamic["test_metrics"]["cldice_mean_patch"])
        delta = dyn_cl - fixed_cl
        report = {
            "fixed": fixed,
            "dynamic": dynamic,
            "cldice_delta_dynamic_minus_fixed": delta,
            "dynamic_better": bool(delta > 0),
        }
        (out / "comparison_report.json").write_text(json.dumps(report, indent=2))
        print("\n=== Comparison ===")
        print(f"Fixed clDice:   {fixed_cl:.6f}")
        print(f"Dynamic clDice: {dyn_cl:.6f}")
        print(f"Delta:          {delta:+.6f}")
        print(f"Dynamic better: {delta > 0}")
        print(f"Report: {out / 'comparison_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
