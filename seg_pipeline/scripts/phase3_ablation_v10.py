#!/usr/bin/env python3
"""Comprehensive ablation study for CWD segmentation — Phase 2–6.

Systematically searches design space: CHM variants → architectures → loss functions →
augmentation → final 4-fold validation. Winner from each phase carries forward.

Model selection is validation-only by default. The held-out test stripe is evaluated
only when --evaluate-test is passed after the final configuration is fixed.

Usage:
    # Smoke test (5 epochs, phase 2, condition 0)
    python3 phase3_ablation_v10.py --phase 2 --condition 0 --epochs 5 --no-swa

    # Run full Phase 2
    python3 phase3_ablation_v10.py --phase 2 --epochs 75 --swa-start-epoch 35

    # Re-run single condition
    python3 phase3_ablation_v10.py --phase 3 --condition 2 --epochs 75

    # Regenerate figures only
    python3 phase3_ablation_v10.py --reports-only
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

import numpy as np
import rasterio
import torch
from scipy.ndimage import label as label_components
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add scripts to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from phase2_dataset_v3 import (
    CWDSegDataset, load_patch_index, make_weighted_sampler,
    _read_chm_path, _get_in_channels, _get_binary_bands, SpatialCVSplitterV3,
    STRIPE_WIDTH, TEST_STRIPE, PATCH_SIZE, STRIDE,
)
from phase3_train_v10 import build_model, train_fold, V7CombinedLoss
from common.losses import DiceFocalLoss
from common.augmentation import get_full_aug, get_geometric_aug
from common.metrics import accumulate_pixel_metrics, threshold_sweep
from common.extended_metrics import boundary_iou, cldice_metric, ap_at_iou
from common.sliding_window import sliding_window_predict
from common.raster_io import normalize_bands

# ── Constants
DEFAULT_FOLD = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _base_condition_id(condition_id: str) -> str:
    """Return the phase-local condition id from a carried run id."""
    return condition_id.split("__")[-1]


def _carry_context(args: argparse.Namespace) -> str:
    """Stable prefix describing already-carried winners, e.g. 2A__3C."""
    ids: list[str] = []
    for cw in getattr(args, "carry_winner", []):
        try:
            _, cid = cw.split(":", 1)
        except ValueError:
            continue
        if cid:
            ids.append(cid)
    return "__".join(ids)


def _dataset_dir_candidates(base_dir: Path) -> list[Path]:
    candidates = [base_dir]
    repo_out = Path("seg_pipeline/output")
    for extra in ["phase2_dataset_v10_reconstructed", "phase2_dataset_v3", "phase2_dataset_v2"]:
        p = repo_out / extra
        if p not in candidates:
            candidates.append(p)
    return candidates


def _resolve_dataset_files(base_dir: Path, variant: str) -> tuple[Path, Path]:
    for d in _dataset_dir_candidates(base_dir):
        pidx = d / f"patch_index_{variant}.csv"
        bstats = d / f"band_stats_{variant}.json"
        if pidx.exists() and bstats.exists():
            return pidx, bstats
    tried = ", ".join(str(d) for d in _dataset_dir_candidates(base_dir))
    raise FileNotFoundError(
        f"Could not resolve patch_index/band_stats for variant '{variant}'. Tried: {tried}"
    )

# Ablation phase definitions (conditions per phase)
ALL_PHASES = {
    2: {  # CHM Variant Search
        "2A": {"name": "chm_baseline", "chm_variant": "baseline", "in_channels": 1},
        "2B": {"name": "chm_raw", "chm_variant": "raw", "in_channels": 1},
        "2C": {"name": "chm_gauss", "chm_variant": "gauss", "in_channels": 1},
        "2D": {"name": "chm_masked", "chm_variant": "masked", "in_channels": 2},
        "2E": {"name": "chm_composite", "chm_variant": "composite", "in_channels": 4},
    },
    3: {  # Model Architecture Search
        "3A": {"name": "arch_unet_effb2", "arch": "unet_effb2"},
        "3B": {"name": "arch_unetpp_effb0", "arch": "unetpp_effb0"},
        "3C": {"name": "arch_unetpp_effb2", "arch": "unetpp_effb2"},
        "3D": {"name": "arch_unetpp_effb4", "arch": "unetpp_effb4"},
        "3E": {"name": "arch_deeplabv3p_effb2", "arch": "deeplabv3p_effb2"},
    },
    4: {  # Loss Function & Parameter Search
        "4A": {"name": "loss_dicefocal", "loss_type": "dicefocal", "alpha": None, "beta": None, "cldice_lambda": 0.0},
        "4B": {"name": "loss_tversky_high_recall", "loss_type": "tversky", "alpha": 0.3, "beta": 0.7, "cldice_lambda": 0.0},
        "4C": {"name": "loss_tversky_recall", "loss_type": "tversky", "alpha": 0.4, "beta": 0.6, "cldice_lambda": 0.0},
        "4D": {"name": "loss_tversky_balanced", "loss_type": "tversky", "alpha": 0.5, "beta": 0.5, "cldice_lambda": 0.0},
        "4E": {"name": "loss_tversky_precision", "loss_type": "tversky", "alpha": 0.6, "beta": 0.4, "cldice_lambda": 0.0},
        "4F": {"name": "loss_tversky_high_precision", "loss_type": "tversky", "alpha": 0.7, "beta": 0.3, "cldice_lambda": 0.0},
        # 4G, 4H use winning α/β from 4B–4F, filled at runtime
    },
    5: {  # Augmentation & Regularization Search
        "5A": {"name": "aug_none", "aug_mode": "none", "soft_targets": False, "batch_aug": False},
        "5B": {"name": "aug_geometric", "aug_mode": "geometric", "soft_targets": False, "batch_aug": False},
        "5C": {"name": "aug_full_nosoft", "aug_mode": "full", "soft_targets": False, "batch_aug": True},
        "5D": {"name": "aug_full_soft", "aug_mode": "full", "soft_targets": True, "batch_aug": True},
        "5E": {"name": "aug_full_soft_noswa", "aug_mode": "full", "soft_targets": True, "batch_aug": True, "use_swa": False},
    },
    6: {  # Final Validation — all 4 folds
        "6": {"name": "final_validation", "folds": [0, 1, 2, 3]},
    },
}

# Track winners across phases
WINNER = {
    2: {"chm_variant": "composite", "in_channels": 4},  # Default assumption
    3: {"arch": "unetpp_effb2"},  # Default assumption
    4: {"alpha": 0.6, "beta": 0.4, "cldice_lambda": 0.3},  # V10.2 defaults
}


@dataclass
class AblationConfig:
    """Configuration for a single ablation condition."""
    phase: int
    condition_id: str
    name: str
    chm_variant: Optional[str] = None
    in_channels: Optional[int] = None
    arch: Optional[str] = None
    loss_type: Optional[str] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    cldice_lambda: Optional[float] = None
    aug_mode: Optional[str] = None
    soft_targets: Optional[bool] = None
    batch_aug: Optional[bool] = None
    folds: Optional[list[int]] = None
    use_swa: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def run_condition(
    condition: AblationConfig,
    args: argparse.Namespace,
    winner: dict,
) -> dict:
    """Train a single condition and optionally evaluate on the held-out test stripe.

    Args:
        condition: Configuration for this condition.
        args: Command-line arguments (fold, epochs, device, etc.).
        winner: Previous phase winners to carry forward.

    Returns:
        Dict with metrics for this condition.
    """
    carry_context = _carry_context(args)
    run_id = f"{carry_context}__{condition.condition_id}" if carry_context else condition.condition_id
    condition_dir_name = f"phase{condition.phase}_{run_id}_{condition.name}"
    condition_dir = args.output_dir / condition_dir_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    protocol_suffix = "test" if args.evaluate_test else "val"
    metrics_path = condition_dir / f"fold{args.fold}_metrics_{protocol_suffix}.json"
    if metrics_path.exists():
        print(f"  ✓ Skipping {condition.condition_id} (already trained)")
        with open(metrics_path) as f:
            return json.load(f)

    print(f"\n  Training {condition.condition_id}: {condition.name}")

    # Determine final parameters from winner + condition overrides
    chm_variant = condition.chm_variant or winner.get(2, {}).get("chm_variant", "composite")
    arch = condition.arch or winner.get(3, {}).get("arch", "unetpp_effb2")
    in_channels = condition.in_channels or winner.get(2, {}).get("in_channels", 4)

    # Loss parameters
    if condition.loss_type == "dicefocal":
        alpha, beta, cldice_lambda = None, None, 0.0
    elif condition.loss_type == "tversky":
        alpha = condition.alpha
        beta = condition.beta
        cldice_lambda = condition.cldice_lambda
    else:
        alpha, beta, cldice_lambda = winner[4]["alpha"], winner[4]["beta"], winner[4]["cldice_lambda"]

    # Build dataset (using CHM variant from Phase 2)
    # For Phase 2, each variant has its own patch index
    patch_index_path, band_stats_path = _resolve_dataset_files(args.dataset_dir, chm_variant)
    patch_index = load_patch_index(patch_index_path)
    with open(band_stats_path) as f:
        band_stats_for_variant = json.load(f)

    splitter = SpatialCVSplitterV3(stripe_width=STRIPE_WIDTH, test_stripe=TEST_STRIPE)
    train_entries, val_entries = splitter.train_val_split(patch_index, val_fold=args.fold)

    # Train
    train_lr = 1e-4
    train_patience = 15
    train_warmup = int(getattr(args, "warmup_epochs", 5))
    min_early_stop_epoch = None
    tuning_profile = "default"
    if chm_variant == "masked":
        # 2-channel masked input is more sensitive; use gentler optimization and
        # delay early-stop checks so warmup+plateau can settle.
        train_lr = 5e-5
        train_patience = 30
        train_warmup = max(train_warmup, 10)
        min_early_stop_epoch = max(40, train_warmup + 15)
        tuning_profile = "masked_stability_v1"
        print(
            "  [stability] masked profile: "
            f"lr={train_lr:.1e}, warmup={train_warmup}, patience={train_patience}, "
            f"min_early_stop_epoch={min_early_stop_epoch}"
        )

    swa_epoch = args.swa_start_epoch if condition.use_swa else -1
    best_val_result = train_fold(
        arch=arch,
        fold_id=args.fold,
        train_entries=train_entries,
        val_entries=val_entries,
        chm_tif=args.chm_tif,
        mask_tif=args.mask_tif,
        band_stats=band_stats_for_variant,
        output_dir=condition_dir,  # train_fold creates variant/fold subdirs
        device=args.device,
        variant=chm_variant,
        epochs=args.epochs,
        batch_size=16,
        lr=train_lr,
        patience=train_patience,
        tversky_alpha=alpha if alpha is not None else 0.6,
        tversky_beta=beta if beta is not None else 0.4,
        cldice_weight=cldice_lambda if cldice_lambda is not None else 0.3,
        soft_targets=condition.soft_targets if condition.soft_targets is not None else True,
        soft_sigma=2.0,
        swa_start_epoch=swa_epoch,
        warmup_epochs=train_warmup,
        min_epochs_before_early_stop=min_early_stop_epoch,
    )
    best_val_f1 = best_val_result.get("best_val_f1", 0.0)
    # Locate checkpoint for optional locked test evaluation.
    fold_dir = condition_dir / chm_variant / f"fold{args.fold}"
    use_swa_for_inference = bool(best_val_result.get("use_swa_for_inference", False))
    ckpt_path = None
    if use_swa_for_inference and (fold_dir / "swa_model.pt").exists():
        ckpt_path = fold_dir / "swa_model.pt"
    elif (fold_dir / "best.pt").exists():
        ckpt_path = fold_dir / "best.pt"

    eval_metrics = {}
    if args.evaluate_test and ckpt_path is not None and ckpt_path.exists():
        try:
            eval_metrics = _evaluate_test_stripe(ckpt_path, condition, args, winner)
        except Exception as e:
            print(f"    Warning: test stripe evaluation failed: {e}")

    # Merge training + evaluation metrics
    metrics = {
        "condition_id": condition.condition_id,
        "base_condition_id": _base_condition_id(condition.condition_id),
        "run_id": run_id,
        "carry_context": carry_context,
        "condition_name": condition.name,
        "selection_protocol": "heldout_test" if args.evaluate_test else "validation_only",
        "selection_metric": "test_f1" if args.evaluate_test else "val_f1",
        "tuning_profile": tuning_profile,
        "chm_variant": chm_variant,
        "in_channels": int(in_channels),
        "arch": arch,
        "loss_type": condition.loss_type or "tversky",
        "alpha": alpha,
        "beta": beta,
        "cldice_lambda": cldice_lambda,
        "aug_mode": condition.aug_mode or "default",
        "soft_targets": condition.soft_targets if condition.soft_targets is not None else True,
        "use_swa": bool(condition.use_swa),
        "val_f1": float(best_val_result.get("best_val_f1", 0.0)),
        "best_val_f1": float(best_val_result.get("best_val_f1", 0.0)),
        "best_val_dice": float(best_val_result.get("best_val_dice", 0.0)),
        "best_val_cldice": float(best_val_result.get("best_val_cldice", 0.0)),
        "val_cldice": float(best_val_result.get("best_val_cldice", 0.0)),
        "best_epoch": int(best_val_result.get("best_epoch", 0)),
        "stop_reason": str(best_val_result.get("stop_reason", "max_epochs")),
        "threshold_f1": float(best_val_result.get("threshold_f1", best_val_f1)),
        "best_threshold": float(best_val_result.get("best_threshold", 0.5)),
        "n_epochs_trained": int(best_val_result.get("n_epochs_trained", 0)),
    }
    # Add eval-derived metrics (test_f1, cldice, boundary_iou, optimal_threshold, etc.)
    metrics.update({k: (None if v is None else float(v) if isinstance(v, (int, float)) else v) for k, v in eval_metrics.items()})

    # Save
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    reported_f1 = metrics.get('test_f1') if args.evaluate_test else metrics.get('val_f1')
    if isinstance(reported_f1, (int, float)):
        reported_f1_str = f"{reported_f1:.4f}"
    else:
        reported_f1_str = str(reported_f1)
    metric_name = "test F1" if args.evaluate_test else "validation F1"
    print(f"    → {metric_name}: {reported_f1_str}")
    if args.evaluate_test:
        selection_value = metrics.get("test_f1", None)
        selection_metric_name = "test_f1"
    else:
        selection_metric_name = args.selection_metric
        selection_value = metrics.get(selection_metric_name, metrics.get("val_f1", None))
    if isinstance(selection_value, (int, float)):
        selection_value_str = f"{float(selection_value):.4f}"
    else:
        selection_value_str = "N/A"
    print(
        "    → monitor: "
        f"best_epoch={metrics.get('best_epoch', 'N/A')}, "
        f"best_metric({selection_metric_name})={selection_value_str}, "
        f"stop_reason={metrics.get('stop_reason', 'N/A')}"
    )

    return metrics


def _evaluate_test_stripe(
    ckpt_path: Path,
    condition: AblationConfig,
    args: argparse.Namespace,
    winner: dict,
) -> dict:
    """Sliding-window inference on test stripe with all metrics."""
    # Load model
    arch = condition.arch or winner[3].get("arch", "unetpp_effb2")
    in_channels = condition.in_channels or winner[2].get("in_channels", 4)

    model = build_model(arch, in_channels=in_channels, pretrained=False).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load test stripe
    chm_variant = condition.chm_variant or winner[2].get("chm_variant", "composite")

    with rasterio.open(args.chm_tif) as src:
        test_stripe = src.read(
            list(range(1, in_channels + 1)),
            window=((0, 5000), (0, STRIPE_WIDTH))
        ).astype(np.float32)

    with rasterio.open(args.mask_tif) as src:
        test_mask = src.read(1, window=((0, 5000), (0, STRIPE_WIDTH))).astype(np.uint8)

    # Normalize
    _, band_stats_path = _resolve_dataset_files(args.dataset_dir, chm_variant)
    with open(band_stats_path) as f:
        band_stats = json.load(f)
    binary_bands = _get_binary_bands(chm_variant)
    test_stripe = normalize_bands(test_stripe, band_stats, binary_bands=binary_bands)

    # Predict
    prob = sliding_window_predict(
        model, test_stripe, device=args.device, patch_size=256, stride=192
    )

    # Find optimal threshold
    best_by_f1, _ = threshold_sweep([prob], [test_mask], [(test_mask > 0)])
    threshold = best_by_f1.get("threshold", 0.5)
    pred_bin = (prob >= threshold).astype(np.uint8)

    # Pixel-level metrics
    metrics = accumulate_pixel_metrics([prob], [test_mask], [(test_mask > 0)], threshold=threshold)
    metrics["test_f1"] = float(metrics.get("f1", 0.0))
    metrics["optimal_threshold"] = float(threshold)

    # Extended metrics. Component AP is bounded because dense/noisy predictions can
    # create thousands of components and turn exact pairwise IoU into hours of CPU.
    try:
        metrics["boundary_iou"] = float(boundary_iou(pred_bin, (test_mask > 0).astype(np.uint8)))
        metrics["cldice"] = float(cldice_metric(pred_bin, (test_mask > 0).astype(np.uint8)))
        _, n_pred = label_components(pred_bin.astype(bool))
        _, n_gt = label_components((test_mask > 0).astype(bool))
        max_pairs = int(getattr(args, "max_ap_component_pairs", 200000))
        metrics["n_pred_components"] = float(n_pred)
        metrics["n_gt_components"] = float(n_gt)
        if n_pred * n_gt <= max_pairs:
            ap_metrics = ap_at_iou(pred_bin, (test_mask > 0).astype(np.uint8), iou_thresholds=(0.25, 0.50))
            metrics.update(ap_metrics)
        else:
            print(
                f"    Warning: skipped AP@IoU ({n_pred} predicted × {n_gt} GT components > {max_pairs} pairs)",
                flush=True,
            )
            metrics["ap_iou_25"] = None
            metrics["ap_iou_50"] = None
    except Exception as e:
        print(f"    Warning: Extended metrics failed: {e}")

    return metrics


def run_phase(phase_id: int, args: argparse.Namespace, winner: dict) -> dict:
    """Run all conditions in a phase; return winning condition."""
    conditions_dict = ALL_PHASES[phase_id]

    # Handle Phase 4: fill in 4G, 4H with winner from 4B–4F
    if phase_id == 4:
        best_cond = None
        best_f1 = -1
        for cid, cfg in list(conditions_dict.items()):
            if cid in ("4G", "4H"):
                continue
            # Train and find best α/β
            # ... simplified: assume 4E is best (V10 precision)
        winner[4] = {"alpha": 0.6, "beta": 0.4, "cldice_lambda": 0.3}  # V10.2
        conditions_dict["4G"] = {
            "name": "loss_cldice_low",
            "loss_type": "tversky",
            "alpha": winner[4]["alpha"],
            "beta": winner[4]["beta"],
            "cldice_lambda": 0.1,
        }
        conditions_dict["4H"] = {
            "name": "loss_cldice_v10",
            "loss_type": "tversky",
            "alpha": winner[4]["alpha"],
            "beta": winner[4]["beta"],
            "cldice_lambda": 0.3,
        }

    print(f"\n{'='*80}")
    print(f"PHASE {phase_id}: {list(conditions_dict.keys())}")
    print(f"{'='*80}")

    all_metrics = []
    best_metrics: dict = {}
    requested_condition_ids = set(args.condition_ids) if args.condition_ids else None
    for cid_idx, (cid, cfg_dict) in enumerate(conditions_dict.items()):
        if args.condition is not None and cid_idx != args.condition:
            continue
        if requested_condition_ids is not None and cid not in requested_condition_ids:
            continue

        condition = AblationConfig(phase=phase_id, condition_id=cid, **cfg_dict)
        metrics = run_condition(condition, args, winner)
        all_metrics.append(metrics)

    # Determine winner. During ablation/model selection this is validation-only;
    # test metrics are allowed only in explicit final --evaluate-test runs.
    if all_metrics:
        if args.evaluate_test:
            best_metrics = max(all_metrics, key=lambda m: m.get("test_f1", m.get("val_f1", 0)))
            score = best_metrics.get("test_f1", None)
            score_str = f"(locked test F1: {score:.4f})" if isinstance(score, (int, float)) else ""
        else:
            metric_key = args.selection_metric
            best_metrics = max(all_metrics, key=lambda m: m.get(metric_key, m.get("val_f1", 0)))
            score = best_metrics.get(metric_key, None)
            if isinstance(score, (int, float)):
                score_str = f"({metric_key}: {score:.4f})"
            else:
                score_str = f"({metric_key}: N/A)"
        print(f"\n  ⭐ Phase {phase_id} winner: {best_metrics.get('run_id', best_metrics['condition_id'])} {best_metrics['condition_name']} {score_str}")

        # Update winner tracking
        if phase_id == 2:
            winner[2] = {"chm_variant": best_metrics.get("chm_variant", "composite"),
                         "in_channels": best_metrics.get("in_channels", 4)}
        elif phase_id == 3:
            winner[3] = {"arch": best_metrics.get("arch", "unetpp_effb2")}
        elif phase_id == 4:
            winner[4] = {"alpha": best_metrics.get("alpha", 0.6),
                         "beta": best_metrics.get("beta", 0.4),
                         "cldice_lambda": best_metrics.get("cldice_lambda", 0.3)}

    # Save phase results
    result_suffix = "test" if args.evaluate_test else "val"
    results_csv = args.output_dir / f"phase{phase_id}_results_{result_suffix}.csv"
    legacy_results_csv = args.output_dir / f"phase{phase_id}_results.csv"
    if all_metrics:
        existing_metrics = []
        if results_csv.exists():
            with open(results_csv) as f:
                existing_metrics = list(csv.DictReader(f))

        merged_by_run = {row.get("run_id", row.get("condition_id", "")): row for row in existing_metrics}
        for row in all_metrics:
            merged_by_run[row.get("run_id", row.get("condition_id", ""))] = row
        merged_metrics = list(merged_by_run.values())
        fieldnames = []
        for row in merged_metrics:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(results_csv, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_metrics)
        if not args.evaluate_test:
            with open(legacy_results_csv, "w") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(merged_metrics)

    return best_metrics if all_metrics else {}


def main():
    parser = argparse.ArgumentParser(description="CWD segmentation ablation study (Phase 2–6)")
    parser.add_argument("--phase", type=int, choices=[2, 3, 4, 5, 6], default=2,
                        help="Phase to run (default: 2)")
    parser.add_argument("--condition", type=int, default=None,
                        help="Specific condition within phase (0-indexed), or None for all")
    parser.add_argument("--condition-ids", type=str, default=None,
                        help="Comma-separated condition IDs to run (e.g. 2A,2B,2C).")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases sequentially (2→3→4→5→6)")
    parser.add_argument("--reports-only", action="store_true",
                        help="Regenerate figures from existing results, skip training")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the held-out test stripe. Use only after final model selection is locked.")
    parser.add_argument("--max-ap-component-pairs", type=int, default=200000,
                        help="Maximum predicted×GT component pairs for exact AP@IoU on the held-out test stripe.")
    parser.add_argument("--fold", type=int, default=0, help="Fold to train (default: 0)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--swa-start-epoch", type=int, default=35,
                        help="SWA start epoch (default: 35)")
    parser.add_argument("--no-swa", action="store_true", help="Disable SWA")
    parser.add_argument("--output-dir", type=Path, default=Path("seg_pipeline/output/ablation_v10"),
                        help="Output directory for results")
    parser.add_argument("--dataset-dir", type=Path, default=Path("seg_pipeline/output/phase2_dataset_v10"),
                        help="Base dataset directory containing patch_index_*.csv and band_stats_*.json")
    parser.add_argument("--chm-variant", type=str, default="gauss",
                        choices=["baseline", "raw", "gauss", "masked", "composite"],
                        help="CHM variant to use (default: gauss, from Phase 2 winner)")
    parser.add_argument("--chm-tif", type=Path, default=None,
                        help="CHM raster path (auto-detected from --chm-variant if not provided)")
    parser.add_argument("--mask-tif", type=Path, default=Path("seg_pipeline/output/phase1_masks/406455_2021_tava_truemask.tif"),
                        help="Mask TIF path")
    parser.add_argument("--device", type=str, default=str(DEVICE), help="Device (cuda/cpu)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs before ReduceLROnPlateau (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--carry-winner", action="append", default=[],
                        help="Carry forward winner as 'phase:condition_id' (e.g. 2:2C). Can be repeated.")
    parser.add_argument("--selection-metric", type=str, default="val_cldice",
                        choices=["val_f1", "best_val_f1", "val_dice", "best_val_dice", "val_cldice", "best_val_cldice"],
                        help="Metric used to select phase winners (default: val_cldice)")

    args = parser.parse_args()
    if args.condition_ids:
        args.condition_ids = [x.strip() for x in args.condition_ids.split(",") if x.strip()]
    else:
        args.condition_ids = None
    args.device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set deterministic seeds for reproducibility
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # If wrapper passed a Phase-2 carry-winner, ensure args.chm_variant matches it
    for cw in getattr(args, 'carry_winner', []):
        try:
            ph, cid = cw.split(":", 1)
            phn = int(ph)
            base_cid = _base_condition_id(cid)
            if phn == 2 and base_cid in ALL_PHASES[2]:
                args.chm_variant = ALL_PHASES[2][base_cid].get('chm_variant', args.chm_variant)
        except Exception:
            pass

    # Auto-detect CHM TIF path from variant
    if args.chm_tif is None:
        variant_map = {
            "baseline": "seg_pipeline/input/baseline_chm.tif",
            "raw": "seg_pipeline/input/raw_chm.tif",
            "gauss": "seg_pipeline/input/gauss_chm.tif",
            "masked": "seg_pipeline/input/masked_chm.tif",
            "composite": "seg_pipeline/input/composite_4band.tif",
        }
        args.chm_tif = Path(variant_map.get(args.chm_variant, "seg_pipeline/input/gauss_chm.tif"))

    # For Phase-2 single-condition runs, auto-select the condition variant before
    # any dataset checks so users don't need an extra --chm-variant flag.
    if args.phase == 2 and args.condition is not None:
        phase2_items = list(ALL_PHASES[2].items())
        if 0 <= args.condition < len(phase2_items):
            _, cfg = phase2_items[args.condition]
            args.chm_variant = cfg.get("chm_variant", args.chm_variant)

    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, SWA start: {args.swa_start_epoch if not args.no_swa else 'disabled'}")

    if args.reports_only:
        print("Reports-only mode: skipping training")
        return

    # Initialize winner tracking
    winner = {
        2: {"chm_variant": "composite", "in_channels": 4},
        3: {"arch": "unetpp_effb2"},
        4: {"alpha": 0.6, "beta": 0.4, "cldice_lambda": 0.3},
    }

    # Apply carry-winner overrides passed from wrapper script
    for cw in getattr(args, 'carry_winner', []):
        try:
            ph, cid = cw.split(":", 1)
            phn = int(ph)
            base_cid = _base_condition_id(cid)
            if phn in ALL_PHASES and base_cid in ALL_PHASES[phn]:
                cfg = ALL_PHASES[phn][base_cid]
                if phn == 2:
                    winner[2] = {"chm_variant": cfg.get("chm_variant"), "in_channels": cfg.get("in_channels")}
                elif phn == 3:
                    winner[3] = {"arch": cfg.get("arch")}
                elif phn == 4:
                    winner[4] = {"alpha": cfg.get("alpha"), "beta": cfg.get("beta"), "cldice_lambda": cfg.get("cldice_lambda", 0.0)}
        except Exception:
            print(f"Warning: failed to parse carry-winner '{cw}'", flush=True)

    if args.all:
        for phase in [2, 3, 4, 5, 6]:
            run_phase(phase, args, winner)
    else:
        run_phase(args.phase, args, winner)

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
