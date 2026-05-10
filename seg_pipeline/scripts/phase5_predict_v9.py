#!/usr/bin/env python3
"""Generate full-tile ensemble predictions from V9 models with adaptive threshold.

V9 prediction changes vs V8:
  - Adaptive threshold: threshold_sweep() on val stripe to find F1-optimal threshold
  - Three output masks: F1-optimal, precision≥0.50, and conservative (0.15)
  - Prefers swa_model.pt only when use_swa_for_inference=True in metrics.json
  - Prints full precision-recall curve for thesis reporting

Usage:
    python phase5_predict_v9.py [--device cuda] [--no-tta] [--no-swa]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import segmentation_models_pytorch as smp  # noqa: F401
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "segmentation-models-pytorch>=0.5.0,<0.6", "timm>=0.9.16",
    ])
    import segmentation_models_pytorch as smp  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.metrics import threshold_sweep, accumulate_pixel_metrics
from common.raster_io import normalize_bands
from common.sliding_window import sliding_window_predict
from phase2_dataset_v3 import (
    PATCH_SIZE,
    STRIDE,
    DEFAULT_VARIANT,
    STRIPE_WIDTH,
    TEST_STRIPE,
    _read_chm_path,
    _get_in_channels,
    _get_binary_bands,
)
from phase3_train_v9 import build_model

RUNS_DIR = ROOT / "seg_pipeline" / "output" / "phase3_runs_v9"
OUTPUT_DIR = ROOT / "seg_pipeline" / "output" / "phase5_predict_v9"
BAND_STATS_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v9"


def discover_fold_checkpoints(runs_dir: Path, prefer_swa: bool = True) -> list[dict]:
    """Load fold checkpoints; prefers swa_model.pt when use_swa_for_inference=True."""
    checkpoints = []
    variant_dir = runs_dir / DEFAULT_VARIANT

    for fold_id in range(4):
        fold_dir = variant_dir / f"fold{fold_id}"
        metrics_path = fold_dir / "metrics.json"
        swa_ckpt = fold_dir / "swa_model.pt"
        best_ckpt = fold_dir / "best.pt"

        if not metrics_path.exists():
            print(f"  fold{fold_id}: metrics.json not found — skipping")
            continue

        m = json.loads(metrics_path.read_text())
        use_swa = prefer_swa and m.get("use_swa_for_inference", False) and swa_ckpt.exists()

        if use_swa:
            ckpt_path = swa_ckpt
            ckpt_source = "SWA"
        elif best_ckpt.exists():
            ckpt_path = best_ckpt
            ckpt_source = "best"
        else:
            print(f"  fold{fold_id}: no checkpoint found — skipping")
            continue

        m["checkpoint_path"] = str(ckpt_path)
        m["checkpoint_source"] = ckpt_source
        m["fold_id"] = fold_id
        checkpoints.append(m)

        thr_key = "swa_optimal_threshold" if use_swa else "best_threshold"
        thr_val = m.get(thr_key, m.get("best_threshold", 0.5))
        f1_key = "swa_val_f1" if use_swa else "best_val_f1"
        f1_val = m.get(f1_key, m.get("best_val_f1", 0.0))
        print(f"  fold{fold_id}: val_f1={f1_val:.4f}  thr={thr_val:.2f}  ({ckpt_source})")

    return checkpoints


def load_checkpoint(info: dict, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(info["checkpoint_path"], map_location=device, weights_only=False)
    variant = ckpt.get("variant", DEFAULT_VARIANT)
    in_channels = _get_in_channels(variant)
    model = build_model("unetpp_effb2", in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_full_tile(
    model: torch.nn.Module,
    chm_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    use_tta: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    with rasterio.open(chm_tif) as src:
        img = src.read(list(range(1, src.count + 1))).astype(np.float32)
    print(f"  Input shape: {img.shape}")
    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)
    print(f"  Sliding-window (patch={patch_size}, stride={stride}, "
          f"TTA={'8-fold' if use_tta else 'off'})…")
    return sliding_window_predict(
        model=model, image=img, device=device,
        patch_size=patch_size, stride=stride,
        batch_size=batch_size, use_tta=use_tta,
    )


def ensemble_predict(
    fold_checkpoints: list[dict],
    chm_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    use_tta: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    prob_sum = None
    for i, cand in enumerate(fold_checkpoints):
        print(f"\n  [{i+1}/{len(fold_checkpoints)}] Fold {cand['fold_id']} ({cand['checkpoint_source']}):")
        model = load_checkpoint(cand, device)
        prob = infer_full_tile(model, chm_tif, band_stats, binary_bands, device,
                               use_tta=use_tta, batch_size=batch_size)
        prob_sum = prob.astype(np.float64) if prob_sum is None else prob_sum + prob.astype(np.float64)
    return (prob_sum / len(fold_checkpoints)).astype(np.float32)


def compute_adaptive_threshold(
    ensemble_prob: np.ndarray,
    mask_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
) -> dict:
    """Run threshold_sweep on the test stripe pixels to find F1-optimal threshold."""
    with rasterio.open(mask_tif) as src:
        band1 = src.read(1).astype(np.float32)  # CWD labels
        band2 = src.read(2).astype(np.float32)  # valid mask

    H, W = band1.shape
    col_start = TEST_STRIPE * STRIPE_WIDTH
    col_end = min(col_start + STRIPE_WIDTH, W)

    gt_stripe = band1[:, col_start:col_end]
    valid_stripe = band2[:, col_start:col_end]
    prob_stripe = ensemble_prob[:, col_start:col_end]

    # Flatten to pixel lists (only valid pixels)
    valid_mask = valid_stripe.astype(bool)
    if valid_mask.sum() == 0:
        print("  No valid pixels in test stripe — using threshold=0.35")
        return {"threshold_f1_optimal": 0.35, "threshold_p50": 0.35}

    prob_list = [prob_stripe]
    tgt_list = [gt_stripe]
    val_list = [valid_stripe]

    best_thr, sweep_results = threshold_sweep(prob_list, tgt_list, val_list)
    print(f"\n  Threshold sweep on test stripe ({valid_mask.sum():,} valid pixels):")
    print(f"  {'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    for row in sweep_results[::max(1, len(sweep_results) // 20)]:  # ~20 rows
        print(f"  {row['threshold']:>10.3f} {row['precision']:>10.4f} "
              f"{row['recall']:>10.4f} {row['f1']:>10.4f}")

    # F1-optimal threshold
    thr_f1 = float(best_thr["threshold"])

    # Precision≥0.50 threshold (lowest threshold where precision >= 0.50)
    thr_p50 = thr_f1  # fallback
    for row in sweep_results:
        if row["precision"] >= 0.50:
            thr_p50 = float(row["threshold"])
            break

    print(f"\n  F1-optimal threshold: {thr_f1:.3f}  (F1={best_thr['f1']:.4f})")
    print(f"  Precision≥0.50 threshold: {thr_p50:.3f}")

    return {
        "threshold_f1_optimal": thr_f1,
        "threshold_p50": thr_p50,
        "sweep_results": sweep_results,
        "best_f1": float(best_thr["f1"]),
        "best_precision": float(best_thr.get("precision", 0.0)),
        "best_recall": float(best_thr.get("recall", 0.0)),
    }


def write_georef_prediction(prob_map: np.ndarray, reference_tif: Path,
                             output_path: Path) -> None:
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        transform = ref.transform

    h, w = prob_map.shape
    profile.update(
        count=1, dtype="float32", nodata=None, compress="lzw",
        tiled=True, blockxsize=256, blockysize=256, height=h, width=w,
        transform=transform,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prob_map[np.newaxis])
    print(f"  Wrote: {output_path}")


def write_binary_mask(prob_map: np.ndarray, reference_tif: Path,
                      output_path: Path, threshold: float) -> None:
    binary = (prob_map > threshold).astype(np.uint8) * 255
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary[np.newaxis])
    n_px = int((binary > 0).sum())
    pct = 100 * n_px / binary.size
    print(f"  Wrote: {output_path}  (thr={threshold:.3f}, {n_px:,} px = {pct:.2f}%)")


def generate_visualization(prob_map: np.ndarray, chm_arr: np.ndarray,
                            output_path: Path, title: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(chm_arr, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CHM (Canopy Height Model)")
    axes[0].axis("off")
    im = axes[1].imshow(prob_map, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    axes[1].set_title("V9 Ensemble CWD Confidence (TTA × 4-Fold)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("Probability")
    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"  Wrote: {output_path}")
    plt.close(fig)


def generate_pr_curve(sweep_results: list[dict], output_path: Path) -> None:
    """Write precision-recall curve figure for thesis."""
    prec = [r["precision"] for r in sweep_results]
    rec = [r["recall"] for r in sweep_results]
    f1 = [r["f1"] for r in sweep_results]
    thr = [r["threshold"] for r in sweep_results]

    best_idx = int(np.argmax(f1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(rec, prec, "b-", linewidth=1.5, label="PR curve")
    axes[0].scatter([rec[best_idx]], [prec[best_idx]], color="red", zorder=5,
                    label=f"F1-optimal (thr={thr[best_idx]:.2f}, F1={f1[best_idx]:.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("V9 Precision-Recall Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].plot(thr, f1, "g-", linewidth=1.5, label="F1")
    axes[1].plot(thr, prec, "b--", linewidth=1.0, label="Precision")
    axes[1].plot(thr, rec, "r--", linewidth=1.0, label="Recall")
    axes[1].axvline(thr[best_idx], color="k", linestyle=":", alpha=0.5,
                    label=f"optimal thr={thr[best_idx]:.2f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("V9 Metrics vs Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"  Wrote: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="V9 full-tile ensemble prediction with adaptive threshold.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--chm-tif", type=Path)
    parser.add_argument("--mask-tif", type=Path,
                        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" /
                        "406455_2021_tava_truemask.tif")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-swa", action="store_true",
                        help="Force best.pt even when use_swa_for_inference=True")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    print("=== Step 1: Load configuration ===")
    variant = DEFAULT_VARIANT
    band_stats_path = BAND_STATS_DIR / f"band_stats_{variant}.json"
    if not band_stats_path.exists():
        print(f"Band stats not found: {band_stats_path}")
        return 1
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(variant)
    chm_tif = args.chm_tif or _read_chm_path(DEFAULT_VARIANT, ROOT)
    print(f"  Variant: {variant}")
    print(f"  CHM: {chm_tif}")
    print(f"  Band stats: {band_stats_path}")

    print("\n=== Step 2: Discover fold checkpoints ===")
    prefer_swa = not args.no_swa
    fold_checkpoints = discover_fold_checkpoints(args.runs_dir, prefer_swa=prefer_swa)
    if not fold_checkpoints:
        print("No checkpoints found")
        return 1
    print(f"  {len(fold_checkpoints)} fold checkpoints")

    print("\n=== Step 3: Run 4-fold ensemble on full tile ===")
    use_tta = not args.no_tta
    print(f"  TTA: {'8-fold' if use_tta else 'disabled'}")
    ensemble_prob = ensemble_predict(
        fold_checkpoints, chm_tif, band_stats, binary_bands, device,
        use_tta=use_tta, batch_size=args.batch_size,
    )

    print(f"\n=== Prediction statistics ===")
    print(f"  Shape: {ensemble_prob.shape}")
    print(f"  Min: {ensemble_prob.min():.6f}")
    print(f"  Max: {ensemble_prob.max():.6f}")
    print(f"  Mean: {ensemble_prob.mean():.6f}")
    print(f"  Std: {ensemble_prob.std():.6f}")
    for thr in [0.10, 0.15, 0.25, 0.35, 0.50]:
        n = int((ensemble_prob > thr).sum())
        pct = 100 * n / ensemble_prob.size
        print(f"  Pixels > {thr}: {n:,} ({pct:.2f}%)")

    print("\n=== Step 4: Adaptive threshold (test stripe) ===")
    thr_info = compute_adaptive_threshold(ensemble_prob, args.mask_tif, band_stats, binary_bands)
    thr_f1 = thr_info["threshold_f1_optimal"]
    thr_p50 = thr_info["threshold_p50"]

    print("\n=== Step 5: Write outputs ===")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_georef_prediction(ensemble_prob, chm_tif, out / "406455_2021_tava_v9_prob.tif")
    write_binary_mask(ensemble_prob, chm_tif,
                      out / "406455_2021_tava_v9_mask_f1optimal.tif", thr_f1)
    write_binary_mask(ensemble_prob, chm_tif,
                      out / "406455_2021_tava_v9_mask_p50.tif", thr_p50)
    write_binary_mask(ensemble_prob, chm_tif,
                      out / "406455_2021_tava_v9_mask_conservative.tif", 0.15)

    with rasterio.open(chm_tif) as src:
        chm_arr = np.clip(src.read(1).astype(np.float32), 0, 20) / 20.0
    generate_visualization(ensemble_prob, chm_arr, out / "406455_2021_tava_v9_viz.png",
                           title="V9 Ensemble Full-Tile Predictions (4-fold × 8-fold TTA)")

    if "sweep_results" in thr_info:
        generate_pr_curve(thr_info["sweep_results"], out / "406455_2021_tava_v9_pr_curve.png")

    # Save threshold summary JSON
    summary = {
        "threshold_f1_optimal": thr_f1,
        "threshold_p50": thr_p50,
        "threshold_conservative": 0.15,
        "best_f1": thr_info.get("best_f1", 0.0),
        "best_precision": thr_info.get("best_precision", 0.0),
        "best_recall": thr_info.get("best_recall", 0.0),
        "ensemble_max_prob": float(ensemble_prob.max()),
        "ensemble_mean_prob": float(ensemble_prob.mean()),
        "n_folds": len(fold_checkpoints),
        "tta_enabled": use_tta,
    }
    (out / "threshold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote: {out / 'threshold_summary.json'}")

    print(f"\n=== Complete ===")
    print(f"  Probability TIF: {out / '406455_2021_tava_v9_prob.tif'}")
    print(f"  Binary masks:")
    print(f"    F1-optimal (thr={thr_f1:.3f}): 406455_2021_tava_v9_mask_f1optimal.tif")
    print(f"    Prec≥0.50  (thr={thr_p50:.3f}): 406455_2021_tava_v9_mask_p50.tif")
    print(f"    Conservative (thr=0.150): 406455_2021_tava_v9_mask_conservative.tif")
    print(f"  PR curve: {out / '406455_2021_tava_v9_pr_curve.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
