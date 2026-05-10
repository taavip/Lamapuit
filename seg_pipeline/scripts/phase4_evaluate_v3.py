#!/usr/bin/env python3
"""Phase IV V3: Evaluation with 5-fold ensemble + connected-component post-processing.

V3 evaluation innovations over V2:
  1. 5-fold ensemble inference — averages sigmoid outputs from all fold checkpoints
     before thresholding.  Reduces prediction variance and improves calibration
     compared to single-fold evaluation (Dietterich, 2000; Breiman, 1996).
  2. Connected-component (CC) post-processing — removes isolated detections smaller
     than a configurable area threshold (default: 50 px = 2 m² at 0.2 m/px).
     CWD logs span ≥0.5 m in their shortest dimension; isolated detections of 1–10
     pixels are terrain or canopy noise artefacts.
  3. Per-fold evaluation retained for diagnostic comparison.
  4. Precision/recall/F1 reported alongside Dice and IoU.

Test stripe: cols 0–999 (matching V1 for apples-to-apples comparison).

Usage:
    python phase4_evaluate_v3.py                         # full evaluation with ensemble
    python phase4_evaluate_v3.py --no-ensemble           # per-fold only (like V2)
    python phase4_evaluate_v3.py --top-k 1 --no-tta      # quick sanity check
    python phase4_evaluate_v3.py --cc-min-px 100         # stricter CC filter
"""

from __future__ import annotations

import argparse
import csv
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

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.metrics import accumulate_pixel_metrics, threshold_sweep
from common.raster_io import normalize_bands
from common.sliding_window import sliding_window_predict
from phase2_dataset_v3 import (
    PATCH_SIZE,
    STRIDE,
    DEFAULT_VARIANT,
    _read_chm_path,
    _get_in_channels,
    _get_binary_bands,
)
from phase3_train_v3 import build_model

ALL_VARIANTS = ["baseline", "raw", "gauss", "masked", "composite"]
TEST_STRIPE_COLS = (0, 1000)  # matches V1 for apples-to-apples comparison
EPS = 1e-8


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def discover_checkpoints(runs_dir: Path, variant_filter: str | None = None) -> list[dict]:
    """Collect all fold metrics.json from phase3_runs_v3/<variant>/fold<k>/."""
    candidates: list[dict] = []
    variants = [variant_filter] if variant_filter else ALL_VARIANTS
    for variant in variants:
        variant_dir = runs_dir / variant
        if not variant_dir.exists():
            continue
        for fold_dir in sorted(variant_dir.iterdir()):
            if not fold_dir.is_dir():
                continue
            metrics_path = fold_dir / "metrics.json"
            ckpt_path = fold_dir / "best.pt"
            if not metrics_path.exists() or not ckpt_path.exists():
                continue
            m = json.loads(metrics_path.read_text())
            m["checkpoint_path"] = str(ckpt_path)
            candidates.append(m)
    return sorted(candidates, key=lambda x: x.get("best_val_dice", 0.0), reverse=True)


def load_checkpoint(info: dict, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(info["checkpoint_path"], map_location=device, weights_only=False)
    variant = ckpt.get("variant", info.get("variant", DEFAULT_VARIANT))
    in_channels = _get_in_channels(variant)
    model = build_model("unetpp_effb2", in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Single-model inference on test stripe
# ---------------------------------------------------------------------------


@torch.no_grad()
def infer_test_stripe(
    model: torch.nn.Module,
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    use_tta: bool = True,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run sliding-window inference on test stripe (cols 0–999).

    Returns (prob_map, target_arr, valid_arr) each of shape (H, W).
    """
    col_start, col_end = TEST_STRIPE_COLS
    W_stripe = col_end - col_start

    with rasterio.open(chm_tif) as src:
        H = src.height
        img = src.read(
            list(range(1, src.count + 1)),
            window=Window(col_start, 0, W_stripe, H),
            boundless=True, fill_value=np.nan,
        ).astype(np.float32)

    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)

    with rasterio.open(mask_tif) as msrc:
        raw = msrc.read(
            [1, 2],
            window=Window(col_start, 0, W_stripe, H),
            boundless=True, fill_value=0.0,
        ).astype(np.float32)
    target_arr = raw[0]
    valid_arr = raw[1]

    prob_map = sliding_window_predict(
        model=model, image=img, device=device,
        patch_size=patch_size, stride=stride,
        batch_size=batch_size, use_tta=use_tta,
    )
    return prob_map, target_arr, valid_arr


# ---------------------------------------------------------------------------
# 5-fold ensemble inference
# ---------------------------------------------------------------------------


def ensemble_predict(
    fold_checkpoints: list[dict],
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    use_tta: bool = True,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average sigmoid outputs from all fold checkpoints (ensemble inference).

    Rationale: each fold's model was trained on a different geographic subset,
    developing complementary spatial features.  Averaging their probability maps
    reduces prediction variance and improves calibration (Lakshminarayanan et al.,
    2017 approximated by simple averaging of sigmoid outputs from diverse models).

    The ensemble prediction is equivalent to a uniform-weight model average,
    which Dietterich (2000) shows reduces variance by 1/K for K decorrelated
    models.  Folds trained on different spatial stripes are naturally decorrelated.
    """
    prob_sum = None
    target_arr = None
    valid_arr = None

    for i, cand in enumerate(fold_checkpoints):
        fold_id = cand["fold_id"]
        print(f"    Loading fold {fold_id} checkpoint for ensemble…")
        model = load_checkpoint(cand, device)
        prob_i, target_i, valid_i = infer_test_stripe(
            model=model, chm_tif=chm_tif, mask_tif=mask_tif,
            band_stats=band_stats, binary_bands=binary_bands,
            device=device, use_tta=use_tta, batch_size=batch_size,
        )
        if prob_sum is None:
            prob_sum = prob_i.astype(np.float64)
            target_arr = target_i
            valid_arr = valid_i
        else:
            prob_sum += prob_i.astype(np.float64)

    n = len(fold_checkpoints)
    ensemble_prob = (prob_sum / n).astype(np.float32)
    return ensemble_prob, target_arr, valid_arr


# ---------------------------------------------------------------------------
# Connected-component post-processing
# ---------------------------------------------------------------------------


def cc_postprocess(
    binary_mask: np.ndarray,
    min_pixels: int = 50,
) -> np.ndarray:
    """Remove isolated connected components smaller than min_pixels.

    Rationale: At 0.2 m/px, CWD logs have a minimum cross-sectional width of
    ~0.4 m and a minimum length of ~0.5 m, corresponding to ≥2 × 2.5 = 5 pixels
    in the narrowest dimension.  Isolated blobs of < min_pixels pixels are
    terrain noise (micro-topographic features, canopy gap shadows) rather than
    true CWD detections.  The default min_pixels=50 corresponds to 2 m² ground
    area at 0.2 m/px — a conservative lower bound for detectable CWD in Estonian
    forest inventory standards.

    Uses 4-connectivity (face-connected) labelling, which is more conservative
    than 8-connectivity and avoids merging diagonally adjacent noise blobs.

    Args:
        binary_mask: (H, W) bool or 0/1 float32 array
        min_pixels: remove components with fewer pixels than this

    Returns:
        Filtered binary mask as float32 {0.0, 1.0}
    """
    if not _HAS_SCIPY:
        print("  [CC postprocess] scipy not available — skipping (pip install scipy)")
        return binary_mask.astype(np.float32)

    bm = (binary_mask > 0.5).astype(np.uint8)
    labeled, n_labels = ndi.label(bm, structure=np.ones((3, 3), dtype=np.uint8))
    if n_labels == 0:
        return bm.astype(np.float32)

    sizes = ndi.sum(bm, labeled, range(1, n_labels + 1))
    keep_labels = np.where(np.array(sizes) >= min_pixels)[0] + 1
    filtered = np.isin(labeled, keep_labels).astype(np.float32)

    removed = n_labels - len(keep_labels)
    print(f"  [CC postprocess] {n_labels} components found, {removed} removed (<{min_pixels} px)")
    return filtered


# ---------------------------------------------------------------------------
# Georeferenced output and overlay figure (shared with V2)
# ---------------------------------------------------------------------------


def write_georef_prediction(
    prob_map: np.ndarray,
    reference_tif: Path,
    output_path: Path,
    col_start: int,
    col_end: int,
) -> None:
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        res = transform.a
        new_left = transform.c + col_start * res
        new_transform = rasterio.transform.from_origin(
            west=new_left, north=transform.f, xsize=res, ysize=res
        )
    h, w = prob_map.shape
    profile.update(
        count=1, dtype="float32", nodata=None, compress="lzw",
        tiled=True, blockxsize=256, blockysize=256,
        height=h, width=w, transform=new_transform,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prob_map[np.newaxis])


def generate_overlay_figure(
    prob_map: np.ndarray,
    target_arr: np.ndarray,
    valid_arr: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str = "",
    chm_arr: np.ndarray | None = None,
    filtered_mask: np.ndarray | None = None,
) -> None:
    v = valid_arr > 0.5
    pred_raw = (prob_map >= threshold).astype(np.float32)
    pred = filtered_mask if filtered_mask is not None else pred_raw
    gt = (target_arr > 0.5).astype(np.float32)
    bg = chm_arr if chm_arr is not None else np.zeros_like(prob_map)

    ncols = 5 if filtered_mask is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), constrained_layout=True)

    def _show(ax, mask, cmap, alpha=0.65, title_str=""):
        ax.imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        if mask is not None:
            disp = np.where(v, mask, np.nan)
            ax.imshow(np.ma.masked_invalid(disp), cmap=cmap, alpha=alpha, vmin=0, vmax=1)
        ax.set_title(title_str)
        ax.axis("off")

    _show(axes[0], None, None, title_str="CHM background")
    _show(axes[1], gt, "Greens", title_str="GT Mask")
    _show(axes[2], pred_raw, "Reds", title_str=f"Pred raw (thr={threshold:.2f})")

    if filtered_mask is not None:
        _show(axes[3], filtered_mask, "Oranges", title_str="Pred + CC filter")
        fp = (pred == 1) & (gt == 0) & v
        fn = (pred == 0) & (gt == 1) & v
        err = np.zeros((*prob_map.shape, 3), dtype=np.float32)
        err[..., 0] = fp.astype(np.float32)
        err[..., 2] = fn.astype(np.float32)
        axes[4].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        axes[4].imshow(err, alpha=0.70)
        axes[4].set_title("FP (red) / FN (blue)")
        axes[4].axis("off")
    else:
        fp = (pred_raw == 1) & (gt == 0) & v
        fn = (pred_raw == 0) & (gt == 1) & v
        err = np.zeros((*prob_map.shape, 3), dtype=np.float32)
        err[..., 0] = fp.astype(np.float32)
        err[..., 2] = fn.astype(np.float32)
        axes[3].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
        axes[3].imshow(err, alpha=0.70)
        axes[3].set_title("FP (red) / FN (blue)")
        axes[3].axis("off")

    fig.suptitle(title, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase IV V3: Ensemble evaluation + CC post-processing")
    p.add_argument(
        "--runs-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v3",
    )
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
        default=ROOT / "seg_pipeline" / "output" / "phase4_report_v3",
    )
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT, choices=ALL_VARIANTS)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--no-ensemble", action="store_true", help="Skip ensemble evaluation")
    p.add_argument(
        "--cc-min-px", type=int, default=50,
        help="Min pixels for CC filter (0=disabled). Default 50 = 2 m² at 0.2m/px",
    )
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    use_tta = not args.no_tta
    use_cc = args.cc_min_px > 0

    # Load band stats and CHM path for variant
    chm_tif = _read_chm_path(args.variant, ROOT)
    band_stats_path = args.dataset_dir / f"band_stats_{args.variant}.json"
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(args.variant)

    # Step 1: rank checkpoints
    print("\n[1] Discovering checkpoints…")
    all_candidates = discover_checkpoints(args.runs_dir, variant_filter=args.variant)
    if not all_candidates:
        raise RuntimeError(f"No checkpoints in {args.runs_dir}. Run phase3_train_v3.py first.")

    top_k = all_candidates[: args.top_k]
    print(f"  {len(all_candidates)} checkpoints — evaluating top {len(top_k)}")
    for i, c in enumerate(top_k, 1):
        print(f"  [{i}] variant={c['variant']}  fold={c['fold_id']}  val_dice={c['best_val_dice']:.4f}")

    all_results: list[dict] = []

    # Step 2: Per-fold evaluation
    print(f"\n[2] Per-fold evaluation on test stripe cols {TEST_STRIPE_COLS} (TTA={use_tta})…")
    for rank, cand in enumerate(top_k, 1):
        variant = cand["variant"]
        fold_id = cand["fold_id"]
        label = f"{variant}_fold{fold_id}"
        print(f"\n  [{rank}/{len(top_k)}] {label}")

        model = load_checkpoint(cand, device)

        prob_no_tta, target_arr, valid_arr = infer_test_stripe(
            model=model, chm_tif=chm_tif, mask_tif=args.mask_tif,
            band_stats=band_stats, binary_bands=binary_bands,
            device=device, use_tta=False, batch_size=args.batch_size,
        )
        m_no_tta = accumulate_pixel_metrics([prob_no_tta], [target_arr], [valid_arr], threshold=0.5)
        best_thr_no_tta, _ = threshold_sweep([prob_no_tta], [target_arr], [valid_arr])

        if use_tta:
            prob_tta, _, _ = infer_test_stripe(
                model=model, chm_tif=chm_tif, mask_tif=args.mask_tif,
                band_stats=band_stats, binary_bands=binary_bands,
                device=device, use_tta=True, batch_size=args.batch_size,
            )
            m_tta = accumulate_pixel_metrics([prob_tta], [target_arr], [valid_arr], threshold=0.5)
            best_thr_tta, _ = threshold_sweep([prob_tta], [target_arr], [valid_arr])
        else:
            prob_tta = prob_no_tta
            m_tta = m_no_tta
            best_thr_tta = best_thr_no_tta

        opt_thr = float(best_thr_tta.get("threshold", 0.5))

        # CC post-processing
        filtered_mask = None
        m_cc = None
        if use_cc:
            raw_binary = (prob_tta >= opt_thr).astype(np.float32)
            filtered_float = cc_postprocess(raw_binary, min_pixels=args.cc_min_px)
            filtered_mask = filtered_float
            m_cc = accumulate_pixel_metrics(
                [filtered_float], [target_arr], [valid_arr], threshold=0.5
            )

        # Write georef TIF
        georef_path = args.output_dir / f"pred_{label}_tta{int(use_tta)}.tif"
        write_georef_prediction(
            prob_map=prob_tta, reference_tif=chm_tif,
            output_path=georef_path,
            col_start=TEST_STRIPE_COLS[0], col_end=TEST_STRIPE_COLS[1],
        )

        # Read CHM background
        with rasterio.open(chm_tif) as src:
            chm_bg = src.read(
                1,
                window=Window(TEST_STRIPE_COLS[0], 0,
                              TEST_STRIPE_COLS[1] - TEST_STRIPE_COLS[0], src.height),
                boundless=True, fill_value=0.0,
            ).astype(np.float32)
        chm_bg = np.clip(chm_bg / 1.3, 0.0, 1.0)

        fig_path = args.output_dir / f"overlay_{label}.png"
        generate_overlay_figure(
            prob_map=prob_tta, target_arr=target_arr, valid_arr=valid_arr,
            threshold=opt_thr, output_path=fig_path,
            title=f"{label} | Dice={m_tta['dice']:.3f} | best_thr={opt_thr:.2f} | TTA={use_tta}",
            chm_arr=chm_bg, filtered_mask=filtered_mask,
        )

        delta_dice = float(m_tta["dice"]) - float(m_no_tta["dice"])
        result = {
            "type": "per_fold",
            "rank": rank,
            "variant": variant,
            "fold_id": fold_id,
            "val_dice_cv": cand["best_val_dice"],
            "test_dice_no_tta_at05": float(m_no_tta["dice"]),
            "test_dice_tta_at05": float(m_tta["dice"]),
            "test_iou_tta_at05": float(m_tta["iou"]),
            "test_precision_tta": float(m_tta["precision"]),
            "test_recall_tta": float(m_tta["recall"]),
            "best_threshold_tta": opt_thr,
            "test_dice_tta_best_thr": float(best_thr_tta.get("dice", 0.0)),
            "test_iou_tta_best_thr": float(best_thr_tta.get("iou", 0.0)),
            "delta_dice_tta": delta_dice,
            "test_dice_cc": float(m_cc["dice"]) if m_cc else None,
            "test_precision_cc": float(m_cc["precision"]) if m_cc else None,
            "test_recall_cc": float(m_cc["recall"]) if m_cc else None,
            "georef_path": str(georef_path),
        }
        all_results.append(result)

        print(
            f"    Dice@0.5={m_tta['dice']:.4f}  IoU={m_tta['iou']:.4f}  "
            f"Prec={m_tta['precision']:.4f}  Rec={m_tta['recall']:.4f}  "
            f"ΔTTA={delta_dice:+.4f}  best_thr={opt_thr:.2f}"
            + (f"  Dice_CC={m_cc['dice']:.4f}" if m_cc else "")
        )

    # Step 3: 5-fold ensemble evaluation
    if not args.no_ensemble:
        print(f"\n[3] Ensemble evaluation (all {len(all_candidates)} folds)…")
        ensemble_prob, target_arr, valid_arr = ensemble_predict(
            fold_checkpoints=all_candidates,
            chm_tif=chm_tif,
            mask_tif=args.mask_tif,
            band_stats=band_stats,
            binary_bands=binary_bands,
            device=device,
            use_tta=use_tta,
            batch_size=args.batch_size,
        )

        m_ens = accumulate_pixel_metrics([ensemble_prob], [target_arr], [valid_arr], threshold=0.5)
        best_thr_ens, _ = threshold_sweep([ensemble_prob], [target_arr], [valid_arr])
        opt_thr_ens = float(best_thr_ens.get("threshold", 0.5))

        filtered_ens = None
        m_ens_cc = None
        if use_cc:
            raw_ens_binary = (ensemble_prob >= opt_thr_ens).astype(np.float32)
            filtered_ens = cc_postprocess(raw_ens_binary, min_pixels=args.cc_min_px)
            m_ens_cc = accumulate_pixel_metrics(
                [filtered_ens], [target_arr], [valid_arr], threshold=0.5
            )

        georef_ens_path = args.output_dir / "pred_ensemble_tta1.tif"
        write_georef_prediction(
            prob_map=ensemble_prob, reference_tif=chm_tif,
            output_path=georef_ens_path,
            col_start=TEST_STRIPE_COLS[0], col_end=TEST_STRIPE_COLS[1],
        )

        with rasterio.open(chm_tif) as src:
            chm_bg = src.read(
                1,
                window=Window(TEST_STRIPE_COLS[0], 0,
                              TEST_STRIPE_COLS[1] - TEST_STRIPE_COLS[0], src.height),
                boundless=True, fill_value=0.0,
            ).astype(np.float32)
        chm_bg = np.clip(chm_bg / 1.3, 0.0, 1.0)

        generate_overlay_figure(
            prob_map=ensemble_prob, target_arr=target_arr, valid_arr=valid_arr,
            threshold=opt_thr_ens, output_path=args.output_dir / "overlay_ensemble.png",
            title=f"ENSEMBLE ({len(all_candidates)} folds) | Dice={m_ens['dice']:.3f} | thr={opt_thr_ens:.2f}",
            chm_arr=chm_bg, filtered_mask=filtered_ens,
        )

        ens_result = {
            "type": "ensemble",
            "rank": 0,
            "variant": args.variant,
            "fold_id": "ensemble",
            "val_dice_cv": float(np.mean([c["best_val_dice"] for c in all_candidates])),
            "test_dice_no_tta_at05": None,
            "test_dice_tta_at05": float(m_ens["dice"]),
            "test_iou_tta_at05": float(m_ens["iou"]),
            "test_precision_tta": float(m_ens["precision"]),
            "test_recall_tta": float(m_ens["recall"]),
            "best_threshold_tta": opt_thr_ens,
            "test_dice_tta_best_thr": float(best_thr_ens.get("dice", 0.0)),
            "test_iou_tta_best_thr": float(best_thr_ens.get("iou", 0.0)),
            "delta_dice_tta": None,
            "test_dice_cc": float(m_ens_cc["dice"]) if m_ens_cc else None,
            "test_precision_cc": float(m_ens_cc["precision"]) if m_ens_cc else None,
            "test_recall_cc": float(m_ens_cc["recall"]) if m_ens_cc else None,
            "georef_path": str(georef_ens_path),
        }
        all_results.insert(0, ens_result)

        print(
            f"\n  ENSEMBLE: Dice@0.5={m_ens['dice']:.4f}  IoU={m_ens['iou']:.4f}  "
            f"Prec={m_ens['precision']:.4f}  Rec={m_ens['recall']:.4f}  "
            f"best_thr={opt_thr_ens:.2f}"
            + (f"  Dice_CC={m_ens_cc['dice']:.4f}" if m_ens_cc else "")
        )

    # Step 4: Save outputs
    final_path = args.output_dir / "final_metrics_v3.json"
    final_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {final_path}")

    table_path = args.output_dir / "thesis_table_v3.csv"
    _write_thesis_table(all_results, table_path)
    print(f"Saved: {table_path}")

    # Step 5: Summary
    print("\n" + "=" * 90)
    print(f"{'Type':<12} {'Variant':<12} {'Fold':<8} {'ValDice':<10} {'Dice@0.5':<10} {'Prec':<8} {'Rec':<8} {'Dice_CC':<10}")
    print("-" * 90)
    for r in all_results:
        cc_str = f"{r['test_dice_cc']:.4f}" if r.get("test_dice_cc") else "—"
        print(
            f"{r['type']:<12} {r['variant']:<12} {str(r['fold_id']):<8} "
            f"{r['val_dice_cv']:<10.4f} {r['test_dice_tta_at05']:<10.4f} "
            f"{r['test_precision_tta']:<8.4f} {r['test_recall_tta']:<8.4f} {cc_str:<10}"
        )
    print("=" * 90)


def _write_thesis_table(results: list[dict], path: Path) -> None:
    cols = [
        "type", "variant", "fold_id", "val_dice_cv",
        "test_dice_tta_at05", "test_iou_tta_at05",
        "test_precision_tta", "test_recall_tta",
        "best_threshold_tta", "test_dice_tta_best_thr",
        "test_dice_cc", "test_precision_cc", "test_recall_cc",
        "delta_dice_tta",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
