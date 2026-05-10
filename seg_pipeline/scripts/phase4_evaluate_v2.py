#!/usr/bin/env python3
"""Phase IV V2: Evaluation & Reporting — multi-variant top-k deathmatch + TTA inference.

Workflow:
    1. Rank all (variant, fold) checkpoints by validation Dice from metrics.json
    2. Select top-k candidates
    3. Evaluate each on the held-out test stripe (Stripe 0, cols 0-1666)
       with and without 8-fold TTA
    4. Run threshold sweep (0.1-0.9) to find best-F1 operating point
    5. Write georeferenced probability GeoTIFF for QGIS review
    6. Generate 4-panel overlay figure per model
    7. Produce thesis_table_v2.csv (LaTeX-ready) + final_metrics_v2.json

Usage:
    python phase4_evaluate_v2.py                          # all variants, top-5
    python phase4_evaluate_v2.py --variant composite      # single variant only
    python phase4_evaluate_v2.py --top-k 1 --no-tta       # quick sanity check
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
from common.raster_io import normalize_bands, write_raster_like
from common.sliding_window import sliding_window_predict
from phase2_dataset_v2 import (
    PATCH_SIZE,
    STRIDE,
    _read_chm_path,
    _get_in_channels,
    _get_binary_bands,
)
from phase3_train_v2 import build_model

ALL_VARIANTS = ["baseline", "raw", "gauss", "masked", "composite"]

# Match V1 test stripe for apples-to-apples comparison: cols 0-999
TEST_STRIPE_COLS = (0, 1000)

EPS = 1e-8


# ---------------------------------------------------------------------------
# Checkpoint discovery and loading
# ---------------------------------------------------------------------------


def discover_checkpoints(runs_dir: Path, variant_filter: str | None = None) -> list[dict]:
    """Collect all fold metrics from phase3_runs_v2/<variant>/fold<k>/metrics.json."""
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
    """Load best.pt and reconstruct the model with correct in_channels."""
    ckpt = torch.load(info["checkpoint_path"], map_location=device, weights_only=False)
    variant = ckpt.get("variant", info.get("variant", "composite"))
    in_channels = _get_in_channels(variant)
    model = build_model("unetpp_effb2", in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test stripe evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_test_stripe(
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
    """Run sliding-window inference on the test stripe (cols 0-1666).

    Returns:
        prob_map:    (H, W) float32 — predicted probabilities
        target_arr:  (H, W) float32 — ground truth target mask
        valid_arr:   (H, W) float32 — valid pixel mask
    """
    col_start, col_end = TEST_STRIPE_COLS
    W_stripe = col_end - col_start

    with rasterio.open(chm_tif) as src:
        H = src.height
        img = src.read(
            list(range(1, src.count + 1)),
            window=Window(col_start, 0, W_stripe, H),
            boundless=True,
            fill_value=np.nan,
        ).astype(np.float32)

    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)

    with rasterio.open(mask_tif) as msrc:
        raw = msrc.read(
            [1, 2],
            window=Window(col_start, 0, W_stripe, H),
            boundless=True,
            fill_value=0.0,
        ).astype(np.float32)
    target_arr = raw[0]
    valid_arr = raw[1]

    prob_map = sliding_window_predict(
        model=model,
        image=img,
        device=device,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        use_tta=use_tta,
    )

    return prob_map, target_arr, valid_arr


# ---------------------------------------------------------------------------
# Georeferenced output
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
        orig_left = transform.c
        orig_top = transform.f
        new_left = orig_left + col_start * res
        new_transform = rasterio.transform.from_origin(
            west=new_left, north=orig_top, xsize=res, ysize=res
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


# ---------------------------------------------------------------------------
# Overlay figure
# ---------------------------------------------------------------------------


def generate_overlay_figure(
    prob_map: np.ndarray,
    target_arr: np.ndarray,
    valid_arr: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str = "",
    chm_arr: np.ndarray | None = None,
) -> None:
    v = valid_arr > 0.5
    pred = (prob_map >= threshold).astype(np.float32)
    gt = (target_arr > 0.5).astype(np.float32)
    bg = chm_arr if chm_arr is not None else np.zeros_like(prob_map)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    axes[0].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CHM background")
    axes[0].axis("off")

    axes[1].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
    gt_disp = np.where(v, gt, np.nan)
    axes[1].imshow(np.ma.masked_invalid(gt_disp), cmap="Greens", alpha=0.65, vmin=0, vmax=1)
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
    pred_disp = np.where(v, pred, np.nan)
    axes[2].imshow(np.ma.masked_invalid(pred_disp), cmap="Reds", alpha=0.65, vmin=0, vmax=1)
    axes[2].set_title(f"Pred (thr={threshold:.2f})")
    axes[2].axis("off")

    fp = (pred == 1) & (gt == 0) & v
    fn = (pred == 0) & (gt == 1) & v
    err = np.zeros((*prob_map.shape, 3), dtype=np.float32)
    err[..., 0] = fp.astype(np.float32)
    err[..., 2] = fn.astype(np.float32)
    axes[3].imshow(bg, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].imshow(err, alpha=0.70)
    axes[3].set_title("FP (red) / FN (blue)")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase IV V2: Evaluation & Reporting")
    p.add_argument(
        "--runs-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase3_runs_v2",
    )
    p.add_argument(
        "--mask-tif", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif",
    )
    p.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v2",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase4_report_v2",
    )
    p.add_argument("--variant", type=str, default=None, choices=ALL_VARIANTS,
                   help="Evaluate a single variant (default: all)")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")
    use_tta = not args.no_tta

    # Step 1: Rank checkpoints
    print("\n[1] Discovering and ranking checkpoints…")
    all_candidates = discover_checkpoints(args.runs_dir, variant_filter=args.variant)
    if not all_candidates:
        raise RuntimeError(f"No checkpoints found in {args.runs_dir}.")

    top_k = all_candidates[: args.top_k]
    print(f"  {len(all_candidates)} checkpoints found — evaluating top {len(top_k)}")
    for i, c in enumerate(top_k, 1):
        print(f"  [{i}] variant={c['variant']}  fold={c['fold_id']}  val_dice={c['best_val_dice']:.4f}")

    # Step 2: Evaluate each on the test stripe
    print(f"\n[2] Evaluating top-{len(top_k)} on test stripe cols {TEST_STRIPE_COLS} (TTA={use_tta})…")
    all_results: list[dict] = []

    for rank, cand in enumerate(top_k, 1):
        variant = cand["variant"]
        fold_id = cand["fold_id"]
        label = f"{variant}_fold{fold_id}"
        print(f"\n  [{rank}/{len(top_k)}] {label}")

        # Load variant-specific CHM and band stats
        chm_tif = _read_chm_path(variant, ROOT)
        band_stats_path = args.dataset_dir / f"band_stats_{variant}.json"
        band_stats = json.loads(band_stats_path.read_text())
        binary_bands = _get_binary_bands(variant)

        model = load_checkpoint(cand, device)

        print("    Running inference (no TTA)…")
        prob_no_tta, target_arr, valid_arr = evaluate_test_stripe(
            model=model, chm_tif=chm_tif, mask_tif=args.mask_tif,
            band_stats=band_stats, binary_bands=binary_bands,
            device=device, use_tta=False, batch_size=args.batch_size,
        )
        m_no_tta = accumulate_pixel_metrics([prob_no_tta], [target_arr], [valid_arr], threshold=0.5)
        best_thr_no_tta, _ = threshold_sweep([prob_no_tta], [target_arr], [valid_arr])

        if use_tta:
            print("    Running inference (8-fold TTA)…")
            prob_tta, _, _ = evaluate_test_stripe(
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

        # Georeferenced GeoTIFF
        georef_path = args.output_dir / f"pred_{label}_tta{int(use_tta)}.tif"
        write_georef_prediction(
            prob_map=prob_tta, reference_tif=chm_tif,
            output_path=georef_path,
            col_start=TEST_STRIPE_COLS[0], col_end=TEST_STRIPE_COLS[1],
        )

        # CHM background for overlay (band 1, clipped to [0,1.3])
        with rasterio.open(chm_tif) as src:
            chm_bg = src.read(
                1,
                window=Window(TEST_STRIPE_COLS[0], 0,
                              TEST_STRIPE_COLS[1] - TEST_STRIPE_COLS[0], src.height),
                boundless=True, fill_value=0.0,
            ).astype(np.float32)
        chm_bg = np.clip(chm_bg / 1.3, 0.0, 1.0)

        # Overlay figure
        fig_path = args.output_dir / f"overlay_{label}.png"
        opt_thr = float(best_thr_tta.get("threshold", 0.5))
        generate_overlay_figure(
            prob_map=prob_tta, target_arr=target_arr, valid_arr=valid_arr,
            threshold=opt_thr, output_path=fig_path,
            title=(
                f"{label}  |  TTA={use_tta}  |  Dice@0.5={m_tta['dice']:.3f}  "
                f"IoU@0.5={m_tta['iou']:.3f}  best_thr={opt_thr:.2f}"
            ),
            chm_arr=chm_bg,
        )

        delta_dice = float(m_tta["dice"]) - float(m_no_tta["dice"])
        result = {
            "rank": rank,
            "variant": variant,
            "fold_id": fold_id,
            "val_dice_cv": cand["best_val_dice"],
            "test_dice_no_tta_at05": float(m_no_tta["dice"]),
            "test_iou_no_tta_at05": float(m_no_tta["iou"]),
            "test_dice_tta_at05": float(m_tta["dice"]),
            "test_iou_tta_at05": float(m_tta["iou"]),
            "test_precision_tta": float(m_tta["precision"]),
            "test_recall_tta": float(m_tta["recall"]),
            "best_threshold_tta": opt_thr,
            "test_dice_tta_best_thr": float(best_thr_tta.get("dice", 0.0)),
            "test_iou_tta_best_thr": float(best_thr_tta.get("iou", 0.0)),
            "delta_dice_tta": delta_dice,
            "georef_path": str(georef_path),
            "overlay_path": str(fig_path),
        }
        all_results.append(result)

        print(
            f"    Dice@0.5: {m_tta['dice']:.4f}  IoU@0.5: {m_tta['iou']:.4f}  "
            f"ΔTTA: {delta_dice:+.4f}  best_thr: {opt_thr:.2f}"
        )

    # Step 3: Save final metrics
    final_path = args.output_dir / "final_metrics_v2.json"
    final_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {final_path}")

    # Step 4: LaTeX-ready CSV
    table_path = args.output_dir / "thesis_table_v2.csv"
    _write_thesis_table(all_results, table_path)
    print(f"Saved: {table_path}")

    # Step 5: Summary
    print("\n" + "=" * 80)
    print(f"{'Rank':<6} {'Variant':<12} {'Fold':<6} {'ValDice':<10} {'Dice@0.5':<12} {'IoU@0.5':<12} {'ΔTTA':<10}")
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['rank']:<6} {r['variant']:<12} {r['fold_id']:<6} "
            f"{r['val_dice_cv']:<10.4f} {r['test_dice_tta_at05']:<12.4f} "
            f"{r['test_iou_tta_at05']:<12.4f} {r['delta_dice_tta']:+<10.4f}"
        )
    print("=" * 80)
    best = all_results[0]
    print(f"\nBest test model: {best['variant']}_fold{best['fold_id']}  "
          f"Dice@0.5={best['test_dice_tta_at05']:.4f}  "
          f"Dice@best_thr={best['test_dice_tta_best_thr']:.4f} (thr={best['best_threshold_tta']:.2f})")


def _write_thesis_table(results: list[dict], path: Path) -> None:
    cols = [
        "rank", "variant", "fold_id", "val_dice_cv",
        "test_dice_tta_at05", "test_iou_tta_at05",
        "test_precision_tta", "test_recall_tta",
        "best_threshold_tta", "test_dice_tta_best_thr", "test_iou_tta_best_thr",
        "delta_dice_tta",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
