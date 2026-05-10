#!/usr/bin/env python3
"""Generate full-tile (5000×5000+) ensemble predictions from V8 models with TTA.

Loads 4-fold checkpoints (preferring SWA-averaged weights if available),
runs 8-fold TTA (4 rotations × 2 flips), and ensemble-averages predictions
across all folds. Writes georeferenced GeoTIFF covering the full mapsheet.

Usage:
    python phase5_predict_v8.py [--device cuda] [--no-tta] [--no-swa] [--threshold 0.35]
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
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.metrics import threshold_sweep
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
from phase3_train_v8 import build_model

RUNS_DIR = ROOT / "seg_pipeline" / "output" / "phase3_runs_v8"
OUTPUT_DIR = ROOT / "seg_pipeline" / "output" / "phase5_predict_v8"
BAND_STATS_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v8"


def discover_fold_checkpoints(prefer_swa: bool = True) -> list[dict]:
    """Load all 4 fold checkpoints from phase3_runs_v8/composite/.

    Prefers swa_model.pt if it exists, falls back to best.pt.
    """
    checkpoints = []
    variant_dir = RUNS_DIR / DEFAULT_VARIANT

    for fold_id in range(4):
        fold_dir = variant_dir / f"fold{fold_id}"
        metrics_path = fold_dir / "metrics.json"
        swa_ckpt_path = fold_dir / "swa_model.pt"
        best_ckpt_path = fold_dir / "best.pt"

        if not metrics_path.exists():
            print(f"⚠️  Fold {fold_id}: metrics.json not found")
            continue

        # Choose checkpoint: prefer SWA if requested and available
        ckpt_path = best_ckpt_path
        ckpt_source = "best"
        if prefer_swa and swa_ckpt_path.exists():
            ckpt_path = swa_ckpt_path
            ckpt_source = "SWA"
        elif not best_ckpt_path.exists():
            print(f"⚠️  Fold {fold_id}: neither swa_model.pt nor best.pt found")
            continue

        m = json.loads(metrics_path.read_text())
        m["checkpoint_path"] = str(ckpt_path)
        m["checkpoint_source"] = ckpt_source
        m["fold_id"] = fold_id
        checkpoints.append(m)
        print(f"✓ Fold {fold_id}: val_f1={m['best_val_f1']:.4f} ({ckpt_source} checkpoint)")

    return checkpoints


def load_checkpoint(info: dict, device: torch.device) -> torch.nn.Module:
    """Load a saved checkpoint and return model in eval mode."""
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
    """Run sliding-window inference on FULL tile with optional TTA.

    Returns prob_map of shape (H, W).
    """
    with rasterio.open(chm_tif) as src:
        H, W = src.height, src.width
        img = src.read(list(range(1, src.count + 1))).astype(np.float32)

    print(f"  Input shape: {img.shape} (CHM variant: {src.count} band(s))")

    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)

    print(f"  Running sliding-window inference (patch={patch_size}, stride={stride}, "
          f"TTA={'8-fold' if use_tta else 'off'})…")
    prob_map = sliding_window_predict(
        model=model, image=img, device=device,
        patch_size=patch_size, stride=stride,
        batch_size=batch_size, use_tta=use_tta,
    )
    return prob_map


def ensemble_predict_full_tile(
    fold_checkpoints: list[dict],
    chm_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    use_tta: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    """Average sigmoid outputs from all fold checkpoints (4-fold ensemble on full tile)."""
    prob_sum = None

    for i, cand in enumerate(fold_checkpoints):
        fold_id = cand["fold_id"]
        source = cand.get("checkpoint_source", "unknown")
        print(f"\n  [{i+1}/{len(fold_checkpoints)}] Fold {fold_id} ({source}):")
        model = load_checkpoint(cand, device)

        prob_map = infer_full_tile(
            model=model, chm_tif=chm_tif, band_stats=band_stats,
            binary_bands=binary_bands, device=device,
            use_tta=use_tta, batch_size=batch_size,
        )

        if prob_sum is None:
            prob_sum = prob_map.astype(np.float64)
        else:
            prob_sum += prob_map.astype(np.float64)

    n = len(fold_checkpoints)
    ensemble_prob = (prob_sum / n).astype(np.float32)
    return ensemble_prob


def write_georef_prediction(
    prob_map: np.ndarray,
    reference_tif: Path,
    output_path: Path,
    col_start: int = 0,
) -> None:
    """Write georeferenced prediction raster with correct geotransform.

    For full tile: col_start=0 (no offset adjustment needed).
    """
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        res = transform.a  # pixel width (0.2 m)
        # Adjust western edge based on col_start offset
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
    print(f"✓ Wrote: {output_path}")


def write_binary_mask(
    prob_map: np.ndarray,
    reference_tif: Path,
    output_path: Path,
    threshold: float = 0.35,
) -> None:
    """Write binary mask at threshold."""
    binary = (prob_map > threshold).astype(np.uint8) * 255

    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary[np.newaxis])
    print(f"✓ Wrote: {output_path} (threshold={threshold})")


def generate_full_tile_visualization(
    prob_map: np.ndarray,
    chm_arr: np.ndarray,
    output_path: Path,
    title: str = "",
) -> None:
    """Generate a simple 2-panel visualization: CHM + prediction heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Panel 1: CHM background
    axes[0].imshow(chm_arr, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CHM (Canopy Height Model)")
    axes[0].axis("off")

    # Panel 2: Prediction heatmap
    im = axes[1].imshow(prob_map, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    axes[1].set_title("V8 Ensemble CWD Confidence (TTA × 4-Fold)")
    axes[1].axis("off")

    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Probability")

    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"✓ Wrote: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate full-tile (5000×N) ensemble predictions from V8 models (4-fold × 8-fold TTA)."
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help=f"Base runs directory (default: {RUNS_DIR})")
    parser.add_argument("--chm-tif", type=Path,
                        help="CHM TIF to use (default: composite variant)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--no-swa", action="store_true", help="Use best.pt instead of swa_model.pt")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Probability threshold for binary mask (default: 0.35)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Override defaults from CLI
    RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
    chm_tif = args.chm_tif or _read_chm_path(DEFAULT_VARIANT, ROOT)

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Load metadata
    print("=== Step 1: Load band statistics and configuration ===")
    variant = DEFAULT_VARIANT
    band_stats_path = BAND_STATS_DIR / f"band_stats_{variant}.json"
    if not band_stats_path.exists():
        print(f"❌ Band stats not found: {band_stats_path}")
        return 1
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(variant)
    print(f"✓ Variant: {variant}")
    print(f"✓ Band stats: {band_stats_path}")
    print(f"✓ CHM input: {chm_tif}")

    # Discover checkpoints
    print("\n=== Step 2: Discover fold checkpoints ===")
    prefer_swa = not args.no_swa
    fold_checkpoints = discover_fold_checkpoints(prefer_swa=prefer_swa)
    if not fold_checkpoints:
        print("❌ No checkpoints found")
        return 1
    print(f"✓ Found {len(fold_checkpoints)} fold checkpoints\n")

    # Run ensemble inference on full tile
    print("=== Step 3: Run 4-fold ensemble on full tile ===")
    use_tta = not args.no_tta
    print(f"TTA: {'enabled (8 augmentations)' if use_tta else 'disabled'}")
    print(f"Ensemble: 4-fold averaging\n")

    ensemble_prob = ensemble_predict_full_tile(
        fold_checkpoints=fold_checkpoints,
        chm_tif=chm_tif,
        band_stats=band_stats,
        binary_bands=binary_bands,
        device=device,
        use_tta=use_tta,
        batch_size=args.batch_size,
    )

    # Statistics
    print(f"\n=== Prediction statistics ===")
    print(f"Shape: {ensemble_prob.shape}")
    print(f"Min: {ensemble_prob.min():.6f}")
    print(f"Max: {ensemble_prob.max():.6f}")
    print(f"Mean: {ensemble_prob.mean():.6f}")
    print(f"Std: {ensemble_prob.std():.6f}")

    for thr in [0.25, 0.35, 0.50, 0.65]:
        count = (ensemble_prob > thr).sum()
        pct = 100 * count / ensemble_prob.size
        print(f"Pixels > {thr}: {count:,} ({pct:.2f}%)")

    # Write output
    print(f"\n=== Step 4: Write outputs ===")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prob_tif = output_dir / "406455_2021_tava_prob_ensemble_tta.tif"
    write_georef_prediction(
        ensemble_prob,
        reference_tif=chm_tif,
        output_path=output_prob_tif,
        col_start=0,
    )

    output_mask_tif = output_dir / "406455_2021_tava_mask_ensemble_tta.tif"
    write_binary_mask(
        ensemble_prob,
        reference_tif=chm_tif,
        output_path=output_mask_tif,
        threshold=args.threshold,
    )

    # Visualization
    with rasterio.open(chm_tif) as src:
        chm_arr = src.read(1).astype(np.float32)
        chm_arr = np.clip(chm_arr, 0, 20) / 20.0  # normalize to [0, 1]

    output_png = output_dir / "406455_2021_tava_viz_ensemble_tta.png"
    generate_full_tile_visualization(
        ensemble_prob, chm_arr,
        output_path=output_png,
        title="V8 Ensemble Full-Tile Predictions (4-fold × 8-fold TTA)",
    )

    print(f"\n=== Complete ===")
    print(f"✓ Full-tile predictions ready for inspection in QGIS:")
    print(f"  Probability: {output_prob_tif}")
    print(f"  Binary mask: {output_mask_tif}")
    print(f"\n✓ Quick preview (PNG):")
    print(f"  {output_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
