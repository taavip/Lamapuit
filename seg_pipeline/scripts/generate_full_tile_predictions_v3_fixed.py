#!/usr/bin/env python3
"""Generate full-tile predictions using EXACT SAME CODE as phase4_evaluate_v3.py

This script reuses the exact inference and ensemble functions from phase4_evaluate_v3.py
but runs them on the full tile (all columns) instead of just the test stripe.

The only difference from test stripe version:
- Read full image (all 5000 columns) instead of window (cols 0-999)
- All else is identical: same models, same band stats, same TTA, same ensemble method
"""

from __future__ import annotations

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

# Import exact same functions as phase4_evaluate_v3.py
from common.metrics import accumulate_pixel_metrics
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

OUTPUT_DIR = ROOT / "seg_pipeline" / "output" / "phase4_report_v3"
BAND_STATS_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v3"


def discover_checkpoints(runs_dir: Path) -> list[dict]:
    """Discover all fold checkpoints in order (fold 0, 1, 2, 3)."""
    candidates = []
    variant_dir = runs_dir / DEFAULT_VARIANT

    for fold_id in range(4):
        fold_dir = variant_dir / f"fold{fold_id}"
        metrics_path = fold_dir / "metrics.json"
        ckpt_path = fold_dir / "best.pt"

        if not metrics_path.exists() or not ckpt_path.exists():
            continue

        m = json.loads(metrics_path.read_text())
        m["checkpoint_path"] = str(ckpt_path)
        m["fold_id"] = fold_id
        candidates.append(m)

    return candidates


def load_checkpoint(info: dict, device: torch.device) -> torch.nn.Module:
    """Load checkpoint (exact same as phase4_evaluate_v3.py)."""
    ckpt = torch.load(info["checkpoint_path"], map_location=device, weights_only=False)
    variant = ckpt.get("variant", DEFAULT_VARIANT)
    in_channels = _get_in_channels(variant)
    model = build_model("unetpp_effb2", in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_full_tile_same_as_test_stripe(
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
    """Run sliding-window inference on FULL TILE.

    Uses EXACT SAME inference code as infer_test_stripe() in phase4_evaluate_v3.py,
    but reads full image instead of window.
    """
    # Read FULL image (all columns) instead of window
    with rasterio.open(chm_tif) as src:
        H, W = src.height, src.width
        img = src.read(
            list(range(1, src.count + 1)),
            boundless=True, fill_value=np.nan,
        ).astype(np.float32)

    # Normalize EXACTLY as phase4_evaluate_v3.py does
    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)

    # Run sliding window EXACTLY as phase4_evaluate_v3.py does
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
    """5-fold ensemble using EXACT SAME method as phase4_evaluate_v3.py."""
    prob_sum = None

    for i, cand in enumerate(fold_checkpoints):
        fold_id = cand["fold_id"]
        print(f"  Loading fold {fold_id} checkpoint for ensemble…")
        model = load_checkpoint(cand, device)

        prob_i = infer_full_tile_same_as_test_stripe(
            model=model, chm_tif=chm_tif, band_stats=band_stats,
            binary_bands=binary_bands, device=device,
            use_tta=use_tta, batch_size=batch_size,
        )

        if prob_sum is None:
            prob_sum = prob_i.astype(np.float64)
        else:
            prob_sum += prob_i.astype(np.float64)

    n = len(fold_checkpoints)
    ensemble_prob = (prob_sum / n).astype(np.float32)
    return ensemble_prob


def write_georef_prediction(
    prob_map: np.ndarray,
    reference_tif: Path,
    output_path: Path,
    col_start: int,
    col_end: int,
) -> None:
    """Write prediction using EXACT SAME method as phase4_evaluate_v3.py."""
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
    print(f"✓ Wrote: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load metadata
    print("=== Load band statistics ===")
    variant = DEFAULT_VARIANT
    band_stats_path = BAND_STATS_DIR / f"band_stats_{variant}.json"
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(variant)
    chm_tif = _read_chm_path(variant, ROOT)
    print(f"✓ Variant: {variant}")
    print(f"✓ CHM: {chm_tif}")

    # Discover checkpoints
    print("\n=== Discover checkpoints ===")
    runs_dir = ROOT / "seg_pipeline" / "output" / "phase3_runs_v3"
    fold_checkpoints = discover_checkpoints(runs_dir)
    print(f"✓ Found {len(fold_checkpoints)} folds")
    for c in fold_checkpoints:
        print(f"  Fold {c['fold_id']}: val_dice={c['best_val_dice']:.4f}")

    # Run ensemble on full tile
    print("\n=== Ensemble inference on full tile (EXACT same method) ===")
    ensemble_prob = ensemble_predict_full_tile(
        fold_checkpoints=fold_checkpoints,
        chm_tif=chm_tif,
        band_stats=band_stats,
        binary_bands=binary_bands,
        device=device,
        use_tta=True,
        batch_size=8,
    )

    # Statistics
    print(f"\n=== Prediction statistics ===")
    print(f"Shape: {ensemble_prob.shape}")
    print(f"Min:   {ensemble_prob.min():.6f}")
    print(f"Max:   {ensemble_prob.max():.6f}")
    print(f"Mean:  {ensemble_prob.mean():.6f}")

    # Write output
    print(f"\n=== Write output ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(chm_tif) as src:
        tile_width = src.width

    output_tif = OUTPUT_DIR / "pred_ensemble_full_tile_tta1.tif"
    write_georef_prediction(
        ensemble_prob,
        reference_tif=chm_tif,
        output_path=output_tif,
        col_start=0,
        col_end=tile_width,
    )

    print(f"\n✅ COMPLETE: {output_tif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
