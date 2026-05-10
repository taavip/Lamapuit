#!/usr/bin/env python3
"""Generate full-tile predictions by tiling 1000-pixel-wide stripes.

Since sliding_window_predict's padding depends on image width, we process the tile
in 1000-pixel-wide stripes (same as test stripe) to ensure EXACT consistency.

Process: cols 0-999 (stripe A), 1000-1999 (stripe B), 2000-2999 (stripe C), etc.
Then stitch all stripes together.
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

import numpy as np
import rasterio
import torch

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
from phase3_train_v3 import build_model

OUTPUT_DIR = ROOT / "seg_pipeline" / "output" / "phase4_report_v3"
BAND_STATS_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v3"
STRIPE_WIDTH = 1000  # Process in 1000-pixel-wide stripes (same as test stripe)


def load_checkpoint(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = ckpt.get("variant", DEFAULT_VARIANT)
    in_channels = _get_in_channels(variant)
    model = build_model("unetpp_effb2", in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def infer_stripe(
    model: torch.nn.Module,
    chm_tif: Path,
    col_start: int,
    col_end: int,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    use_tta: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    """Run inference on a single stripe (col_start to col_end)."""
    w_stripe = col_end - col_start

    with rasterio.open(chm_tif) as src:
        H = src.height
        img = src.read(
            list(range(1, src.count + 1)),
            window=rasterio.windows.Window(col_start, 0, w_stripe, H),
            boundless=True, fill_value=np.nan,
        ).astype(np.float32)

    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)

    prob_map = sliding_window_predict(
        model=model, image=img, device=device,
        patch_size=PATCH_SIZE, stride=STRIDE,
        batch_size=batch_size, use_tta=use_tta,
    )
    return prob_map


def ensemble_predict_stripe(
    fold_checkpoints: list[Path],
    chm_tif: Path,
    col_start: int,
    col_end: int,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    use_tta: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    """5-fold ensemble on a single stripe."""
    prob_sum = None

    for i, ckpt_path in enumerate(fold_checkpoints):
        fold_id = ckpt_path.parent.name.replace("fold", "")
        print(f"    Stripe {col_start}–{col_end}: Fold {fold_id}")
        model = load_checkpoint(str(ckpt_path), device)

        prob_i = infer_stripe(
            model=model, chm_tif=chm_tif,
            col_start=col_start, col_end=col_end,
            band_stats=band_stats, binary_bands=binary_bands,
            device=device, use_tta=use_tta, batch_size=batch_size,
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
    """Write prediction with geotransform adjusted for column offset."""
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load metadata
    print("=== Setup ===")
    variant = DEFAULT_VARIANT
    band_stats_path = BAND_STATS_DIR / f"band_stats_{variant}.json"
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(variant)
    chm_tif = _read_chm_path(variant, ROOT)
    print(f"✓ Variant: {variant}")
    print(f"✓ CHM: {chm_tif.name}")
    print(f"✓ Stripe width: {STRIPE_WIDTH} px")

    # Discover folds
    print("\n=== Discover folds ===")
    runs_dir = ROOT / "seg_pipeline" / "output" / "phase3_runs_v3"
    fold_checkpoints = []
    for fold_id in range(4):
        ckpt_path = runs_dir / variant / f"fold{fold_id}" / "best.pt"
        if ckpt_path.exists():
            fold_checkpoints.append(ckpt_path)
            print(f"✓ Fold {fold_id}: {ckpt_path}")

    # Get tile dimensions
    with rasterio.open(chm_tif) as src:
        tile_height, tile_width = src.height, src.width
    print(f"✓ Tile size: {tile_height} × {tile_width} px")

    # Process stripes
    print(f"\n=== Process stripes (width={STRIPE_WIDTH} px) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine stripe boundaries
    n_stripes = (tile_width + STRIPE_WIDTH - 1) // STRIPE_WIDTH
    stripes = []
    for i in range(n_stripes):
        col_start = i * STRIPE_WIDTH
        col_end = min((i + 1) * STRIPE_WIDTH, tile_width)
        stripes.append((col_start, col_end))

    print(f"Stripes: {n_stripes}")
    for col_start, col_end in stripes:
        print(f"  [{col_start:4d}–{col_end:4d}]")

    # Ensemble predictions for each stripe
    print(f"\n=== Ensemble inference (5-fold average) ===")
    stripe_predictions = []

    for col_start, col_end in stripes:
        print(f"\n  Cols {col_start}–{col_end}:")
        stripe_prob = ensemble_predict_stripe(
            fold_checkpoints=fold_checkpoints,
            chm_tif=chm_tif,
            col_start=col_start, col_end=col_end,
            band_stats=band_stats, binary_bands=binary_bands,
            device=device, use_tta=True, batch_size=8,
        )
        stripe_predictions.append((col_start, col_end, stripe_prob))
        print(f"    Shape: {stripe_prob.shape}, Min: {stripe_prob.min():.4f}, Max: {stripe_prob.max():.4f}")

    # Stitch stripes together
    print(f"\n=== Stitch stripes ===")
    full_tile = np.hstack([pred for _, _, pred in stripe_predictions])
    print(f"Full tile shape: {full_tile.shape}")
    print(f"Min: {full_tile.min():.6f}, Max: {full_tile.max():.6f}, Mean: {full_tile.mean():.6f}")

    # Write full tile
    print(f"\n=== Write output ===")
    output_tif = OUTPUT_DIR / "pred_ensemble_full_tile_tta1.tif"
    write_georef_prediction(
        full_tile, reference_tif=chm_tif, output_path=output_tif,
        col_start=0, col_end=tile_width,
    )

    print(f"\n✅ COMPLETE: {output_tif}")
    print(f"   Shape: {full_tile.shape}")
    print(f"   Size: {output_tif.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
