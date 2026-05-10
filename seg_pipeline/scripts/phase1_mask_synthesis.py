#!/usr/bin/env python3
"""Phase I: True Mask Synthesis — GPKG rasterization + ensemble denoising.

Converts sparse polygon labels + tile-classifier ensemble into a clean 3-band
supervision raster used by all downstream training phases:
    Band 1 (target):       1=CWD, 0=background
    Band 2 (valid_mask):   1=use pixel in loss, 0=ignore (noisy or nodata)
    Band 3 (ensemble_prob): raw ensemble soft-vote probability [0, 1]

Conflict resolution logic:
    Positive (target=1, valid=1):  pixel inside GPKG polygon
    Negative (target=0, valid=1):  ensemble_prob < neg_threshold AND outside GPKG
    Noisy/ignore (valid=0):        ensemble_prob > noisy_threshold AND outside GPKG
    Nodata (valid=0):              CHM is nodata / nofinite
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.features import rasterize
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]  # project root
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))


# ---------------------------------------------------------------------------
# Ensemble model loading — reuses label_tiles._get_build_fn pattern
# ---------------------------------------------------------------------------


def _load_label_tiles_helpers():
    """Import _get_build_fn and _instantiate_model_from_build_fn from label_tiles.py."""
    spec = importlib.util.spec_from_file_location(
        "label_tiles", ROOT / "scripts" / "label_tiles.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._get_build_fn, mod._instantiate_model_from_build_fn


def load_ensemble(
    meta_path: Path,
    device: torch.device,
) -> list[nn.Module]:
    """Load the 4-model ensemble from ensemble_meta.json.

    Reuses the exact loading logic from recalculate_model_probs_tta_ensemble.py:60-74.
    """
    get_build_fn, instantiate = _load_label_tiles_helpers()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checkpoint_paths = meta.get("checkpoints", {})

    models: list[nn.Module] = []
    for name, ckpt_rel in checkpoint_paths.items():
        ckpt_path = ROOT / ckpt_rel
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        build_fn_name = ckpt.get("build_fn_name", "_build_deep_cnn_attn")
        build_fn = get_build_fn(build_fn_name)
        if build_fn is None:
            raise RuntimeError(f"Cannot resolve build_fn '{build_fn_name}' for {name}")

        model = instantiate(build_fn).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded {name} ({build_fn_name})")

    return models


# ---------------------------------------------------------------------------
# Per-tile TTA inference — matches recalculate_model_probs_tta_ensemble.py:77-108
# ---------------------------------------------------------------------------


def _normalize_for_ensemble(tile: np.ndarray) -> np.ndarray:
    """clip(0, 20)/20 normalization matching train_ensemble.py."""
    return np.clip(tile.astype(np.float32), 0.0, 20.0) / 20.0


@torch.no_grad()
def predict_tile_ensemble_tta(
    models: list[nn.Module],
    chm_tile: np.ndarray,
    device: torch.device,
) -> float:
    """Soft-vote ensemble probability P(CWD) with 8-fold TTA.

    Exactly matches the TTA logic from train_ensemble.py lines 285-298:
    4 rotations × 2 flips = 8 views per model; average across all views and models.
    """
    if not np.any(np.isfinite(chm_tile)):
        return float("nan")

    norm = _normalize_for_ensemble(chm_tile)
    x = torch.tensor(norm[np.newaxis, np.newaxis], dtype=torch.float32).to(device)

    all_probs: list[float] = []
    for model in models:
        for k in range(4):
            v = torch.rot90(x, k=k, dims=[-2, -1])
            all_probs.append(torch.softmax(model(v), dim=1)[0, 1].item())
            all_probs.append(torch.softmax(model(torch.flip(v, dims=[-1])), dim=1)[0, 1].item())

    return float(np.mean(all_probs))


# ---------------------------------------------------------------------------
# Dense sliding-window inference over the full CHM tile
# ---------------------------------------------------------------------------


def run_ensemble_sliding_window(
    chm_path: Path,
    models: list[nn.Module],
    device: torch.device,
    chunk_size: int = 128,
    stride: int = 64,
    batch_size: int = 32,
    smoke_rows: int | None = None,
    smoke_cols: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile the CHM into chunk_size×chunk_size windows and run ensemble TTA.

    Returns:
        prob_map: (H, W) float32 soft-vote probability map (Gaussian-weighted accumulation)
        coverage: (H, W) bool — True where CHM has valid (finite) data
    """
    with rasterio.open(chm_path) as src:
        full = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        full[full == nodata] = np.nan

    H, W = full.shape
    if smoke_rows:
        H = smoke_rows
        full = full[:H, :]
    if smoke_cols:
        W = smoke_cols
        full = full[:, :W]
        full = full[:H, :]

    coverage = np.isfinite(full)

    # Gaussian blend weight
    w1d = np.hanning(chunk_size).astype(np.float32)
    weight = np.maximum(np.outer(w1d, w1d), 1e-3)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    ys = list(range(0, max(1, H - chunk_size + 1), stride))
    xs = list(range(0, max(1, W - chunk_size + 1), stride))
    if ys and ys[-1] != H - chunk_size:
        ys.append(max(0, H - chunk_size))
    if xs and xs[-1] != W - chunk_size:
        xs.append(max(0, W - chunk_size))

    coords_batch: list[tuple[int, int]] = []
    tiles_batch: list[np.ndarray] = []

    def flush(coords_b, tiles_b):
        for i, tile in enumerate(tiles_b):
            prob = predict_tile_ensemble_tta(models, tile, device)
            y0, x0 = coords_b[i]
            if np.isfinite(prob):
                prob_sum[y0 : y0 + chunk_size, x0 : x0 + chunk_size] += prob * weight
                weight_sum[y0 : y0 + chunk_size, x0 : x0 + chunk_size] += weight

    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + chunk_size, H)
            x1 = min(x0 + chunk_size, W)
            chunk = np.full((chunk_size, chunk_size), np.nan, dtype=np.float32)
            chunk[: y1 - y0, : x1 - x0] = full[y0:y1, x0:x1]
            coords_batch.append((y0, x0))
            tiles_batch.append(chunk)

            if len(tiles_batch) >= batch_size:
                flush(coords_batch, tiles_batch)
                coords_batch.clear()
                tiles_batch.clear()

            done += 1
            if done % max(1, total // 20) == 0:
                pct = 100.0 * done / total
                print(f"  [{done}/{total}] {pct:.0f}%", flush=True)

    if tiles_batch:
        flush(coords_batch, tiles_batch)

    with np.errstate(invalid="ignore"):
        prob_map = prob_sum / np.maximum(weight_sum, 1e-6)
        prob_map[~coverage] = np.nan

    return prob_map, coverage


# ---------------------------------------------------------------------------
# GPKG rasterization
# ---------------------------------------------------------------------------


def rasterize_gpkg(
    gpkg_path: Path,
    reference_tif: Path,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize CWD polygons to match the reference raster grid.

    Returns float32 (H, W) with 1.0=CWD, 0.0=background, nan=nodata.
    """
    import geopandas as gpd

    with rasterio.open(reference_tif) as src:
        transform = src.transform
        height, width = src.height, src.width
        crs = src.crs

    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError(f"Labels GPKG has no CRS: {gpkg_path}")

    # Use 2D EPSG:3301 for rasterization since CHM CRS may be compound (3D)
    target_epsg = 3301
    gdf = gdf.to_crs(epsg=target_epsg)

    from rasterio.crs import CRS as RioCRS
    ref_crs_2d = RioCRS.from_epsg(target_epsg)

    shapes = [
        (geom, 1.0)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        raise ValueError("No valid geometries in GPKG after reprojection")

    gpkg_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=all_touched,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    return gpkg_mask


# ---------------------------------------------------------------------------
# Three-way conflict resolution
# ---------------------------------------------------------------------------


def synthesize_mask(
    coverage: np.ndarray,
    gpkg_mask: np.ndarray,
    ensemble_prob: np.ndarray,
    neg_threshold: float = 0.15,
    noisy_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conflict resolution logic → (target, valid_mask).

    - target=1, valid=1:  inside GPKG polygon (Positive rule)
    - target=0, valid=1:  prob < neg_threshold AND outside GPKG (Confirmed negative)
    - valid=0:            prob > noisy_threshold AND outside GPKG (Noisy/ignore)
    - valid=0:            nodata pixels in CHM (coverage == False)
    """
    target = np.zeros_like(gpkg_mask, dtype=np.float32)
    valid = np.zeros_like(gpkg_mask, dtype=np.float32)

    inside = gpkg_mask > 0.5
    outside = ~inside
    has_data = coverage & np.isfinite(ensemble_prob)

    # Positives
    pos = inside & has_data
    target[pos] = 1.0
    valid[pos] = 1.0

    # Confirmed negatives
    neg = outside & has_data & (ensemble_prob < neg_threshold)
    target[neg] = 0.0
    valid[neg] = 1.0

    # Noisy/ignore: high-confidence CWD detections without label — exclude from loss
    # valid remains 0 (already initialized)

    return target, valid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase I: True Mask Synthesis")
    p.add_argument(
        "--baseline-chm",
        type=Path,
        default=ROOT / "seg_pipeline" / "input" / "baseline_chm.tif",
    )
    p.add_argument(
        "--gpkg",
        type=Path,
        default=ROOT / "seg_pipeline" / "input" / "cdw_labels_MP.gpkg",
    )
    p.add_argument(
        "--ensemble-meta",
        type=Path,
        default=ROOT / "output" / "tile_labels" / "ensemble_meta.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "seg_pipeline" / "output" / "phase1_masks",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--neg-threshold", type=float, default=0.15)
    p.add_argument("--noisy-threshold", type=float, default=0.85)
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a 512×512 crop of the CHM for fast iteration",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    smoke_rows = 512 if args.smoke_test else None
    smoke_cols = 512 if args.smoke_test else None
    suffix = "_smoke" if args.smoke_test else ""

    # Step 1: Rasterize GPKG
    print("\n[1/4] Rasterizing GPKG labels…")
    gpkg_mask = rasterize_gpkg(args.gpkg, args.baseline_chm)
    if args.smoke_test:
        gpkg_mask = gpkg_mask[:512, :512]
    n_pos_px = int((gpkg_mask > 0.5).sum())
    print(f"  GPKG rasterized: {n_pos_px:,} positive pixels ({100*n_pos_px/gpkg_mask.size:.2f}%)")

    # Step 2: Load ensemble
    print("\n[2/4] Loading ensemble models…")
    models = load_ensemble(args.ensemble_meta, device)
    print(f"  {len(models)} models loaded")

    # Step 3: Dense sliding-window inference
    print("\n[3/4] Running ensemble inference (this takes several minutes)…")
    prob_map, coverage = run_ensemble_sliding_window(
        chm_path=args.baseline_chm,
        models=models,
        device=device,
        chunk_size=args.chunk_size,
        stride=args.stride,
        batch_size=args.batch_size,
        smoke_rows=smoke_rows,
        smoke_cols=smoke_cols,
    )
    valid_px = int(coverage.sum())
    prob_finite = prob_map[coverage & np.isfinite(prob_map)]
    print(f"  Valid CHM pixels: {valid_px:,}  |  prob mean={prob_finite.mean():.3f} std={prob_finite.std():.3f}")

    # Step 4: Conflict resolution
    print("\n[4/4] Applying conflict resolution…")
    target, valid_mask = synthesize_mask(
        coverage=coverage,
        gpkg_mask=gpkg_mask,
        ensemble_prob=prob_map,
        neg_threshold=args.neg_threshold,
        noisy_threshold=args.noisy_threshold,
    )
    n_valid = int(valid_mask.sum())
    n_pos = int((target * valid_mask).sum())
    n_neg = int(((1 - target) * valid_mask).sum())
    n_noisy = int(coverage.sum()) - n_valid
    print(f"  Valid pixels:    {n_valid:,}")
    print(f"  Positives (CWD): {n_pos:,} ({100*n_pos/max(1,n_valid):.1f}%)")
    print(f"  Negatives:       {n_neg:,} ({100*n_neg/max(1,n_valid):.1f}%)")
    print(f"  Noisy/ignored:   {n_noisy:,}")

    # Save 3-band raster
    out_path = args.output_dir / f"406455_2021_tava_truemask{suffix}.tif"
    bands = np.stack([target, valid_mask, np.nan_to_num(prob_map, nan=0.0)], axis=0)

    with rasterio.open(args.baseline_chm) as ref:
        profile = ref.profile.copy()

    # Restrict to smoke region if needed
    if args.smoke_test:
        with rasterio.open(args.baseline_chm) as ref:
            transform = ref.transform
            from rasterio.transform import from_bounds
            left = transform.c
            top = transform.f
            res = transform.a
            new_transform = rasterio.transform.from_origin(left, top, res, res)

    profile.update(
        count=3,
        dtype="float32",
        nodata=None,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    profile["height"] = bands.shape[1]
    profile["width"] = bands.shape[2]

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands)

    # Save metadata alongside the raster
    meta_out = {
        "source_chm": str(args.baseline_chm),
        "source_gpkg": str(args.gpkg),
        "ensemble_meta": str(args.ensemble_meta),
        "neg_threshold": args.neg_threshold,
        "noisy_threshold": args.noisy_threshold,
        "n_valid": n_valid,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_noisy_ignored": n_noisy,
        "smoke_test": args.smoke_test,
    }
    meta_path = args.output_dir / f"406455_2021_tava_truemask{suffix}_meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2))

    print(f"\nSaved: {out_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
