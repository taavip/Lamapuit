#!/usr/bin/env python3
"""Offline Grad-CAM++ + Otsu mask generation for CHM tiles.

This script precomputes per-tile CAM confidence maps and binary masks:
- ``*_cam.npy``: continuous confidence map in ``[0, 1]``
- ``*_mask.npy``: Otsu-thresholded binary pseudo-mask in ``{0, 1}``

Generated artifacts can be consumed by
``src/cdw_detect/cwd_partialconv_pipeline.py --cam-mask-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from skimage.filters import frangi
    from skimage.morphology import skeletonize

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency fallback
    frangi = None
    skeletonize = None
    _HAS_SKIMAGE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cdw_detect.cwd_partialconv_pipeline import CHMDataset
from cdw_detect.cwd_partialconv_pipeline import PseudoLabelStore
from cdw_detect.cwd_partialconv_pipeline import TileRecord
from cdw_detect.cwd_partialconv_pipeline import cam_to_binary_mask
from cdw_detect.cwd_partialconv_pipeline import compute_gradcam_hotspots
from cdw_detect.cwd_partialconv_pipeline import create_strict_splits
from cdw_detect.cwd_partialconv_pipeline import load_legacy_model
from cdw_detect.cwd_partialconv_pipeline import seed_everything
from cdw_detect.cwd_partialconv_pipeline import setup_logging
from cdw_detect.cwd_partialconv_pipeline import tile_id_to_artifact_stem


def _parse_float_list(text: str, default: list[float]) -> list[float]:
    """Parse comma-separated float list with fallback default."""
    values: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values if values else list(default)


def _normalize01(arr: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize array to [0,1], optionally over valid pixels only."""
    x = np.asarray(arr, dtype=np.float32)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if np.any(valid):
            vals = x[valid]
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            if mx <= mn + 1e-8:
                out = np.zeros_like(x, dtype=np.float32)
            else:
                out = (x - mn) / (mx - mn)
            out = np.clip(out, 0.0, 1.0)
            out[~valid] = 0.0
            return out.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def _robust_norm01(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Robust percentile normalization to [0,1] using valid pixels."""
    x = np.asarray(arr, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[valid]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if hi <= lo + 1e-8:
        hi = lo + 1.0
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 0.0
    return out.astype(np.float32)


def _estimate_component_shape(comp_mask: np.ndarray) -> tuple[float, float, float]:
    """Estimate component area, length, and width in pixels."""
    comp = np.asarray(comp_mask, dtype=bool)
    ys, xs = np.where(comp)
    area = float(xs.size)
    if area <= 0:
        return 0.0, 0.0, 0.0

    if xs.size >= 5:
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        (_, _), (w_rect, h_rect), _ = cv2.minAreaRect(pts)
        rect_len = float(max(w_rect, h_rect))
        rect_wid = float(min(w_rect, h_rect))
    else:
        rect_len = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        rect_wid = float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))

    skel_len = 0.0
    if _HAS_SKIMAGE:
        try:
            skel_len = float(np.count_nonzero(skeletonize(comp)))
        except Exception:
            skel_len = 0.0

    length_px = max(rect_len, skel_len)
    width_from_skel = area / max(length_px, 1.0)
    geom_width = min(rect_wid, rect_len)
    # Blend skeleton and geometric estimates for robustness on curved/short segments.
    width_px = 0.6 * width_from_skel + 0.4 * geom_width
    return area, float(length_px), float(width_px)


def _enforce_max_positive_ratio(mask: np.ndarray, score_map: np.ndarray, max_ratio: float) -> np.ndarray:
    """Cap positive area fraction by keeping highest-score pixels only."""
    out = np.asarray(mask, dtype=np.float32).copy()
    if max_ratio <= 0.0:
        return np.zeros_like(out, dtype=np.float32)
    if float(out.mean()) <= float(max_ratio):
        return out

    score = np.asarray(score_map, dtype=np.float32)
    keep = int(max(1, round(max_ratio * out.size)))
    flat_idx = np.argsort(score.reshape(-1))[::-1]
    selected = np.zeros(out.size, dtype=np.float32)
    selected[flat_idx[:keep]] = 1.0
    return selected.reshape(out.shape)


def _line_aware_mask_from_cam(
    cam_map: np.ndarray,
    chm_map: np.ndarray,
    valid_map: np.ndarray,
    *,
    line_sigmas: list[float],
    cam_quantile: float,
    ridge_quantile: float,
    min_width_px: float,
    max_width_px: float,
    min_length_px: float,
    max_length_px: float,
    min_occluded_length_px: float,
    occluded_cam_mean_min: float,
    max_positive_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Create line-structured pseudo-mask and confidence map from CAM + CHM.

    The method is tuned for thin elongated structures (e.g., 1-6 px width,
    20-200 px length) with optional support for shorter occluded fragments.
    """
    cam = np.clip(np.asarray(cam_map, dtype=np.float32), 0.0, 1.0)
    valid = np.asarray(valid_map, dtype=np.float32) >= 0.5
    if not np.any(valid):
        zero = np.zeros_like(cam, dtype=np.float32)
        return zero, zero, 0.0, {
            "components_total": 0.0,
            "components_kept": 0.0,
            "positive_ratio": 0.0,
        }

    _, otsu_thr = cam_to_binary_mask(cam)
    valid_vals = cam[valid]
    q_thr = float(np.quantile(valid_vals, np.clip(cam_quantile, 0.0, 1.0)))
    high_thr = max(float(otsu_thr), q_thr)
    seed = (cam >= high_thr) & valid

    chm_01 = _robust_norm01(chm_map, valid)

    if _HAS_SKIMAGE:
        ridge_cam = frangi(cam, sigmas=line_sigmas, black_ridges=False).astype(np.float32)
        ridge_chm = frangi(chm_01, sigmas=line_sigmas, black_ridges=False).astype(np.float32)
        ridge_mix = _normalize01(np.maximum(ridge_cam, ridge_chm), valid)
    else:
        # Fallback: Sobel gradients provide weaker but still localized ridge cues.
        gx = cv2.Sobel(chm_01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(chm_01, cv2.CV_32F, 0, 1, ksize=3)
        ridge_mix = _normalize01(np.sqrt(gx * gx + gy * gy), valid)

    ridge_vals = ridge_mix[valid]
    ridge_thr = float(np.quantile(ridge_vals, np.clip(ridge_quantile, 0.0, 1.0)))
    ridge_mask = (ridge_mix >= ridge_thr) & valid

    seed_u8 = seed.astype(np.uint8)
    seed_u8 = cv2.dilate(seed_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)

    candidate = ((seed_u8 > 0) & ridge_mask) | seed
    candidate_u8 = candidate.astype(np.uint8)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    candidate_u8 = (candidate_u8 > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    kept = np.zeros_like(candidate_u8, dtype=np.float32)
    total_components = max(0, int(num_labels - 1))
    kept_components = 0

    for comp_id in range(1, num_labels):
        area = float(stats[comp_id, cv2.CC_STAT_AREA])
        if area <= 0.0:
            continue
        comp = labels == comp_id
        _, length_px, width_px = _estimate_component_shape(comp)
        cam_mean = float(cam[comp].mean()) if np.any(comp) else 0.0

        keep_main = (
            (min_width_px <= width_px <= max_width_px)
            and (min_length_px <= length_px <= max_length_px)
        )
        keep_occluded = (
            (min_width_px <= width_px <= max_width_px)
            and (min_occluded_length_px <= length_px < min_length_px)
            and (cam_mean >= occluded_cam_mean_min)
        )

        if keep_main or keep_occluded:
            kept[comp] = 1.0
            kept_components += 1

    if kept_components == 0:
        # Fallback to high-confidence CAM to avoid empty targets.
        kept = ((cam >= high_thr) & valid).astype(np.float32)

    score_mix = _normalize01((0.55 * cam) + (0.45 * ridge_mix), valid)
    kept = _enforce_max_positive_ratio(kept, score_map=score_mix, max_ratio=max_positive_ratio)
    kept = (kept > 0.5).astype(np.float32)
    kept[~valid] = 0.0

    cam_refined = (score_mix * valid.astype(np.float32)).astype(np.float32)
    stats_out = {
        "components_total": float(total_components),
        "components_kept": float(kept_components),
        "positive_ratio": float(kept.mean()),
    }
    return kept, cam_refined, float(high_thr), stats_out


def _compute_cam_batch(
    legacy_model: torch.nn.Module,
    chm_batch: torch.Tensor,
    smooth_samples: int,
    noise_std: float,
) -> np.ndarray:
    """Compute CAM batch, optionally using Smooth Grad-CAM++ averaging."""
    cam_sum = compute_gradcam_hotspots(legacy_model, chm_batch).detach()
    n = 1

    n_smooth = max(0, int(smooth_samples))
    if n_smooth > 0 and float(noise_std) > 0.0:
        std = float(noise_std)
        for _ in range(n_smooth):
            noisy = torch.clamp(chm_batch + std * torch.randn_like(chm_batch), min=0.0)
            cam_sum = cam_sum + compute_gradcam_hotspots(legacy_model, noisy).detach()
            n += 1

    cam_avg = torch.clamp(cam_sum / float(n), 0.0, 1.0)
    return cam_avg.cpu().numpy()[:, 0].astype(np.float32)


def _select_tiles(split_name: str, *, train: list[TileRecord], val: list[TileRecord], test: list[TileRecord]) -> list[TileRecord]:
    """Select tiles from split partitions.

    Parameters
    ----------
    split_name : str
        One of ``train``, ``val``, ``test``, ``trainval``, ``all``.
    train : list[TileRecord]
        Training tiles.
    val : list[TileRecord]
        Validation tiles.
    test : list[TileRecord]
        Test tiles.

    Returns
    -------
    list[TileRecord]
        Selected tiles.
    """
    if split_name == "train":
        return train
    if split_name == "val":
        return val
    if split_name == "test":
        return test
    if split_name == "trainval":
        return train + val
    return train + val + test


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate offline Grad-CAM++ + Otsu pseudo-masks")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--registry-dir", type=str, required=True)
    parser.add_argument("--legacy-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "trainval", "all"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tiles-per-raster", type=int, default=2500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--buffer-meters", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional hard cap on number of tiles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--mask-method",
        type=str,
        default="line_aware",
        choices=["otsu", "line_aware"],
        help="Pseudo-mask extraction method from CAM maps",
    )
    parser.add_argument(
        "--line-sigmas",
        type=str,
        default="1,2,3",
        help="Comma-separated Frangi scales (pixels) for line-aware masking",
    )
    parser.add_argument("--cam-quantile", type=float, default=0.90, help="High-CAM quantile used for seed threshold")
    parser.add_argument("--ridge-quantile", type=float, default=0.88, help="Ridge quantile used for candidate line map")
    parser.add_argument("--min-width-px", type=float, default=1.0)
    parser.add_argument("--max-width-px", type=float, default=6.0)
    parser.add_argument("--min-length-px", type=float, default=20.0)
    parser.add_argument("--max-length-px", type=float, default=200.0)
    parser.add_argument("--min-occluded-length-px", type=float, default=8.0)
    parser.add_argument("--occluded-cam-mean-min", type=float, default=0.72)
    parser.add_argument("--max-positive-ratio", type=float, default=0.12)
    parser.add_argument("--cam-smooth-samples", type=int, default=4, help="Additional noisy CAM samples for Smooth Grad-CAM++")
    parser.add_argument("--cam-noise-std", type=float, default=0.20, help="Gaussian noise std in CHM meters for CAM smoothing")
    parser.add_argument("--save-raw-cam", action="store_true", help="Also save *_cam_raw.npy before refinement")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    seed_everything(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    split = create_strict_splits(
        data_dir=args.data_dir,
        test_size=args.test_size,
        buffer_meters=args.buffer_meters,
        registry_dir=args.registry_dir,
        val_size=args.val_size,
        max_tiles_per_raster=args.max_tiles_per_raster,
        seed=args.seed,
    )

    selected_tiles = _select_tiles(args.split, train=split.train, val=split.val, test=split.test)
    if args.limit > 0:
        selected_tiles = selected_tiles[: args.limit]

    if not selected_tiles:
        raise RuntimeError("No tiles selected for CAM mask generation")

    legacy_model = load_legacy_model(args.legacy_checkpoint, device=device)
    if legacy_model is None:
        raise RuntimeError(f"Failed to load legacy classifier from {args.legacy_checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    line_sigmas = _parse_float_list(args.line_sigmas, default=[1.0, 2.0, 3.0])

    dataset = CHMDataset(
        tiles=selected_tiles,
        label_store=PseudoLabelStore(default_confidence=0.5),
        augment=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    tile_by_id = {tile.tile_id: tile for tile in selected_tiles}
    manifest_path = output_dir / "manifest.csv"
    total = 0
    mean_thresholds: list[float] = []
    mean_positive: list[float] = []
    mean_components_total: list[float] = []
    mean_components_kept: list[float] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tile_id",
                "sample_id",
                "mapsheet",
                "year",
                "raster_path",
                "row_off",
                "col_off",
                "chunk_size",
                "mask_file",
                "cam_file",
                "otsu_threshold",
                "mask_positive_ratio",
                "cam_mean",
                "cam_max",
                "mask_method",
                "components_total",
                "components_kept",
            ],
        )
        writer.writeheader()

        for batch in dataloader:
            inputs = batch["input"].to(device)
            chm_inputs = inputs[:, 0:1]
            valid_inputs = inputs[:, 1:2]
            cam_batch = _compute_cam_batch(
                legacy_model,
                chm_inputs,
                smooth_samples=args.cam_smooth_samples,
                noise_std=args.cam_noise_std,
            )
            chm_batch = chm_inputs.detach().cpu().numpy()[:, 0].astype(np.float32)
            valid_batch = valid_inputs.detach().cpu().numpy()[:, 0].astype(np.float32)
            tile_ids = [str(tile_id) for tile_id in batch["tile_id"]]

            for idx, tile_id in enumerate(tile_ids):
                tile = tile_by_id.get(tile_id)
                if tile is None:
                    continue

                cam_raw = np.clip(np.asarray(cam_batch[idx], dtype=np.float32), 0.0, 1.0)
                chm_map = np.asarray(chm_batch[idx], dtype=np.float32)
                valid_map = np.asarray(valid_batch[idx], dtype=np.float32)

                if str(args.mask_method).strip().lower() == "line_aware":
                    mask, cam, threshold, comp_stats = _line_aware_mask_from_cam(
                        cam_map=cam_raw,
                        chm_map=chm_map,
                        valid_map=valid_map,
                        line_sigmas=line_sigmas,
                        cam_quantile=float(args.cam_quantile),
                        ridge_quantile=float(args.ridge_quantile),
                        min_width_px=float(args.min_width_px),
                        max_width_px=float(args.max_width_px),
                        min_length_px=float(args.min_length_px),
                        max_length_px=float(args.max_length_px),
                        min_occluded_length_px=float(args.min_occluded_length_px),
                        occluded_cam_mean_min=float(args.occluded_cam_mean_min),
                        max_positive_ratio=float(args.max_positive_ratio),
                    )
                else:
                    mask, threshold = cam_to_binary_mask(cam_raw)
                    cam = cam_raw
                    comp_stats = {
                        "components_total": 0.0,
                        "components_kept": 0.0,
                        "positive_ratio": float(mask.mean()),
                    }

                stem = tile_id_to_artifact_stem(tile_id)
                if args.save_raw_cam:
                    np.save(output_dir / f"{stem}_cam_raw.npy", cam_raw)
                np.save(output_dir / f"{stem}_cam.npy", cam)
                np.save(output_dir / f"{stem}_mask.npy", mask.astype(np.float32))

                writer.writerow(
                    {
                        "tile_id": tile.tile_id,
                        "sample_id": tile.sample_id,
                        "mapsheet": tile.mapsheet,
                        "year": tile.year,
                        "raster_path": tile.raster_path,
                        "row_off": tile.row_off,
                        "col_off": tile.col_off,
                        "chunk_size": tile.chunk_size,
                        "mask_file": f"{stem}_mask.npy",
                        "cam_file": f"{stem}_cam.npy",
                        "otsu_threshold": f"{threshold:.6f}",
                        "mask_positive_ratio": f"{float(mask.mean()):.6f}",
                        "cam_mean": f"{float(cam.mean()):.6f}",
                        "cam_max": f"{float(cam.max()):.6f}",
                        "mask_method": str(args.mask_method),
                        "components_total": f"{float(comp_stats.get('components_total', 0.0)):.2f}",
                        "components_kept": f"{float(comp_stats.get('components_kept', 0.0)):.2f}",
                    }
                )

                total += 1
                mean_thresholds.append(float(threshold))
                mean_positive.append(float(mask.mean()))
                mean_components_total.append(float(comp_stats.get("components_total", 0.0)))
                mean_components_kept.append(float(comp_stats.get("components_kept", 0.0)))

    summary = {
        "total_tiles": total,
        "split": args.split,
        "device": str(device),
        "mask_method": str(args.mask_method),
        "line_sigmas": [float(x) for x in line_sigmas],
        "mean_otsu_threshold": float(np.mean(mean_thresholds)) if mean_thresholds else 0.0,
        "mean_mask_positive_ratio": float(np.mean(mean_positive)) if mean_positive else 0.0,
        "mean_components_total": float(np.mean(mean_components_total)) if mean_components_total else 0.0,
        "mean_components_kept": float(np.mean(mean_components_kept)) if mean_components_kept else 0.0,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Generated CAM masks: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
