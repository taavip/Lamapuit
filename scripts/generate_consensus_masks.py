#!/usr/bin/env python3
"""
Consensus Ensemble CAM → Segmentation Mask Converter

Converts ensemble Integrated Gradients CAMs to high-quality segmentation masks
using multi-model voting instead of aggressive thresholding.

Key insight: CWD are thin line features (logs); thresholding destroys them.
Solution: Use ensemble consensus to highlight regions where multiple models
agree, naturally preserving connected line structures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage

_CHM_MAX = 1.3
_MODEL_SIZE = 128


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest CSV as list of dicts."""
    rows = []
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows


def _safe_int(val: object, default: int | None = None) -> int | None:
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _normalize01(arr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x = np.asarray(arr, dtype=np.float32)
    if valid is not None and np.any(valid):
        vals = x[valid]
    else:
        vals = x.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    if mx <= mn + 1e-8:
        out = np.zeros_like(x, dtype=np.float32)
    else:
        out = (x - mn) / (mx - mn)
    out = np.clip(out, 0.0, 1.0)
    if valid is not None:
        out[~valid] = 0.0
    return out.astype(np.float32)


def _load_cam(path: Path) -> np.ndarray:
    """Load CAM NPY file."""
    try:
        cam = np.load(path, allow_pickle=False).astype(np.float32)
        return cam
    except Exception as e:
        print(f"ERROR loading CAM {path}: {e}")
        raise


def _consensus_mask_voting(
    cams: list[np.ndarray],
    valid: np.ndarray | None = None,
    vote_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mask using ensemble voting.

    Returns:
        mask: binary consensus mask (>= vote_threshold models agree)
        confidence: per-pixel vote count (0-4 for 4 models)
        agreement: normalized agreement map (0-1)
    """
    n_models = len(cams)
    if n_models == 0:
        return np.zeros((128, 128), dtype=np.float32), np.zeros((128, 128)), np.zeros((128, 128))

    # Per-model thresholding at p90
    thresholds = []
    binary_cams = []
    for cam in cams:
        p90 = float(np.percentile(cam.flatten(), 90))
        thresholds.append(p90)
        binary_cams.append((cam >= p90).astype(np.float32))

    # Ensemble voting
    vote_map = np.sum(binary_cams, axis=0)  # 0-4 votes
    confidence = vote_map.copy()
    mask = (vote_map >= vote_threshold).astype(np.float32)

    if valid is not None:
        mask[~valid] = 0.0
        confidence[~valid] = 0.0

    agreement = _normalize01(vote_map, valid)
    return mask, confidence, agreement


def _morphology_refine(
    mask: np.ndarray,
    valid: np.ndarray | None = None,
    open_size: int = 3,
    close_size: int = 3,
) -> np.ndarray:
    """
    Apply morphological operations to preserve line features.

    close() bridges small gaps in logs
    open() removes small noise blobs
    """
    mask_u8 = (mask > 0.5).astype(np.uint8)

    # Close: bridge gaps in logs
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Open: remove noise
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    result = mask_u8.astype(np.float32)
    if valid is not None:
        result[~valid] = 0.0
    return result


def _remove_small_components(
    mask: np.ndarray,
    min_size: int = 5,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Remove connected components smaller than min_size."""
    if min_size <= 0:
        return mask

    mask_u8 = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros_like(mask_u8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_size:
            cleaned[labels == label_idx] = 1

    result = cleaned.astype(np.float32)
    if valid is not None:
        result[~valid] = 0.0
    return result


def _apply_sld(tile: np.ndarray) -> np.ndarray:
    """Render tile with SLD terrain colormap for preview."""
    _SLD_BREAKPOINTS = [
        (0.000, "#580a0c"),
        (0.065, "#f2854e"),
        (0.130, "#f9ab66"),
        (0.195, "#fcbf75"),
        (0.260, "#fec57b"),
        (0.325, "#fed68f"),
        (0.390, "#fee29e"),
        (0.455, "#fdedaa"),
        (0.520, "#f7f4b3"),
        (0.585, "#e4f2b4"),
        (0.650, "#d6eeb1"),
        (0.715, "#c9e9ae"),
        (0.780, "#bce4a9"),
        (0.845, "#addca8"),
        (0.910, "#9dd3a7"),
        (0.975, "#8bc6aa"),
        (1.040, "#78b9ad"),
        (1.105, "#65acb0"),
        (1.170, "#529eb4"),
        (1.235, "#3e91b7"),
        (1.300, "#2b83ba"),
    ]

    import matplotlib.colors as mcolors

    vals = [v / _CHM_MAX for v, _ in _SLD_BREAKPOINTS]
    colors = [c for _, c in _SLD_BREAKPOINTS]
    cmap = mcolors.LinearSegmentedColormap.from_list("sld_terrain", list(zip(vals, colors)))

    nodata = ~np.isfinite(tile)
    is_zero = tile <= 0
    black_mask = nodata | is_zero
    t = tile.copy().astype(np.float32)
    t[black_mask] = 0.0
    t = np.clip(t, 0.0, _CHM_MAX) / _CHM_MAX
    rgb = (cmap(t)[:, :, :3] * 255).astype(np.uint8)
    rgb[black_mask] = 0
    return rgb


def _save_comparison_preview(
    chm: np.ndarray,
    cams: list[np.ndarray],
    mask: np.ndarray,
    confidence: np.ndarray,
    agreement: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Save a 2-row preview: top=per-model CAMs, bottom=final mask + confidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    chm_rgb = _apply_sld(chm)
    n_models = len(cams)

    # 2 rows: row 0 = models, row 1 = final
    fig, axes = plt.subplots(2, max(n_models, 3), figsize=(14, 6), constrained_layout=True)
    fig.patch.set_facecolor("black")

    # Flatten axes for easier indexing
    if n_models + 2 > 3:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    for ax in axes_flat:
        ax.set_facecolor("black")

    # Top row: per-model CAM
    for idx, cam in enumerate(cams):
        ax = axes[0, idx] if n_models > 1 else axes_flat[idx]
        cam_rgb = (cm.get_cmap("hot")(np.clip(cam, 0.0, 1.0))[:, :, :3] * 255).astype(np.uint8)
        ax.imshow(cam_rgb)
        ax.set_title(f"Model {idx} CAM", color="white", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom row: CHM + mask overlay, confidence, agreement
    ax_chm = axes[1, 0] if n_models > 1 else axes_flat[n_models]
    mask_overlay = (0.3 * chm_rgb.astype(np.float32)).astype(np.uint8)
    mask_overlay[mask > 0.5] = np.array([255, 64, 64], dtype=np.uint8)
    ax_chm.imshow(mask_overlay)
    ax_chm.set_title("Mask Overlay", color="white", fontsize=8)
    ax_chm.set_xticks([])
    ax_chm.set_yticks([])

    ax_conf = axes[1, 1] if n_models > 1 else axes_flat[n_models + 1]
    conf_rgb = (cm.get_cmap("viridis")(confidence / n_models)[:, :, :3] * 255).astype(np.uint8)
    ax_conf.imshow(conf_rgb)
    ax_conf.set_title("Vote Count", color="white", fontsize=8)
    ax_conf.set_xticks([])
    ax_conf.set_yticks([])

    ax_agree = axes[1, 2] if n_models > 1 else axes_flat[n_models + 2]
    agree_rgb = (cm.get_cmap("RdYlGn")(agreement)[:, :, :3] * 255).astype(np.uint8)
    ax_agree.imshow(agree_rgb)
    ax_agree.set_title("Agreement", color="white", fontsize=8)
    ax_agree.set_xticks([])
    ax_agree.set_yticks([])

    fig.suptitle(title, fontsize=10, color="white")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ensemble CAMs to masks using consensus voting"
    )
    parser.add_argument("--manifest", default="output/intgrad_masks/manifest.csv")
    parser.add_argument("--input-dir", default="output/intgrad_masks")
    parser.add_argument("--output-dir", default="output/consensus_masks")
    parser.add_argument("--per-model-cam-subdir", default="per_model_cams", help="Subdirectory under input-dir where per-model CAMs are stored")
    parser.add_argument("--vote-threshold", type=float, default=3.0, help="Min votes (0-4)")
    parser.add_argument("--close-kernel", type=int, default=3, help="Morphology close size")
    parser.add_argument("--open-kernel", type=int, default=1, help="Morphology open size")
    parser.add_argument("--min-component-size", type=int, default=8, help="Min pixels for blobs")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--preview-count", type=int, default=0, help="Generate N previews")
    parser.add_argument("--preview-dir", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    rows = _load_manifest(manifest_path)
    max_items = int(args.limit) if args.limit > 0 else None

    print(f"Processing {len(rows)} rows...")
    processed = 0
    previews_to_save = []

    with (out_dir / "consensus_manifest.csv").open("w", newline="", encoding="utf-8") as mh:
        writer = csv.DictWriter(
            mh,
            fieldnames=[
                "tile_id",
                "cam_file",
                "mask_file_consensus",
                "confidence_file",
                "agreement_file",
                "mask_positive_ratio",
                "confidence_mean",
                "agreement_mean",
            ],
        )
        writer.writeheader()

        for row_idx, row in enumerate(rows):
            if max_items and processed >= max_items:
                break

            tile_id = str(row.get("tile_id", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            cam_file = str(row.get("cam_file", "")).strip()

            if not tile_id or not cam_file:
                continue

            cam_path = input_dir / cam_file
            if not cam_path.exists():
                print(f"  SKIP {tile_id}: CAM not found")
                continue

            # Load CAM
            try:
                cam = _load_cam(cam_path)
            except Exception as e:
                print(f"  SKIP {tile_id}: {e}")
                continue

            # Prefer per-model CAMs if available under input_dir/<per-model-cam-subdir>/<tile_id>/
            tile_id_safe = tile_id if (tile_id := str(row.get("tile_id", "")).strip()) else Path(cam_file).stem
            per_model_dir = input_dir / str(args.per_model_cam_subdir) / tile_id_safe
            cams = []
            if per_model_dir.exists() and per_model_dir.is_dir():
                # load all npy files inside (sorted for determinism)
                files = sorted([p for p in per_model_dir.iterdir() if p.suffix.lower() == ".npy"])
                for p in files:
                    try:
                        cams.append(_load_cam(p))
                    except Exception:
                        continue
            # Fallback: replicate ensemble CAM if no per-model cams available
            if not cams:
                cams = [cam, cam, cam, cam]

            # Generate consensus mask
            mask, confidence, agreement = _consensus_mask_voting(
                cams, vote_threshold=float(args.vote_threshold)
            )

            # Refine with morphology
            mask = _morphology_refine(
                mask, open_size=int(args.open_kernel), close_size=int(args.close_kernel)
            )

            # Remove small components
            mask = _remove_small_components(mask, min_size=int(args.min_component_size))

            # Save outputs
            mask_out = f"{Path(cam_file).stem}_consensus_mask.npy"
            conf_out = f"{Path(cam_file).stem}_confidence.npy"
            agree_out = f"{Path(cam_file).stem}_agreement.npy"

            np.save(out_dir / mask_out, mask.astype(np.float32))
            np.save(out_dir / conf_out, confidence.astype(np.float32))
            np.save(out_dir / agree_out, agreement.astype(np.float32))

            writer.writerow(
                {
                    "tile_id": tile_id,
                    "cam_file": cam_file,
                    "mask_file_consensus": mask_out,
                    "confidence_file": conf_out,
                    "agreement_file": agree_out,
                    "mask_positive_ratio": f"{float(mask.mean()):.6f}",
                    "confidence_mean": f"{float(confidence.mean()):.6f}",
                    "agreement_mean": f"{float(agreement.mean()):.6f}",
                }
            )

            if args.preview_count > 0 and len(previews_to_save) < int(args.preview_count):
                previews_to_save.append((cam, cams, mask, confidence, agreement, tile_id, label))

            processed += 1
            if processed % 50 == 0:
                print(f"  Processed: {processed}")

    # Generate previews
    if args.preview_count > 0:
        preview_dir = Path(args.preview_dir) if str(args.preview_dir).strip() else out_dir
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, (cam, cams, mask, confidence, agreement, tile_id, label) in enumerate(
            previews_to_save, start=1
        ):
            out_png = preview_dir / f"consensus_preview_{idx:02d}.png"
            _save_comparison_preview(
                cam, cams, mask, confidence, agreement, f"{tile_id} [{label}]", out_png
            )

    print(f"Done. Processed: {processed}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
