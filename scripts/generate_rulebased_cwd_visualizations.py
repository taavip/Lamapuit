#!/usr/bin/env python3
"""Generate CWD visualization panels with rule-based true masks.

Rule for true mask per tile:
- If tile label is no-CWD (label < 0.5), use an all-background mask.
- If tile label is CWD (label >= 0.5), derive mask from per-tile hotspot map.

Each output PNG contains 3 panels:
1) Original CHM tile
2) Rule-based true mask
3) Model predicted mask
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cdw_detect.cwd_partialconv_pipeline import CHMDataset
from cdw_detect.cwd_partialconv_pipeline import PseudoLabelStore
from cdw_detect.cwd_partialconv_pipeline import cam_to_binary_mask
from cdw_detect.cwd_partialconv_pipeline import create_strict_splits
from cdw_detect.cwd_partialconv_pipeline import load_partialconv_checkpoint


def _slug(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in text)


def _pick_indices(
    pos_idx: list[int],
    neg_idx: list[int],
    total_count: int,
    n_positive: int,
    seed: int,
) -> list[int]:
    n_positive = max(0, min(n_positive, total_count))
    n_negative = total_count - n_positive

    if len(pos_idx) < n_positive:
        raise RuntimeError(
            f"Requested {n_positive} positive tiles but only {len(pos_idx)} available"
        )
    if len(neg_idx) < n_negative:
        raise RuntimeError(
            f"Requested {n_negative} negative tiles but only {len(neg_idx)} available"
        )

    rng = random.Random(seed)
    selected = rng.sample(pos_idx, n_positive) + rng.sample(neg_idx, n_negative)
    rng.shuffle(selected)
    return selected


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return arr[y0 : y0 + size, x0 : x0 + size]


def _save_panel(
    chm: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    finite = np.isfinite(chm)
    if finite.any():
        lo = float(np.percentile(chm[finite], 2.0))
        hi = float(np.percentile(chm[finite], 98.0))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(chm, cmap="viridis", vmin=lo, vmax=hi)
    axes[0].set_title("Original CHM")
    axes[1].imshow(true_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("True CWD Mask")
    axes[2].imshow(pred_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Predicted CWD Mask")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_cam_manifest(manifest_path: Path, cam_root: Path) -> dict[str, dict[str, object]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"CAM manifest not found: {manifest_path}")

    out: dict[str, dict[str, object]] = {}
    with manifest_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            tile_id = str(row.get("tile_id", "")).strip()
            cam_file = str(row.get("cam_file", "")).strip()
            if not tile_id or not cam_file:
                continue
            out[tile_id] = {
                "cam_path": cam_root / cam_file,
                "cam_file": cam_file,
                "otsu_threshold": float(row.get("otsu_threshold") or 0.0),
            }
    return out


def _to_chunk_size(mask: np.ndarray, chunk_size: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.shape == (chunk_size, chunk_size):
        return arr

    h, w = arr.shape
    if h >= chunk_size and w >= chunk_size:
        y0 = (h - chunk_size) // 2
        x0 = (w - chunk_size) // 2
        return arr[y0 : y0 + chunk_size, x0 : x0 + chunk_size]

    out = np.zeros((chunk_size, chunk_size), dtype=np.float32)
    y0 = max(0, (chunk_size - h) // 2)
    x0 = max(0, (chunk_size - w) // 2)
    out[y0 : y0 + h, x0 : x0 + w] = arr
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rule-based CWD visualization PNGs")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/chm_dataset_harmonized_0p8m_raw_gauss",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default="registry/chm_dataset_harmonized_0p8m_raw_gauss",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/cwd_visualizations_rulebased",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/cwd_partialconv_gpu_multiepoch_20260417_sota_es/best_partialconv_unet.pt",
    )
    parser.add_argument(
        "--cam-manifest",
        type=str,
        default="output/cam_masks_gradcam_lineaware_train_full_sotaes/manifest.csv",
        help="Manifest containing per-tile hotspot-derived masks",
    )
    parser.add_argument(
        "--cam-root",
        type=str,
        default="",
        help="Root directory for mask files from manifest (defaults to manifest parent)",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--positive-count", type=int, default=3)
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--buffer-meters", type=float, default=50.0)
    parser.add_argument("--max-tiles-per-raster", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = create_strict_splits(
        data_dir=args.data_dir,
        registry_dir=args.registry_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        buffer_meters=args.buffer_meters,
        max_tiles_per_raster=args.max_tiles_per_raster,
        seed=args.seed,
    )

    tiles = split.train + split.val
    label_store = PseudoLabelStore.from_tiles(split.train + split.val + split.test)
    dataset = CHMDataset(tiles=tiles, label_store=label_store, augment=False)

    manifest_path = Path(args.cam_manifest)
    cam_root = Path(args.cam_root) if str(args.cam_root).strip() else manifest_path.parent
    cam_by_tile = _load_cam_manifest(manifest_path=manifest_path, cam_root=cam_root)

    # Positive candidates must have a non-empty mask after converting their hotspot CAM map.
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    pos_cam_cache: dict[int, tuple[np.ndarray, float, str]] = {}
    for i, tile in enumerate(tiles):
        entry = label_store.get_tile_label(tile)
        if entry is None:
            continue
        if float(entry[0]) >= 0.5:
            item = cam_by_tile.get(tile.tile_id)
            if item is None:
                continue
            cam_path = Path(item["cam_path"])
            if not cam_path.exists():
                continue
            cam_raw = np.load(cam_path).astype(np.float32)
            cs = int(tile.chunk_size)
            cam_chunk = np.clip(_to_chunk_size(cam_raw, cs), 0.0, 1.0)
            mask, otsu_thr = cam_to_binary_mask(cam_chunk)
            if int((mask > 0.5).sum()) <= 0:
                continue
            pos_idx.append(i)
            pos_cam_cache[i] = (
                cam_chunk,
                float(otsu_thr),
                str(item["cam_file"]),
            )
        else:
            neg_idx.append(i)

    selected_idx = _pick_indices(
        pos_idx=pos_idx,
        neg_idx=neg_idx,
        total_count=max(1, int(args.count)),
        n_positive=max(0, int(args.positive_count)),
        seed=int(args.seed),
    )

    device = torch.device(
        args.device if str(args.device).strip() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    seg_model = load_partialconv_checkpoint(args.checkpoint, device=device)

    summary: list[dict[str, object]] = []

    for rank, idx in enumerate(selected_idx, start=1):
        tile = tiles[idx]
        entry = label_store.get_tile_label(tile)
        label_value = float(entry[0]) if entry is not None else 0.0

        sample = dataset[idx]
        input_map = sample["input"].numpy().astype(np.float32)
        chm_256 = input_map[0]

        x = sample["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            _, seg = seg_model(x)
        pred_prob_256 = seg[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_mask_256 = (pred_prob_256 >= float(args.pred_threshold)).astype(np.float32)

        if label_value >= 0.5:
            cached = pos_cam_cache.get(idx)
            if cached is None:
                raise RuntimeError(f"Positive tile missing cached hotspot CAM: {tile.tile_id}")
            cam_chunk, _, cam_file = cached
            true_mask, otsu_thr = cam_to_binary_mask(cam_chunk)
            true_rule = "cwd_hotspot_to_mask_from_best_classifier"
        else:
            cs = int(tile.chunk_size)
            true_mask = np.zeros((cs, cs), dtype=np.float32)
            otsu_thr = 0.0
            cam_file = ""
            true_rule = "no_cwd_background"

        cs = int(tile.chunk_size)
        chm = _center_crop(chm_256, cs)
        pred_mask = _center_crop(pred_mask_256, cs)

        inter = float((pred_mask * true_mask).sum())
        den = float(pred_mask.sum() + true_mask.sum())
        dice = (2.0 * inter / den) if den > 0 else 1.0
        union = float(((pred_mask + true_mask) > 0).sum())
        iou = (inter / union) if union > 0 else 1.0

        out_png = out_dir / f"cwd_original_true_pred_mask_rulebased_{rank:02d}_{_slug(tile.tile_id)}.png"
        title = (
            f"tile={tile.tile_id} | label={int(label_value >= 0.5)} | "
            f"rule={true_rule} | dice={dice:.3f} | iou={iou:.3f}"
        )
        _save_panel(chm, true_mask, pred_mask, title, out_png)

        summary.append(
            {
                "rank": rank,
                "tile_id": tile.tile_id,
                "label_value": label_value,
                "true_mask_rule": true_rule,
                "chunk_size": cs,
                "otsu_threshold": float(otsu_thr),
                "cam_file": cam_file,
                "true_mask_positive_px": int((true_mask > 0.5).sum()),
                "pred_mask_positive_px": int((pred_mask > 0.5).sum()),
                "dice": dice,
                "iou": iou,
                "output_png": str(out_png),
            }
        )

    summary_path = out_dir / "cwd_original_true_pred_mask_rulebased_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(summary)} rule-based PNGs")
    for row in summary:
        print(row["output_png"])
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
