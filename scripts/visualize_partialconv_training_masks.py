#!/usr/bin/env python3
"""Visualize masks used by PartialConv training.

The generated panels show, per selected tile:
- CHM input channel
- valid-data mask from nodata filtering
- curriculum-masked valid map at chosen epochs
- label supervision map
- ConvNeXt Grad-CAM++ hotspot map (when legacy checkpoint is provided)
- effective segmentation target + confidence used for supervision mode

This script is read-only for model/data artifacts and writes visual outputs only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cdw_detect.cwd_partialconv_pipeline import CHMDataset
from cdw_detect.cwd_partialconv_pipeline import DEFAULT_LEGACY_CHECKPOINT
from cdw_detect.cwd_partialconv_pipeline import PseudoLabelStore
from cdw_detect.cwd_partialconv_pipeline import TileRecord
from cdw_detect.cwd_partialconv_pipeline import apply_curriculum_masking
from cdw_detect.cwd_partialconv_pipeline import compute_gradcam_hotspots
from cdw_detect.cwd_partialconv_pipeline import create_strict_splits
from cdw_detect.cwd_partialconv_pipeline import hotspot_to_confidence_map
from cdw_detect.cwd_partialconv_pipeline import load_legacy_model
from cdw_detect.cwd_partialconv_pipeline import seed_everything
from cdw_detect.cwd_partialconv_pipeline import setup_logging

CELL_SIZE = 192
TITLE_BAR_HEIGHT = 26
HEADER_HEIGHT = 44


def _parse_epoch_list(text: str, max_epochs: int) -> list[int]:
    raw = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            raw.append(int(token))
        except ValueError:
            continue

    if not raw:
        raw = [1, max(1, max_epochs // 2), max_epochs]

    epochs = sorted({max(1, min(max_epochs, value)) for value in raw})
    return epochs


def _slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _to_u8_chm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    values = arr[finite]
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _to_u8_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_cell(arr: np.ndarray, title: str, kind: str) -> np.ndarray:
    if kind == "chm":
        base = cv2.applyColorMap(_to_u8_chm(arr), cv2.COLORMAP_VIRIDIS)
        interp = cv2.INTER_LINEAR
    elif kind == "conf":
        base = cv2.applyColorMap(_to_u8_mask(arr), cv2.COLORMAP_TURBO)
        interp = cv2.INTER_LINEAR
    else:
        base = cv2.cvtColor(_to_u8_mask(arr), cv2.COLOR_GRAY2BGR)
        interp = cv2.INTER_NEAREST

    cell = cv2.resize(base, (CELL_SIZE, CELL_SIZE), interpolation=interp)
    bar = np.full((TITLE_BAR_HEIGHT, CELL_SIZE, 3), 235, dtype=np.uint8)
    cv2.putText(
        bar,
        title,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, cell])


def _apply_mask_snapshot(input_map: np.ndarray, epoch: int, max_epochs: int, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    x = torch.from_numpy(np.ascontiguousarray(input_map[None, ...])).float()
    with torch.no_grad():
        out = apply_curriculum_masking(x, epoch=epoch, max_epochs=max_epochs)
    return out[0].cpu().numpy()


def _select_split_tiles(split_name: str, train: list[TileRecord], val: list[TileRecord], test: list[TileRecord]) -> list[TileRecord]:
    if split_name == "train":
        return train
    if split_name == "val":
        return val
    if split_name == "test":
        return test
    if split_name == "trainval":
        return train + val
    return train + val + test


def _compute_hotspot_map(
    legacy_model: Optional[torch.nn.Module],
    input_map: np.ndarray,
    device: torch.device,
) -> Optional[np.ndarray]:
    if legacy_model is None:
        return None

    chm = np.ascontiguousarray(input_map[0:1][None, ...], dtype=np.float32)
    chm_tensor = torch.from_numpy(chm).to(device)
    hotspots = compute_gradcam_hotspots(legacy_model, chm_tensor)
    return hotspots[0, 0].detach().cpu().numpy().astype(np.float32)


def _resolve_seg_target_for_viz(
    seg_target_label: np.ndarray,
    confidence_label: np.ndarray,
    has_label: bool,
    hotspot_map: Optional[np.ndarray],
    seg_target_source: str,
    hotspot_conf_min: float,
    hotspot_conf_gamma: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    mode = str(seg_target_source).strip().lower()
    if mode not in {"labels", "hotspot", "labels_or_hotspot"}:
        mode = "labels"

    if mode == "labels":
        return seg_target_label, confidence_label, mode

    if hotspot_map is None:
        # Fallback keeps visualization robust when hotspot generation is unavailable.
        return seg_target_label, confidence_label, f"{mode}_fallback_labels"

    hotspot = np.clip(np.asarray(hotspot_map, dtype=np.float32), 0.0, 1.0)
    hotspot_conf = (
        hotspot_to_confidence_map(
            torch.from_numpy(hotspot[None, None, ...]),
            min_conf=float(np.clip(hotspot_conf_min, 0.0, 1.0)),
            gamma=float(max(hotspot_conf_gamma, 1e-8)),
        )
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if mode == "hotspot":
        return hotspot, hotspot_conf, mode

    if has_label:
        return seg_target_label, confidence_label, mode
    return hotspot, hotspot_conf, mode


def _group_indices(tiles: list[TileRecord], label_store: PseudoLabelStore) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"positive": [], "negative": [], "unlabeled": []}
    for idx, tile in enumerate(tiles):
        entry = label_store.get_tile_label(tile)
        if entry is None:
            groups["unlabeled"].append(idx)
            continue
        if float(entry[0]) >= 0.5:
            groups["positive"].append(idx)
        else:
            groups["negative"].append(idx)
    return groups


def _pick_examples(groups: dict[str, list[int]], per_group: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    selected: list[tuple[str, int]] = []
    for group in ["positive", "negative", "unlabeled"]:
        options = groups[group]
        if not options:
            continue
        take = min(per_group, len(options))
        for idx in rng.sample(options, take):
            selected.append((group, idx))
    return selected


def _panel_for_sample(
    tile: TileRecord,
    group: str,
    sample: dict,
    label_entry: Optional[tuple[float, float, str]],
    hotspot_map: Optional[np.ndarray],
    seg_target_used: np.ndarray,
    conf_used: np.ndarray,
    seg_mode_used: str,
    snapshots: dict[int, np.ndarray],
    first_epoch: int,
    last_epoch: int,
) -> np.ndarray:
    input_map = sample["input"].numpy()
    seg_target = sample["seg_target"].numpy()[0]
    confidence = sample["confidence"].numpy()[0]
    has_label = float(sample["has_label"].item()) > 0.5

    valid_raw = input_map[1]
    valid_first = snapshots[first_epoch][1]
    valid_last = snapshots[last_epoch][1]
    chm_raw = input_map[0]
    chm_first = snapshots[first_epoch][0]
    chm_last = snapshots[last_epoch][0]

    row1 = np.hstack(
        [
            _render_cell(chm_raw, "CHM", "chm"),
            _render_cell(valid_raw, "valid raw", "mask"),
            _render_cell(valid_first, f"valid e{first_epoch}", "mask"),
            _render_cell(valid_last, f"valid e{last_epoch}", "mask"),
        ]
    )
    row2 = np.hstack(
        [
            _render_cell(seg_target, "seg label(raw)", "mask"),
            _render_cell(hotspot_map if hotspot_map is not None else np.zeros_like(seg_target), "hotspot", "conf"),
            _render_cell(seg_target_used, "seg used(loss)", "mask"),
            _render_cell(conf_used, "conf used", "conf"),
        ]
    )

    panel = np.vstack([row1, row2])

    label_text = "none"
    conf_text = "n/a"
    src_text = "n/a"
    if label_entry is not None:
        label_text = str(int(float(label_entry[0]) >= 0.5))
        conf_text = f"{float(label_entry[1]):.3f}"
        src_text = str(label_entry[2])

    header = np.full((HEADER_HEIGHT, panel.shape[1], 3), 255, dtype=np.uint8)
    line1 = f"tile={tile.tile_id}  sample={tile.sample_id}  group={group}  has_label={int(has_label)}"
    line2 = (
        f"tile_label={label_text}  tile_conf={conf_text}  source={src_text}  "
        f"seg_source={seg_mode_used}  has_hotspot={int(hotspot_map is not None)}"
    )

    cv2.putText(header, line1[:180], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(header, line2[:180], (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    return np.vstack([header, panel])


def _write_contact_sheet(images: list[np.ndarray], output_path: Path, n_cols: int = 2) -> None:
    if not images:
        return

    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    gap = 12
    n_rows = int(math.ceil(len(images) / max(1, n_cols)))

    sheet_h = n_rows * h + (n_rows - 1) * gap
    sheet_w = n_cols * w + (n_cols - 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        y = row * (h + gap)
        x = col * (w + gap)
        padded = np.full((h, w, 3), 255, dtype=np.uint8)
        padded[: img.shape[0], : img.shape[1]] = img
        sheet[y : y + h, x : x + w] = padded

    cv2.imwrite(str(output_path), sheet)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize masks used during PartialConv training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--registry-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "trainval", "all"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--buffer-meters", type=float, default=50.0)
    parser.add_argument("--max-tiles-per-raster", type=int, default=2500)
    parser.add_argument("--cam-mask-dir", type=str, default="")
    parser.add_argument("--legacy-checkpoint", type=str, default=str(DEFAULT_LEGACY_CHECKPOINT))
    parser.add_argument(
        "--seg-target-source",
        type=str,
        default="labels",
        choices=["labels", "hotspot", "labels_or_hotspot"],
        help="Match training segmentation target source semantics",
    )
    parser.add_argument(
        "--hotspot-conf-min",
        type=float,
        default=0.05,
        help="Minimum confidence floor for hotspot-derived supervision",
    )
    parser.add_argument(
        "--hotspot-conf-gamma",
        type=float,
        default=1.0,
        help="Hotspot confidence exponent: >1 sharper borders, <1 flatter confidence",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--epochs-to-show", type=str, default="1,3,5")
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    seed_everything(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.output_dir)
    panel_dir = out_dir / "sample_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    split = create_strict_splits(
        data_dir=args.data_dir,
        test_size=args.test_size,
        buffer_meters=args.buffer_meters,
        registry_dir=args.registry_dir,
        val_size=args.val_size,
        max_tiles_per_raster=args.max_tiles_per_raster,
        seed=args.seed,
    )

    label_store = PseudoLabelStore.from_tiles(split.train + split.val + split.test)

    cam_mask_dir: Optional[Path] = None
    if str(args.cam_mask_dir).strip():
        candidate = Path(args.cam_mask_dir)
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"CAM mask directory does not exist: {candidate}")
        cam_mask_dir = candidate

    legacy_model: Optional[torch.nn.Module] = None
    if str(args.legacy_checkpoint).strip():
        legacy_model = load_legacy_model(args.legacy_checkpoint, device=device)

    if args.seg_target_source != "labels" and legacy_model is None:
        raise RuntimeError(
            "seg-target-source requires a loadable legacy checkpoint for hotspot generation"
        )

    tiles = _select_split_tiles(args.split, split.train, split.val, split.test)
    if not tiles:
        raise RuntimeError("No tiles available for selected split")

    dataset = CHMDataset(
        tiles=tiles,
        label_store=label_store,
        augment=False,
        cam_mask_dir=cam_mask_dir,
    )

    epochs_to_show = _parse_epoch_list(args.epochs_to_show, max_epochs=max(1, int(args.epochs)))
    first_epoch = epochs_to_show[0]
    last_epoch = epochs_to_show[-1]

    groups = _group_indices(tiles, label_store)
    selected = _pick_examples(groups, per_group=max(1, args.samples_per_group), seed=args.seed)

    written_panels: list[str] = []
    loaded_panels: list[np.ndarray] = []

    for rank, (group, idx) in enumerate(selected, start=1):
        tile = tiles[idx]
        sample = dataset[idx]
        input_map = sample["input"].numpy()
        seg_target_label = sample["seg_target"].numpy()[0]
        confidence_label = sample["confidence"].numpy()[0]
        has_label = float(sample["has_label"].item()) > 0.5

        hotspot_map = _compute_hotspot_map(legacy_model, input_map, device=device)
        seg_target_used, conf_used, seg_mode_used = _resolve_seg_target_for_viz(
            seg_target_label=seg_target_label,
            confidence_label=confidence_label,
            has_label=has_label,
            hotspot_map=hotspot_map,
            seg_target_source=args.seg_target_source,
            hotspot_conf_min=args.hotspot_conf_min,
            hotspot_conf_gamma=args.hotspot_conf_gamma,
        )

        snapshots: dict[int, np.ndarray] = {}
        for epoch in epochs_to_show:
            snap_seed = args.seed + idx * 997 + epoch * 37
            snapshots[epoch] = _apply_mask_snapshot(input_map, epoch, max(1, args.epochs), snap_seed)

        panel = _panel_for_sample(
            tile=tile,
            group=group,
            sample=sample,
            label_entry=label_store.get_tile_label(tile),
            hotspot_map=hotspot_map,
            seg_target_used=seg_target_used,
            conf_used=conf_used,
            seg_mode_used=seg_mode_used,
            snapshots=snapshots,
            first_epoch=first_epoch,
            last_epoch=last_epoch,
        )
        name = f"{rank:03d}_{group}_{_slug(tile.tile_id)}.png"
        path = panel_dir / name
        cv2.imwrite(str(path), panel)
        written_panels.append(name)
        loaded_panels.append(panel)

    contact_sheet_path = out_dir / "contact_sheet.png"
    _write_contact_sheet(loaded_panels, contact_sheet_path, n_cols=2)

    stats_pool = list(range(len(dataset)))
    rng = random.Random(args.seed + 123)
    if args.stats_samples > 0 and len(stats_pool) > args.stats_samples:
        stats_pool = rng.sample(stats_pool, args.stats_samples)

    raw_keep: list[float] = []
    keep_by_epoch: dict[int, list[float]] = {epoch: [] for epoch in epochs_to_show}
    has_label_values: list[float] = []
    seg_positive: list[float] = []
    seg_constant: list[float] = []
    seg_used_positive: list[float] = []
    seg_used_constant: list[float] = []
    seg_conf_abs_diff: list[float] = []
    hotspot_available: list[float] = []

    for idx in stats_pool:
        sample = dataset[idx]
        input_map = sample["input"].numpy()
        seg_map = sample["seg_target"].numpy()[0]
        conf_map = sample["confidence"].numpy()[0]
        has_label = float(sample["has_label"].item()) > 0.5
        hotspot_map = _compute_hotspot_map(legacy_model, input_map, device=device)
        seg_used_map, conf_used_map, _ = _resolve_seg_target_for_viz(
            seg_target_label=seg_map,
            confidence_label=conf_map,
            has_label=has_label,
            hotspot_map=hotspot_map,
            seg_target_source=args.seg_target_source,
            hotspot_conf_min=args.hotspot_conf_min,
            hotspot_conf_gamma=args.hotspot_conf_gamma,
        )

        raw_keep.append(float(input_map[1].mean()))
        has_label_values.append(1.0 if has_label else 0.0)
        hotspot_available.append(1.0 if hotspot_map is not None else 0.0)
        seg_used_positive.append(float(seg_used_map.mean()))
        seg_used_constant.append(1.0 if float(np.std(seg_used_map)) < 1e-8 else 0.0)
        seg_conf_abs_diff.append(float(np.mean(np.abs(seg_used_map - conf_used_map))))

        if has_label:
            seg_positive.append(float(seg_map.mean()))
            seg_constant.append(1.0 if float(np.std(seg_map)) < 1e-8 else 0.0)

        for epoch in epochs_to_show:
            snap_seed = args.seed + idx * 1499 + epoch * 41
            masked = _apply_mask_snapshot(input_map, epoch, max(1, args.epochs), snap_seed)
            keep_by_epoch[epoch].append(float(masked[1].mean()))

    summary = {
        "args": {
            "data_dir": args.data_dir,
            "registry_dir": args.registry_dir,
            "split": args.split,
            "cam_mask_dir": str(cam_mask_dir) if cam_mask_dir is not None else "",
            "legacy_checkpoint": args.legacy_checkpoint,
            "seg_target_source": args.seg_target_source,
            "hotspot_conf_min": float(args.hotspot_conf_min),
            "hotspot_conf_gamma": float(args.hotspot_conf_gamma),
            "device": str(device),
            "epochs": int(args.epochs),
            "epochs_to_show": epochs_to_show,
            "samples_per_group": int(args.samples_per_group),
            "stats_samples": int(len(stats_pool)),
            "seed": int(args.seed),
        },
        "split_metadata": split.metadata,
        "tile_counts": {
            "selected_split_tiles": len(tiles),
            "group_positive": len(groups["positive"]),
            "group_negative": len(groups["negative"]),
            "group_unlabeled": len(groups["unlabeled"]),
            "panels_written": len(written_panels),
        },
        "mask_coverage": {
            "valid_keep_raw": _mean_std(raw_keep),
            "valid_keep_epoch": {str(epoch): _mean_std(values) for epoch, values in keep_by_epoch.items()},
        },
        "supervision": {
            "has_label_fraction": _mean_std(has_label_values),
            "hotspot_available_fraction": _mean_std(hotspot_available),
            "seg_positive_fraction_on_labeled": _mean_std(seg_positive),
            "seg_constant_fraction_on_labeled": _mean_std(seg_constant),
            "seg_used_positive_fraction": _mean_std(seg_used_positive),
            "seg_used_constant_fraction": _mean_std(seg_used_constant),
            "seg_conf_mean_abs_diff": _mean_std(seg_conf_abs_diff),
        },
        "outputs": {
            "panel_dir": str(panel_dir),
            "panels": written_panels,
            "contact_sheet": str(contact_sheet_path),
        },
    }

    summary_path = out_dir / "mask_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logging.info("Wrote %d panels to %s", len(written_panels), panel_dir)
    logging.info("Contact sheet: %s", contact_sheet_path)
    logging.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
