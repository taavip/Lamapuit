#!/usr/bin/env python3
"""Onboarding-label weak-supervision segmentation experiment on harmonized CHM.

Design summary:
- Training supervision: tile-level labels from output/onboarding_labels_v2_drop13
- Training imagery: harmonized CHM raw/gauss from output/chm_dataset_harmonized_0p8m_raw_gauss_stable
- Validation: 23 manual RGBA masks from output/manual_masks
- Model: DeepLabV3+ with non-ConvNeXt encoder (by default resnet18)
- Leakage control: remove onboarding train tiles overlapping manual validation windows
- Iterative study: run 3 progressively improved iterations and write reproducible reports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "segmentation_models_pytorch is required. Install in docker conda env: "
        "pip install 'segmentation-models-pytorch>=0.5.0,<0.6'"
    ) from exc


EPS = 1e-8


# SLD terrain style used by labeler and legacy tile tools.
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
_SLD_MAX_HAG_M = 1.3
_SLD_DARK_THRESHOLD = 0.15 / _SLD_MAX_HAG_M


def _make_sld_cmap() -> mcolors.LinearSegmentedColormap:
    vals = [v / _SLD_MAX_HAG_M for v, _ in _SLD_BREAKPOINTS]
    colors = [c for _, c in _SLD_BREAKPOINTS]
    return mcolors.LinearSegmentedColormap.from_list("sld_terrain", list(zip(vals, colors)))


_SLD_CMAP = _make_sld_cmap()


def _apply_sld_from_normalized_chm(chm_norm: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Render normalized CHM tile with SLD symbology in HAG meters.

    Training/eval CHM is stored as [0,1] after clipping to 20 m; convert back to
    approximate meters for display and apply SLD palette in the 0-1.3 m band.
    """
    chm_m = np.clip(chm_norm.astype(np.float32), 0.0, 1.0) * 20.0
    black_mask = (~valid_mask) | (chm_m <= 0.0)

    t = chm_m.copy()
    t[black_mask] = 0.0
    t = np.clip(t, 0.0, _SLD_MAX_HAG_M) / _SLD_MAX_HAG_M

    rgb = (_SLD_CMAP(t)[:, :, :3] * 255).astype(np.uint8)
    dark_factor = np.where(
        t < _SLD_DARK_THRESHOLD,
        (t / _SLD_DARK_THRESHOLD) ** 0.7,
        1.0,
    ).astype(np.float32)
    rgb = (rgb.astype(np.float32) * dark_factor[:, :, np.newaxis]).astype(np.uint8)
    rgb[black_mask] = 0
    return rgb


@dataclass
class OnboardingTile:
    sample_id: str
    variant: str
    raster_path: Path
    row_off: int
    col_off: int
    chunk_size: int
    label: int
    source: str
    model_prob: float | None


@dataclass
class ManualValTile:
    tile_id: str
    sample_id: str
    variant: str
    row_off: int
    col_off: int
    h: int
    w: int
    raster_path: Path
    chm: np.ndarray
    target: np.ndarray
    valid: np.ndarray


@dataclass
class IterationConfig:
    name: str
    variants: tuple[str, ...]
    positive_mask_mode: str  # full_tile | q80_sparse
    encoder_name: str
    max_per_class: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dice_weight: float
    focal_weight: float
    augment: bool


class WeakLabelArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        augment: bool,
    ) -> None:
        self.images = images
        self.targets = targets
        self.weights = weights
        self.augment = augment

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = self.images[idx].copy()
        tgt = self.targets[idx].copy()
        wgt = self.weights[idx].copy()

        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, axis=2).copy()
                tgt = np.flip(tgt, axis=2).copy()
                wgt = np.flip(wgt, axis=2).copy()
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                tgt = np.flip(tgt, axis=1).copy()
                wgt = np.flip(wgt, axis=1).copy()
            if random.random() < 0.75:
                k = random.randint(1, 3)
                img = np.rot90(img, k=k, axes=(1, 2)).copy()
                tgt = np.rot90(tgt, k=k, axes=(1, 2)).copy()
                wgt = np.rot90(wgt, k=k, axes=(1, 2)).copy()

        return {
            "image": torch.from_numpy(img).float(),
            "target": torch.from_numpy(tgt).float(),
            "weight": torch.from_numpy(wgt).float(),
        }


class WeightedDiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float,
        focal_weight: float,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.alpha = float(focal_alpha)
        self.gamma = float(focal_gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(weight_map, min=0.0)
        wsum = torch.clamp(w.sum(), min=1.0)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_t * ((1.0 - p_t).pow(self.gamma)) * bce
        focal_loss = (focal * w).sum() / wsum

        pw = prob * w
        tw = target * w
        inter = (pw * target).sum(dim=(1, 2, 3))
        den = pw.sum(dim=(1, 2, 3)) + tw.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_chm(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    invalid = ~np.isfinite(x)
    x[invalid] = 0.0
    x = np.clip(x, 0.0, 20.0) / 20.0
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Onboarding->Harmonized segmentation experiment")
    p.add_argument("--labels-dir", type=Path, default=Path("output/onboarding_labels_v2_drop13"))
    p.add_argument(
        "--chm-root",
        type=Path,
        default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss_stable"),
    )
    p.add_argument("--manual-mask-dir", type=Path, default=Path("output/manual_masks"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/manual_mask_experiments/onboarding_harmonized_deeplab_v1"),
    )
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--overlay-count", type=int, default=8)
    p.add_argument("--max-rows-per-file", type=int, default=0)
    return p.parse_args()


def resolve_harmonized_raster(chm_root: Path, sample_id: str, variant: str) -> Path | None:
    tif = chm_root / f"chm_{variant}" / f"{sample_id}_harmonized_dem_last_{variant}_chm.tif"
    if tif.exists():
        return tif
    return None


def parse_sample_id_from_legacy_raster(raster_name: str) -> str | None:
    m = re.match(r"^(?P<sample>.+?)_chm_max_hag_20cm\.tif$", raster_name)
    if m is None:
        return None
    return m.group("sample")


def read_tile_chm_and_valid(raster_path: Path, row_off: int, col_off: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(raster_path) as src:
        arr = src.read(1, window=Window(col_off, row_off, size, size), boundless=True, fill_value=np.nan)
        if arr.shape != (size, size):
            fixed = np.full((size, size), np.nan, dtype=np.float32)
            h = min(size, arr.shape[0])
            w = min(size, arr.shape[1])
            fixed[:h, :w] = arr[:h, :w]
            arr = fixed

        invalid = ~np.isfinite(arr)
        if src.nodata is not None:
            invalid |= np.isclose(arr, float(src.nodata))
        invalid |= arr < 0.0

    chm = normalize_chm(arr)
    valid = (~invalid).astype(np.float32)
    return chm, valid


def load_manual_validation_tiles(mask_dir: Path, chm_root: Path) -> list[ManualValTile]:
    pat = re.compile(r"^(?P<sample>.+?)_harmonized_dem_last_(?P<variant>raw|gauss)_chm__r(?P<row>\d+)_c(?P<col>\d+)$")
    out: list[ManualValTile] = []

    for meta_path in sorted(mask_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tile_id = str(meta.get("tile_id", "")).strip()
        m = pat.match(tile_id)
        if m is None:
            continue

        sample_id = m.group("sample")
        variant = m.group("variant")
        row_off = int(m.group("row"))
        col_off = int(m.group("col"))

        raster = resolve_harmonized_raster(chm_root, sample_id, variant)
        if raster is None:
            continue

        mask_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_mask.png"))
        if not mask_path.exists():
            continue

        rgba = np.asarray(Image.open(mask_path).convert("RGBA"), dtype=np.uint8)
        red = rgba[..., 0]
        alpha = rgba[..., 3]
        pos = (alpha > 0) & (red >= 200)
        neg = (alpha > 0) & (red <= 50)
        valid = (pos | neg).astype(np.float32)
        target = pos.astype(np.float32)

        h, w = target.shape
        try:
            chm, _ = read_tile_chm_and_valid(raster, row_off, col_off, size=h)
        except Exception:
            continue

        out.append(
            ManualValTile(
                tile_id=tile_id,
                sample_id=sample_id,
                variant=variant,
                row_off=row_off,
                col_off=col_off,
                h=h,
                w=w,
                raster_path=raster,
                chm=chm,
                target=target,
                valid=valid,
            )
        )

    return out


def _rects_intersect(a_r: int, a_c: int, a_h: int, a_w: int, b_r: int, b_c: int, b_h: int, b_w: int) -> bool:
    return (a_r < b_r + b_h) and (a_r + a_h > b_r) and (a_c < b_c + b_w) and (a_c + a_w > b_c)


def load_onboarding_tiles(
    labels_dir: Path,
    chm_root: Path,
    variants: tuple[str, ...],
    manual_tiles: list[ManualValTile],
    max_rows_per_file: int = 0,
) -> tuple[list[OnboardingTile], dict[str, Any]]:
    manual_by_key: dict[tuple[str, str], list[ManualValTile]] = {}
    for mt in manual_tiles:
        manual_by_key.setdefault((mt.sample_id, mt.variant), []).append(mt)

    tiles: list[OnboardingTile] = []
    stats: dict[str, Any] = {
        "files": 0,
        "rows_total": 0,
        "rows_mapped": 0,
        "rows_skipped_no_map": 0,
        "rows_skipped_source": 0,
        "rows_skipped_overlap": 0,
        "label_counts": {"cdw": 0, "no_cdw": 0},
    }

    include_sources = {"auto", "manual", "auto_reviewed"}

    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        stats["files"] += 1
        with csv_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            n_rows = 0
            for row in reader:
                n_rows += 1
                if max_rows_per_file > 0 and n_rows > max_rows_per_file:
                    break

                stats["rows_total"] += 1

                label_text = str(row.get("label", "")).strip().lower()
                if label_text == "cdw":
                    label = 1
                elif label_text == "no_cdw":
                    label = 0
                else:
                    continue

                source = str(row.get("source", "")).strip().lower()
                if source not in include_sources:
                    stats["rows_skipped_source"] += 1
                    continue

                raster_name = str(row.get("raster", "")).strip()
                sample_id = parse_sample_id_from_legacy_raster(raster_name)
                if sample_id is None:
                    stats["rows_skipped_no_map"] += 1
                    continue

                row_off = int(float(row.get("row_off", 0)))
                col_off = int(float(row.get("col_off", 0)))
                chunk = int(float(row.get("chunk_size", 128)))

                prob_raw = str(row.get("model_prob", "")).strip()
                model_prob = None
                if prob_raw not in ("", "None", "nan", "NaN"):
                    try:
                        model_prob = float(prob_raw)
                    except ValueError:
                        model_prob = None

                for variant in variants:
                    raster = resolve_harmonized_raster(chm_root, sample_id, variant)
                    if raster is None:
                        continue

                    overlap_list = manual_by_key.get((sample_id, variant), [])
                    overlap = False
                    for mt in overlap_list:
                        if _rects_intersect(
                            row_off,
                            col_off,
                            chunk,
                            chunk,
                            mt.row_off,
                            mt.col_off,
                            mt.h,
                            mt.w,
                        ):
                            overlap = True
                            break
                    if overlap:
                        stats["rows_skipped_overlap"] += 1
                        continue

                    tiles.append(
                        OnboardingTile(
                            sample_id=sample_id,
                            variant=variant,
                            raster_path=raster,
                            row_off=row_off,
                            col_off=col_off,
                            chunk_size=chunk,
                            label=label,
                            source=source,
                            model_prob=model_prob,
                        )
                    )
                    stats["rows_mapped"] += 1
                    if label == 1:
                        stats["label_counts"]["cdw"] += 1
                    else:
                        stats["label_counts"]["no_cdw"] += 1

    return tiles, stats


def _weak_target_from_positive(chm: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full_tile":
        return valid.copy()

    if mode == "q80_sparse":
        vals = chm[valid > 0.5]
        if vals.size < 32:
            return valid.copy()
        thr = float(np.quantile(vals, 0.80))
        target = ((chm >= thr) & (valid > 0.5)).astype(np.float32)
        if float(target.sum()) < 16.0:
            return valid.copy()
        return target

    raise ValueError(f"Unsupported positive mask mode: {mode}")


def _tile_confidence(tile: OnboardingTile) -> float:
    source_defaults = {
        "manual": 1.0,
        "auto_reviewed": 0.9,
        "auto": 0.8,
    }
    base = float(source_defaults.get(tile.source, 0.7))
    if tile.model_prob is not None:
        p = float(np.clip(tile.model_prob, 0.0, 1.0))
        conf = p if tile.label == 1 else (1.0 - p)
        base = max(base * 0.6, conf)
    return float(np.clip(base, 0.05, 1.0))


def materialize_training_arrays(
    tiles: list[OnboardingTile],
    tile_size: int,
    positive_mask_mode: str,
    seed: int,
    max_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    pos = [t for t in tiles if t.label == 1]
    neg = [t for t in tiles if t.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos = pos[: max(1, max_per_class)]
    neg = neg[: max(1, max_per_class)]
    chosen = pos + neg
    rng.shuffle(chosen)

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    skipped_read = 0
    for t in chosen:
        size = int(t.chunk_size)
        if size != tile_size:
            # Keep experiment strict: train exactly on 128x128 onboarding tiles.
            continue
        try:
            chm, valid = read_tile_chm_and_valid(t.raster_path, t.row_off, t.col_off, size=size)
        except Exception:
            skipped_read += 1
            continue

        if float(valid.mean()) < 0.20:
            continue

        if t.label == 1:
            target2d = _weak_target_from_positive(chm=chm, valid=valid, mode=positive_mask_mode)
        else:
            target2d = np.zeros_like(chm, dtype=np.float32)

        conf = _tile_confidence(t)
        weight2d = valid * conf

        images.append(chm[None, ...].astype(np.float32))
        targets.append(target2d[None, ...].astype(np.float32))
        weights.append(weight2d[None, ...].astype(np.float32))

    if not images:
        raise RuntimeError("No training samples materialized. Check mapping/filters.")

    x = np.stack(images, axis=0)
    y = np.stack(targets, axis=0)
    w = np.stack(weights, axis=0)

    stats = {
        "materialized": int(x.shape[0]),
        "materialized_positive": int((y.reshape(y.shape[0], -1).sum(axis=1) > 0).sum()),
        "materialized_negative": int((y.reshape(y.shape[0], -1).sum(axis=1) == 0).sum()),
        "skipped_read": int(skipped_read),
        "mean_weight": float(w.mean()),
    }
    return x, y, w, stats


def create_model(encoder_name: str, aspp_rates: tuple[int, int, int]) -> nn.Module:
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,
    )
    try:
        model = smp.DeepLabV3Plus(decoder_atrous_rates=aspp_rates, **kwargs)
    except TypeError:
        model = smp.DeepLabV3Plus(**kwargs)
    return model


def _gaussian_window(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    g = np.outer(w, w).astype(np.float32)
    g = np.maximum(g, 1e-3)
    return g


@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    overlap: float,
    batch_size: int,
) -> np.ndarray:
    h, w = image.shape
    stride = max(1, int(round(patch_size * (1.0 - overlap))))

    pad_h = (stride - ((h - patch_size) % stride)) % stride if h > patch_size else patch_size - h
    pad_w = (stride - ((w - patch_size) % stride)) % stride if w > patch_size else patch_size - w
    pad_h = max(0, pad_h)
    pad_w = max(0, pad_w)

    img = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    H, W = img.shape

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    weight = _gaussian_window(patch_size)
    prob_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            patches.append(img[y0 : y0 + patch_size, x0 : x0 + patch_size])
            coords.append((y0, x0))

    model.eval()
    for i in range(0, len(patches), batch_size):
        b = np.stack(patches[i : i + batch_size], axis=0)
        bt = torch.from_numpy(b[:, None, ...]).float().to(device)
        logits = model(bt)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
        for j, p in enumerate(probs):
            y0, x0 = coords[i + j]
            prob_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += p * weight
            w_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += weight

    out = prob_sum / np.maximum(w_sum, 1e-6)
    return out[:h, :w]


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = float(np.logical_and(pred == 1, gt == 1).sum())
    fp = float(np.logical_and(pred == 1, gt == 0).sum())
    fn = float(np.logical_and(pred == 0, gt == 1).sum())
    tn = float(np.logical_and(pred == 0, gt == 0).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    acc = (tp + tn) / (tp + tn + fp + fn + EPS)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


@torch.no_grad()
def evaluate_manual_tiles(
    model: nn.Module,
    manual_tiles: list[ManualValTile],
    device: torch.device,
    tile_size: int,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    probs_by_id: dict[str, np.ndarray] = {}
    tp = fp = fn = tn = 0.0

    for mt in manual_tiles:
        probs = sliding_window_predict(
            model=model,
            image=mt.chm,
            device=device,
            patch_size=tile_size,
            overlap=0.5,
            batch_size=batch_size,
        )
        probs_by_id[mt.tile_id] = probs

        valid = mt.valid > 0.5
        gt = (mt.target > 0.5) & valid
        pred = (probs >= threshold) & valid

        tp += float(np.logical_and(pred, gt).sum())
        fp += float(np.logical_and(pred, ~gt).sum())
        fn += float(np.logical_and(~pred, gt).sum())
        tn += float(np.logical_and(~pred, ~gt).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    acc = (tp + tn) / (tp + tn + fp + fn + EPS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }, probs_by_id


def threshold_sweep(
    probs_by_id: dict[str, np.ndarray],
    manual_tiles: list[ManualValTile],
    thresholds: list[float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    by_id = {t.tile_id: t for t in manual_tiles}
    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for thr in thresholds:
        tp = fp = fn = tn = 0.0
        for tile_id, probs in probs_by_id.items():
            mt = by_id[tile_id]
            valid = mt.valid > 0.5
            gt = (mt.target > 0.5) & valid
            pred = (probs >= thr) & valid
            tp += float(np.logical_and(pred, gt).sum())
            fp += float(np.logical_and(pred, ~gt).sum())
            fn += float(np.logical_and(~pred, gt).sum())
            tn += float(np.logical_and(~pred, ~gt).sum())

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
        iou = tp / (tp + fp + fn + EPS)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = (2.0 * precision * recall) / (precision + recall + EPS)
        row = {
            "threshold": float(thr),
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best = row

    if best is None:
        raise RuntimeError("Threshold sweep produced no rows")
    return best, rows


def save_overlays(
    out_dir: Path,
    manual_tiles: list[ManualValTile],
    probs_by_id: dict[str, np.ndarray],
    threshold: float,
    max_count: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(manual_tiles, key=lambda t: float((t.target * t.valid).sum()), reverse=True)
    if max_count <= 0:
        chosen = ranked
    else:
        chosen = ranked[: max(1, min(max_count, len(ranked)))]
    written: list[str] = []
    rows: list[dict[str, Any]] = []

    legend_text = (
        "CHM (SLD): 0 m black, low heights dark red/orange, mid yellow/green, high cyan/blue. "
        "Range clipped to 0-1.3 m for display.\n"
        "GT label mask: CWD=green (value=1), labeled background=yellow (value=0), unlabeled=transparent.\n"
        f"Prediction mask: red where probability >= {threshold:.2f} (binary value=1).\n"
        "FP/FN map: FP red (pred=1, gt=0), FN blue (pred=0, gt=1)."
    )
    (out_dir / "overlay_legend.txt").write_text(legend_text, encoding="utf-8")

    for mt in chosen:
        probs = probs_by_id[mt.tile_id]
        valid = mt.valid > 0.5
        gt_pos = (mt.target > 0.5) & valid
        gt_bg = (mt.target <= 0.5) & valid
        pred = (probs >= threshold) & valid

        valid_count = int(valid.sum())
        if valid_count == 0:
            continue

        m = compute_binary_metrics(pred[valid].astype(np.uint8), gt_pos[valid].astype(np.uint8))

        fp = np.logical_and(pred, ~gt_pos)
        fn = np.logical_and(~pred, gt_pos)
        tp = np.logical_and(pred, gt_pos)
        tn = np.logical_and(~pred, ~gt_pos) & valid

        chm_m = np.clip(mt.chm.astype(np.float32), 0.0, 1.0) * 20.0
        chm_vals = chm_m[valid]
        chm_min = float(np.min(chm_vals))
        chm_mean = float(np.mean(chm_vals))
        chm_max = float(np.max(chm_vals))
        chm_p95 = float(np.quantile(chm_vals, 0.95))

        base_rgb = _apply_sld_from_normalized_chm(mt.chm, valid_mask=valid)

        gt_rgba = np.zeros((*mt.chm.shape, 4), dtype=np.float32)
        gt_rgba[gt_bg] = [1.00, 0.95, 0.20, 0.35]
        gt_rgba[gt_pos] = [0.05, 1.00, 0.35, 0.78]

        pred_rgba = np.zeros((*mt.chm.shape, 4), dtype=np.float32)
        pred_rgba[pred] = [1.00, 0.10, 0.10, 0.72]

        err_rgba = np.zeros((*mt.chm.shape, 4), dtype=np.float32)
        err_rgba[fp] = [1.00, 0.00, 0.00, 0.90]
        err_rgba[fn] = [0.00, 0.35, 1.00, 0.90]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.subplots_adjust(bottom=0.25, wspace=0.03)

        axes[0].imshow(base_rgb)
        axes[0].set_title(
            "CHM (SLD)\n"
            f"min/mean/max={chm_min:.2f}/{chm_mean:.2f}/{chm_max:.2f} m, p95={chm_p95:.2f}",
            fontsize=9,
        )
        axes[0].axis("off")

        sm = plt.cm.ScalarMappable(cmap=_SLD_CMAP, norm=plt.Normalize(0.0, _SLD_MAX_HAG_M))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0], fraction=0.045, pad=0.015)
        cbar.set_label("HAG meters (display clipped 0-1.3)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        axes[1].imshow(base_rgb)
        axes[1].imshow(gt_rgba)
        axes[1].set_title("GT label mask\nCWD=green, BG=yellow", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(base_rgb)
        axes[2].imshow(pred_rgba)
        axes[2].set_title(
            f"Prediction at threshold {threshold:.2f}\n"
            f"pred+={int(pred.sum())} ({100.0*float(pred.sum())/max(1, valid_count):.1f}% valid)",
            fontsize=9,
        )
        axes[2].axis("off")

        axes[3].imshow(base_rgb)
        axes[3].imshow(err_rgba)
        axes[3].set_title(
            "FP/FN map\n"
            f"FP={int(fp.sum())}, FN={int(fn.sum())}, TP={int(tp.sum())}, TN={int(tn.sum())}",
            fontsize=9,
        )
        axes[3].axis("off")

        fig.suptitle(mt.tile_id, fontsize=9)

        desc = (
            f"Metrics on valid GT pixels: Dice={m['dice']:.3f}, IoU={m['iou']:.3f}, "
            f"Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}, Accuracy={m['accuracy']:.3f}. "
            "CHM colors follow SLD (0-1.3 m display range)."
        )
        fig.text(
            0.01,
            0.03,
            desc,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "alpha": 0.90, "edgecolor": "#777"},
        )

        path = out_dir / f"overlay_{len(written) + 1:02d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

        rows.append(
            {
                "overlay_file": path.name,
                "tile_id": mt.tile_id,
                "threshold": float(threshold),
                "valid_pixels": int(valid_count),
                "chm_min_m": chm_min,
                "chm_mean_m": chm_mean,
                "chm_p95_m": chm_p95,
                "chm_max_m": chm_max,
                "tp": int(tp.sum()),
                "fp": int(fp.sum()),
                "fn": int(fn.sum()),
                "tn": int(tn.sum()),
                "dice": float(m["dice"]),
                "iou": float(m["iou"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "accuracy": float(m["accuracy"]),
            }
        )

    metrics_csv = out_dir / "overlay_metrics.csv"
    if rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return written


def run_one_iteration(
    cfg: IterationConfig,
    all_onboarding_tiles: list[OnboardingTile],
    manual_tiles: list[ManualValTile],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    tiles = [t for t in all_onboarding_tiles if t.variant in cfg.variants]
    x, y, w, material_stats = materialize_training_arrays(
        tiles=tiles,
        tile_size=args.tile_size,
        positive_mask_mode=cfg.positive_mask_mode,
        seed=args.seed,
        max_per_class=cfg.max_per_class,
    )

    train_ds = WeakLabelArrayDataset(images=x, targets=y, weights=w, augment=cfg.augment)
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, cfg.batch_size),
        shuffle=True,
        num_workers=max(0, args.num_workers),
        drop_last=False,
    )

    model = create_model(encoder_name=cfg.encoder_name, aspp_rates=(6, 12, 18)).to(device)
    criterion = WeightedDiceFocalLoss(
        dice_weight=cfg.dice_weight,
        focal_weight=cfg.focal_weight,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = max(1, cfg.epochs * len(train_loader))
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=20.0,
        final_div_factor=50.0,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dice = -1.0
    no_improve = 0
    patience = 3

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["target"].to(device)
            wgt = batch["weight"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img)
                loss = criterion(logits, tgt, wgt)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()

            running += float(loss.detach().item())
            steps += 1

        train_loss = running / max(1, steps)
        val05, _ = evaluate_manual_tiles(
            model=model,
            manual_tiles=manual_tiles,
            device=device,
            tile_size=args.tile_size,
            batch_size=max(1, cfg.batch_size),
            threshold=0.5,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_dice@0.5": float(val05["dice"]),
            "val_f1@0.5": float(val05["f1"]),
            "val_iou@0.5": float(val05["iou"]),
        }
        history.append(row)
        print(
            f"[{cfg.name}] epoch={epoch:02d} train_loss={train_loss:.5f} "
            f"val_dice@0.5={val05['dice']:.4f} val_f1@0.5={val05['f1']:.4f}"
        )

        if val05["dice"] > best_dice + 1e-6:
            best_dice = float(val05["dice"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    val05, probs = evaluate_manual_tiles(
        model=model,
        manual_tiles=manual_tiles,
        device=device,
        tile_size=args.tile_size,
        batch_size=max(1, cfg.batch_size),
        threshold=0.5,
    )
    best_thr, sweep = threshold_sweep(
        probs_by_id=probs,
        manual_tiles=manual_tiles,
        thresholds=[round(float(x), 2) for x in np.arange(0.20, 0.91, 0.05)],
    )
    val_best, _ = evaluate_manual_tiles(
        model=model,
        manual_tiles=manual_tiles,
        device=device,
        tile_size=args.tile_size,
        batch_size=max(1, cfg.batch_size),
        threshold=float(best_thr["threshold"]),
    )

    ckpt_path = out_dir / "best_model.pt"
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "best_threshold": float(best_thr["threshold"]),
        },
        ckpt_path,
    )

    overlays = save_overlays(
        out_dir=out_dir / "overlays",
        manual_tiles=manual_tiles,
        probs_by_id=probs,
        threshold=float(best_thr["threshold"]),
        max_count=args.overlay_count,
    )

    report = {
        "iteration": asdict(cfg),
        "material_stats": material_stats,
        "train_samples_variant_filtered": len(tiles),
        "history": history,
        "metrics": {
            "val_at_0p5": val05,
            "best_threshold": best_thr,
            "val_at_best_threshold": val_best,
            "threshold_sweep": sweep,
        },
        "outputs": {
            "checkpoint": str(ckpt_path),
            "overlay_files": overlays,
        },
    }
    (out_dir / "iteration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def render_summary_markdown(
    args: argparse.Namespace,
    onboarding_stats: dict[str, Any],
    manual_tiles: list[ManualValTile],
    reports: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Onboarding -> Harmonized Segmentation Experiment")
    lines.append("")
    lines.append("## Objective")
    lines.append(
        "Train weakly supervised segmentation on onboarding tile labels mapped to harmonized CHM (raw/gauss), "
        "and validate on manually brushed masks."
    )
    lines.append("")
    lines.append("## Data")
    lines.append(f"- Onboarding label rows scanned: {onboarding_stats['rows_total']}")
    lines.append(f"- Onboarding rows mapped after filters: {onboarding_stats['rows_mapped']}")
    lines.append(f"- Rows skipped (no harmonized map): {onboarding_stats['rows_skipped_no_map']}")
    lines.append(f"- Rows skipped (source filter): {onboarding_stats['rows_skipped_source']}")
    lines.append(f"- Rows skipped (manual-overlap leakage filter): {onboarding_stats['rows_skipped_overlap']}")
    lines.append(f"- Validation manual mask tiles used: {len(manual_tiles)}")
    lines.append("")
    lines.append("## Iterations")
    lines.append("")
    lines.append("| Iteration | Variants | Weak Mask Mode | Train Samples | Val Dice@0.5 | Val Dice@BestThr | Best Thr |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for rep in reports:
        cfg = rep["iteration"]
        m05 = rep["metrics"]["val_at_0p5"]
        mb = rep["metrics"]["val_at_best_threshold"]
        bthr = rep["metrics"]["best_threshold"]
        lines.append(
            f"| {cfg['name']} | {','.join(cfg['variants'])} | {cfg['positive_mask_mode']} | "
            f"{rep['material_stats']['materialized']} | {m05['dice']:.4f} | {mb['dice']:.4f} | {bthr['threshold']:.2f} |"
        )

    lines.append("")
    lines.append("## Critical Notes")
    lines.append(
        "- Best Model Search V3 classifier (ConvNeXt-Small) is a tile classifier; it was not forced as segmentation backbone here, "
        "to avoid architecture mismatch and CHM-domain transfer assumptions."
    )
    lines.append(
        "- Training supervision is weak (tile labels), so segmentation quality is bounded by weak-mask heuristic quality."
    )
    lines.append(
        f"- Manual validation set is small ({len(manual_tiles)} usable tiles in this run); "
        "use this for model selection trend, not final performance claims."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("Run command used:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "docker run --rm -v \"$PWD\":/workspace -w /workspace lamapuit-dev bash -lc \"source /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true; conda activate cwd-detect >/dev/null 2>&1 || true; PYTHONPATH=/workspace/src python scripts/run_onboarding_harmonized_seg_experiment.py\""
    )
    lines.append("```")
    lines.append("")
    lines.append("## Next Experiment Suggestions")
    lines.append("1. Retrain the tile classifier on harmonized CHM (raw/gauss) before generating weak masks.")
    lines.append("2. Replace q80 weak positives with CAM/attention-derived soft masks.")
    lines.append("3. Expand manual masks to >100 tiles and reserve an untouched test subset.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manual_tiles = load_manual_validation_tiles(args.manual_mask_dir, args.chm_root)
    if len(manual_tiles) < 4:
        raise RuntimeError(f"Too few manual validation tiles loaded: {len(manual_tiles)}")

    # Load onboarding tiles once with both variants; per-iteration filtering is applied later.
    onboarding_tiles, onboarding_stats = load_onboarding_tiles(
        labels_dir=args.labels_dir,
        chm_root=args.chm_root,
        variants=("raw", "gauss"),
        manual_tiles=manual_tiles,
        max_rows_per_file=args.max_rows_per_file,
    )
    if len(onboarding_tiles) < 200:
        raise RuntimeError(f"Too few onboarding tiles after mapping/filtering: {len(onboarding_tiles)}")

    iterations = [
        IterationConfig(
            name="iter1_raw_fulltile",
            variants=("raw",),
            positive_mask_mode="full_tile",
            encoder_name="resnet18",
            max_per_class=700,
            epochs=2,
            batch_size=8,
            lr=3e-4,
            weight_decay=1e-4,
            dice_weight=1.0,
            focal_weight=1.0,
            augment=True,
        ),
        IterationConfig(
            name="iter2_raw_gauss_fulltile",
            variants=("raw", "gauss"),
            positive_mask_mode="full_tile",
            encoder_name="resnet18",
            max_per_class=900,
            epochs=2,
            batch_size=8,
            lr=2.5e-4,
            weight_decay=1e-4,
            dice_weight=1.0,
            focal_weight=1.0,
            augment=True,
        ),
        IterationConfig(
            name="iter3_raw_gauss_q80",
            variants=("raw", "gauss"),
            positive_mask_mode="q80_sparse",
            encoder_name="resnet18",
            max_per_class=1000,
            epochs=3,
            batch_size=8,
            lr=2e-4,
            weight_decay=1e-4,
            dice_weight=1.2,
            focal_weight=1.0,
            augment=True,
        ),
    ]

    reports: list[dict[str, Any]] = []
    for cfg in iterations:
        iter_out = args.output_dir / cfg.name
        iter_out.mkdir(parents=True, exist_ok=True)
        print(f"=== Running {cfg.name} ===")
        rep = run_one_iteration(
            cfg=cfg,
            all_onboarding_tiles=onboarding_tiles,
            manual_tiles=manual_tiles,
            args=args,
            device=device,
            out_dir=iter_out,
        )
        reports.append(rep)

    summary = {
        "args": {
            "labels_dir": str(args.labels_dir),
            "chm_root": str(args.chm_root),
            "manual_mask_dir": str(args.manual_mask_dir),
            "output_dir": str(args.output_dir),
            "tile_size": int(args.tile_size),
            "seed": int(args.seed),
            "device": str(device),
        },
        "onboarding_stats": onboarding_stats,
        "manual_validation_tiles": len(manual_tiles),
        "iterations": reports,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = render_summary_markdown(
        args=args,
        onboarding_stats=onboarding_stats,
        manual_tiles=manual_tiles,
        reports=reports,
    )
    (args.output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print("Experiment complete")
    print(f"Summary JSON: {args.output_dir / 'summary.json'}")
    print(f"Summary MD: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
