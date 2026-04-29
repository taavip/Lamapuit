#!/usr/bin/env python3
"""Train DeepLabV3+ (ConvNeXt-Small encoder) on manual CWD masks.

This experiment is intentionally separate from the existing PartialConv pipeline.
It reuses the best Model Search V3 ConvNeXt-Small classifier checkpoint as
encoder initialization, then fine-tunes a segmentation head on manual RGBA masks.

Key design choices for thin, elongated logs:
- DeepLabV3+ with low-level skip connections
- ASPP atrous rates tuned for 128px tiles: (6, 12, 18)
- Hybrid Dice + Focal loss with ignore-region support
- Sliding-window inference with 50% overlap (stride = tile_size / 2)
- Rotation-heavy augmentation for orientation robustness
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
except Exception:  # pragma: no cover - optional fallback
    A = None

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover - clearer runtime guidance
    raise RuntimeError(
        "segmentation_models_pytorch is required. "
        "Install in docker conda env: pip install segmentation-models-pytorch timm"
    ) from exc


EPS = 1e-8


@dataclass
class ManualTile:
    tile_id: str
    sample_id: str
    variant: str
    row_off: int
    col_off: int
    raster_path: Path
    chm: np.ndarray  # float32, HxW in [0, 1]
    target: np.ndarray  # float32, HxW in {0, 1}
    valid: np.ndarray  # float32, HxW in {0, 1}


class RandomPatchDataset(Dataset):
    def __init__(
        self,
        tiles: list[ManualTile],
        tile_size: int,
        patches_per_tile: int,
        augment: bool,
        min_labeled_pixels: int,
    ) -> None:
        self.tiles = tiles
        self.tile_size = int(tile_size)
        self.patches_per_tile = int(max(1, patches_per_tile))
        self.augment = bool(augment)
        self.min_labeled_pixels = int(max(1, min_labeled_pixels))

        self.albu = None
        if self.augment and A is not None:
            # Mask interpolation for additional targets remains nearest-neighbor.
            self.albu = A.Compose(
                [
                    A.Rotate(
                        limit=180,
                        interpolation=1,
                        border_mode=2,
                        p=0.8,
                    ),
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.10,
                        rotate_limit=20,
                        interpolation=1,
                        border_mode=2,
                        p=0.5,
                    ),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                ],
                additional_targets={"target": "mask", "valid": "mask"},
            )

    def __len__(self) -> int:
        return len(self.tiles) * self.patches_per_tile

    def _random_crop(self, tile: ManualTile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = tile.chm.shape
        ts = self.tile_size
        if h < ts or w < ts:
            raise RuntimeError(f"Tile smaller than requested patch size: {tile.tile_id}")

        max_y = h - ts
        max_x = w - ts

        for _ in range(12):
            y0 = random.randint(0, max_y) if max_y > 0 else 0
            x0 = random.randint(0, max_x) if max_x > 0 else 0
            chm = tile.chm[y0 : y0 + ts, x0 : x0 + ts]
            target = tile.target[y0 : y0 + ts, x0 : x0 + ts]
            valid = tile.valid[y0 : y0 + ts, x0 : x0 + ts]
            if int(valid.sum()) >= self.min_labeled_pixels:
                return chm, target, valid

        # Fallback to center crop if random tries did not meet label coverage.
        y0 = max(0, (h - ts) // 2)
        x0 = max(0, (w - ts) // 2)
        return (
            tile.chm[y0 : y0 + ts, x0 : x0 + ts],
            tile.target[y0 : y0 + ts, x0 : x0 + ts],
            tile.valid[y0 : y0 + ts, x0 : x0 + ts],
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        tile = self.tiles[idx % len(self.tiles)]
        chm, target, valid = self._random_crop(tile)

        if self.albu is not None:
            aug = self.albu(image=chm, target=target, valid=valid)
            chm = aug["image"]
            target = aug["target"]
            valid = aug["valid"]
        elif self.augment:
            if random.random() < 0.5:
                chm = np.flip(chm, axis=1).copy()
                target = np.flip(target, axis=1).copy()
                valid = np.flip(valid, axis=1).copy()
            if random.random() < 0.5:
                chm = np.flip(chm, axis=0).copy()
                target = np.flip(target, axis=0).copy()
                valid = np.flip(valid, axis=0).copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                chm = np.rot90(chm, k=k).copy()
                target = np.rot90(target, k=k).copy()
                valid = np.rot90(valid, k=k).copy()

        return {
            "image": torch.from_numpy(chm[None, ...]).float(),
            "target": torch.from_numpy(target[None, ...]).float(),
            "valid": torch.from_numpy(valid[None, ...]).float(),
            "tile_id": tile.tile_id,
        }


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.smooth = float(smooth)

    def _masked_focal(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - p_t).pow(self.focal_gamma)) * bce
        denom = valid.sum().clamp_min(1.0)
        return (focal * valid).sum() / denom

    def _masked_dice(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits) * valid
        truth = targets * valid
        inter = (probs * truth).sum(dim=(1, 2, 3))
        den = probs.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        focal = self._masked_focal(logits, targets, valid)
        dice = self._masked_dice(logits, targets, valid)
        return self.focal_weight * focal + self.dice_weight * dice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepLabV3+ ConvNeXt-Small transfer on manual masks")
    parser.add_argument(
        "--v3-output",
        type=Path,
        default=Path("output/model_search_v3_academic_leakage26"),
        help="Model Search V3 output root used to auto-select the best classifier checkpoint.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit classifier checkpoint. If omitted, auto-select from --v3-output.",
    )
    parser.add_argument("--mask-dir", type=Path, default=Path("output/manual_masks"))
    parser.add_argument(
        "--chm-dir",
        type=Path,
        default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/manual_mask_experiments/deeplabv3plus_convnext_transfer"),
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--aspp-rates", type=int, nargs=3, default=[6, 12, 18])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patches-per-tile", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-labeled-pixels", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--overlay-count", type=int, default=8)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_chm(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    invalid = ~np.isfinite(x)
    x[invalid] = 0.0
    # Match classifier training scale from fine_tune_cnn.py
    x = np.clip(x, 0.0, 20.0) / 20.0
    return x


def build_raster_index(chm_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for tif in chm_dir.rglob("*.tif"):
        idx[tif.stem] = tif
    return idx


def load_rgba_mask(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(mask_path).convert("RGBA"), dtype=np.uint8)
    red = rgba[..., 0]
    alpha = rgba[..., 3]

    pos = (alpha > 0) & (red >= 200)
    neg = (alpha > 0) & (red <= 50)
    valid = pos | neg

    target = pos.astype(np.float32)
    valid_f = valid.astype(np.float32)
    return target, valid_f


def read_chm_chip(raster_path: Path, row_off: int, col_off: int, h: int, w: int) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        arr = src.read(1, window=Window(col_off, row_off, w, h), boundless=True, fill_value=np.nan)
    if arr.shape != (h, w):
        fixed = np.full((h, w), np.nan, dtype=np.float32)
        hh = min(h, arr.shape[0])
        ww = min(w, arr.shape[1])
        fixed[:hh, :ww] = arr[:hh, :ww]
        arr = fixed
    return normalize_chm(arr)


def parse_manual_tiles(mask_dir: Path, chm_dir: Path, max_tiles: int = 0) -> list[ManualTile]:
    raster_index = build_raster_index(chm_dir)
    pat = re.compile(r"^(?P<sample>.+)_harmonized_dem_last_(?P<variant>raw|gauss)_chm__r(?P<row>\d+)_c(?P<col>\d+)$")

    out: list[ManualTile] = []
    for meta_path in sorted(mask_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tile_id = str(meta.get("tile_id", "")).strip()
        m = pat.match(tile_id)
        if not m:
            continue

        mask_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_mask.png"))
        if not mask_path.exists():
            continue

        raster_stem = f"{m.group('sample')}_harmonized_dem_last_{m.group('variant')}_chm"
        raster_path = raster_index.get(raster_stem)
        if raster_path is None:
            continue

        target, valid = load_rgba_mask(mask_path)
        h, w = target.shape
        chm = read_chm_chip(
            raster_path=raster_path,
            row_off=int(m.group("row")),
            col_off=int(m.group("col")),
            h=h,
            w=w,
        )

        out.append(
            ManualTile(
                tile_id=tile_id,
                sample_id=m.group("sample"),
                variant=m.group("variant"),
                row_off=int(m.group("row")),
                col_off=int(m.group("col")),
                raster_path=raster_path,
                chm=chm,
                target=target,
                valid=valid,
            )
        )

        if max_tiles > 0 and len(out) >= max_tiles:
            break

    return out


def stratified_split(tiles: list[ManualTile], val_fraction: float, seed: int) -> tuple[list[ManualTile], list[ManualTile]]:
    positives = [t for t in tiles if float((t.target * t.valid).sum()) > 0.0]
    negatives = [t for t in tiles if float((t.target * t.valid).sum()) <= 0.0]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_pos_val = max(1, int(round(len(positives) * val_fraction))) if positives else 0
    n_neg_val = max(1, int(round(len(negatives) * val_fraction))) if negatives else 0

    val = positives[:n_pos_val] + negatives[:n_neg_val]
    train = positives[n_pos_val:] + negatives[n_neg_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _suffix_score(src_key: str, dst_key: str) -> int:
    s = src_key.split(".")
    d = dst_key.split(".")
    score = 0
    for a, b in zip(reversed(s), reversed(d)):
        if a == b:
            score += 1
        else:
            break
    return score


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk in {"n_averaged", "param_averages"}:
            continue
        out[nk] = v
    return out


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for k in ("state_dict", "model_state_dict"):
            state = payload.get(k)
            if isinstance(state, dict):
                return _clean_state_dict(state)
        return _clean_state_dict(payload)
    raise RuntimeError("Unsupported checkpoint format for classifier state_dict")


def find_best_v3_classifier_checkpoint(v3_output: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = v3_output / "experiment_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing V3 summary CSV: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError(f"No rows in {summary_path}")

    ranked = df.sort_values(["mean_cv_f1", "mean_cv_auc"], ascending=[False, False]).reset_index(drop=True)

    for _, row in ranked.iterrows():
        model_file = f"{row['model_name']}_{row['data_strategy']}_{row['loss_name']}_{row['regularization']}.pt"
        final_path = v3_output / "final_models" / model_file
        if final_path.exists():
            return final_path, {
                "experiment_id": str(row["experiment_id"]),
                "model_name": str(row["model_name"]),
                "mean_cv_f1": float(row["mean_cv_f1"]),
                "mean_cv_auc": float(row["mean_cv_auc"]),
                "source": "final_models",
            }

    # Fallback: best ranked row fold1 checkpoint.
    row = ranked.iloc[0]
    ckpt = v3_output / "checkpoints" / str(row["experiment_id"]) / "fold1.pt"
    if ckpt.exists():
        return ckpt, {
            "experiment_id": str(row["experiment_id"]),
            "model_name": str(row["model_name"]),
            "mean_cv_f1": float(row["mean_cv_f1"]),
            "mean_cv_auc": float(row["mean_cv_auc"]),
            "source": "checkpoints/fold1",
        }

    raise FileNotFoundError("Unable to locate a usable checkpoint from V3 output")


def _pick_convnext_encoder_name() -> str:
    names = set(smp.encoders.get_encoder_names())

    candidates = [
        "convnext_small",
        "tu-convnext_small",  # SMP timm-universal path (works in newer SMP)
        "timm-convnext_small",
    ]
    for cand in candidates:
        if cand in names:
            return cand

    # Some SMP versions do not list tu-* names but still support them via get_encoder.
    for cand in ["tu-convnext_small", "convnext_small", "timm-convnext_small"]:
        try:
            smp.encoders.get_encoder(cand, in_channels=1, depth=5, weights=None)
            return cand
        except Exception:
            continue

    available_convnext = sorted([n for n in names if "convnext" in n.lower()])
    raise RuntimeError(
        "No ConvNeXt-Small encoder found in segmentation_models_pytorch. "
        f"Detected convnext-like encoders: {available_convnext}. "
        "Install/upgrade with: pip install -U 'segmentation-models-pytorch>=0.5.0,<0.6' timm"
    )


def create_model(aspp_rates: tuple[int, int, int]) -> tuple[nn.Module, str]:
    encoder_name = _pick_convnext_encoder_name()
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )
    try:
        model = smp.DeepLabV3Plus(decoder_atrous_rates=aspp_rates, **kwargs)
    except TypeError:
        model = smp.DeepLabV3Plus(**kwargs)
    return model, encoder_name


def transfer_classifier_to_encoder(model: nn.Module, classifier_ckpt: Path) -> dict[str, Any]:
    payload = torch.load(classifier_ckpt, map_location="cpu", weights_only=False)
    src_state = extract_state_dict(payload)
    src_state = {
        k: v
        for k, v in src_state.items()
        if not re.search(r"(^|\.)(classifier|head|fc|pre_logits|cls_head)(\.|$)", k, flags=re.IGNORECASE)
    }

    encoder_state = model.encoder.state_dict()
    merged = dict(encoder_state)
    used_dst: set[str] = set()
    mapped: dict[str, str] = {}
    unmapped_src: list[str] = []

    shape_to_dst: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for dk, dv in encoder_state.items():
        shape_to_dst[tuple(dv.shape)].append(dk)

    def _candidate_keys(src_key: str) -> list[str]:
        out = [
            src_key,
            f"model.{src_key}",
            f"backbone.{src_key}",
            src_key.replace("features.", "model.features."),
            src_key.replace("features.", "backbone.features."),
        ]
        uniq = []
        seen = set()
        for key in out:
            if key not in seen:
                uniq.append(key)
                seen.add(key)
        return uniq

    # Pass 1: direct/prefix mapping.
    for sk, sv in src_state.items():
        hit = None
        for dk in _candidate_keys(sk):
            if dk in encoder_state and dk not in used_dst and tuple(encoder_state[dk].shape) == tuple(sv.shape):
                hit = dk
                break
        if hit is not None:
            merged[hit] = sv
            used_dst.add(hit)
            mapped[sk] = hit
        else:
            unmapped_src.append(sk)

    # Pass 2: shape + suffix heuristic mapping.
    deferred_shape_order: dict[tuple[int, ...], list[str]] = defaultdict(list)
    still_unmapped: list[str] = []
    for sk in unmapped_src:
        sv = src_state[sk]
        cands = [k for k in shape_to_dst.get(tuple(sv.shape), []) if k not in used_dst]
        if not cands:
            still_unmapped.append(sk)
            continue

        scored = sorted(cands, key=lambda dk: _suffix_score(sk, dk), reverse=True)
        best_key = scored[0]
        best_score = _suffix_score(sk, best_key)

        if best_score <= 0 and len(cands) > 1:
            deferred_shape_order[tuple(sv.shape)].append(sk)
            continue

        merged[best_key] = sv
        used_dst.add(best_key)
        mapped[sk] = best_key

    # Pass 2b: for repeated shapes with no lexical overlap, map by stable order.
    for shape, src_keys in deferred_shape_order.items():
        dst_keys = sorted([k for k in shape_to_dst.get(shape, []) if k not in used_dst])
        for sk, dk in zip(sorted(src_keys), dst_keys):
            merged[dk] = src_state[sk]
            used_dst.add(dk)
            mapped[sk] = dk
        for sk in sorted(src_keys)[len(dst_keys) :]:
            still_unmapped.append(sk)

    # Pass 3: channel adaptation for first conv if needed.
    for sk in list(still_unmapped):
        sv = src_state[sk]
        if sv.ndim != 4:
            continue
        out_c, in_c, kh, kw = sv.shape
        for dk, dv in encoder_state.items():
            if dk in used_dst or dv.ndim != 4:
                continue
            if tuple(dv.shape[:1]) != (out_c,) or tuple(dv.shape[2:]) != (kh, kw):
                continue
            if dv.shape[1] == in_c:
                merged[dk] = sv
            elif in_c > dv.shape[1]:
                merged[dk] = sv.mean(dim=1, keepdim=True).repeat(1, dv.shape[1], 1, 1)
            else:
                reps = int(math.ceil(dv.shape[1] / in_c))
                merged[dk] = sv.repeat(1, reps, 1, 1)[:, : dv.shape[1], :, :]
            used_dst.add(dk)
            mapped[sk] = dk
            still_unmapped.remove(sk)
            break

    load_result = model.encoder.load_state_dict(merged, strict=False)

    return {
        "classifier_checkpoint": str(classifier_ckpt),
        "mapped_count": int(len(mapped)),
        "unmapped_source_count": int(len(still_unmapped)),
        "unmapped_source_keys": still_unmapped,
        "encoder_missing_keys": list(getattr(load_result, "missing_keys", [])),
        "encoder_unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }


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
    overlap: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected HxW single-channel image for sliding-window prediction")

    h, w = image.shape
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    if stride <= 0:
        stride = patch_size // 2

    pad_h = (stride - ((h - patch_size) % stride)) % stride if h > patch_size else patch_size - h
    pad_w = (stride - ((w - patch_size) % stride)) % stride if w > patch_size else patch_size - w
    pad_h = max(0, pad_h)
    pad_w = max(0, pad_w)

    img_p = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    H, W = img_p.shape

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    weight = _gaussian_window(patch_size)
    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            patch = img_p[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patches.append(patch)
            coords.append((y0, x0))

    model.eval()
    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i : i + batch_size], axis=0)
        batch_t = torch.from_numpy(batch[:, None, ...]).float().to(device)
        logits = model(batch_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        for j, prob in enumerate(probs):
            y0, x0 = coords[i + j]
            prob_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += prob * weight
            weight_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += weight

    out = prob_sum / np.maximum(weight_sum, 1e-6)
    return out[:h, :w]


def compute_metrics(probs: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, float]:
    v = valid > 0.5
    if not np.any(v):
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    pred = (probs >= threshold) & v
    gt = (target > 0.5) & v

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    acc = (tp + tn) / (tp + fp + fn + tn + EPS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


@torch.no_grad()
def evaluate_tiles(
    model: nn.Module,
    tiles: list[ManualTile],
    device: torch.device,
    patch_size: int,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    probs_by_tile: dict[str, np.ndarray] = {}

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    for tile in tiles:
        probs = sliding_window_predict(
            model=model,
            image=tile.chm,
            device=device,
            patch_size=patch_size,
            overlap=0.5,
            batch_size=batch_size,
        )
        probs_by_tile[tile.tile_id] = probs

        v = tile.valid > 0.5
        if not np.any(v):
            continue
        pred = (probs >= threshold) & v
        gt = (tile.target > 0.5) & v
        total_tp += float(np.logical_and(pred, gt).sum())
        total_fp += float(np.logical_and(pred, ~gt).sum())
        total_fn += float(np.logical_and(~pred, gt).sum())
        total_tn += float(np.logical_and(~pred, ~gt).sum())

    dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + EPS)
    iou = total_tp / (total_tp + total_fp + total_fn + EPS)
    precision = total_tp / (total_tp + total_fp + EPS)
    recall = total_tp / (total_tp + total_fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + EPS)

    metrics = {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }
    return metrics, probs_by_tile


def threshold_sweep(
    probs_by_tile: dict[str, np.ndarray],
    tiles: list[ManualTile],
    thresholds: list[float],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    tile_lookup = {t.tile_id: t for t in tiles}
    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for thr in thresholds:
        tp = fp = fn = tn = 0.0
        for tile_id, probs in probs_by_tile.items():
            tile = tile_lookup[tile_id]
            v = tile.valid > 0.5
            if not np.any(v):
                continue
            pred = (probs >= thr) & v
            gt = (tile.target > 0.5) & v
            tp += float(np.logical_and(pred, gt).sum())
            fp += float(np.logical_and(pred, ~gt).sum())
            fn += float(np.logical_and(~pred, gt).sum())
            tn += float(np.logical_and(~pred, ~gt).sum())

        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = (2.0 * precision * recall) / (precision + recall + EPS)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
        iou = tp / (tp + fp + fn + EPS)
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
        raise RuntimeError("Threshold sweep failed: no metrics")
    return best, rows


def save_overlay_examples(
    tiles: list[ManualTile],
    probs_by_tile: dict[str, np.ndarray],
    threshold: float,
    out_dir: Path,
    max_count: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prioritize tiles with labeled positives so thin CWD structures are visible.
    ranked = sorted(
        tiles,
        key=lambda t: float((t.target * t.valid).sum()),
        reverse=True,
    )
    chosen = ranked[: max(1, min(max_count, len(ranked)))]

    written: list[str] = []
    for i, tile in enumerate(chosen, start=1):
        probs = probs_by_tile[tile.tile_id]
        pred = (probs >= threshold).astype(np.float32)

        v = tile.valid > 0.5
        gt = np.where(v, tile.target, np.nan)
        pd = np.where(v, pred, np.nan)

        metrics = compute_metrics(probs, tile.target, tile.valid, threshold)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

        axes[0].imshow(tile.chm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("CHM")
        axes[0].axis("off")

        axes[1].imshow(tile.chm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].imshow(np.ma.masked_invalid(gt), cmap="Greens", alpha=0.65, vmin=0, vmax=1)
        axes[1].set_title("GT Mask")
        axes[1].axis("off")

        axes[2].imshow(tile.chm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2].imshow(np.ma.masked_invalid(pd), cmap="Reds", alpha=0.65, vmin=0, vmax=1)
        axes[2].set_title(f"Pred Mask (thr={threshold:.2f})")
        axes[2].axis("off")

        fp = np.logical_and(pd == 1, gt == 0)
        fn = np.logical_and(pd == 0, gt == 1)
        err = np.zeros((*tile.chm.shape, 3), dtype=np.float32)
        err[..., 0] = fp.astype(np.float32)  # red for FP
        err[..., 2] = fn.astype(np.float32)  # blue for FN
        axes[3].imshow(tile.chm, cmap="gray", vmin=0.0, vmax=1.0)
        axes[3].imshow(err, alpha=0.70)
        axes[3].set_title(f"FP/FN Dice={metrics['dice']:.3f}")
        axes[3].axis("off")

        fig.suptitle(tile.tile_id, fontsize=10)
        out_path = out_dir / f"overlay_{i:02d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(str(out_path))

    return written


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.classifier_checkpoint is not None:
        classifier_ckpt = args.classifier_checkpoint
        best_row = {
            "source": "explicit-arg",
            "path": str(classifier_ckpt),
        }
    else:
        classifier_ckpt, best_row = find_best_v3_classifier_checkpoint(args.v3_output)

    if not classifier_ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {classifier_ckpt}")

    tiles = parse_manual_tiles(args.mask_dir, args.chm_dir, max_tiles=args.max_tiles)
    if len(tiles) < 4:
        raise RuntimeError(f"Need >=4 parsed manual tiles, found {len(tiles)}")

    train_tiles, val_tiles = stratified_split(tiles, val_fraction=args.val_fraction, seed=args.seed)
    if not train_tiles or not val_tiles:
        raise RuntimeError("Train/val split is empty. Increase manual masks or adjust --val-fraction")

    train_ds = RandomPatchDataset(
        tiles=train_tiles,
        tile_size=args.tile_size,
        patches_per_tile=args.patches_per_tile,
        augment=not args.no_augment,
        min_labeled_pixels=args.min_labeled_pixels,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=max(0, args.num_workers),
        drop_last=False,
    )

    model, encoder_name = create_model(aspp_rates=tuple(int(x) for x in args.aspp_rates))
    model = model.to(device)
    transfer_report = transfer_classifier_to_encoder(model, classifier_ckpt)

    criterion = DiceFocalLoss(
        dice_weight=1.0,
        focal_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        smooth=1e-6,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history: list[dict[str, float]] = []
    best_dice = -1.0
    no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_probs: dict[str, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            valid = batch["valid"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(image)
                loss = criterion(logits, target, valid)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.detach().item())
            steps += 1

        train_loss = running / max(1, steps)
        val_metrics, val_probs = evaluate_tiles(
            model=model,
            tiles=val_tiles,
            device=device,
            patch_size=args.tile_size,
            batch_size=max(1, args.batch_size),
            threshold=0.5,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_dice@0.5": float(val_metrics["dice"]),
            "val_iou@0.5": float(val_metrics["iou"]),
            "val_f1@0.5": float(val_metrics["f1"]),
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_dice@0.5={val_metrics['dice']:.4f} "
            f"val_f1@0.5={val_metrics['f1']:.4f}"
        )

        if val_metrics["dice"] > best_dice + 1e-6:
            best_dice = float(val_metrics["dice"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = {k: np.array(v, copy=True) for k, v in val_probs.items()}
            no_improve = 0
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"early stopping at epoch {epoch} (patience={args.patience})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid best state")
    model.load_state_dict(best_state)

    # Re-evaluate best model for robust threshold selection.
    val_metrics_05, val_probs = evaluate_tiles(
        model=model,
        tiles=val_tiles,
        device=device,
        patch_size=args.tile_size,
        batch_size=max(1, args.batch_size),
        threshold=0.5,
    )
    best_probs = val_probs if best_probs is None else best_probs

    thresholds = [round(float(x), 2) for x in np.arange(0.15, 0.91, 0.05)]
    best_thr_row, sweep_rows = threshold_sweep(best_probs, val_tiles, thresholds)

    tuned_thr = float(best_thr_row["threshold"])
    tuned_metrics = evaluate_tiles(
        model=model,
        tiles=val_tiles,
        device=device,
        patch_size=args.tile_size,
        batch_size=max(1, args.batch_size),
        threshold=tuned_thr,
    )[0]

    ckpt_out = args.output_dir / "best_deeplabv3plus_convnext_transfer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_checkpoint": str(classifier_ckpt),
            "transfer_report": transfer_report,
            "config": vars(args),
            "best_threshold": tuned_thr,
        },
        ckpt_out,
    )

    overlay_dir = args.output_dir / "overlays"
    overlay_files = save_overlay_examples(
        tiles=val_tiles,
        probs_by_tile=best_probs,
        threshold=tuned_thr,
        out_dir=overlay_dir,
        max_count=args.overlay_count,
    )

    report = {
        "selected_v3_model": best_row,
        "segmentation_encoder_name": encoder_name,
        "classifier_checkpoint": str(classifier_ckpt),
        "dataset": {
            "total_tiles": int(len(tiles)),
            "train_tiles": int(len(train_tiles)),
            "val_tiles": int(len(val_tiles)),
            "positive_total": int(sum(1 for t in tiles if float((t.target * t.valid).sum()) > 0.0)),
            "positive_train": int(sum(1 for t in train_tiles if float((t.target * t.valid).sum()) > 0.0)),
            "positive_val": int(sum(1 for t in val_tiles if float((t.target * t.valid).sum()) > 0.0)),
        },
        "training": {
            "epochs_requested": int(args.epochs),
            "epochs_ran": int(len(history)),
            "history": history,
        },
        "transfer_report": transfer_report,
        "metrics": {
            "val_at_0p5": val_metrics_05,
            "best_threshold": best_thr_row,
            "val_at_best_threshold": tuned_metrics,
            "threshold_sweep": sweep_rows,
        },
        "outputs": {
            "checkpoint": str(ckpt_out),
            "overlay_dir": str(overlay_dir),
            "overlay_files": overlay_files,
        },
    }

    report_path = args.output_dir / "deeplabv3plus_manual_mask_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DeepLabV3+ manual-mask experiment complete")
    print(f"Selected V3 checkpoint: {classifier_ckpt}")
    print(f"Transfer mapped: {transfer_report['mapped_count']} params")
    print(f"Val Dice@0.5: {val_metrics_05['dice']:.4f} | Val F1@0.5: {val_metrics_05['f1']:.4f}")
    print(
        "Best threshold metrics: "
        f"thr={best_thr_row['threshold']:.2f} "
        f"dice={best_thr_row['dice']:.4f} f1={best_thr_row['f1']:.4f}"
    )
    print(f"Checkpoint: {ckpt_out}")
    print(f"Report: {report_path}")
    print(f"Overlay examples: {overlay_dir}")


if __name__ == "__main__":
    main()
