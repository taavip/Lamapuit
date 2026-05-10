"""Gaussian-weighted sliding-window inference with multi-channel input and 8-fold TTA.

Core sliding_window_predict ported from scripts/train_deeplabv3plus_manual_masks.py
lines 627-690, extended to handle (C, H, W) inputs and batched TTA.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_TTA_VARIANTS = [
    (0, False),   # identity
    (0, True),    # hflip
    (1, False),   # rot90
    (1, True),    # rot90 + hflip
    (2, False),   # rot180
    (2, True),    # rot180 + hflip
    (3, False),   # rot270
    (3, True),    # rot270 + hflip
]


def _gaussian_window(size: int) -> np.ndarray:
    """Hanning-windowed 2-D weight for overlap-add reconstruction."""
    w = np.hanning(size).astype(np.float32)
    g = np.outer(w, w).astype(np.float32)
    return np.maximum(g, 1e-3)


def _apply_tta(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply rotation and optional horizontal flip to a (B, C, H, W) batch."""
    if k:
        x = torch.rot90(x, k=k, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


def _invert_tta(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Invert the TTA transform applied to a (B, 1, H, W) prediction."""
    if flip:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, k=(4 - k) % 4, dims=[-2, -1])
    return x


@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int | None = None,
    batch_size: int = 8,
    use_tta: bool = False,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Sliding-window probability map for a multi-channel raster image.

    Args:
        image: Float32 array, shape (C, H, W) or (H, W) for single-channel.
        stride: Pixels between patch starts; defaults to patch_size // 2 (50% overlap).
        use_tta: If True, average 8-fold TTA predictions.
        valid_mask: Optional (H, W) boolean mask; patches with zero valid coverage
            are skipped. If use_tta, predictions are zeroed outside the intersection
            of all 8 transformed valid masks.

    Returns:
        Float32 probability map of shape (H, W).
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    c, h, w = image.shape
    stride = stride if stride is not None else patch_size // 2

    pad_h = (stride - ((h - patch_size) % stride)) % stride if h > patch_size else patch_size - h
    pad_w = (stride - ((w - patch_size) % stride)) % stride if w > patch_size else patch_size - w
    pad_h, pad_w = max(0, pad_h), max(0, pad_w)

    img_p = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    H, W = img_p.shape[1], img_p.shape[2]

    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    weight = _gaussian_window(patch_size)
    tta_variants = _TTA_VARIANTS if use_tta else [_TTA_VARIANTS[0]]
    n_aug = len(tta_variants)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            patch = img_p[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
            patches.append(patch)
            coords.append((y0, x0))

    model.eval()
    for aug_k, aug_flip in tta_variants:
        aug_prob_sum = np.zeros((H, W), dtype=np.float32)
        aug_weight_sum = np.zeros((H, W), dtype=np.float32)

        for i in range(0, len(patches), batch_size):
            batch_np = np.stack(patches[i : i + batch_size], axis=0)
            batch_t = torch.from_numpy(batch_np).float().to(device)
            batch_aug = _apply_tta(batch_t, aug_k, aug_flip)

            logits = model(batch_aug)
            probs_t = torch.sigmoid(logits)
            probs_t = _invert_tta(probs_t, aug_k, aug_flip)
            probs_np = probs_t[:, 0].detach().cpu().numpy()

            for j, prob in enumerate(probs_np):
                y0, x0 = coords[i + j]
                aug_prob_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += prob * weight
                aug_weight_sum[y0 : y0 + patch_size, x0 : x0 + patch_size] += weight

        with np.errstate(invalid="ignore"):
            aug_out = aug_prob_sum / np.maximum(aug_weight_sum, 1e-6)

        prob_sum += aug_out
        weight_sum += 1.0

    out = prob_sum / np.maximum(weight_sum, 1e-6)
    return out[:h, :w]
