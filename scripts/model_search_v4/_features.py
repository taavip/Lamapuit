"""Hand-crafted classical features for CHM tiles.

V3/V4 used a thin 12-feature-per-channel bank (moments, quantiles, mean
gradient magnitudes). For academic comparability we extend this with
**texture** and **ridge** descriptors that are physically plausible for
CWD on CHM: logs on the ground appear as short, thin, locally low ridges.

Feature groups per channel (``n_ch`` channels → ``n_ch * N_PER_CH`` features):

    Moments (10)            mean, std, min, max, quantiles 10/25/50/75/90,
                            nodata-fraction (≤ 1e-6 as CHM zero-marker).
    Gradients (4)           mean |d/dx|, mean |d/dy|, std |d/dx|, std |d/dy|.
    Diagonal grads (4)      mean & std of |45°| and |135°| finite differences.
                            Combined with the axis-aligned gradients this
                            gives four orientation bins — the simplest
                            anisotropy descriptor. CWD logs have a dominant
                            orientation; isotropic scenes do not.
    Laplacian (2)           Laplacian mean and variance. Laplacian variance
                            is a classical "focus / edge density" proxy
                            that correlates with thin ridge presence.
    Morphological (2)       top-hat residual mean & std (img - open(img)).
                            Isolates small bright structures above the
                            local low-pass; thin logs survive, canopy does
                            not. Uses a 3×3 box open as a cheap stand-in
                            for a true structuring element.
    Ridgeness proxy (2)     max(|∂²/∂x²| + |∂²/∂y²|) over 3×3 neighborhoods
                            (mean & std across the tile) — captures
                            concentrated 2nd-derivative energy without
                            requiring a full Hessian eigendecomposition.

Total: 24 features per channel (2× the V3 bank), still pure NumPy so the
classical baseline remains a fair "no deep learning, no fancy libraries"
reference point.
"""

from __future__ import annotations

import numpy as np


N_FEATURES_PER_CHANNEL = 24


def _diag_diff(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """|arr[r+dr, c+dc] - arr[r, c]| with appropriate cropping."""
    n, h, w = arr.shape
    r0 = max(0, -dr); r1 = h - max(0, dr)
    c0 = max(0, -dc); c1 = w - max(0, dc)
    a = arr[:, r0:r1, c0:c1]
    b = arr[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return np.abs(a - b).reshape(n, -1)


def _box_open_3x3(a: np.ndarray) -> np.ndarray:
    """3×3 morphological opening via erosion-then-dilation; same shape."""
    n, h, w = a.shape
    pad = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    # Erosion: local min over 3x3.
    er = np.stack(
        [
            pad[:, 0:h, 0:w], pad[:, 0:h, 1:w+1], pad[:, 0:h, 2:w+2],
            pad[:, 1:h+1, 0:w], pad[:, 1:h+1, 1:w+1], pad[:, 1:h+1, 2:w+2],
            pad[:, 2:h+2, 0:w], pad[:, 2:h+2, 1:w+1], pad[:, 2:h+2, 2:w+2],
        ],
        axis=0,
    ).min(axis=0)
    # Dilation: local max over 3x3.
    pad = np.pad(er, ((0, 0), (1, 1), (1, 1)), mode="edge")
    dl = np.stack(
        [
            pad[:, 0:h, 0:w], pad[:, 0:h, 1:w+1], pad[:, 0:h, 2:w+2],
            pad[:, 1:h+1, 0:w], pad[:, 1:h+1, 1:w+1], pad[:, 1:h+1, 2:w+2],
            pad[:, 2:h+2, 0:w], pad[:, 2:h+2, 1:w+1], pad[:, 2:h+2, 2:w+2],
        ],
        axis=0,
    ).max(axis=0)
    return dl


def _laplacian(a: np.ndarray) -> np.ndarray:
    """4-connected Laplacian (same shape, edge-padded)."""
    n, h, w = a.shape
    pad = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    up = pad[:, 0:h, 1:w+1]
    down = pad[:, 2:h+2, 1:w+1]
    left = pad[:, 1:h+1, 0:w]
    right = pad[:, 1:h+1, 2:w+2]
    center = pad[:, 1:h+1, 1:w+1]
    return up + down + left + right - 4.0 * center


def _ridgeness_proxy(lap: np.ndarray) -> np.ndarray:
    """Local absolute Laplacian maxed over a 3x3 window."""
    a = np.abs(lap)
    n, h, w = a.shape
    pad = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")
    stack = np.stack(
        [
            pad[:, 0:h, 0:w], pad[:, 0:h, 1:w+1], pad[:, 0:h, 2:w+2],
            pad[:, 1:h+1, 0:w], pad[:, 1:h+1, 1:w+1], pad[:, 1:h+1, 2:w+2],
            pad[:, 2:h+2, 0:w], pad[:, 2:h+2, 1:w+1], pad[:, 2:h+2, 2:w+2],
        ],
        axis=0,
    )
    return stack.max(axis=0)


def tile_feature_matrix(x: np.ndarray) -> np.ndarray:
    """Per-tile feature vector stack. Input: (N, C, H, W) float32. Output: (N, C*24)."""
    n, c, _, _ = x.shape
    feat_cols: list[np.ndarray] = []

    for ch in range(c):
        arr = x[:, ch, :, :].astype(np.float32)
        flat = arr.reshape(n, -1)

        # Moments (10)
        feat_cols.append(flat.mean(axis=1))
        feat_cols.append(flat.std(axis=1))
        feat_cols.append(flat.min(axis=1))
        feat_cols.append(flat.max(axis=1))
        feat_cols.append(np.quantile(flat, 0.10, axis=1))
        feat_cols.append(np.quantile(flat, 0.25, axis=1))
        feat_cols.append(np.quantile(flat, 0.50, axis=1))
        feat_cols.append(np.quantile(flat, 0.75, axis=1))
        feat_cols.append(np.quantile(flat, 0.90, axis=1))
        feat_cols.append((flat <= 1e-6).mean(axis=1))

        # Axis-aligned gradients (4)
        gr = np.abs(np.diff(arr, axis=1)).reshape(n, -1)
        gc = np.abs(np.diff(arr, axis=2)).reshape(n, -1)
        feat_cols.append(gr.mean(axis=1))
        feat_cols.append(gc.mean(axis=1))
        feat_cols.append(gr.std(axis=1))
        feat_cols.append(gc.std(axis=1))

        # Diagonal gradients (4): 45° and 135° mean & std (0° / 90° already covered above).
        d45 = _diag_diff(arr, 1, 1)
        d135 = _diag_diff(arr, 1, -1)
        feat_cols.append(d45.mean(axis=1))
        feat_cols.append(d45.std(axis=1))
        feat_cols.append(d135.mean(axis=1))
        feat_cols.append(d135.std(axis=1))

        # Laplacian (2)
        lap = _laplacian(arr)
        lap_flat = lap.reshape(n, -1)
        feat_cols.append(lap_flat.mean(axis=1))
        feat_cols.append(lap_flat.var(axis=1))

        # Top-hat residual (2)
        tophat = arr - _box_open_3x3(arr)
        thf = tophat.reshape(n, -1)
        feat_cols.append(thf.mean(axis=1))
        feat_cols.append(thf.std(axis=1))

        # Ridgeness proxy (2)
        ridge = _ridgeness_proxy(lap)
        rf = ridge.reshape(n, -1)
        feat_cols.append(rf.mean(axis=1))
        feat_cols.append(rf.std(axis=1))

    return np.column_stack(feat_cols).astype(np.float32)
