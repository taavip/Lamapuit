"""Distance-transform soft target generation for thin-structure segmentation.

Replaces hard binary masks with Gaussian-decaying heatmaps so that a 1-pixel
prediction offset receives a small gradient penalty instead of a 100% miss.
Critical for CWD trunks which are 1–5 px wide at 0.2 m/px resolution.
"""

from __future__ import annotations
import numpy as np


def binary_to_soft_target(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Convert a hard binary mask to a Gaussian distance-transform heatmap.

    The centre (medial axis) of each CWD object gets value 1.0; values decay
    smoothly to 0 toward the object boundary and are 0 outside the mask.

    Args:
        mask:  Binary array (H, W), dtype float32 or uint8. 1 = CWD, 0 = background.
        sigma: Gaussian decay parameter in pixels. sigma=2 means the value halves
               ~2.8 px from the centre. Use smaller values for sharper supervision.

    Returns:
        Soft target array (H, W) float32, values in [0, 1].
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=np.float32)
    if mask.sum() == 0:
        return mask

    # Distance from each foreground pixel to the nearest background pixel
    dist = distance_transform_edt(mask)
    max_dist = dist.max()
    if max_dist == 0:
        return mask

    # Normalise to [0, 1]: centre = 1.0, edge = 0.0
    dist_norm = dist / max_dist

    # Gaussian decay: pixels at the medial axis → 1.0, edges → near 0
    soft = np.exp(-((1.0 - dist_norm) ** 2) / (2.0 * sigma ** 2))

    # Zero outside the original mask
    return (soft * mask).astype(np.float32)
