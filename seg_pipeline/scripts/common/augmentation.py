"""Augmentation pipelines for CWD segmentation training.

Ported and extended from scripts/train_deeplabv3plus_manual_masks.py lines 84-110.
All augmentations are applied jointly to (image, target_mask, valid_mask) via
albumentations additional_targets to preserve spatial consistency.
"""

from __future__ import annotations

import numpy as np

try:
    import albumentations as A
    from albumentations import ImageOnlyTransform as _ImageOnlyTransform

    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False
    _ImageOnlyTransform = object


_EXTRA_TARGETS = {"target": "mask", "valid": "mask"}


if _HAS_ALBU:
    class NodataDropout(_ImageOnlyTransform):
        """Simulate sparse/missing data by randomly zeroing pixels to nodata value.

        Simulates the effect of sparse point clouds or missing measurements on dense
        rasters. Randomly selects min_drop to max_drop fraction of valid pixels,
        zeros their values, and marks them as invalid in the valid_mask.

        This is applied to the image only; the valid_mask is then updated by the
        parent pipeline to reflect these newly invalid pixels.
        """

        def __init__(self, min_drop=0.05, max_drop=0.15, p=0.4):
            """Initialize NodataDropout.

            Args:
                min_drop: Minimum fraction of pixels to drop [0, 1]
                max_drop: Maximum fraction of pixels to drop [0, 1]
                p: Probability of applying this transform [0, 1]
            """
            super().__init__(p=p)
            self.min_drop = min_drop
            self.max_drop = max_drop

        def apply(self, img, **kwargs):
            """Zero out random pixels in the image.

            Args:
                img: Input image (H, W, C) or (H, W)
            """
            h, w = img.shape[:2]
            n_pixels = h * w
            drop_fraction = np.random.uniform(self.min_drop, self.max_drop)
            n_drop = int(n_pixels * drop_fraction)

            # Randomly select pixel indices to drop
            flat_indices = np.random.choice(n_pixels, size=n_drop, replace=False)
            rows = flat_indices // w
            cols = flat_indices % w

            # Zero out selected pixels (all channels if multi-channel)
            img_out = img.copy()
            img_out[rows, cols] = 0
            return img_out

        def get_transform_init_args_names(self):
            return ("min_drop", "max_drop", "p")
else:
    class NodataDropout:
        """Stub for when albumentations is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("albumentations is required for NodataDropout")


def get_geometric_aug() -> "A.Compose":
    """Strong geometric augmentation for rotation-invariant CWD structures.

    Identical to train_deeplabv3plus_manual_masks.py lines 86-110.
    """
    if not _HAS_ALBU:
        raise ImportError("albumentations is required — pip install albumentations")
    return A.Compose(
        [
            A.Rotate(limit=180, interpolation=1, border_mode=2, p=0.8),
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
        additional_targets=_EXTRA_TARGETS,
    )


def get_radiometric_aug() -> "A.Compose":
    """Mild radiometric augmentation for CHM intensity robustness."""
    if not _HAS_ALBU:
        raise ImportError("albumentations is required — pip install albumentations")
    return A.Compose(
        [
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.3
            ),
        ],
        additional_targets=_EXTRA_TARGETS,
    )


def get_full_aug() -> "A.Compose":
    """Combined geometric + radiometric + nodata dropout pipeline for training.

    Applies geometric transforms, radiometric transforms, and nodata dropout in
    a single Compose so all transforms share the same spatial random state,
    ensuring image/target/valid remain aligned. Nodata dropout is applied first
    to simulate sparse measurements.
    """
    if not _HAS_ALBU:
        raise ImportError("albumentations is required — pip install albumentations")
    return A.Compose(
        [
            NodataDropout(min_drop=0.05, max_drop=0.15, p=0.4),
            A.Rotate(limit=180, interpolation=1, border_mode=2, p=0.8),
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
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.3
            ),
        ],
        additional_targets=_EXTRA_TARGETS,
    )
