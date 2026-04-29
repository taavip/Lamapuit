"""
PyTorch Dataset for tiled CHM images with YOLO-format segmentation labels.

Reads PNG tiles (grayscale L-mode) and converts YOLO polygon annotations
to Mask R-CNN targets:
  boxes  – float32 [N, 4] xyxy pixel coords
  masks  – uint8   [N, H, W] binary instance masks
  labels – int64   [N]  (all 1 = CDW class)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet mean/std used for all timm pretrained models.
# Single-channel grayscale is replicated to 3 ch before normalisation.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_to_tensor = T.ToTensor()  # uint8 H×W×C → float32 C×H×W  [0,1]
_normalize = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


def _yolo_poly_to_mask(
    poly_coords: np.ndarray,  # shape (N, 2) normalised [0,1]
    h: int,
    w: int,
) -> np.ndarray:
    """Fill a polygon on an H×W binary uint8 canvas."""
    pts = (poly_coords * np.array([w, h])).astype(np.int32)
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas


def _parse_yolo_seg_label(
    txt_path: Path | str,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a YOLO segmentation .txt file.

    Returns
    -------
    boxes  : float32 [N, 4]  xyxy pixel
    masks  : uint8   [N, H, W]
    labels : int64   [N]
    """
    lines = Path(txt_path).read_text().strip().splitlines()
    boxes_list, masks_list, labels_list = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:  # need at least class + 3 xy pairs
            continue
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        mask = _yolo_poly_to_mask(coords, h, w)
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_list.append([x1, y1, x2, y2])
        masks_list.append(mask)
        labels_list.append(cls + 1)  # 0 = background in Mask R-CNN

    if not boxes_list:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, h, w), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.array(boxes_list, dtype=np.float32),
        np.stack(masks_list, axis=0).astype(np.uint8),
        np.array(labels_list, dtype=np.int64),
    )


def _load_image_as_rgb_tensor(img_path: Path | str) -> torch.Tensor:
    """Load PNG (grayscale or RGB) → float32 [3, H, W] in [0,1], ImageNet-normalised."""
    img = Image.open(img_path)
    if img.mode == "L":
        img = img.convert("RGB")  # replicate single channel × 3
    elif img.mode != "RGB":
        img = img.convert("RGB")
    tensor = _to_tensor(img)  # [3, H, W] float32 [0,1]
    return _normalize(tensor)


class TileDataset(Dataset):
    """Dataset of image tiles with YOLO segmentation labels.

    Parameters
    ----------
    images_dir : directory containing *.png files
    labels_dir : directory containing *.txt YOLO seg files (same stems)
    tile_size   : expected H & W (default 640)
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        tile_size: int = 640,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.tile_size = tile_size

        self.samples: list[tuple[Path, Path | None]] = []
        for img_path in sorted(self.images_dir.glob("*.png")):
            lbl_path = self.labels_dir / (img_path.stem + ".txt")
            self.samples.append((img_path, lbl_path if lbl_path.exists() else None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path, lbl_path = self.samples[idx]
        h = w = self.tile_size

        image = _load_image_as_rgb_tensor(img_path)

        if lbl_path is not None:
            boxes, masks, labels = _parse_yolo_seg_label(lbl_path, h, w)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros((0,), dtype=np.int64)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            # Mask R-CNN needs area for COCO eval; compute from box
            "area": (
                torch.as_tensor(
                    (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                    dtype=torch.float32,
                )
                if len(boxes)
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    """Collate variable-size target dicts — required by torchvision detectors."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets
