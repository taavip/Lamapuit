"""
Instance segmentation data preparation for YOLO CDW detection.

Handles polygon annotations (not lines) with:
- Source raster linking via attributes
- Combined augmentations (rotation, nodata drop, noise, flip, brightness)
- Overlap handling via z_order
- Train/val/test splits
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely import affinity
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
import csv
import json
import random
import itertools
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Augmentation primitives & pipeline
# ---------------------------------------------------------------------------


def _aug_rotate(
    img: np.ndarray, labels: List[str], angle: float, tile_size: int
) -> Tuple[np.ndarray, List[str]]:
    """Rotate image and labels by *angle* degrees around centre."""
    h, w = img.shape[:2]
    centre = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)

    new_labels: List[str] = []
    for line in labels:
        parts = line.strip().split()
        cls_id = parts[0]
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]

        # coords are normalised 0-1 → convert to pixel space
        pxs = [x * w for x in xs]
        pys = [y * h for y in ys]

        # apply rotation matrix
        rot_pxs, rot_pys = [], []
        for px, py in zip(pxs, pys):
            rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            rot_pxs.append(rx)
            rot_pys.append(ry)

        # normalise back & clip
        nx = [np.clip(px / w, 0, 1) for px in rot_pxs]
        ny = [np.clip(py / h, 0, 1) for py in rot_pys]

        if len(nx) < 3:
            continue
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(nx, ny))
        new_labels.append(f"{cls_id} {coord_str}")

    return rotated, new_labels


def _aug_drop_nodata(
    img: np.ndarray, labels: List[str], ratio: float, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """Randomly zero-out *ratio* fraction of pixels (simulate nodata)."""
    out = img.copy()
    mask = np.random.random(out.shape[:2]) < ratio
    out[mask] = 0
    return out, labels  # labels unchanged


def _aug_noise(
    img: np.ndarray, labels: List[str], sigma: float = 10.0, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """Add Gaussian noise with standard-deviation *sigma* (uint8 scale)."""
    out = img.astype(np.float32)
    out += np.random.normal(0, sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, labels


def _aug_flip_h(img: np.ndarray, labels: List[str], **_kw) -> Tuple[np.ndarray, List[str]]:
    """Horizontal flip."""
    out = cv2.flip(img, 1)
    new_labels: List[str] = []
    for line in labels:
        parts = line.strip().split()
        cls_id = parts[0]
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]
        xs = [1.0 - x for x in xs]
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        new_labels.append(f"{cls_id} {coord_str}")
    return out, new_labels


def _aug_flip_v(img: np.ndarray, labels: List[str], **_kw) -> Tuple[np.ndarray, List[str]]:
    """Vertical flip."""
    out = cv2.flip(img, 0)
    new_labels: List[str] = []
    for line in labels:
        parts = line.strip().split()
        cls_id = parts[0]
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]
        ys = [1.0 - y for y in ys]
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        new_labels.append(f"{cls_id} {coord_str}")
    return out, new_labels


def _aug_brightness(
    img: np.ndarray, labels: List[str], factor: float = 1.2, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """Multiply pixel values by *factor* (brightness jitter)."""
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, labels


def _aug_blur(
    img: np.ndarray, labels: List[str], ksize: int = 3, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """Gaussian blur with kernel size *ksize* (simulates sensor/motion blur)."""
    if ksize % 2 == 0:
        ksize += 1
    out = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return out, labels


def _aug_gamma(
    img: np.ndarray, labels: List[str], gamma: float = 1.5, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """Gamma correction: >1 brightens shadows, <1 compresses highlights."""
    inv_gamma = 1.0 / gamma
    table = np.array([int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(img, table)
    return out, labels


def _aug_contrast(
    img: np.ndarray, labels: List[str], alpha: float = 1.5, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """Contrast stretch around mean: alpha>1 increases contrast."""
    mean_val = float(np.mean(img))
    out = np.clip((img.astype(np.float32) - mean_val) * alpha + mean_val, 0, 255).astype(np.uint8)
    return out, labels


def _aug_scale(
    img: np.ndarray, labels: List[str], factor: float = 1.3, **_kw
) -> Tuple[np.ndarray, List[str]]:
    """
    Scale augmentation.

    factor > 1  →  zoom-in:  crop central region and resize to full tile.
                   Objects appear larger; handles small CDW fragments.
    factor < 1  →  zoom-out: shrink content and pad with zeros.
                   Objects appear smaller; handles large / partially-visible CDW.
    """
    h, w = img.shape[:2]

    if factor > 1.0:
        # --- Zoom in: crop centre, resize to (w, h) ---
        crop_w = max(1, int(w / factor))
        crop_h = max(1, int(h / factor))
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = img[y1 : y1 + crop_h, x1 : x1 + crop_w]
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Transform labels: new_norm = (old_norm - x1_n) * factor
        x1_n, y1_n = x1 / w, y1 / h
        new_labels: List[str] = []
        for line in labels:
            parts = line.strip().split()
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))
            xs = [(x - x1_n) * factor for x in coords[0::2]]
            ys = [(y - y1_n) * factor for y in coords[1::2]]
            valid_pts = [
                (np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0))
                for x, y in zip(xs, ys)
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            ]
            if len(valid_pts) < 3:
                continue
            vx, vy = zip(*valid_pts)
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(vx, vy))
            new_labels.append(f"{cls_id} {coord_str}")
    else:
        # --- Zoom out: shrink and centre-pad with zeros ---
        new_w = max(1, int(w * factor))
        new_h = max(1, int(h * factor))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out = np.zeros_like(img)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        out[y1 : y1 + new_h, x1 : x1 + new_w] = small

        # Transform labels: new_norm = old_norm * factor + x1_n
        x1_n, y1_n = x1 / w, y1 / h
        new_labels = []
        for line in labels:
            parts = line.strip().split()
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))
            xs = [x * factor + x1_n for x in coords[0::2]]
            ys = [y * factor + y1_n for y in coords[1::2]]
            valid_pts = [(np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)) for x, y in zip(xs, ys)]
            if len(valid_pts) < 3:
                continue
            vx, vy = zip(*valid_pts)
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(vx, vy))
            new_labels.append(f"{cls_id} {coord_str}")

    return out, new_labels


# ---------------------------------------------------------------------------
# Registry: name → (function, default kwargs)
# ---------------------------------------------------------------------------
AUGMENTATION_REGISTRY: Dict[str, Tuple] = {
    # --- rotations (arbitrary angles help with CDW in any direction) ---
    "rotate_15": (_aug_rotate, {"angle": 15}),
    "rotate_30": (_aug_rotate, {"angle": 30}),
    "rotate_45": (_aug_rotate, {"angle": 45}),
    "rotate_90": (_aug_rotate, {"angle": 90}),
    "rotate_135": (_aug_rotate, {"angle": 135}),
    "rotate_180": (_aug_rotate, {"angle": 180}),
    "rotate_225": (_aug_rotate, {"angle": 225}),
    "rotate_270": (_aug_rotate, {"angle": 270}),
    # --- nodata drop ---
    "drop_5": (_aug_drop_nodata, {"ratio": 0.05}),
    "drop_10": (_aug_drop_nodata, {"ratio": 0.10}),
    "drop_20": (_aug_drop_nodata, {"ratio": 0.20}),
    "drop_30": (_aug_drop_nodata, {"ratio": 0.30}),
    # --- noise ---
    "noise_low": (_aug_noise, {"sigma": 5.0}),
    "noise_med": (_aug_noise, {"sigma": 10.0}),
    "noise_high": (_aug_noise, {"sigma": 20.0}),
    # --- geometric flips ---
    "flip_h": (_aug_flip_h, {}),
    "flip_v": (_aug_flip_v, {}),
    # --- photometric ---
    "bright_up": (_aug_brightness, {"factor": 1.3}),
    "bright_down": (_aug_brightness, {"factor": 0.7}),
    "blur_3": (_aug_blur, {"ksize": 3}),
    "blur_5": (_aug_blur, {"ksize": 5}),
    "gamma_up": (_aug_gamma, {"gamma": 1.5}),
    "gamma_down": (_aug_gamma, {"gamma": 0.67}),
    "contrast_hi": (_aug_contrast, {"alpha": 1.5}),
    "contrast_lo": (_aug_contrast, {"alpha": 0.7}),
    # --- scale jitter (important for CDW of varying lengths) ---
    "scale_up": (_aug_scale, {"factor": 1.3}),
    "scale_down": (_aug_scale, {"factor": 0.75}),
}


# ---------------------------------------------------------------------------
# Pre-defined combination pipelines  (55 total)
# Each pipeline is a list of augmentation names applied sequentially.
# Rule of thumb:
#   - Singles: expose the model to each transform in isolation
#   - Doubles: realistic co-occurrences (rotation + noise, scale + flip, …)
#   - Triples: hard cases combining geometric + photometric + artefacts
# ---------------------------------------------------------------------------
DEFAULT_AUGMENTATION_COMBOS: List[List[str]] = [
    # === Singles (17) ===
    ["rotate_90"],
    ["rotate_180"],
    ["rotate_270"],
    ["rotate_45"],
    ["flip_h"],
    ["flip_v"],
    ["drop_10"],
    ["drop_20"],
    ["noise_med"],
    ["bright_up"],
    ["bright_down"],
    ["scale_up"],
    ["scale_down"],
    ["gamma_up"],
    ["gamma_down"],
    ["contrast_hi"],
    ["blur_3"],
    # === Doubles (25) ===
    ["rotate_90", "drop_10"],
    ["rotate_180", "drop_20"],
    ["rotate_270", "noise_med"],
    ["rotate_45", "flip_h"],
    ["flip_h", "drop_10"],
    ["flip_h", "noise_low"],
    ["flip_v", "drop_10"],
    ["flip_v", "bright_up"],
    ["rotate_90", "bright_down"],
    ["drop_10", "noise_low"],
    ["drop_20", "noise_med"],
    ["scale_up", "flip_h"],
    ["scale_up", "rotate_90"],
    ["scale_up", "noise_low"],
    ["scale_down", "noise_med"],
    ["scale_down", "rotate_45"],
    ["gamma_up", "blur_3"],
    ["gamma_down", "noise_low"],
    ["contrast_hi", "drop_10"],
    ["contrast_lo", "noise_med"],
    ["flip_h", "gamma_up"],
    ["flip_v", "contrast_hi"],
    ["rotate_135", "drop_10"],
    ["rotate_225", "noise_low"],
    ["blur_3", "drop_10"],
    # === Triples (13) ===
    ["rotate_90", "drop_10", "noise_low"],
    ["rotate_180", "drop_20", "noise_med"],
    ["flip_h", "drop_10", "bright_up"],
    ["flip_v", "drop_20", "noise_low"],
    ["rotate_270", "drop_10", "bright_down"],
    ["scale_up", "flip_h", "noise_low"],
    ["scale_down", "rotate_90", "drop_10"],
    ["rotate_45", "gamma_up", "noise_low"],
    ["flip_h", "scale_up", "drop_10"],
    ["rotate_90", "contrast_hi", "noise_low"],
    ["flip_v", "gamma_down", "blur_3"],
    ["scale_up", "rotate_180", "bright_down"],
    ["rotate_45", "flip_v", "drop_20"],
]


def apply_augmentation_pipeline(
    img: np.ndarray,
    labels: List[str],
    pipeline: List[str],
    tile_size: int = 640,
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply a sequence of named augmentations to *img* and *labels*.

    Args:
        img: uint8 image array
        labels: YOLO label lines
        pipeline: ordered list of augmentation names from AUGMENTATION_REGISTRY
        tile_size: image side length (needed for rotation)

    Returns:
        (augmented_img, augmented_labels)
    """
    cur_img, cur_labels = img.copy(), list(labels)
    for name in pipeline:
        if name not in AUGMENTATION_REGISTRY:
            raise ValueError(f"Unknown augmentation: {name}")
        func, kwargs = AUGMENTATION_REGISTRY[name]
        cur_img, cur_labels = func(cur_img, cur_labels, tile_size=tile_size, **kwargs)
    return cur_img, cur_labels


# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Statistics for the prepared dataset."""

    total_tiles: int = 0
    positive_tiles: int = 0
    negative_tiles: int = 0
    skipped_nodata: int = 0
    total_instances: int = 0
    augmented_tiles: int = 0
    augmentation_breakdown: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})

    def to_dict(self) -> dict:
        return {
            "total_tiles": self.total_tiles,
            "positive_tiles": self.positive_tiles,
            "negative_tiles": self.negative_tiles,
            "skipped_nodata": self.skipped_nodata,
            "total_instances": self.total_instances,
            "augmented_tiles": self.augmented_tiles,
            "augmentation_breakdown": self.augmentation_breakdown,
            "splits": self.splits,
        }


class InstanceSegmentationPreparer:
    """
    Prepare YOLO instance segmentation data from polygon annotations.

    Features:
    - Links annotations to source rasters via source_raster attribute
    - Handles overlapping instances via z_order
    - Generates nodata augmented versions
    - Proper train/val/test splits
    """

    def __init__(
        self,
        output_dir: str,
        tile_size: int = 640,
        overlap: float = 0.50,
        min_instance_pixels: int = 50,
        val_split: float = 0.20,
        test_split: float = 0.10,
        include_negative_samples: bool = True,
        negative_ratio: float = 0.20,
        negative_exclusion_radius: float = 40.0,
        nodata_augment_ratios: List[float] = None,
        augmentation_combos: List[List[str]] = None,
        max_aug_per_tile: int = 15,
        random_seed: int = 42,
    ):
        """
        Initialize the instance segmentation data preparer.

        Args:
            output_dir: Output directory for YOLO dataset
            tile_size: Size of image tiles in pixels (640 for YOLO)
            overlap: Overlap fraction between tiles (0.25 = 25%)
            min_instance_pixels: Minimum pixels for a valid instance
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            include_negative_samples: Include tiles without CDW
            negative_ratio: Ratio of negative samples to positive
            nodata_augment_ratios: Legacy – list of nodata fractions
                (converted to drop_X combos automatically if
                augmentation_combos is not given)
            augmentation_combos: List of augmentation pipelines.  Each
                pipeline is a list of augmentation names from
                AUGMENTATION_REGISTRY.  When provided, this takes
                priority over nodata_augment_ratios.
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap = overlap
        # stride guarantees any object ≤ tile*(1-overlap) px is fully visible in ≥1 tile
        # With overlap=0.5 and tile=640: guaranteed for objects ≤320px = 64 m @ 20cm/px
        self.stride = int(tile_size * (1 - overlap))
        self.min_instance_pixels = min_instance_pixels
        self.val_split = val_split
        self.test_split = test_split
        self.include_negative_samples = include_negative_samples
        self.negative_ratio = negative_ratio
        # Tiles whose centre falls within this radius (metres) of any labeled polygon
        # are excluded from the negative pool – they may contain unlabeled CDW.
        self.negative_exclusion_radius = negative_exclusion_radius
        self.max_aug_per_tile = max_aug_per_tile
        self.random_seed = random_seed

        # Resolve augmentation specification
        if augmentation_combos is not None:
            self.augmentation_combos = augmentation_combos
        elif nodata_augment_ratios:
            # Legacy fallback: convert old-style ratios to combo pipelines
            self.augmentation_combos = [
                [f"drop_{int(r * 100)}"]
                for r in nodata_augment_ratios
                if f"drop_{int(r * 100)}" in AUGMENTATION_REGISTRY
            ]
        else:
            self.augmentation_combos = []
        # keep for metadata / backward compat
        self.nodata_augment_ratios = nodata_augment_ratios or []

        self.metadata = []
        self.stats = DatasetStats()

        random.seed(random_seed)
        np.random.seed(random_seed)

        self._setup_directories()

    def _setup_directories(self):
        """Create YOLO dataset directory structure."""
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def prepare_from_gpkg(
        self,
        labels_gpkg: str,
        chm_dir: str,
        layer_name: str = "cdw_labels_examples",
    ) -> DatasetStats:
        """
        Prepare dataset from GeoPackage with source_raster attribute.

        Args:
            labels_gpkg: Path to GeoPackage with polygon labels
            chm_dir: Directory containing CHM rasters
            layer_name: Layer name in GeoPackage

        Returns:
            DatasetStats with preparation statistics
        """
        labels_gpkg = Path(labels_gpkg)
        chm_dir = Path(chm_dir)

        if not labels_gpkg.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_gpkg}")
        if not chm_dir.exists():
            raise FileNotFoundError(f"CHM directory not found: {chm_dir}")

        # Load labels
        labels_gdf = gpd.read_file(labels_gpkg, layer=layer_name)

        if len(labels_gdf) == 0:
            raise ValueError(f"No labels found in layer '{layer_name}'")

        # Check for source_raster column
        if "source_raster" not in labels_gdf.columns:
            raise ValueError("Labels must have 'source_raster' column")

        # Ensure instance_id
        if "instance_id" not in labels_gdf.columns:
            labels_gdf["instance_id"] = range(len(labels_gdf))

        # Group labels by source raster
        raster_groups = labels_gdf.groupby("source_raster")

        all_tiles = []

        for raster_name, group_gdf in tqdm(raster_groups, desc="Processing rasters"):
            raster_path = chm_dir / raster_name

            if not raster_path.exists():
                print(f"WARNING: Raster not found: {raster_path}")
                continue

            # Process this raster
            tiles = self._process_raster(raster_path, group_gdf)
            all_tiles.extend(tiles)

        if len(all_tiles) == 0:
            raise ValueError("No valid tiles generated!")

        # -----------------------------------------------------------------
        # Instance-based train / val / test split
        # -----------------------------------------------------------------
        # Tiles that share a CDW instance must all go to the same split so
        # that validation metrics are honest (no leakage from overlapping tiles
        # that contain the same real-world CDW object).
        # For negative tiles (no instances) we fall through to random split.
        # -----------------------------------------------------------------
        self._assign_splits_by_instance(all_tiles)

        # Save tiles
        print(f"\nSaving {len(all_tiles)} tiles...")
        for tile in tqdm(all_tiles, desc="Saving"):
            self._save_tile(tile)

        # Generate combined augmentations (only for train)
        if self.augmentation_combos:
            self._generate_combined_augmentations(all_tiles)

        # Save dataset files
        self._save_dataset_yaml()
        self._save_metadata()
        self._save_statistics()

        return self.stats

    def _process_raster(
        self,
        raster_path: Path,
        labels_gdf: gpd.GeoDataFrame,
    ) -> List[dict]:
        """Process a single raster and its associated labels."""
        tiles = []

        with rasterio.open(raster_path) as src:
            crs = src.crs
            transform = src.transform
            pixel_size = abs(transform.a)
            width, height = src.width, src.height
            nodata = src.nodata if src.nodata is not None else -9999.0

            # Reproject labels if needed
            if crs and labels_gdf.crs and labels_gdf.crs != crs:
                labels_gdf = labels_gdf.to_crs(crs)

            # Ensure z_order exists for overlap handling
            if "z_order" not in labels_gdf.columns:
                labels_gdf["z_order"] = 0

            # Build spatial index
            labels_gdf.sindex

            # Calculate tile grid
            n_cols = max(1, (width - self.tile_size) // self.stride + 1)
            n_rows = max(1, (height - self.tile_size) // self.stride + 1)

            for row in range(n_rows):
                for col in range(n_cols):
                    row_off = min(row * self.stride, height - self.tile_size)
                    col_off = min(col * self.stride, width - self.tile_size)

                    window = Window(col_off, row_off, self.tile_size, self.tile_size)
                    tile_bounds = rasterio.windows.bounds(window, transform)
                    tile_box = box(*tile_bounds)

                    # Read tile
                    tile_data = src.read(1, window=window)

                    # Check nodata
                    nodata_mask = np.isnan(tile_data) | (tile_data == nodata) | (tile_data < 0)
                    nodata_fraction = nodata_mask.sum() / nodata_mask.size

                    if nodata_fraction > 0.7:
                        self.stats.skipped_nodata += 1
                        continue

                    # Find intersecting instances
                    candidates = labels_gdf.sindex.query(tile_box)
                    intersecting = labels_gdf.iloc[candidates]
                    intersecting = intersecting[intersecting.intersects(tile_box)]

                    # Create tile transform for rasterization
                    tile_transform = rasterio.transform.from_bounds(
                        *tile_bounds, self.tile_size, self.tile_size
                    )

                    # Normalize image
                    tile_img = self._normalize_tile(tile_data, nodata_mask)

                    # Process instances
                    label_lines, n_instances = self._process_instances(
                        intersecting, tile_box, tile_bounds, tile_transform
                    )

                    # Decide whether to include negative samples
                    is_positive = n_instances > 0

                    if not is_positive and not self.include_negative_samples:
                        continue

                    tiles.append(
                        {
                            "raster": raster_path.name,
                            "row_off": row_off,
                            "col_off": col_off,
                            "bounds": tile_bounds,
                            "crs": str(crs),
                            "image": tile_img,
                            "labels": label_lines,
                            "n_instances": n_instances,
                            "nodata_fraction": nodata_fraction,
                            "is_positive": is_positive,
                        }
                    )

        # Balance negative samples with spatial exclusion
        positive = [t for t in tiles if t["is_positive"]]
        candidate_neg = [t for t in tiles if not t["is_positive"]]

        # Collect all labeled polygon centroids for distance filtering
        label_centroids = (
            np.array([[g.centroid.x, g.centroid.y] for g in labels_gdf.geometry])
            if len(labels_gdf) > 0
            else np.empty((0, 2))
        )

        safe_negative = []
        for t in candidate_neg:
            if len(label_centroids) > 0:
                # Tile centre in map coordinates
                minx, miny, maxx, maxy = t["bounds"]
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                dists = np.sqrt(
                    (label_centroids[:, 0] - cx) ** 2 + (label_centroids[:, 1] - cy) ** 2
                )
                if dists.min() < self.negative_exclusion_radius:
                    # This tile is too close to a label – may have unlabeled CDW
                    continue
            safe_negative.append(t)

        n_negative_keep = int(len(positive) * self.negative_ratio)
        if len(safe_negative) > n_negative_keep:
            safe_negative = random.sample(safe_negative, n_negative_keep)

        return positive + safe_negative

    def _normalize_tile(
        self,
        tile_data: np.ndarray,
        nodata_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize tile to 0-255 uint8 with percentile stretch + CLAHE.

        Percentile stretch (p2–p98) is more robust than min-max because it
        avoids single extreme pixels dominating the dynamic range.
        CLAHE then enhances local contrast, making low-lying CDW (0.2–0.5 m
        above neighbouring ground) more visually distinct.
        """
        tile_clean = tile_data.copy().astype(np.float32)
        tile_clean[nodata_mask] = np.nan

        valid = tile_clean[~nodata_mask]

        if len(valid) > 10:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                tile_norm = (tile_clean - p2) / (p98 - p2)
            else:
                tile_norm = (tile_clean - float(valid.min())) / (
                    float(valid.max() - valid.min()) + 1e-6
                )
            tile_norm[nodata_mask] = 0.0  # suppress NaN → clean cast
            tile_img = (np.clip(tile_norm, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            tile_img = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

        tile_img[nodata_mask] = 0

        # CLAHE for enhanced local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tile_img = clahe.apply(tile_img)
        tile_img[nodata_mask] = 0  # restore zero-nodata after CLAHE

        return tile_img

    def _process_instances(
        self,
        instances_gdf: gpd.GeoDataFrame,
        tile_box: Polygon,
        tile_bounds: Tuple[float, float, float, float],
        tile_transform: rasterio.Affine,
    ) -> Tuple[List[str], int]:
        """
        Process instances for a tile, handling overlaps.

        Returns:
            Tuple of (label_lines, n_instances)
        """
        label_lines = []
        n_instances = 0

        if len(instances_gdf) == 0:
            return label_lines, n_instances

        # Sort by z_order (bottom first) for proper overlap handling
        instances_gdf = instances_gdf.sort_values("z_order", ascending=True)
        instances_list = list(instances_gdf.iterrows())

        for i, (idx, row) in enumerate(instances_list):
            geom = row.geometry

            # Clip to tile bounds
            clipped = geom.intersection(tile_box)
            if clipped.is_empty:
                continue

            # For visible-only mode: subtract higher instances
            visible_geom = clipped
            for j in range(i + 1, len(instances_list)):
                _, upper_row = instances_list[j]
                upper_geom = upper_row.geometry.intersection(tile_box)
                if not upper_geom.is_empty and visible_geom.intersects(upper_geom):
                    visible_geom = visible_geom.difference(upper_geom)

            if visible_geom.is_empty:
                continue

            # Convert to YOLO format
            lines = self._geometry_to_yolo(visible_geom, tile_bounds)
            label_lines.extend(lines)
            n_instances += len(lines)

        return label_lines, n_instances

    def _geometry_to_yolo(
        self,
        geom: Union[Polygon, MultiPolygon],
        tile_bounds: Tuple[float, float, float, float],
    ) -> List[str]:
        """Convert geometry to YOLO polygon format."""
        results = []
        minx, miny, maxx, maxy = tile_bounds
        tile_width = maxx - minx
        tile_height = maxy - miny

        # Handle geometry types
        if geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        elif geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "GeometryCollection":
            polygons = [g for g in geom.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
        else:
            return results

        for poly in polygons:
            if poly.is_empty or poly.area < 0.01:
                continue

            # Simplify if needed
            if len(poly.exterior.coords) > 100:
                poly = poly.simplify(0.1, preserve_topology=True)

            coords = np.array(poly.exterior.coords)

            if len(coords) < 4:
                continue

            # Normalize to [0, 1] relative to tile
            norm_x = (coords[:, 0] - minx) / tile_width
            norm_y = (coords[:, 1] - miny) / tile_height

            # Flip Y for image coordinates (origin top-left)
            norm_y = 1.0 - norm_y

            # Clip to valid range
            norm_x = np.clip(norm_x, 0, 1)
            norm_y = np.clip(norm_y, 0, 1)

            # Remove duplicate last point
            if np.allclose([norm_x[0], norm_y[0]], [norm_x[-1], norm_y[-1]]):
                norm_x = norm_x[:-1]
                norm_y = norm_y[:-1]

            if len(norm_x) < 3:
                continue

            # Check minimum area in pixels
            # Approximate area check
            pixel_area = poly.area / (tile_width * tile_height) * (self.tile_size**2)
            if pixel_area < self.min_instance_pixels:
                continue

            # Format: class_id x1 y1 x2 y2 ...
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(norm_x, norm_y))
            results.append(f"0 {coord_str}")

        return results

    def _save_tile(self, tile: dict):
        """Save a tile and its labels."""
        split = tile["split"]
        tile_idx = self.stats.total_tiles
        tile_name = f"tile_{tile_idx:05d}"

        # Save image
        img_path = self.output_dir / "images" / split / f"{tile_name}.png"
        cv2.imwrite(str(img_path), tile["image"])

        # Save labels
        label_path = self.output_dir / "labels" / split / f"{tile_name}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(tile["labels"]))

        # Update stats
        self.stats.total_tiles += 1
        self.stats.splits[split] += 1
        self.stats.total_instances += tile["n_instances"]

        if tile["is_positive"]:
            self.stats.positive_tiles += 1
        else:
            self.stats.negative_tiles += 1

        # Store metadata
        self.metadata.append(
            {
                "tile": tile_name,
                "split": split,
                "raster": tile["raster"],
                "col_off": tile["col_off"],
                "row_off": tile["row_off"],
                "bounds": tile["bounds"],
                "crs": tile["crs"],
                "n_instances": tile["n_instances"],
                "nodata_fraction": tile["nodata_fraction"],
                "augmented": False,
                "augment_type": None,
            }
        )

    def _assign_splits_by_instance(self, all_tiles: List[dict]):
        """
        Assign train / val / test splits so that tiles sharing the same
        CDW instance always end up in the same split.

        Algorithm
        ---------
        1. Build a mapping: instance_id → [tile indices]
        2. Collect unique instance groups and shuffle them
        3. Assign groups to splits proportionally
        4. Negative tiles (no instance) are split randomly after positive ones
        """
        # --- group positive tiles by the *set* of instance ids they contain ---
        # Each tile stores its label lines; we use the tile object identity as
        # the group seed.  Since we don't embed raw instance ids in tiles we
        # use spatial identity: tiles from the same positive source tile that
        # share (raster, row_off, col_off) are "the same object group".
        from collections import defaultdict

        pos_tiles = [t for t in all_tiles if t["is_positive"]]
        neg_tiles = [t for t in all_tiles if not t["is_positive"]]

        # Group by spatial origin (raster + position)
        pos_groups: Dict[Tuple, List[dict]] = defaultdict(list)
        for t in pos_tiles:
            key = (t["raster"], t["row_off"], t["col_off"])
            pos_groups[key].append(t)

        group_keys = list(pos_groups.keys())
        random.shuffle(group_keys)

        n_groups = len(group_keys)
        n_test_g = max(1, int(n_groups * self.test_split))
        n_val_g = max(1, int(n_groups * self.val_split))

        for i, key in enumerate(group_keys):
            if i < n_test_g:
                split = "test"
            elif i < n_test_g + n_val_g:
                split = "val"
            else:
                split = "train"
            for t in pos_groups[key]:
                t["split"] = split

        # Negative tiles: simple random split (they have no CDW anyway)
        random.shuffle(neg_tiles)
        n_neg = len(neg_tiles)
        n_neg_test = max(0, int(n_neg * self.test_split))
        n_neg_val = max(0, int(n_neg * self.val_split))
        for i, t in enumerate(neg_tiles):
            if i < n_neg_test:
                t["split"] = "test"
            elif i < n_neg_test + n_neg_val:
                t["split"] = "val"
            else:
                t["split"] = "train"

    def _generate_combined_augmentations(self, tiles: List[dict]):
        """
        Generate augmented tiles.

        For each training tile we apply a RANDOM SAMPLE of at most
        ``self.max_aug_per_tile`` pipelines from ``self.augmentation_combos``.

        Capping the number of augmentations per tile avoids the overfitting
        that occurs when the model sees the exact same CDW object 40–55 times
        with trivially different transforms.  YOLO's own per-batch random
        augmentation (mosaic, HSV, perspective, copy-paste …) covers the rest.
        """
        train_tiles = [t for t in tiles if t["split"] == "train" and t["is_positive"]]

        if len(train_tiles) == 0:
            return

        n_available = len(self.augmentation_combos)
        n_per_tile = min(self.max_aug_per_tile, n_available)

        print(
            f"\nGenerating augmentations for {len(train_tiles)} training tiles"
            f" (≤{n_per_tile} random combos each, "
            f"≈{n_per_tile * len(train_tiles)} new tiles)..."
        )

        for tile in tqdm(train_tiles, desc="Augmenting tiles"):
            # Random sample – different subset for each tile → more diversity
            selected = random.sample(self.augmentation_combos, k=n_per_tile)

            for combo in selected:
                combo_tag = "+".join(combo)
                aug_img, aug_labels = apply_augmentation_pipeline(
                    tile["image"],
                    tile["labels"],
                    combo,
                    tile_size=self.tile_size,
                )

                # Skip if all labels were lost (e.g. rotated out of frame)
                if tile["n_instances"] > 0 and len(aug_labels) == 0:
                    continue

                tile_idx = self.stats.total_tiles
                tile_name = f"tile_{tile_idx:05d}_aug_{combo_tag}"

                # Save augmented image
                img_path = self.output_dir / "images" / "train" / f"{tile_name}.png"
                cv2.imwrite(str(img_path), aug_img)

                # Save labels
                label_path = self.output_dir / "labels" / "train" / f"{tile_name}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(aug_labels))

                # Update stats
                n_inst = len(aug_labels)
                self.stats.total_tiles += 1
                self.stats.splits["train"] += 1
                self.stats.augmented_tiles += 1
                self.stats.total_instances += n_inst
                self.stats.augmentation_breakdown[combo_tag] = (
                    self.stats.augmentation_breakdown.get(combo_tag, 0) + 1
                )

                self.metadata.append(
                    {
                        "tile": tile_name,
                        "split": "train",
                        "raster": tile["raster"],
                        "col_off": tile["col_off"],
                        "row_off": tile["row_off"],
                        "bounds": tile["bounds"],
                        "crs": tile["crs"],
                        "n_instances": n_inst,
                        "nodata_fraction": tile["nodata_fraction"],
                        "augmented": True,
                        "augment_type": combo_tag,
                    }
                )

    def _save_dataset_yaml(self):
        """Save YOLO dataset.yaml file."""
        yaml_content = f"""# CDW Instance Segmentation Dataset
# Generated with nodata augmentation

path: {self.output_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: cdw

nc: 1
"""
        with open(self.output_dir / "dataset.yaml", "w") as f:
            f.write(yaml_content)

    def _save_metadata(self):
        """Save tile metadata for georeferencing."""
        csv_path = self.output_dir / "tile_metadata.csv"

        with open(csv_path, "w", newline="") as f:
            if len(self.metadata) > 0:
                fieldnames = list(self.metadata[0].keys())
                # Handle bounds tuple
                fieldnames = [fn for fn in fieldnames if fn != "bounds"]
                fieldnames.extend(["minx", "miny", "maxx", "maxy"])

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for m in self.metadata:
                    row = {k: v for k, v in m.items() if k != "bounds"}
                    if "bounds" in m:
                        row["minx"] = m["bounds"][0]
                        row["miny"] = m["bounds"][1]
                        row["maxx"] = m["bounds"][2]
                        row["maxy"] = m["bounds"][3]
                    writer.writerow(row)

    def _save_statistics(self):
        """Save dataset statistics."""
        stats_dict = self.stats.to_dict()
        stats_dict["augmentation_combos"] = ["+".join(c) for c in self.augmentation_combos]
        stats_dict["tile_size"] = self.tile_size
        stats_dict["overlap"] = self.overlap

        with open(self.output_dir / "dataset_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=2)

        print(f"\n{'='*50}")
        print("Dataset Statistics")
        print("=" * 50)
        print(f"Total tiles: {self.stats.total_tiles}")
        print(f"  - Positive (original): {self.stats.positive_tiles}")
        print(f"  - Negative: {self.stats.negative_tiles}")
        print(f"  - Augmented: {self.stats.augmented_tiles}")
        print(f"Total instances: {self.stats.total_instances}")
        print(f"Skipped (nodata): {self.stats.skipped_nodata}")
        print(
            f"Splits: train={self.stats.splits['train']}, "
            f"val={self.stats.splits['val']}, test={self.stats.splits['test']}"
        )
        if self.stats.augmentation_breakdown:
            print("Augmentation breakdown:")
            for tag, count in sorted(self.stats.augmentation_breakdown.items()):
                print(f"    {tag}: {count}")


def prepare_instance_dataset(
    labels_gpkg: str,
    chm_dir: str,
    output_dir: str,
    layer_name: str = "cdw_labels_examples",
    tile_size: int = 640,
    overlap: float = 0.50,
    nodata_ratios: List[float] = None,
    augmentation_combos: List[List[str]] = None,
    max_aug_per_tile: int = 15,
    val_split: float = 0.20,
    test_split: float = 0.10,
    negative_ratio: float = 0.20,
    negative_exclusion_radius: float = 40.0,
    random_seed: int = 42,
) -> DatasetStats:
    """
    Convenience function to prepare instance segmentation dataset.

    Args:
        labels_gpkg: Path to GeoPackage with polygon labels
        chm_dir: Directory containing CHM rasters
        output_dir: Output directory for dataset
        layer_name: Layer name in GeoPackage
        tile_size: Tile size in pixels
        overlap: Tile overlap fraction (0.5 recommended for 44 m trees)
        nodata_ratios: Legacy – list of nodata fractions (ignored when
            augmentation_combos is provided)
        augmentation_combos: List of augmentation pipelines.
            Use ``None`` (default) to apply DEFAULT_AUGMENTATION_COMBOS.
            Pass an empty list ``[]`` to disable augmentation entirely.
        max_aug_per_tile: Maximum number of augmentation pipelines randomly
            sampled per tile.  Limits near-identical copies of the same object.
        val_split: Validation split ratio
        test_split: Test split ratio
        negative_ratio: Fraction of negative tiles relative to positive tiles.
        negative_exclusion_radius: Tiles whose centre is within this many metres
            of any labeled polygon are excluded from the negative pool (they may
            contain unlabeled CDW).
        random_seed: Random seed

    Returns:
        DatasetStats object
    """
    # Resolve augmentation combos
    if augmentation_combos is None and nodata_ratios is None:
        augmentation_combos = DEFAULT_AUGMENTATION_COMBOS

    preparer = InstanceSegmentationPreparer(
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        nodata_augment_ratios=nodata_ratios,
        augmentation_combos=augmentation_combos,
        max_aug_per_tile=max_aug_per_tile,
        val_split=val_split,
        test_split=test_split,
        negative_ratio=negative_ratio,
        negative_exclusion_radius=negative_exclusion_radius,
        random_seed=random_seed,
    )

    return preparer.prepare_from_gpkg(
        labels_gpkg=labels_gpkg,
        chm_dir=chm_dir,
        layer_name=layer_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare YOLO instance segmentation dataset for CDW detection"
    )
    parser.add_argument("--labels", required=True, help="Path to labels GeoPackage")
    parser.add_argument("--chm-dir", required=True, help="Directory with CHM rasters")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--layer", default="cdw_labels_examples", help="Layer name in GeoPackage")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.50,
        help="Tile overlap 0-1 (0.5 recommended for 44 m CDW)",
    )
    parser.add_argument(
        "--nodata-ratios", type=str, default=None, help="Legacy: comma-separated nodata ratios"
    )
    parser.add_argument(
        "--use-default-combos",
        action="store_true",
        default=True,
        help="Use DEFAULT_AUGMENTATION_COMBOS (default: True)",
    )
    parser.add_argument("--no-augmentation", action="store_true", help="Disable all augmentation")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split")
    parser.add_argument("--test-split", type=float, default=0.10, help="Test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    nodata_ratios = None
    aug_combos = None

    if args.no_augmentation:
        aug_combos = []
    elif args.nodata_ratios:
        nodata_ratios = [float(x) for x in args.nodata_ratios.split(",")]
        aug_combos = None  # will use legacy path
    else:
        aug_combos = DEFAULT_AUGMENTATION_COMBOS

    stats = prepare_instance_dataset(
        labels_gpkg=args.labels,
        chm_dir=args.chm_dir,
        output_dir=args.output,
        layer_name=args.layer,
        tile_size=args.tile_size,
        overlap=args.overlap,
        nodata_ratios=nodata_ratios,
        augmentation_combos=aug_combos,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
    )

    print(f"\nDataset created at: {args.output}")
