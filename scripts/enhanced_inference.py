#!/usr/bin/env python
"""
Enhanced inference with ensemble predictions and geometric post-processing for CDW detection.

Features:
- Ensemble of multiple trained models
- Geometric constraints (size, shape, aspect ratio)
- Confidence threshold optimization
- Non-Maximum Suppression (NMS) tuning
- Post-processing filters

Usage:
    python scripts/enhanced_inference.py --chm path/to/chm.tif --models model1.pt model2.pt model3.pt --output results/
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import cv2
import rasterio
from typing import List, Tuple
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class GeometricConstraints:
    """Apply geometric constraints specific to CDW detection."""

    def __init__(
        self,
        min_area_m2: float = 5.0,  # Minimum CDW area
        max_area_m2: float = 500.0,  # Maximum CDW area
        min_length_m: float = 2.0,  # Minimum length
        max_width_m: float = 5.0,  # Maximum width
        min_aspect_ratio: float = 1.5,  # Minimum length/width ratio
        max_solidity: float = 0.8,  # Max convexity (CDW should be elongated)
    ):
        self.min_area_m2 = min_area_m2
        self.max_area_m2 = max_area_m2
        self.min_length_m = min_length_m
        self.max_width_m = max_width_m
        self.min_aspect_ratio = min_aspect_ratio
        self.max_solidity = max_solidity

    def filter_polygon(self, polygon: Polygon, pixel_size: float) -> bool:
        """Check if polygon meets CDW geometric constraints."""
        # Area constraint
        area_m2 = polygon.area * (pixel_size**2)
        if area_m2 < self.min_area_m2 or area_m2 > self.max_area_m2:
            return False

        # Bounding box constraints
        minx, miny, maxx, maxy = polygon.bounds
        length = max((maxx - minx), (maxy - miny)) * pixel_size
        width = min((maxx - minx), (maxy - miny)) * pixel_size

        if length < self.min_length_m:
            return False

        if width > self.max_width_m:
            return False

        # Aspect ratio (elongation)
        if width > 0:
            aspect_ratio = length / width
            if aspect_ratio < self.min_aspect_ratio:
                return False

        # Solidity (convexity) - CDW should be elongated, not circular
        convex_hull = polygon.convex_hull
        if convex_hull.area > 0:
            solidity = polygon.area / convex_hull.area
            if solidity > self.max_solidity:
                return False

        return True


class EnsembleInference:
    """Ensemble inference with multiple YOLO models."""

    def __init__(self, model_paths: List[str], device: str = "0", conf_threshold: float = 0.15):
        from ultralytics import YOLO

        self.models = []
        print(f"Loading {len(model_paths)} models for ensemble...")
        for i, model_path in enumerate(model_paths, 1):
            print(f"  [{i}/{len(model_paths)}] Loading {Path(model_path).name}")
            model = YOLO(model_path)
            self.models.append(model)

        self.device = device
        self.conf_threshold = conf_threshold
        print(f"✓ Ensemble ready with {len(self.models)} models")

    def predict_ensemble(self, image: np.ndarray, iou_threshold: float = 0.3) -> dict:
        """
        Run ensemble prediction and merge results.

        Args:
            image: Input image
            iou_threshold: IoU threshold for merging overlapping predictions

        Returns:
            Dictionary with merged masks and confidence scores
        """
        all_masks = []
        all_scores = []

        # Get predictions from all models
        for model in self.models:
            results = model.predict(
                image,
                conf=self.conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
            )

            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

                all_masks.extend(masks)
                all_scores.extend(scores)

        if len(all_masks) == 0:
            return {"masks": [], "scores": []}

        # Merge overlapping predictions with weighted voting
        merged_masks, merged_scores = self._merge_predictions(all_masks, all_scores, iou_threshold)

        return {"masks": merged_masks, "scores": merged_scores}

    def _merge_predictions(
        self, masks: List[np.ndarray], scores: List[float], iou_threshold: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Merge overlapping predictions using weighted voting."""
        if len(masks) == 0:
            return [], []

        # Convert to binary masks
        binary_masks = [(m > 0.5).astype(np.uint8) for m in masks]

        # Cluster overlapping masks
        merged_masks = []
        merged_scores = []
        used = set()

        for i, mask_i in enumerate(binary_masks):
            if i in used:
                continue

            # Find overlapping masks
            cluster_masks = [mask_i]
            cluster_scores = [scores[i]]
            used.add(i)

            for j, mask_j in enumerate(binary_masks):
                if j <= i or j in used:
                    continue

                # Calculate IoU
                intersection = np.logical_and(mask_i, mask_j).sum()
                union = np.logical_or(mask_i, mask_j).sum()

                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        cluster_masks.append(mask_j)
                        cluster_scores.append(scores[j])
                        used.add(j)

            # Merge cluster with weighted voting
            if len(cluster_masks) > 1:
                # Weight by confidence scores
                weights = np.array(cluster_scores)
                weights = weights / weights.sum()

                merged = np.zeros_like(cluster_masks[0], dtype=float)
                for mask, weight in zip(cluster_masks, weights):
                    merged += mask.astype(float) * weight

                merged = (merged > 0.5).astype(np.uint8)
            else:
                merged = cluster_masks[0]

            merged_masks.append(merged)
            merged_scores.append(np.mean(cluster_scores))

        return merged_masks, merged_scores


def process_chm_with_ensemble(
    chm_path: str,
    model_paths: List[str],
    output_path: str,
    tile_size: int = 640,
    overlap: float = 0.3,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.3,
    device: str = "0",
    apply_geometric_constraints: bool = True,
):
    """
    Process CHM with ensemble models and geometric post-processing.
    """
    print(f"\n{'='*70}")
    print(f"ENHANCED CDW DETECTION WITH ENSEMBLE")
    print(f"{'='*70}")
    print(f"CHM: {chm_path}")
    print(f"Models: {len(model_paths)}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Geometric constraints: {apply_geometric_constraints}")
    print(f"{'='*70}\n")

    # Initialize ensemble
    ensemble = EnsembleInference(model_paths, device, conf_threshold)

    # Initialize geometric constraints
    geo_constraints = GeometricConstraints() if apply_geometric_constraints else None

    # Open CHM
    with rasterio.open(chm_path) as src:
        chm_data = src.read(1)
        transform = src.transform
        crs = src.crs
        pixel_size = src.res[0]
        nodata = src.nodata or -9999.0

        print(f"CHM shape: {chm_data.shape}")
        print(f"Pixel size: {pixel_size:.2f}m")

        # Process with sliding window
        stride = int(tile_size * (1 - overlap))
        height, width = chm_data.shape

        all_geometries = []
        all_confidences = []

        n_tiles_x = (width - tile_size) // stride + 1
        n_tiles_y = (height - tile_size) // stride + 1
        total_tiles = n_tiles_x * n_tiles_y

        print(f"\nProcessing {total_tiles} tiles...")

        tile_count = 0
        for row in range(0, height - tile_size + 1, stride):
            for col in range(0, width - tile_size + 1, stride):
                tile_count += 1

                # Extract tile
                tile = chm_data[row : row + tile_size, col : col + tile_size]

                # Skip tiles with too much nodata
                nodata_mask = (tile == nodata) | (tile < 0)
                if nodata_mask.sum() / nodata_mask.size > 0.95:
                    continue

                # Normalize tile
                valid = tile[~nodata_mask]
                if len(valid) > 0 and valid.max() > valid.min():
                    tile_norm = np.clip((tile - valid.min()) / (valid.max() - valid.min()), 0, 1)
                    tile_img = (tile_norm * 255).astype(np.uint8)
                else:
                    continue

                tile_img[nodata_mask] = 0
                tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2RGB)

                # Run ensemble prediction
                pred = ensemble.predict_ensemble(tile_rgb, iou_threshold)

                if len(pred["masks"]) == 0:
                    continue

                # Convert masks to geometries
                for mask, score in zip(pred["masks"], pred["scores"]):
                    # Resize mask to tile size if needed
                    if mask.shape != (tile_size, tile_size):
                        mask = cv2.resize(mask, (tile_size, tile_size))

                    # Find contours
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        if len(contour) < 3:
                            continue

                        # Convert to image coordinates
                        points = contour.squeeze()
                        if len(points.shape) == 1:
                            continue

                        # Transform to world coordinates
                        world_coords = []
                        for x, y in points:
                            world_x = transform[2] + (col + x) * transform[0]
                            world_y = transform[5] + (row + y) * transform[4]
                            world_coords.append((world_x, world_y))

                        if len(world_coords) < 3:
                            continue

                        polygon = Polygon(world_coords)

                        # Apply geometric constraints
                        if geo_constraints and not geo_constraints.filter_polygon(
                            polygon, pixel_size
                        ):
                            continue

                        all_geometries.append(polygon)
                        all_confidences.append(score)

                if tile_count % 10 == 0:
                    print(
                        f"  Processed {tile_count}/{total_tiles} tiles, found {len(all_geometries)} CDW candidates"
                    )

    print(f"\n✓ Total CDW detections: {len(all_geometries)}")

    # Save results
    if len(all_geometries) > 0:
        gdf = gpd.GeoDataFrame({"confidence": all_confidences}, geometry=all_geometries, crs=crs)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG")
        print(f"✓ Saved results to: {output_path}")

        # Print statistics
        print(f"\nDETECTION STATISTICS:")
        print(f"  Total detections: {len(gdf)}")
        print(f"  Mean confidence: {gdf['confidence'].mean():.3f}")
        print(f"  Min confidence: {gdf['confidence'].min():.3f}")
        print(f"  Max confidence: {gdf['confidence'].max():.3f}")
    else:
        print("⚠ No detections found")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced CDW inference with ensemble and post-processing"
    )
    parser.add_argument("--chm", required=True, help="Path to CHM raster")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to trained model weights")
    parser.add_argument("--output", required=True, help="Output GeoPackage path")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size")
    parser.add_argument("--overlap", type=float, default=0.3, help="Tile overlap")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument("--device", default="0", help="Device (cpu or cuda)")
    parser.add_argument("--no-geometric", action="store_true", help="Disable geometric constraints")

    args = parser.parse_args()

    process_chm_with_ensemble(
        chm_path=args.chm,
        model_paths=args.models,
        output_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        apply_geometric_constraints=not args.no_geometric,
    )


if __name__ == "__main__":
    main()
