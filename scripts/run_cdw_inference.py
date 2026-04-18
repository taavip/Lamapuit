#!/usr/bin/env python
"""
CDW Detection Inference Script.

Run YOLO instance segmentation on CHM rasters with proper tiling
and georeferenced output.
"""

import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, box, Polygon
from shapely.affinity import translate
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional


class CDWDetector:
    """Run CDW detection on CHM rasters with tiled inference."""

    def __init__(
        self,
        model_path: str,
        tile_size: int = 640,
        overlap: float = 0.50,  # 50 % overlap: guarantees 64 m objects fully in ≥1 tile
        confidence: float = 0.25,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to YOLO model weights
            tile_size: Tile size for inference
            overlap: Overlap between tiles
            confidence: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def detect(
        self,
        chm_path: str,
        output_gpkg: str = None,
        output_dir: str = None,
        save_visualizations: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Run CDW detection on a CHM raster.

        Args:
            chm_path: Path to CHM GeoTIFF
            output_gpkg: Path to output GeoPackage (optional)
            output_dir: Directory for visualizations (optional)
            save_visualizations: Whether to save tile visualizations

        Returns:
            GeoDataFrame with detected CDW polygons
        """
        chm_path = Path(chm_path)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        all_detections = []

        with rasterio.open(chm_path) as src:
            crs = src.crs
            transform = src.transform
            pixel_size = abs(transform.a)
            width, height = src.width, src.height
            nodata = src.nodata if src.nodata is not None else -9999.0

            # Calculate tile grid
            n_cols = max(1, (width - self.tile_size) // self.stride + 1)
            n_rows = max(1, (height - self.tile_size) // self.stride + 1)

            print(f"Processing: {chm_path.name}")
            print(
                f"Raster size: {width}x{height} ({width*pixel_size:.0f}m x {height*pixel_size:.0f}m)"
            )
            print(f"Tiles: {n_cols}x{n_rows} = {n_cols * n_rows}")

            tile_idx = 0

            for row in tqdm(range(n_rows), desc="Processing rows"):
                for col in range(n_cols):
                    row_off = min(row * self.stride, height - self.tile_size)
                    col_off = min(col * self.stride, width - self.tile_size)

                    # Read tile
                    window = Window(col_off, row_off, self.tile_size, self.tile_size)
                    tile_data = src.read(1, window=window)

                    # Check for valid data
                    nodata_mask = np.isnan(tile_data) | (tile_data == nodata) | (tile_data < 0)
                    if nodata_mask.sum() / nodata_mask.size > 0.9:
                        continue

                    # Get tile bounds for georeferencing
                    tile_bounds = rasterio.windows.bounds(window, transform)

                    # Normalize tile
                    tile_img = self._normalize_tile(tile_data, nodata_mask)

                    # Run inference
                    detections = self._detect_tile(tile_img, tile_bounds, pixel_size)

                    if detections:
                        all_detections.extend(detections)

                        # Save visualization if requested
                        if save_visualizations and output_dir:
                            self._save_visualization(
                                tile_img,
                                detections,
                                tile_bounds,
                                output_dir / f"tile_{tile_idx:04d}.png",
                            )

                    tile_idx += 1

        # Convert to GeoDataFrame
        if all_detections:
            gdf = gpd.GeoDataFrame(all_detections, crs=crs)

            # Apply NMS across tiles to remove duplicates
            gdf = self._apply_spatial_nms(gdf)

            # Add instance IDs
            gdf["instance_id"] = range(len(gdf))
            gdf["source_raster"] = chm_path.name
            gdf["detection_date"] = datetime.now().strftime("%Y-%m-%d")

            # Save to GeoPackage if requested
            if output_gpkg:
                gdf.to_file(output_gpkg, driver="GPKG", layer="cdw_detections")
                print(f"\nDetections saved to: {output_gpkg}")

            print(f"\nTotal CDW detected: {len(gdf)}")
            return gdf
        else:
            print("\nNo CDW detected")
            return gpd.GeoDataFrame(columns=["geometry", "confidence", "area_m2"], crs=crs)

    def _normalize_tile(
        self,
        tile_data: np.ndarray,
        nodata_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Normalise tile to 0-255 uint8 using percentile stretch + CLAHE.

        Matches the preprocessing applied during training dataset preparation.
        """
        tile_clean = tile_data.copy().astype(np.float32)
        tile_clean[nodata_mask] = np.nan

        valid = tile_clean[~nodata_mask]

        if len(valid) > 10:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                tile_norm = (tile_clean - p2) / (p98 - p2)
            else:
                vmin, vmax = float(valid.min()), float(valid.max())
                tile_norm = (tile_clean - vmin) / (vmax - vmin + 1e-6)
            tile_norm[nodata_mask] = 0.0  # suppress NaN → clean cast
            tile_img = (np.clip(tile_norm, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            tile_img = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)

        tile_img[nodata_mask] = 0

        # CLAHE for enhanced local contrast (same as training preprocessing)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tile_img = clahe.apply(tile_img)
        tile_img[nodata_mask] = 0

        # Convert to 3-channel for YOLO
        tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2RGB)
        return tile_rgb

    def _detect_tile(
        self,
        tile_img: np.ndarray,
        tile_bounds: Tuple[float, float, float, float],
        pixel_size: float,
    ) -> List[Dict]:
        """Run detection on a single tile and georeference results."""
        minx, miny, maxx, maxy = tile_bounds
        tile_width = maxx - minx
        tile_height = maxy - miny

        # Run YOLO inference
        results = self.model.predict(
            source=tile_img,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []

        for r in results:
            if r.masks is None:
                continue

            # Process each detection
            for i, (mask, box_data) in enumerate(zip(r.masks.xy, r.boxes)):
                if len(mask) < 3:
                    continue

                # Convert normalized mask coordinates to georeferenced
                mask_coords = np.array(mask)

                # Scale from pixel to geographic coordinates
                geo_x = minx + (mask_coords[:, 0] / self.tile_size) * tile_width
                geo_y = maxy - (mask_coords[:, 1] / self.tile_size) * tile_height  # Flip Y

                # Create polygon
                coords = list(zip(geo_x, geo_y))
                if len(coords) < 3:
                    continue

                try:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)  # Fix invalid geometry
                    if poly.is_empty or poly.area < 0.1:  # Min 0.1 m²
                        continue

                    detections.append(
                        {
                            "geometry": poly,
                            "confidence": float(box_data.conf),
                            "area_m2": poly.area,
                        }
                    )
                except Exception as e:
                    continue

        return detections

    def _apply_spatial_nms(
        self,
        gdf: gpd.GeoDataFrame,
        iou_threshold: float = 0.40,  # slightly lower than 0.5 to catch offset duplicates
    ) -> gpd.GeoDataFrame:
        """Apply spatial NMS to remove duplicate detections across tiles."""
        if len(gdf) <= 1:
            return gdf

        # Sort by confidence
        gdf = gdf.sort_values("confidence", ascending=False).reset_index(drop=True)

        keep = []
        suppressed = set()

        for i in range(len(gdf)):
            if i in suppressed:
                continue

            keep.append(i)
            geom_i = gdf.iloc[i].geometry

            for j in range(i + 1, len(gdf)):
                if j in suppressed:
                    continue

                geom_j = gdf.iloc[j].geometry

                # Calculate IoU
                if geom_i.intersects(geom_j):
                    intersection = geom_i.intersection(geom_j).area
                    union = geom_i.area + geom_j.area - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > iou_threshold:
                        suppressed.add(j)

        return gdf.iloc[keep].reset_index(drop=True)

    def _save_visualization(
        self,
        tile_img: np.ndarray,
        detections: List[Dict],
        tile_bounds: Tuple[float, float, float, float],
        output_path: Path,
    ):
        """Save tile with detection overlay."""
        vis_img = tile_img.copy()
        minx, miny, maxx, maxy = tile_bounds
        tile_width = maxx - minx
        tile_height = maxy - miny

        for det in detections:
            poly = det["geometry"]
            conf = det["confidence"]

            # Convert geo coordinates back to pixel
            coords = np.array(poly.exterior.coords)
            px_x = ((coords[:, 0] - minx) / tile_width * self.tile_size).astype(int)
            px_y = ((maxy - coords[:, 1]) / tile_height * self.tile_size).astype(int)

            pts = np.column_stack([px_x, px_y]).reshape((-1, 1, 2))

            # Draw filled polygon with transparency
            overlay = vis_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

            # Draw contour
            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)

            # Add confidence label
            centroid = poly.centroid
            cx = int((centroid.x - minx) / tile_width * self.tile_size)
            cy = int((maxy - centroid.y) / tile_height * self.tile_size)
            cv2.putText(
                vis_img, f"{conf:.2f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        cv2.imwrite(str(output_path), vis_img)


def main():
    parser = argparse.ArgumentParser(description="Run CDW detection on CHM rasters")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights")
    parser.add_argument("--chm", required=True, help="Path to CHM GeoTIFF")
    parser.add_argument("--output", help="Output GeoPackage path")
    parser.add_argument("--vis-dir", help="Directory for visualization outputs")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size")
    parser.add_argument(
        "--overlap", type=float, default=0.50, help="Tile overlap (0.5 recommended for 44 m trees)"
    )

    args = parser.parse_args()

    detector = CDWDetector(
        model_path=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        confidence=args.conf,
    )

    gdf = detector.detect(
        chm_path=args.chm,
        output_gpkg=args.output,
        output_dir=args.vis_dir,
        save_visualizations=args.vis_dir is not None,
    )

    return gdf


if __name__ == "__main__":
    main()
