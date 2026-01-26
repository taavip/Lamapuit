"""
CDW detection inference using trained YOLO model.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import cv2
from pathlib import Path
from tqdm import tqdm


class CDWDetector:
    """Detect coarse woody debris in CHM rasters using YOLO segmentation."""
    
    def __init__(
        self,
        model_path: str,
        tile_size: int = 640,
        stride: int = 480,
        confidence: float = 0.15,
        iou_threshold: float = 0.5,
        min_area_m2: float = 0.5,
        device: str = 'cpu',
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained YOLO model (.pt)
            tile_size: Sliding window size in pixels
            stride: Stride for sliding window (< tile_size for overlap)
            confidence: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS
            min_area_m2: Minimum detection area in m²
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.stride = stride
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_area_m2 = min_area_m2
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with memory cleanup."""
        import gc
        import torch
        
        # Clear any lingering memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
    
    def detect(self, chm_path: str, output_path: str = None) -> gpd.GeoDataFrame:
        """
        Run detection on a CHM raster.
        
        Args:
            chm_path: Path to CHM GeoTIFF
            output_path: Optional path to save GeoPackage
            
        Returns:
            GeoDataFrame with detected CDW polygons
        """
        detections = []
        
        with rasterio.open(chm_path) as src:
            crs = src.crs or 'EPSG:3301'
            transform = src.transform
            width, height = src.width, src.height
            nodata = src.nodata or -9999.0
            
            # Calculate tiles
            n_cols = max(1, (width - self.tile_size) // self.stride + 1)
            n_rows = max(1, (height - self.tile_size) // self.stride + 1)
            total = n_cols * n_rows
            
            pbar = tqdm(total=total, desc="Detecting")
            
            for row in range(n_rows):
                for col in range(n_cols):
                    row_off = min(row * self.stride, height - self.tile_size)
                    col_off = min(col * self.stride, width - self.tile_size)
                    
                    # Read tile
                    window = Window(col_off, row_off, self.tile_size, self.tile_size)
                    tile = src.read(1, window=window)
                    
                    # Handle nodata
                    nodata_mask = np.isnan(tile) | (tile == nodata) | (tile < 0)
                    if nodata_mask.sum() / nodata_mask.size > 0.5:
                        pbar.update(1)
                        continue
                    
                    # Normalize
                    tile_clean = tile.copy()
                    tile_clean[nodata_mask] = 0
                    valid = tile_clean[~nodata_mask]
                    
                    if len(valid) == 0 or valid.max() <= valid.min():
                        pbar.update(1)
                        continue
                    
                    tile_norm = np.clip((tile_clean - valid.min()) / (valid.max() - valid.min()), 0, 1)
                    tile_img = (tile_norm * 255).astype(np.uint8)
                    tile_img[nodata_mask] = 0
                    tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2RGB)
                    
                    # Run inference
                    results = self.model.predict(
                        tile_rgb,
                        conf=self.confidence,
                        iou=self.iou_threshold,
                        verbose=False,
                        device=self.device,
                    )
                    
                    # Process results
                    if results and results[0].masks is not None:
                        masks = results[0].masks.data.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        
                        for i in range(len(masks)):
                            mask = masks[i]
                            conf = float(confs[i])
                            
                            # Resize mask
                            if mask.shape != (self.tile_size, self.tile_size):
                                mask = cv2.resize(mask, (self.tile_size, self.tile_size))
                            
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            
                            for contour in contours:
                                if len(contour) < 3:
                                    continue
                                
                                # Convert to geo coordinates
                                geo_coords = []
                                for pt in contour:
                                    px, py = pt[0]
                                    geo_x = transform.c + (col_off + px) * transform.a
                                    geo_y = transform.f + (row_off + py) * transform.e
                                    geo_coords.append((geo_x, geo_y))
                                
                                if len(geo_coords) < 3:
                                    continue
                                
                                try:
                                    poly = Polygon(geo_coords)
                                    if poly.area >= self.min_area_m2:
                                        detections.append({
                                            'geometry': poly.simplify(transform.a * 0.5),
                                            'confidence': conf,
                                            'area_m2': poly.area,
                                        })
                                except:
                                    continue
                    
                    pbar.update(1)
            
            pbar.close()
        
        if not detections:
            print("No detections found")
            return gpd.GeoDataFrame(columns=['geometry', 'confidence', 'area_m2'], crs=crs)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(detections, crs=crs)
        
        # Apply NMS across tiles
        gdf = self._apply_nms(gdf)
        
        # Add metrics
        gdf['length_m'] = gdf.geometry.length
        gdf['perimeter_m'] = gdf.geometry.apply(lambda g: g.exterior.length)
        
        # Save if path provided
        if output_path:
            gdf.to_file(output_path, driver='GPKG')
            print(f"Saved {len(gdf)} detections to {output_path}")
        
        return gdf
    
    def _apply_nms(self, gdf: gpd.GeoDataFrame, iou_thresh: float = 0.4) -> gpd.GeoDataFrame:
        """Apply non-maximum suppression across overlapping detections."""
        if len(gdf) == 0:
            return gdf
        
        tree = STRtree(gdf.geometry)
        keep = set(range(len(gdf)))
        
        for idx in range(len(gdf)):
            if idx not in keep:
                continue
            
            candidates = list(tree.query(gdf.iloc[idx].geometry))
            
            for other_idx in candidates:
                if other_idx == idx or other_idx not in keep:
                    continue
                
                inter = gdf.iloc[idx].geometry.intersection(gdf.iloc[other_idx].geometry).area
                union = gdf.iloc[idx].geometry.union(gdf.iloc[other_idx].geometry).area
                
                if union > 0 and inter / union > iou_thresh:
                    # Keep higher confidence
                    if gdf.iloc[idx]['confidence'] > gdf.iloc[other_idx]['confidence']:
                        keep.discard(other_idx)
                    else:
                        keep.discard(idx)
                        break
        
        return gdf.iloc[list(keep)].reset_index(drop=True)
