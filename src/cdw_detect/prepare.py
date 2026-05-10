"""
Training data preparation for YOLO CDW detection.
Converts vector line annotations to YOLO segmentation format.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box, Polygon
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
import csv


class YOLODataPreparer:
    """Prepare YOLO segmentation training data from line vectors and CHM rasters."""
    
    def __init__(
        self,
        output_dir: str,
        tile_size: int = 640,
        buffer_width: float = 0.5,
        overlap: float = 0.2,
        min_log_pixels: int = 50,
        val_split: float = 0.2,
    ):
        """
        Initialize the data preparer.
        
        Args:
            output_dir: Output directory for YOLO dataset
            tile_size: Size of image tiles in pixels (640 for YOLO)
            buffer_width: Buffer width in meters for line-to-polygon (0.5m = 1m total width)
            overlap: Overlap fraction between tiles (0.2 = 20%)
            min_log_pixels: Minimum pixels for a valid log mask
            val_split: Fraction of data for validation
        """
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.buffer_width = buffer_width
        self.overlap = overlap
        self.min_log_pixels = min_log_pixels
        self.val_split = val_split
        self.metadata = []
        
        self._setup_directories()
        
    def _setup_directories(self):
        """Create YOLO dataset directory structure."""
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
    def prepare(self, chm_path: str, labels_path: str) -> dict:
        """
        Prepare training data from CHM raster and vector labels.
        
        Args:
            chm_path: Path to CHM GeoTIFF
            labels_path: Path to GeoPackage with line labels
            
        Returns:
            Dictionary with dataset statistics
        """
        # Load data
        labels_gdf = gpd.read_file(labels_path)
        
        with rasterio.open(chm_path) as src:
            crs = src.crs
            transform = src.transform
            pixel_size = src.res[0]
            width, height = src.width, src.height
            nodata = src.nodata or -9999.0
            
            # Reproject labels if needed
            if crs is not None and labels_gdf.crs != crs:
                labels_gdf = labels_gdf.to_crs(crs)
            elif crs is None and labels_gdf.crs is not None:
                crs = labels_gdf.crs  # Use labels CRS if raster has none
            
            # Buffer lines to polygons
            labels_gdf['geometry'] = labels_gdf.geometry.buffer(self.buffer_width)
            
            # Calculate tile positions
            stride = int(self.tile_size * (1 - self.overlap))
            n_cols = max(1, (width - self.tile_size) // stride + 1)
            n_rows = max(1, (height - self.tile_size) // stride + 1)
            
            # Process tiles
            stats = {'total': 0, 'with_cdw': 0, 'empty': 0, 'skipped': 0}
            tile_idx = 0
            
            for row in tqdm(range(n_rows), desc="Processing rows"):
                for col in range(n_cols):
                    row_off = row * stride
                    col_off = col * stride
                    
                    # Clip to bounds
                    if col_off + self.tile_size > width:
                        col_off = width - self.tile_size
                    if row_off + self.tile_size > height:
                        row_off = height - self.tile_size
                    
                    # Read tile
                    window = Window(col_off, row_off, self.tile_size, self.tile_size)
                    tile_data = src.read(1, window=window)
                    
                    # Skip tiles with too much nodata
                    nodata_mask = np.isnan(tile_data) | (tile_data == nodata) | (tile_data < 0)
                    if nodata_mask.sum() / nodata_mask.size > 0.5:
                        stats['skipped'] += 1
                        continue
                    
                    # Get tile bounds
                    tile_bounds = rasterio.windows.bounds(window, transform)
                    tile_box = box(*tile_bounds)
                    
                    # Find intersecting labels
                    intersecting = labels_gdf[labels_gdf.intersects(tile_box)]
                    
                    # Create tile transform
                    tile_transform = rasterio.transform.from_bounds(
                        *tile_bounds, self.tile_size, self.tile_size
                    )
                    
                    # Normalize image
                    tile_clean = tile_data.copy()
                    tile_clean[nodata_mask] = 0
                    valid = tile_clean[~nodata_mask]
                    
                    if len(valid) > 0 and valid.max() > valid.min():
                        tile_norm = np.clip((tile_clean - valid.min()) / (valid.max() - valid.min()), 0, 1)
                        tile_img = (tile_norm * 255).astype(np.uint8)
                    else:
                        tile_img = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                    
                    tile_img[nodata_mask] = 0
                    
                    # Determine split
                    split = 'val' if np.random.random() < self.val_split else 'train'
                    
                    # Create label file
                    label_lines = []
                    has_cdw = False
                    
                    for _, label_row in intersecting.iterrows():
                        geom = label_row.geometry.intersection(tile_box)
                        if geom.is_empty:
                            continue
                        
                        # Rasterize mask
                        mask = rasterize(
                            [(geom, 1)],
                            out_shape=(self.tile_size, self.tile_size),
                            transform=tile_transform,
                            fill=0,
                            dtype=np.uint8
                        )
                        
                        if mask.sum() < self.min_log_pixels:
                            continue
                        
                        # Extract contours for YOLO format
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            if len(contour) < 3:
                                continue
                            
                            # Normalize coordinates
                            points = contour.squeeze()
                            if len(points.shape) == 1:
                                continue
                            
                            norm_points = points / self.tile_size
                            coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in norm_points)
                            label_lines.append(f'0 {coords}')
                            has_cdw = True
                    
                    # Save files
                    tile_name = f'tile_{tile_idx:05d}'
                    
                    # Save image
                    img_path = self.output_dir / 'images' / split / f'{tile_name}.png'
                    cv2.imwrite(str(img_path), tile_img)
                    
                    # Save label
                    label_path = self.output_dir / 'labels' / split / f'{tile_name}.txt'
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(label_lines))
                    
                    # Store metadata
                    self.metadata.append({
                        'tile': tile_name,
                        'split': split,
                        'col_off': col_off,
                        'row_off': row_off,
                        'bounds': tile_bounds,
                        'has_cdw': has_cdw,
                    })
                    
                    stats['total'] += 1
                    if has_cdw:
                        stats['with_cdw'] += 1
                    else:
                        stats['empty'] += 1
                    
                    tile_idx += 1
        
        # Save dataset.yaml
        self._save_dataset_yaml()
        
        # Save metadata
        self._save_metadata(crs)
        
        return stats
    
    def _save_dataset_yaml(self):
        """Save YOLO dataset.yaml file."""
        yaml_content = f"""path: {self.output_dir.resolve()}
train: images/train
val: images/val

names:
  0: cdw
"""
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
    
    def _save_metadata(self, crs):
        """Save tile metadata for georeferencing."""
        csv_path = self.output_dir / 'tile_metadata.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'tile', 'split', 'col_off', 'row_off', 
                'minx', 'miny', 'maxx', 'maxy', 'crs', 'has_cdw'
            ])
            writer.writeheader()
            for m in self.metadata:
                writer.writerow({
                    'tile': m['tile'],
                    'split': m['split'],
                    'col_off': m['col_off'],
                    'row_off': m['row_off'],
                    'minx': m['bounds'][0],
                    'miny': m['bounds'][1],
                    'maxx': m['bounds'][2],
                    'maxy': m['bounds'][3],
                    'crs': str(crs),
                    'has_cdw': m['has_cdw'],
                })


def augment_with_nodata(dataset_dir: str, output_dir: str, nodata_fraction: float = 0.3):
    """
    Augment dataset with nodata patterns for robustness.
    
    Args:
        dataset_dir: Source dataset directory
        output_dir: Output directory for augmented dataset
        nodata_fraction: Fraction of images to augment with nodata
    """
    src = Path(dataset_dir)
    dst = Path(output_dir)
    
    # Copy original dataset
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    
    # Augment train images
    train_dir = dst / 'images' / 'train'
    images = list(train_dir.glob('*.png'))
    n_augment = int(len(images) * nodata_fraction)
    
    for img_path in np.random.choice(images, n_augment, replace=False):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Random nodata pattern
        pattern = np.random.choice(['edge', 'corner', 'strip', 'random'])
        mask = np.ones_like(img, dtype=bool)
        h, w = img.shape
        
        if pattern == 'edge':
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            size = np.random.randint(h // 4, h // 2)
            if edge == 'top':
                mask[:size, :] = False
            elif edge == 'bottom':
                mask[-size:, :] = False
            elif edge == 'left':
                mask[:, :size] = False
            else:
                mask[:, -size:] = False
        elif pattern == 'corner':
            size = np.random.randint(h // 3, h // 2)
            corner = np.random.randint(0, 4)
            if corner == 0:
                mask[:size, :size] = False
            elif corner == 1:
                mask[:size, -size:] = False
            elif corner == 2:
                mask[-size:, :size] = False
            else:
                mask[-size:, -size:] = False
        elif pattern == 'strip':
            if np.random.random() > 0.5:
                start = np.random.randint(0, w - w // 4)
                width = np.random.randint(w // 6, w // 3)
                mask[:, start:start + width] = False
            else:
                start = np.random.randint(0, h - h // 4)
                height = np.random.randint(h // 6, h // 3)
                mask[start:start + height, :] = False
        else:
            mask = np.random.random((h, w)) > 0.3
        
        # Apply nodata
        img[~mask] = 0
        cv2.imwrite(str(img_path), img)
    
    print(f"Augmented {n_augment} images with nodata patterns")
