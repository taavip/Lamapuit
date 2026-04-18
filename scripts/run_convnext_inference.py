#!/usr/bin/env python3
"""
Run tiled CDW detection with ConvNeXt V2 + Mask R-CNN on a CHM GeoTIFF.

Produces a GeoPackage of detected CDW polygons (same format as YOLO inference).

Usage
-----
python scripts/run_convnext_inference.py \
    --model   output/cdw_training_convnext/weights/best.pt \
    --chm     chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif \
    --output  output/cdw_detections_convnext_406455.gpkg \
    --conf    0.25 \
    --device  0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ImageNet normalisation – must match training
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_norm = T.Normalize(_MEAN, _STD)
_to_t = T.ToTensor()


# ──────────────────────────────────────────────────────────────────────────────
# Tile normalisation  (matches prepare_instance._normalize_tile)
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_tile(tile: np.ndarray) -> np.ndarray:
    """Float32 CHM tile → uint8 via p2-p98 stretch + CLAHE."""
    import cv2

    nodata_mask = ~np.isfinite(tile)
    tile = tile.copy()
    tile[nodata_mask] = np.nan
    valid = tile[~nodata_mask]
    if valid.size == 0:
        return np.zeros(tile.shape, dtype=np.uint8)
    p2, p98 = np.nanpercentile(tile, 2), np.nanpercentile(tile, 98)
    rng = p98 - p2
    if rng < 1e-6:
        rng = 1.0
    norm = np.clip((tile - p2) / rng, 0.0, 1.0)
    norm[nodata_mask] = 0.0
    uint8 = (norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(uint8)


def _tile_to_tensor(raw_tile: np.ndarray) -> torch.Tensor:
    """Raw float32 CHM tile → normalised [3, H, W] tensor."""
    uint8 = _normalize_tile(raw_tile)
    img = Image.fromarray(uint8, mode="L").convert("RGB")
    return _norm(_to_t(img))


# ──────────────────────────────────────────────────────────────────────────────
# NMS on georeferenced detections
# ──────────────────────────────────────────────────────────────────────────────


def _nms_polygons(
    geoms: list,
    scores: list[float],
    iou_thresh: float = 0.40,
) -> list[int]:
    """Simple polygon IoU NMS – keeps highest-score, removes overlapping."""
    if not geoms:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    removed = set()
    for i in order:
        if i in removed:
            continue
        keep.append(i)
        for j in order:
            if j <= i or j in removed:
                continue
            g1, g2 = geoms[i], geoms[j]
            try:
                inter = g1.intersection(g2).area
                union = g1.union(g2).area
                if union > 0 and inter / union > iou_thresh:
                    removed.add(j)
            except Exception:
                pass
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Main inference
# ──────────────────────────────────────────────────────────────────────────────


def run_inference(
    model_path: str | Path,
    chm_path: str | Path,
    output_path: str | Path,
    conf_thresh: float = 0.25,
    mask_thresh: float = 0.50,
    overlap: float = 0.50,
    tile_size: int = 640,
    iou_thresh: float = 0.40,
    device_id: int | str = 0,
) -> int:
    device = torch.device(
        f"cuda:{device_id}"
        if isinstance(device_id, int)
        else device_id if torch.cuda.is_available() else "cpu"
    )

    # ── load model ───────────────────────────────────────────────────────────
    logger.info("Loading model from %s", model_path)
    from cdw_detect.model_convnext import build_convnext_maskrcnn

    model = build_convnext_maskrcnn(
        num_classes=2,
        box_score_thresh=conf_thresh,
        box_nms_thresh=iou_thresh,
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── tiled inference ──────────────────────────────────────────────────────
    stride = int(tile_size * (1.0 - overlap))
    all_geoms: list = []
    all_scores: list[float] = []
    all_confs: list[float] = []

    with rasterio.open(chm_path) as src:
        crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        res_x, res_y = abs(transform.a), abs(transform.e)

        row_starts = list(range(0, height - tile_size + 1, stride))
        if not row_starts or row_starts[-1] + tile_size < height:
            row_starts.append(max(0, height - tile_size))
        col_starts = list(range(0, width - tile_size + 1, stride))
        if not col_starts or col_starts[-1] + tile_size < width:
            col_starts.append(max(0, width - tile_size))

        n_tiles = len(row_starts) * len(col_starts)
        logger.info("Processing: %s", Path(chm_path).name)
        logger.info(
            "Raster size: %dx%d (%.0fm x %.0fm)",
            width,
            height,
            width * res_x,
            height * res_y,
        )
        logger.info("Tiles: %dx%d = %d", len(col_starts), len(row_starts), n_tiles)

        processed = 0
        for row_off in row_starts:
            for col_off in col_starts:
                tile_h = min(tile_size, height - row_off)
                tile_w = min(tile_size, width - col_off)
                window = Window(col_off, row_off, tile_w, tile_h)
                raw = src.read(1, window=window).astype(np.float32)

                # Pad if edge tile is smaller
                if tile_h < tile_size or tile_w < tile_size:
                    pad = np.zeros((tile_size, tile_size), dtype=np.float32)
                    pad[:tile_h, :tile_w] = raw
                    raw = pad

                img_tensor = _tile_to_tensor(raw).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)

                dets = outputs[0]
                boxes = dets["boxes"].cpu().numpy()  # [N, 4] xyxy pixel
                scores = dets["scores"].cpu().numpy()  # [N]
                masks = dets["masks"].cpu().numpy()  # [N, 1, H, W]

                for box, score, mask_pred in zip(boxes, scores, masks):
                    if score < conf_thresh:
                        continue
                    # Binary mask from sigmoid output
                    binary = (mask_pred[0] > mask_thresh).astype(np.uint8)
                    if binary.sum() < 4:
                        continue
                    # Extract contour → polygon
                    import cv2

                    contours, _ = cv2.findContours(
                        binary[:tile_h, :tile_w],
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if not contours:
                        continue
                    cnt = max(contours, key=cv2.contourArea)
                    if len(cnt) < 4:
                        continue
                    # Pixel coords → geo coords
                    pts_px = cnt[:, 0, :]  # (M, 2) xy
                    world_pts = []
                    for px, py in pts_px:
                        gx = transform.c + (col_off + px) * transform.a
                        gy = transform.f + (row_off + py) * transform.e
                        world_pts.append((gx, gy))
                    if len(world_pts) < 4:
                        continue
                    poly = Polygon(world_pts)
                    if not poly.is_valid or poly.area < 1.0:
                        continue
                    all_geoms.append(poly)
                    all_scores.append(float(score))
                    all_confs.append(float(score))

                processed += 1

        logger.info("Processed %d tiles, %d raw detections", processed, len(all_geoms))

    # ── NMS ─────────────────────────────────────────────────────────────────
    keep = _nms_polygons(all_geoms, all_scores, iou_thresh=iou_thresh)
    logger.info("After NMS: %d detections", len(keep))

    # ── save ─────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if keep:
        gdf = gpd.GeoDataFrame(
            {"confidence": [all_confs[i] for i in keep]},
            geometry=[all_geoms[i] for i in keep],
            crs=crs,
        )
    else:
        gdf = gpd.GeoDataFrame(
            {"confidence": []},
            geometry=[],
            crs=crs,
        )
    gdf.to_file(output_path, driver="GPKG")
    logger.info("Detections saved to: %s", output_path)
    logger.info("Total CDW detected: %d", len(keep))
    return len(keep)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDW inference with ConvNeXt Mask R-CNN")
    p.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--chm", required=True, help="CHM GeoTIFF raster path")
    p.add_argument("--output", required=True, help="Output GeoPackage path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.40)
    p.add_argument("--overlap", type=float, default=0.50)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        device_id: int | str = int(args.device)
    except ValueError:
        device_id = args.device

    run_inference(
        model_path=args.model,
        chm_path=args.chm,
        output_path=args.output,
        conf_thresh=args.conf,
        overlap=args.overlap,
        iou_thresh=args.iou,
        device_id=device_id,
    )


if __name__ == "__main__":
    main()
