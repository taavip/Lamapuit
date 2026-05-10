#!/usr/bin/env python3
"""Phase IV V5: Instance segmentation evaluation + full-tile GeoPackage.

Evaluates YOLO11m-seg and Mask2Former on the held-out test stripe (cols 0–999),
computes instance-level AP@50, AP@75, mAP@50:95, count error, and size bias.
Generates a full-tile GeoPackage of predicted instances for QGIS inspection.

Test stripe: cols 0–999 (matching V3 test geometry for apples-to-apples comparison).
Sliding window: 640×640, stride=480 (same overlap as training patches).
NMS merge: IoU=0.5 across overlapping windows.

Comparison table:
    V3 U-Net++ (semantic): test_dice=0.192 (pixel-level only)
    V5a YOLO11m-seg:       AP@50, mAP@50:95, count error
    V5b Mask2Former:       AP@50, mAP@50:95, count error

Usage:
    python phase4_evaluate_v5.py                    # full evaluation
    python phase4_evaluate_v5.py --no-gpkg          # skip GeoPackage generation
    python phase4_evaluate_v5.py --device cpu       # CPU fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.instance_metrics import InstanceMetrics, polygon_iou, yolo_result_to_polygons
from phase2_dataset_v5 import (
    PATCH_SIZE,
    STRIDE,
    CHM_TIF,
    CHM_BANDS,
    CHM_HAG_MAX,
    LABEL_FILE,
    load_chm_as_uint8,
    load_label_pixel_polygons,
    build_spatial_index,
    get_overlapping_instances,
    grid_positions,
)

RUNS_DIR = ROOT / "seg_pipeline" / "output" / "phase3_runs_v5"
OUTPUT_DIR = ROOT / "seg_pipeline" / "output" / "phase4_report_v5"
TEST_COL_START = 0
TEST_COL_END = 1000
MIN_INSTANCE_PX = 50  # minimum pixel area for a valid predicted instance
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# CHM patch extraction for inference
# ---------------------------------------------------------------------------


def load_test_stripe_uint8() -> tuple[np.ndarray, rasterio.transform.Affine]:
    """Load CHM bands 1-3 for the test stripe as uint8."""
    with rasterio.open(CHM_TIF) as src:
        H = src.height
        img = src.read(
            CHM_BANDS,
            window=rasterio.windows.Window(TEST_COL_START, 0, TEST_COL_END - TEST_COL_START, H),
            boundless=True, fill_value=0.0,
        ).astype(np.float32)
        transform = src.transform

    img[img < 0] = 0.0
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0.0, CHM_HAG_MAX) / CHM_HAG_MAX
    img = (img * 255).astype(np.uint8)
    return img, transform


# ---------------------------------------------------------------------------
# YOLO inference + sliding window NMS
# ---------------------------------------------------------------------------


def infer_yolo_stripe(
    fold_checkpoints: list[Path],
    img_uint8: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD,
) -> tuple[list[float], list]:
    """Run YOLO ensemble inference on the test stripe.

    Returns (scores, polygons) in STRIPE pixel coordinates (row, col).
    Uses ensemble: average polygon sets from all folds, then NMS merge.
    """
    from ultralytics import YOLO
    from shapely.geometry import Polygon as SPolygon

    _, H, W = img_uint8.shape
    rows = grid_positions(H, PATCH_SIZE, STRIDE)
    cols = grid_positions(W, PATCH_SIZE, STRIDE)

    all_scores: list[float] = []
    all_polys: list = []

    for ckpt_path in fold_checkpoints:
        print(f"    YOLO fold: {ckpt_path.parents[1].name}")
        model = YOLO(str(ckpt_path))

        for row0 in rows:
            for col0 in cols:
                patch = img_uint8[:, row0:row0 + PATCH_SIZE, col0:col0 + PATCH_SIZE]
                # Transpose to (H, W, C) for PIL
                from PIL import Image
                pil_patch = Image.fromarray(np.transpose(patch, (1, 2, 0)))

                results = model.predict(
                    pil_patch, conf=conf_threshold, iou=NMS_IOU_THRESHOLD,
                    verbose=False, half=False,
                )
                if not results:
                    continue
                result = results[0]
                s, polys = yolo_result_to_polygons(result, conf_threshold)
                # Translate to stripe pixel coords
                for score, poly in zip(s, polys):
                    from shapely.affinity import translate
                    poly_stripe = translate(poly, xoff=col0, yoff=row0)
                    if poly_stripe.area >= MIN_INSTANCE_PX:
                        all_scores.append(score)
                        all_polys.append(poly_stripe)

    # Global NMS across all windows and folds
    merged_scores, merged_polys = nms_merge(all_scores, all_polys, iou_threshold=NMS_IOU_THRESHOLD)
    return merged_scores, merged_polys


# ---------------------------------------------------------------------------
# Mask2Former inference
# ---------------------------------------------------------------------------


def infer_mask2former_stripe(
    fold_checkpoints: list[Path],
    img_uint8: np.ndarray,
    device: torch.device,
    conf_threshold: float = CONF_THRESHOLD,
) -> tuple[list[float], list]:
    """Run Mask2Former ensemble inference on the test stripe."""
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    from shapely.geometry import Polygon as SPolygon
    from PIL import Image

    _, H, W = img_uint8.shape
    rows = grid_positions(H, PATCH_SIZE, STRIDE)
    cols = grid_positions(W, PATCH_SIZE, STRIDE)

    all_scores: list[float] = []
    all_polys: list = []

    for ckpt_path in fold_checkpoints:
        print(f"    M2F fold: {ckpt_path.parent.name}")
        ckpt_dir = ckpt_path.parent

        processor = Mask2FormerImageProcessor.from_pretrained(str(ckpt_dir))
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-coco-instance",
            num_labels=1,
            ignore_mismatched_sizes=True,
        ).to(device)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        model.eval()

        with torch.no_grad():
            for row0 in rows:
                for col0 in cols:
                    patch = img_uint8[:, row0:row0 + PATCH_SIZE, col0:col0 + PATCH_SIZE]
                    pil_patch = Image.fromarray(np.transpose(patch, (1, 2, 0)))
                    inputs = processor(images=pil_patch, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model(**inputs)
                    result = processor.post_process_instance_segmentation(
                        outputs, target_sizes=[(PATCH_SIZE, PATCH_SIZE)]
                    )[0]

                    for seg_info in result["segments_info"]:
                        label_id = seg_info["label_id"]
                        score = seg_info.get("score", 1.0)
                        if score < conf_threshold:
                            continue
                        mask = (result["segmentation"] == seg_info["id"]).cpu().numpy()
                        poly = mask_to_polygon(mask)
                        if poly is None or poly.area < MIN_INSTANCE_PX:
                            continue
                        from shapely.affinity import translate
                        poly_stripe = translate(poly, xoff=col0, yoff=row0)
                        all_scores.append(float(score))
                        all_polys.append(poly_stripe)

    merged_scores, merged_polys = nms_merge(all_scores, all_polys, iou_threshold=NMS_IOU_THRESHOLD)
    return merged_scores, merged_polys


def mask_to_polygon(mask: np.ndarray):
    """Convert a binary mask to the largest Shapely polygon via contour extraction."""
    try:
        import cv2
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return None
        coords = largest.reshape(-1, 2).tolist()
        from shapely.geometry import Polygon as SPolygon
        poly = SPolygon(coords)
        return poly if poly.is_valid and not poly.is_empty else None
    except ImportError:
        from skimage.measure import find_contours
        contours = find_contours(mask.astype(float), 0.5)
        if not contours:
            return None
        c = max(contours, key=len)
        from shapely.geometry import Polygon as SPolygon
        coords = [(x, y) for y, x in c]
        if len(coords) < 3:
            return None
        poly = SPolygon(coords)
        return poly if poly.is_valid and not poly.is_empty else None


# ---------------------------------------------------------------------------
# NMS merge
# ---------------------------------------------------------------------------


def nms_merge(
    scores: list[float],
    polys: list,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> tuple[list[float], list]:
    """Non-maximum suppression over polygons sorted by score."""
    if not scores:
        return [], []
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j <= i or j in suppressed:
                continue
            if polygon_iou(polys[i], polys[j]) >= iou_threshold:
                suppressed.add(j)

    return [scores[i] for i in keep], [polys[i] for i in keep]


# ---------------------------------------------------------------------------
# GT instance loading for test stripe
# ---------------------------------------------------------------------------


def load_test_gt_instances(transform: rasterio.transform.Affine) -> list:
    """Load ground-truth instances clipped to the test stripe (cols 0-999 → pixel rows/cols)."""
    from shapely.geometry import box

    pixel_polys = load_label_pixel_polygons(LABEL_FILE, transform)
    # Test stripe in pixel coords: cols 0-999
    test_box = box(TEST_COL_START, 0, TEST_COL_END, 5000)
    gt_polys = []
    for poly in pixel_polys:
        clipped = poly.intersection(test_box)
        if not clipped.is_empty and clipped.area >= MIN_INSTANCE_PX:
            gt_polys.append(clipped)
    return gt_polys


# ---------------------------------------------------------------------------
# Polygon → GeoPackage
# ---------------------------------------------------------------------------


def polys_to_geopkg(
    polys: list,
    scores: list[float],
    transform: rasterio.transform.Affine,
    output_path: Path,
    layer_name: str = "predictions",
    col_offset: int = 0,
) -> None:
    """Convert pixel-space polygons to georeferenced GeoPackage."""
    import geopandas as gpd
    from shapely.affinity import affine_transform as shapely_affine

    # pixel → geo: x_geo = transform.c + (col + col_offset) * transform.a
    #              y_geo = transform.f + row * transform.e
    sx = transform.a
    sy = transform.e
    tx = transform.c + col_offset * transform.a
    ty = transform.f

    geo_polys = []
    for poly in polys:
        geo_poly = shapely_affine(poly, [sx, 0, 0, sy, tx, ty])
        geo_polys.append(geo_poly)

    import pyproj
    crs_str = "EPSG:3301"
    gdf = gpd.GeoDataFrame(
        {"score": scores, "geometry": geo_polys},
        crs=crs_str,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG", layer=layer_name)
    print(f"  ✓ GeoPackage: {output_path} ({len(gdf)} instances, layer='{layer_name}')")


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def print_comparison_table(results: list[dict]) -> None:
    print("\n=== V5 Instance Segmentation Results ===")
    header = f"{'Model':<25} {'AP@50':>8} {'AP@75':>8} {'mAP50:95':>10} {'Prec@50':>9} {'Rec@50':>8} {'CntErr':>8} {'SizeΔ':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<25} "
            f"{r.get('ap50', 0):.4f}   "
            f"{r.get('ap75', 0):.4f}   "
            f"{r.get('map_50_95', 0):.4f}     "
            f"{r.get('precision_50', 0):.4f}   "
            f"{r.get('recall_50', 0):.4f}  "
            f"{r.get('count_error', 0):+.2f}   "
            f"{r.get('size_delta_px', 0):+.1f}"
        )
    print("\nBaseline (V3 semantic, pixel-level):")
    print("  V3 U-Net++ ensemble: test_dice=0.192, precision=0.135, recall=0.297")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase IV V5: Instance evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-gpkg", action="store_true", help="Skip GeoPackage output")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CHM test stripe
    print("\n=== Load CHM test stripe ===")
    img_uint8, transform = load_test_stripe_uint8()
    _, H, W = img_uint8.shape
    print(f"  Shape: {H}×{W} uint8 (3-channel)")

    # Load GT instances for test stripe
    print("\n=== Load ground-truth instances ===")
    gt_polys = load_test_gt_instances(transform)
    print(f"  GT instances in test stripe: {len(gt_polys)}")

    all_results = []

    # -----------------------------------------------------------------------
    # YOLO evaluation
    # -----------------------------------------------------------------------
    print("\n=== YOLO11m-seg Evaluation ===")
    yolo_runs = RUNS_DIR / "yolo"
    yolo_ckpts = []
    for fold_id in range(4):
        ckpt = yolo_runs / f"fold{fold_id}" / "weights" / "best.pt"
        if ckpt.exists():
            yolo_ckpts.append(ckpt)
            print(f"  ✓ Fold {fold_id}: {ckpt}")
        else:
            print(f"  ✗ Fold {fold_id}: not found (skipping)")

    if yolo_ckpts:
        print(f"\n  Running YOLO ensemble ({len(yolo_ckpts)} folds)...")
        yolo_scores, yolo_polys = infer_yolo_stripe(yolo_ckpts, img_uint8, args.conf)
        print(f"  Predictions: {len(yolo_polys)} instances")

        # Compute metrics
        yolo_metrics = InstanceMetrics()
        yolo_metrics.update(yolo_scores, yolo_polys, gt_polys)
        yolo_result = yolo_metrics.compute()
        yolo_result["model"] = "YOLO11m-seg (V5a)"
        yolo_result["n_folds"] = len(yolo_ckpts)
        all_results.append(yolo_result)

        print(f"  AP@50={yolo_result['ap50']:.4f}, mAP={yolo_result['map_50_95']:.4f}, "
              f"count_err={yolo_result['count_error']:+.2f}")

        # GeoPackage for test stripe predictions
        if not args.no_gpkg and yolo_polys:
            polys_to_geopkg(
                yolo_polys, yolo_scores, transform,
                OUTPUT_DIR / "pred_yolo_test_stripe.gpkg",
                layer_name="yolo_predictions",
                col_offset=TEST_COL_START,
            )
    else:
        print("  No YOLO checkpoints found — skipping YOLO evaluation")

    # -----------------------------------------------------------------------
    # Mask2Former evaluation
    # -----------------------------------------------------------------------
    print("\n=== Mask2Former Evaluation ===")
    m2f_runs = RUNS_DIR / "mask2former"
    m2f_ckpts = []
    for fold_id in range(4):
        ckpt = m2f_runs / f"fold{fold_id}" / "checkpoint" / "model.pt"
        if ckpt.exists():
            m2f_ckpts.append(ckpt)
            print(f"  ✓ Fold {fold_id}: {ckpt}")
        else:
            print(f"  ✗ Fold {fold_id}: not found (skipping)")

    if m2f_ckpts:
        print(f"\n  Running Mask2Former ensemble ({len(m2f_ckpts)} folds)...")
        m2f_scores, m2f_polys = infer_mask2former_stripe(m2f_ckpts, img_uint8, device, args.conf)
        print(f"  Predictions: {len(m2f_polys)} instances")

        m2f_metrics = InstanceMetrics()
        m2f_metrics.update(m2f_scores, m2f_polys, gt_polys)
        m2f_result = m2f_metrics.compute()
        m2f_result["model"] = "Mask2Former (V5b)"
        m2f_result["n_folds"] = len(m2f_ckpts)
        all_results.append(m2f_result)

        print(f"  AP@50={m2f_result['ap50']:.4f}, mAP={m2f_result['map_50_95']:.4f}, "
              f"count_err={m2f_result['count_error']:+.2f}")

        if not args.no_gpkg and m2f_polys:
            polys_to_geopkg(
                m2f_polys, m2f_scores, transform,
                OUTPUT_DIR / "pred_mask2former_test_stripe.gpkg",
                layer_name="mask2former_predictions",
                col_offset=TEST_COL_START,
            )
    else:
        print("  No Mask2Former checkpoints found — skipping Mask2Former evaluation")

    # -----------------------------------------------------------------------
    # Full-tile GeoPackage (YOLO ensemble on all 5 stripes)
    # -----------------------------------------------------------------------
    if not args.no_gpkg and yolo_ckpts:
        print("\n=== Full-tile GeoPackage (YOLO) ===")
        img_full_uint8, transform_full = load_chm_as_uint8_full()
        print(f"  Full tile shape: {img_full_uint8.shape}")

        all_full_scores, all_full_polys = [], []
        for ckpt_path in yolo_ckpts:
            from ultralytics import YOLO
            from PIL import Image as PILImage
            model = YOLO(str(ckpt_path))
            _, H, W = img_full_uint8.shape
            rows = grid_positions(H, PATCH_SIZE, STRIDE)
            cols = grid_positions(W, PATCH_SIZE, STRIDE)
            print(f"    Fold {ckpt_path.parent.parent.name}: {len(rows)*len(cols)} patches")
            for row0 in rows:
                for col0 in cols:
                    patch = img_full_uint8[:, row0:row0 + PATCH_SIZE, col0:col0 + PATCH_SIZE]
                    pil_patch = PILImage.fromarray(np.transpose(patch, (1, 2, 0)))
                    results = model.predict(pil_patch, conf=args.conf, iou=NMS_IOU_THRESHOLD, verbose=False)
                    if not results:
                        continue
                    s, polys = yolo_result_to_polygons(results[0], args.conf)
                    from shapely.affinity import translate
                    for score, poly in zip(s, polys):
                        poly_full = translate(poly, xoff=col0, yoff=row0)
                        if poly_full.area >= MIN_INSTANCE_PX:
                            all_full_scores.append(score)
                            all_full_polys.append(poly_full)

        merged_scores, merged_polys = nms_merge(all_full_scores, all_full_polys, NMS_IOU_THRESHOLD)
        print(f"  Full-tile predictions: {len(merged_polys)} instances")
        polys_to_geopkg(
            merged_polys, merged_scores, transform_full,
            OUTPUT_DIR / "pred_instances_full_tile.gpkg",
            layer_name="yolo_full_tile",
        )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    print_comparison_table(all_results)

    # JSON
    metrics_path = OUTPUT_DIR / "final_metrics_v5.json"
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n✓ Metrics: {metrics_path}")

    # CSV for thesis
    csv_path = OUTPUT_DIR / "thesis_table_v5.csv"
    if all_results:
        fields = ["model", "ap50", "ap75", "map_50_95", "precision_50", "recall_50",
                  "count_error", "size_delta_px", "n_gt", "n_pred"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"✓ CSV:     {csv_path}")

    print(f"\n✅ COMPLETE — outputs in {OUTPUT_DIR}")
    return 0


def load_chm_as_uint8_full() -> tuple[np.ndarray, rasterio.transform.Affine]:
    """Load the full CHM tile (5000×5000) as uint8."""
    with rasterio.open(CHM_TIF) as src:
        img = src.read(CHM_BANDS, boundless=True, fill_value=0.0).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        img[img == nodata] = 0.0
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0.0, CHM_HAG_MAX) / CHM_HAG_MAX
    img = (img * 255).astype(np.uint8)
    return img, transform


if __name__ == "__main__":
    sys.exit(main())
