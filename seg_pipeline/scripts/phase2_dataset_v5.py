#!/usr/bin/env python3
"""Phase II V5: Instance Segmentation Dataset Preparation.

Produces YOLO-seg and COCO-format patch datasets from the 250 MultiPolygon
CWD instances in data/labels/cdw_labels_MP.gpkg.

Input:
    seg_pipeline/input/composite_4band.tif  — bands 1-3 used as pseudo-RGB
    data/labels/cdw_labels_MP.gpkg          — 250 MultiPolygon instances (EPSG:3301)

Output:
    phase2_dataset_v5/yolo/fold{k}/{train,val,test}/{images,labels}/
    phase2_dataset_v5/coco/fold{k}/{train.json,val.json,test.json}
    phase2_dataset_v5/patch_stats_v5.json

Spatial CV:
    Same 5-stripe split as V3:
      stripe 0 (cols 0–999)   → permanent test set
      folds 0–3: stripes 1–4 rotate as validation

Patch size: 640×640, stride: 480 (25% overlap)
Negative sampling: 1:3 ratio (neg:pos) within train/val splits.

Usage:
    python phase2_dataset_v5.py --validate   # count patches + instances, no output written
    python phase2_dataset_v5.py              # full build
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from phase2_dataset_v3 import (
    STRIPE_WIDTH,
    N_STRIPES,
    TEST_STRIPE,
    BUFFER_PX,
    SpatialCVSplitterV3,
)

try:
    import geopandas as gpd
    from shapely.geometry import box, MultiPolygon, Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_SIZE = 640
STRIDE = 480
MIN_INSTANCE_PX = 10       # discard clipped instances smaller than this area
NEG_TO_POS_RATIO = 3       # number of negative patches per positive patch
LABEL_FILE = ROOT / "data" / "labels" / "cdw_labels_MP.gpkg"
CHM_TIF = ROOT / "seg_pipeline" / "input" / "composite_4band.tif"
OUTPUT_DIR = ROOT / "seg_pipeline" / "output" / "phase2_dataset_v5"
CHM_HAG_MAX = 1.3          # clip CHM to [0, HAG_MAX] before uint8 conversion
CHM_BANDS = [1, 2, 3]      # bands 1-3 of composite_4band.tif → pseudo-RGB
MAX_POLYGON_VERTICES = 100  # simplify polygons to at most this many vertices
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# CHM → uint8 pseudo-RGB
# ---------------------------------------------------------------------------


def load_chm_as_uint8(tif_path: Path) -> tuple[np.ndarray, rasterio.transform.Affine, dict]:
    """Load CHM bands 1-3 and convert to uint8 pseudo-RGB.

    Returns:
        img_uint8: (3, H, W) uint8 array
        transform: rasterio Affine transform
        profile: rasterio profile (for writing)
    """
    with rasterio.open(tif_path) as src:
        img = src.read(CHM_BANDS, boundless=True, fill_value=0.0).astype(np.float32)
        transform = src.transform
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        img[img == nodata] = 0.0
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0.0, CHM_HAG_MAX) / CHM_HAG_MAX  # → [0, 1]
    img = (img * 255).astype(np.uint8)
    return img, transform, profile


# ---------------------------------------------------------------------------
# Label → pixel-space polygons
# ---------------------------------------------------------------------------


def load_label_pixel_polygons(
    gpkg_path: Path,
    transform: rasterio.transform.Affine,
) -> list[Polygon]:
    """Read all label instances and convert to pixel-coordinate Shapely polygons.

    Returns a list of (possibly simplified) Shapely Polygon objects in pixel space.
    Each MultiPolygon is decomposed into its component Polygons.
    """
    gdf = gpd.read_file(gpkg_path)
    pixel_polys: list[Polygon] = []

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geom = make_valid(geom)
        # Convert to pixel coords
        px_geom = geo_to_pixel(geom, transform)
        if px_geom is None or px_geom.is_empty:
            continue
        if isinstance(px_geom, MultiPolygon):
            for part in px_geom.geoms:
                if not part.is_empty and part.area >= MIN_INSTANCE_PX:
                    pixel_polys.append(part)
        elif isinstance(px_geom, Polygon):
            if not px_geom.is_empty and px_geom.area >= MIN_INSTANCE_PX:
                pixel_polys.append(px_geom)

    return pixel_polys


def geo_to_pixel(geom, transform: rasterio.transform.Affine):
    """Convert a geometry from geo coords to pixel coords using CHM transform."""
    from shapely.affinity import affine_transform

    # transform.a = pixel width (x res), transform.e = pixel height (y res, negative)
    # x_geo = transform.c + col * transform.a
    # y_geo = transform.f + row * transform.e
    # → col = (x_geo - transform.c) / transform.a
    # → row = (y_geo - transform.f) / transform.e

    sx = 1.0 / transform.a
    sy = 1.0 / transform.e
    tx = -transform.c / transform.a
    ty = -transform.f / transform.e
    # shapely affine_transform params: [a, b, d, e, xoff, yoff]
    # new_x = a*x + b*y + xoff
    # new_y = d*x + e*y + yoff
    return affine_transform(geom, [sx, 0, 0, sy, tx, ty])


# ---------------------------------------------------------------------------
# Spatial indexing for fast patch lookup
# ---------------------------------------------------------------------------


def build_spatial_index(pixel_polys: list[Polygon]):
    """Build an STRtree for fast intersection queries."""
    from shapely.strtree import STRtree
    return STRtree(pixel_polys)


# ---------------------------------------------------------------------------
# Patch grid + overlap detection
# ---------------------------------------------------------------------------


def grid_positions(dim: int, patch_size: int, stride: int) -> list[int]:
    positions = list(range(0, max(1, dim - patch_size + 1), stride))
    if not positions or positions[-1] + patch_size < dim:
        positions.append(max(0, dim - patch_size))
    return positions


def get_overlapping_instances(
    strtree,
    pixel_polys: list[Polygon],
    row0: int,
    col0: int,
    patch_size: int = PATCH_SIZE,
    min_area: float = MIN_INSTANCE_PX,
) -> list[Polygon]:
    """Return clipped instance polygons that overlap significantly with the patch."""
    patch_box = box(col0, row0, col0 + patch_size, row0 + patch_size)
    candidates = strtree.query(patch_box)
    result: list[Polygon] = []
    for idx in candidates:
        poly = pixel_polys[idx]
        clipped = poly.intersection(patch_box)
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiPolygon):
            for part in clipped.geoms:
                if part.area >= min_area:
                    result.append(part)
        elif clipped.area >= min_area:
            result.append(clipped)
    return result


# ---------------------------------------------------------------------------
# YOLO label format
# ---------------------------------------------------------------------------


def polygon_to_yolo(poly: Polygon, col0: int, row0: int, patch_size: int) -> str | None:
    """Convert a pixel-space polygon (already clipped to patch) to YOLO label line.

    Returns: '0 x1_norm y1_norm x2_norm y2_norm ...' or None if degenerate.
    """
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return None

    # Simplify to at most MAX_POLYGON_VERTICES
    if len(coords) > MAX_POLYGON_VERTICES + 1:
        from shapely.geometry import Polygon as SPolygon
        simplified = poly.simplify(
            tolerance=max(1.0, poly.length / MAX_POLYGON_VERTICES),
            preserve_topology=True,
        )
        coords = list(simplified.exterior.coords) if not simplified.is_empty else coords

    tokens = ["0"]
    for px, py in coords[:-1]:  # skip repeated closing vertex
        x_norm = (px - col0) / patch_size
        y_norm = (py - row0) / patch_size
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        tokens.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

    if len(tokens) < 7:  # class + at least 3 x,y pairs
        return None
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# COCO annotation format
# ---------------------------------------------------------------------------


def build_coco_annotation(
    poly: Polygon,
    image_id: int,
    ann_id: int,
    col0: int,
    row0: int,
) -> dict:
    """Build a COCO annotation dict for one clipped instance polygon."""
    coords = list(poly.exterior.coords)
    segmentation = []
    for px, py in coords[:-1]:
        segmentation.extend([px - col0, py - row0])

    bbox_x, bbox_y, bbox_x2, bbox_y2 = poly.bounds
    w = bbox_x2 - bbox_x
    h = bbox_y2 - bbox_y
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [segmentation],
        "area": float(poly.area),
        "bbox": [bbox_x - col0, bbox_y - row0, w, h],
        "iscrowd": 0,
    }


# ---------------------------------------------------------------------------
# Image writing
# ---------------------------------------------------------------------------


def write_patch_image(
    img_uint8: np.ndarray,
    row0: int,
    col0: int,
    patch_size: int,
    out_path: Path,
) -> None:
    """Write a (3, H, W) uint8 patch as JPEG."""
    from PIL import Image
    patch = img_uint8[:, row0:row0 + patch_size, col0:col0 + patch_size]
    C, H, W = patch.shape
    pil_img = Image.fromarray(np.transpose(patch, (1, 2, 0)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path, quality=95)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_dataset(validate_only: bool = False) -> dict:
    """Build YOLO + COCO datasets for all 4 folds.

    Returns a dict of statistics.
    """
    if not HAS_GEO:
        raise ImportError("geopandas and shapely are required: pip install geopandas shapely")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")

    print("=== Phase II V5 Dataset Build ===")
    print(f"Labels: {LABEL_FILE} ({LABEL_FILE.exists()})")
    print(f"CHM:    {CHM_TIF}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Patch:  {PATCH_SIZE}×{PATCH_SIZE}, stride={STRIDE}")

    # Load CHM
    print("\nLoading CHM...")
    img_uint8, transform, profile = load_chm_as_uint8(CHM_TIF)
    _, H, W = img_uint8.shape
    print(f"  CHM shape: {H}×{W} (uint8 pseudo-RGB from bands 1-3)")

    # Load labels
    print("\nLoading label instances...")
    pixel_polys = load_label_pixel_polygons(LABEL_FILE, transform)
    print(f"  Instances: {len(pixel_polys)} (from {LABEL_FILE.name})")

    # Build spatial index
    strtree = build_spatial_index(pixel_polys)

    # Grid positions
    rows = grid_positions(H, PATCH_SIZE, STRIDE)
    cols = grid_positions(W, PATCH_SIZE, STRIDE)
    splitter = SpatialCVSplitterV3()
    print(f"  Grid: {len(rows)} rows × {len(cols)} cols = {len(rows)*len(cols)} candidates")

    # Enumerate all patches and find overlapping instances
    # Note: we do NOT use V3's is_in_buffer — it assumes 256px patches in 1000-col stripes
    # (buffer becomes degenerate with 640px patches). Instead, split by patch center stripe.
    print("\nScanning patches...")
    all_patches: list[dict] = []
    for row0 in rows:
        for col0 in cols:
            instances = get_overlapping_instances(strtree, pixel_polys, row0, col0)
            stripe = splitter.stripe_of(col0, PATCH_SIZE)
            fold_id = splitter.fold_id_of(stripe)
            all_patches.append({
                "row0": row0, "col0": col0,
                "stripe": stripe, "fold_id": fold_id,
                "instances": instances,
                "n_instances": len(instances),
            })

    n_positive = sum(1 for p in all_patches if p["n_instances"] > 0)
    print(f"  Total patches: {len(all_patches)}, positive: {n_positive}")

    # Determine which split each patch belongs to
    # test: fold_id == -1
    # train/val: based on fold
    stats = {"n_instances": len(pixel_polys), "folds": {}}

    for fold_id in range(4):
        fold_key = f"fold{fold_id}"
        val_stripe = splitter._val_stripes[fold_id]

        train_patches, val_patches, test_patches = [], [], []
        for p in all_patches:
            if p["fold_id"] == -1:
                test_patches.append(p)
            elif p["stripe"] == val_stripe:
                val_patches.append(p)
            else:
                train_patches.append(p)

        # Negative sampling for train and val
        train_patches = sample_patches(train_patches, NEG_TO_POS_RATIO, RANDOM_SEED + fold_id)
        val_patches = sample_patches(val_patches, NEG_TO_POS_RATIO, RANDOM_SEED + fold_id + 100)

        n_train_pos = sum(1 for p in train_patches if p["n_instances"] > 0)
        n_val_pos = sum(1 for p in val_patches if p["n_instances"] > 0)
        n_test_pos = sum(1 for p in test_patches if p["n_instances"] > 0)

        print(f"\n  Fold {fold_id}: train={len(train_patches)} ({n_train_pos} pos), "
              f"val={len(val_patches)} ({n_val_pos} pos), test={len(test_patches)} ({n_test_pos} pos)")

        stats["folds"][fold_key] = {
            "n_train": len(train_patches), "n_train_pos": n_train_pos,
            "n_val": len(val_patches), "n_val_pos": n_val_pos,
            "n_test": len(test_patches), "n_test_pos": n_test_pos,
        }

        if validate_only:
            continue

        # Write datasets
        fold_yolo_dir = OUTPUT_DIR / "yolo" / fold_key
        fold_coco_dir = OUTPUT_DIR / "coco" / fold_key

        for split_name, patches in [("train", train_patches), ("val", val_patches), ("test", test_patches)]:
            print(f"    Writing {split_name}...", end=" ", flush=True)
            coco_images, coco_annotations = [], []
            ann_id = 0

            for img_id, p in enumerate(patches):
                patch_name = f"r{p['row0']:04d}_c{p['col0']:04d}"

                # Write image
                img_path = fold_yolo_dir / split_name / "images" / f"{patch_name}.jpg"
                write_patch_image(img_uint8, p["row0"], p["col0"], PATCH_SIZE, img_path)

                # Write YOLO label
                lbl_path = fold_yolo_dir / split_name / "labels" / f"{patch_name}.txt"
                lbl_path.parent.mkdir(parents=True, exist_ok=True)
                yolo_lines = []
                for inst in p["instances"]:
                    line = polygon_to_yolo(inst, p["col0"], p["row0"], PATCH_SIZE)
                    if line:
                        yolo_lines.append(line)
                lbl_path.write_text("\n".join(yolo_lines))

                # COCO metadata
                coco_images.append({
                    "id": img_id,
                    "file_name": f"{split_name}/images/{patch_name}.jpg",
                    "height": PATCH_SIZE,
                    "width": PATCH_SIZE,
                })
                for inst in p["instances"]:
                    ann = build_coco_annotation(inst, img_id, ann_id, p["col0"], p["row0"])
                    coco_annotations.append(ann)
                    ann_id += 1

            # Write COCO JSON
            coco_json = {
                "info": {"description": "V5 CWD Instance Segmentation", "version": "5.0"},
                "categories": [{"id": 1, "name": "cwd", "supercategory": "object"}],
                "images": coco_images,
                "annotations": coco_annotations,
            }
            coco_path = fold_coco_dir / f"{split_name}.json"
            coco_path.parent.mkdir(parents=True, exist_ok=True)
            coco_path.write_text(json.dumps(coco_json, indent=2))

            print(f"done ({len(patches)} patches, {ann_id} annotations)")

        # Write YOLO data.yaml
        data_yaml = fold_yolo_dir / "data.yaml"
        data_yaml.write_text(
            f"path: {fold_yolo_dir}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"nc: 1\n"
            f"names: ['cwd']\n"
        )

    # Write stats
    if not validate_only:
        stats_path = OUTPUT_DIR / "patch_stats_v5.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"\n✓ Stats: {stats_path}")

    return stats


def sample_patches(patches: list[dict], neg_to_pos_ratio: int, seed: int) -> list[dict]:
    """Keep all positive patches; sample negatives at neg_to_pos_ratio:1 ratio."""
    rng = random.Random(seed)
    positives = [p for p in patches if p["n_instances"] > 0]
    negatives = [p for p in patches if p["n_instances"] == 0]
    n_neg = min(len(negatives), len(positives) * neg_to_pos_ratio)
    sampled_neg = rng.sample(negatives, n_neg)
    combined = positives + sampled_neg
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase II V5: Instance dataset preparation")
    parser.add_argument("--validate", action="store_true", help="Count patches only, no output")
    args = parser.parse_args()

    stats = build_dataset(validate_only=args.validate)

    print("\n=== Summary ===")
    for fold_key, s in stats["folds"].items():
        print(f"  {fold_key}: train={s['n_train']} ({s['n_train_pos']} pos), "
              f"val={s['n_val']} ({s['n_val_pos']} pos), "
              f"test={s['n_test']} ({s['n_test_pos']} pos)")

    if args.validate:
        print("\n[validate mode — no files written]")
    else:
        print(f"\n✅ Dataset written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
