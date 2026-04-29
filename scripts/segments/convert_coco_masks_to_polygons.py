#!/usr/bin/env python3
"""
Convert COCO RLE/mask segmentations inside a ZIP to polygon segmentations.

Usage:
  python scripts/segments/convert_coco_masks_to_polygons.py INPUT_ZIP OUTPUT_ZIP --simplify-tol 2.0 --min-seg-len 3.0
"""
import argparse
import json
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
from pycocotools import mask as maskUtils
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def flatten_coords(coords):
    flat = []
    for x, y in coords:
        flat.append(float(x))
        flat.append(float(y))
    return flat

def max_edge_length(coords):
    # coords: sequence of (x,y)
    if len(coords) < 2:
        return 0.0
    maxl = 0.0
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i+1]
        d = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        if d > maxl:
            maxl = d
    return maxl

def find_annotation_json(extracted_dir):
    # return path to a JSON file that contains top-level "annotations" list
    for root, _, files in os.walk(extracted_dir):
        for f in files:
            if f.lower().endswith('.json'):
                p = os.path.join(root, f)
                try:
                    with open(p, 'r') as fh:
                        j = json.load(fh)
                    if isinstance(j, dict) and 'annotations' in j and isinstance(j['annotations'], list):
                        return p, j
                except Exception:
                    continue
    return None, None

def process_annotations(ann_data, simplify_tol, min_seg_len):
    logs = []
    anns = ann_data['annotations']
    img_index = defaultdict(list)
    for a in anns:
        img_index[a['image_id']].append(a)

    processed = 0
    for ann in anns:
        seg = ann.get('segmentation')
        if seg is None:
            logs.append(f"ann id={ann.get('id')} no segmentation -> skip")
            continue

        is_rle = isinstance(seg, dict) or (isinstance(seg, list) and len(seg) == 1 and isinstance(seg[0], dict))
        if not is_rle:
            # Already polygon(s) — skip
            logs.append(f"ann id={ann.get('id')} segmentation already polygon/list -> keep")
            continue

        # Normalize RLE input for maskUtils.decode; if list with dict, merge/choose:
        rle = seg
        if isinstance(seg, list) and len(seg) == 1 and isinstance(seg[0], dict):
            rle = seg[0]

        try:
            mask = maskUtils.decode(rle)
        except Exception as e:
            logs.append(f"ann id={ann.get('id')} mask decode failed: {e} -> skip")
            continue

        if mask is None:
            logs.append(f"ann id={ann.get('id')} decode returned None -> skip")
            continue

        # If mask has shape (H,W,1) or (H,W,n), collapse OR across channels
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # find contours at 0.5 level
        conts = measure.find_contours(mask, 0.5)
        polys = []
        kept = 0
        for c in conts:
            if c.shape[0] < 3:
                continue
            # c is array of (row, col). Convert to (x=col, y=row)
            coords = [(float(pt[1]), float(pt[0])) for pt in c]
            try:
                poly = Polygon(coords)
                if not poly.is_valid or poly.is_empty:
                    continue
                poly = poly.simplify(simplify_tol)
                if poly.is_empty:
                    continue
                # if result is MultiPolygon, iterate parts
                parts = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
                for p in parts:
                    ext_coords = list(p.exterior.coords)
                    if max_edge_length(ext_coords) < min_seg_len:
                        continue
                    if p.area <= 0:
                        continue
                    polys.append(p)
                    kept += 1
            except Exception:
                continue

        if not polys:
            logs.append(f"ann id={ann.get('id')} -> no polygons kept after simplify/filter")
            # clear segmentation, set area to 0 and bbox to [0,0,0,0]
            ann['segmentation'] = []
            ann['area'] = 0.0
            ann['bbox'] = [0.0, 0.0, 0.0, 0.0]
            ann['iscrowd'] = 0
            continue

        # Combine polygons to get bbox and area
        union = unary_union(polys)
        if isinstance(union, (Polygon, MultiPolygon)):
            minx, miny, maxx, maxy = union.bounds
            area = union.area
        else:
            minx = miny = maxx = maxy = 0.0
            area = 0.0

        # Update annotation segmentation to list of polygons (list-of-lists)
        seg_out = []
        for p in polys:
            coords = list(p.exterior.coords)
            # flatten x,y pairs
            seg_out.append(flatten_coords(coords))
        ann['segmentation'] = seg_out
        ann['bbox'] = [float(minx), float(miny), float(maxx - minx), float(maxy - miny)]
        ann['area'] = float(area)
        ann['iscrowd'] = 0
        processed += 1
        logs.append(f"ann id={ann.get('id')} processed: contours={len(conts)} kept_polys={kept}")

    return ann_data, logs, processed

def main():
    p = argparse.ArgumentParser(description="Convert COCO mask/RLE segmentations to polygons")
    p.add_argument('input_zip', help='Input COCO ZIP (export)')
    p.add_argument('output_zip', help='Output ZIP path')
    p.add_argument('--simplify-tol', type=float, default=2.0, help='shapely simplify tolerance (pixels)')
    p.add_argument('--min-seg-len', type=float, default=3.0, help='Minimum segment edge length to keep (pixels)')
    args = p.parse_args()

    if not os.path.exists(args.input_zip):
        print(f"Error: input zip not found: {args.input_zip}")
        return 2

    tmpdir = tempfile.mkdtemp(prefix='coco_mask2poly_')
    try:
        with zipfile.ZipFile(args.input_zip, 'r') as zin:
            zin.extractall(tmpdir)

        ann_path, ann_data = find_annotation_json(tmpdir)
        if ann_path is None:
            print("Error: No COCO annotations JSON found inside the ZIP.")
            return 3

        print(f"Loaded annotations JSON: {ann_path}")
        print(f"Found {len(ann_data.get('annotations', []))} annotations")

        new_ann_data, logs, processed = process_annotations(ann_data, args.simplify_tol, args.min_seg_len)

        # Write new annotations.json in tmpdir root as "annotations.json"
        out_ann_name = 'annotations.json'
        out_ann_path = os.path.join(tmpdir, out_ann_name)
        with open(out_ann_path, 'w') as fh:
            json.dump(new_ann_data, fh)

        # Create output zip including images and the new annotations.json
        with zipfile.ZipFile(args.output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            # Add new annotations file
            zout.write(out_ann_path, arcname=out_ann_name)
            # Walk tmpdir and add image files (avoid re-adding annotations.json if multiple)
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, tmpdir)
                    if rel == out_ann_name:
                        continue
                    # include typical image extensions and any files under 'images' folder
                    if rel.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')) or 'image' in rel.lower() or 'images' in root.lower():
                        zout.write(fp, arcname=rel)
            # Also include any other files from the original zip that might be needed (optional)
            # For determinism, not re-adding everything; user can modify if needed.

        print(f"Wrote output ZIP: {args.output_zip}")
        print(f"Processed annotations: {processed}")
        for L in logs:
            print(L)
        return 0
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

if __name__ == '__main__':
    raise SystemExit(main())
