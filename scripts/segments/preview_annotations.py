#!/usr/bin/env python3
"""Preview annotations from a COCO ZIP.

Creates a single-image preview and a rows x cols grid of images with drawn polygons and bboxes.

Usage:
  python scripts/segments/preview_annotations.py INPUT_ZIP --outdir output/segments/preview --rows 3 --cols 4
"""
import argparse
import io
import json
import os
import zipfile
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_coco_from_zip(zip_path):
    # reopen ZIP when loading images to avoid closed-handle errors
    with zipfile.ZipFile(zip_path, 'r') as z:
        # find annotations json
        ann_name = None
        for name in z.namelist():
            if name.lower().endswith('.json'):
                try:
                    with z.open(name) as fh:
                        j = json.load(fh)
                    if isinstance(j, dict) and 'annotations' in j:
                        ann_name = name
                        ann_data = j
                        break
                except Exception:
                    continue
        if ann_name is None:
            raise RuntimeError('No annotations JSON found in ZIP')

    images = {img['id']: img for img in ann_data.get('images', [])}
    anns_by_image = defaultdict(list)
    for a in ann_data.get('annotations', []):
        anns_by_image[a['image_id']].append(a)

    # return zip_path and leave image loading to a separate helper that reopens the ZIP
    return ann_data, images, anns_by_image


def load_image_from_zip(zip_path, fname):
    with zipfile.ZipFile(zip_path, 'r') as z:
        try:
            with z.open(fname) as f:
                data = f.read()
            return Image.open(io.BytesIO(data)).convert('RGB')
        except Exception:
            # try base name fallback
            bname = os.path.basename(fname)
            for name in z.namelist():
                if os.path.basename(name) == bname and name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    with z.open(name) as f:
                        return Image.open(io.BytesIO(f.read())).convert('RGB')
    return None


def draw_annotations(ax, img, anns, category_map=None, show_bbox=False):
    ax.imshow(img)
    ax.axis('off')
    h, w = img.size[1], img.size[0]
    for ann in anns:
        segs = ann.get('segmentation', [])
        # segmentation is list-of-lists (x0,y0,x1,y1...)
        if isinstance(segs, list) and segs and isinstance(segs[0], list):
            for seg in segs:
                xs = seg[0::2]
                ys = seg[1::2]
                poly = patches.Polygon(xy=list(zip(xs, ys)), closed=True, fill=False, edgecolor='lime', linewidth=1)
                ax.add_patch(poly)
        # draw bbox
        bbox = ann.get('bbox')
        if show_bbox and bbox and len(bbox) >= 4:
            x, y, bw, bh = bbox
            rect = patches.Rectangle((x, y), bw, bh, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        # label
        cid = ann.get('category_id')
        label = None
        if category_map:
            label = category_map.get(cid)
        if label:
            ax.text(5, 10, label, color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_zip')
    p.add_argument('--outdir', default='output/segments/preview')
    p.add_argument('--rows', type=int, default=3)
    p.add_argument('--cols', type=int, default=4)
    p.add_argument('--single', action='store_true', help='also save a single-image preview')
    p.add_argument('--show-bbox', action='store_true', help='draw bbox rectangles')
    p.add_argument('--category', type=str, help='filter to this category name (e.g., pen)')
    p.add_argument('--random', type=int, help='select N random images for the grid')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ann_data, images, anns_by_image = load_coco_from_zip(args.input_zip)

    # build category map
    cat_map = {c['id']: c.get('name') for c in ann_data.get('categories', [])}

    img_items = sorted(images.items(), key=lambda kv: kv[0])
    if not img_items:
        raise RuntimeError('No images found in COCO JSON')

    # optionally filter by category name
    if args.category:
        # find category id(s)
        cat_name = args.category
        cat_map = {c['id']: c.get('name') for c in ann_data.get('categories', [])}
        name_to_id = {v: k for k, v in cat_map.items()}
        target_id = name_to_id.get(cat_name)
        if target_id is None:
            print('Warning: category not found:', cat_name)
        else:
            # keep only images that have at least one annotation with this category
            img_items = [it for it in img_items if any(a.get('category_id') == target_id for a in anns_by_image.get(it[0], []))]

    # single image preview: first image
    if not img_items:
        raise RuntimeError('No images left after filtering')
    first_id, first_img = img_items[0]
    fname = first_img.get('file_name')
    pil = load_image_from_zip(args.input_zip, fname)
    if pil is None:
        print('Warning: could not load image', fname)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_annotations(ax, pil, anns_by_image.get(first_id, []), cat_map)
        single_out = os.path.join(args.outdir, f'preview_single_{os.path.basename(fname)}.png')
        fig.savefig(single_out, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print('Wrote', single_out)

    # grid preview
    total = args.rows * args.cols
    if args.random:
        import random
        n = min(args.random, len(img_items))
        sel = random.sample(img_items, n)
        # if less than grid size, pad with first images
        if len(sel) < total:
            sel = sel + img_items[:(total - len(sel))]
        sel = sel[:total]
    else:
        sel = img_items[:total]
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 3, args.rows * 3))
    axes = axes.flatten()
    for ax, (img_id, img_meta) in zip(axes, sel):
        pil = load_image_from_zip(args.input_zip, img_meta.get('file_name'))
        if pil is None:
            ax.text(0.5, 0.5, 'missing', ha='center')
            ax.axis('off')
            continue
        draw_annotations(ax, pil, anns_by_image.get(img_id, []), cat_map, show_bbox=args.show_bbox)
        ax.set_title(os.path.basename(img_meta.get('file_name'))[:20], fontsize=8)

    # hide remaining axes if any
    for ax in axes[len(sel):]:
        ax.axis('off')

    grid_out = os.path.join(args.outdir, f'preview_grid_{args.rows}x{args.cols}.png')
    fig.savefig(grid_out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('Wrote', grid_out)


if __name__ == '__main__':
    main()
