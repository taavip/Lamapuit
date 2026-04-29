#!/usr/bin/env python3
"""Debug heatmaps for CNN-Deep-Attn on a small set of tiles.
Saves heatmaps and overlays to output/heatmap_debug.
"""

from pathlib import Path
import json
import argparse
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

# Import label_tiles functions by adding scripts dir to sys.path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import label_tiles as lt

CHUNK = 128


def apply_sld_and_save(raw_tile, out_rgb_path):
    rgb = lt._apply_sld(raw_tile)
    cv2.imwrite(str(out_rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return rgb


def save_heatmap_gray(hm_uint8, out_path):
    cv2.imwrite(str(out_path), hm_uint8)


def save_overlay(rgb, hm_uint8, out_path, alpha=0.6):
    # convert heatmap to color
    cmap = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    overlay = (rgb.astype(float) * (1 - alpha) + cmap.astype(float) * alpha).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-list", default="output/audits/audit_review_queue.csv")
    p.add_argument("--chm-dir", default="chm_max_hag")
    p.add_argument("--output", default="output/heatmap_debug")
    p.add_argument("--model", default="output/tile_labels/ensemble_model.pt")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--methods", default="HiResCAM,GradCAM+,IntGrad,RISE")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read tile list csv
    import csv

    entries = []
    with open(args.tile_list, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            entries.append(row)
    if not entries:
        print("No entries in tile list")
        return

    # Build predictor
    predictor = lt.CNNPredictor()
    model_path = Path(args.model)
    if not model_path.exists():
        print("Model not found:", model_path)
        return
    ok = predictor.load_from_disk(model_path)
    print("Loaded model ok=", ok)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    count = 0
    for row in entries:
        if count >= args.n:
            break
        raster = row["raster"]
        row_off = int(row["row_off"])
        col_off = int(row["col_off"])
        chm_path = Path(args.chm_dir) / raster
        if not chm_path.exists():
            print("CHM not found:", chm_path)
            continue
        try:
            with rasterio.open(chm_path) as src:
                raw = src.read(
                    1, window=Window(col_off, row_off, CHUNK, CHUNK), boundless=True, fill_value=0
                ).astype(np.float32)
        except Exception as exc:
            print("Failed reading tile", exc)
            continue

        # Save original SLD RGB
        base = out_dir / f"{raster}_{row_off}_{col_off}"
        base.mkdir(exist_ok=True)
        rgb = apply_sld_and_save(raw, base / "tile_rgb.png")

        prob = predictor.predict_proba_cdw(raw)
        print(f"Tile {raster} @ ({row_off},{col_off}) prob={prob}")

        meta = {"raster": raster, "row_off": row_off, "col_off": col_off, "prob": prob}
        (base / "meta.json").write_text(json.dumps(meta, indent=2))

        cache = {}
        for m in methods:
            try:
                hm = lt._compute_heatmap(m, predictor, raw, row_off, col_off, cache)
                if hm is None:
                    print("hm None for", m)
                    continue
                # ensure uint8
                if hm.dtype != np.uint8:
                    hm = lt._to_saliency_map(hm)
                save_heatmap_gray(hm, base / f"heat_{m}.png")
                save_overlay(rgb, hm, base / f"overlay_{m}.png")
            except Exception as exc:
                print("Error computing heatmap", m, exc)
        count += 1


if __name__ == "__main__":
    main()
