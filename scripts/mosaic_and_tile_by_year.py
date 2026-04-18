#!/usr/bin/env python3
"""
mosaic_and_tile_by_year.py

Merge contiguous CHM tiles per year, run the Label Studio PNG tiler,
and produce a manifest CSV per year with tile metadata.

Outputs:
- mosaics -> data/car/chm_2m_mosaic/<year>/{source}.tif
- PNG tiles -> data/car/chm_2m_label/<year>/tiles/
- manifest -> data/car/chm_2m_label/<year>/manifest.csv

Usage: python scripts/mosaic_and_tile_by_year.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

def parse_args():
    p = argparse.ArgumentParser(description="Mosaic CHM tiles by year and produce Label Studio PNGs + manifests")
    p.add_argument("--input-dir", type=Path, default=Path("data/car/chm_2m"))
    p.add_argument("--mosaic-out", type=Path, default=Path("data/car/chm_2m_mosaic"))
    p.add_argument("--label-out", type=Path, default=Path("data/car/chm_2m_label"))
    p.add_argument("--tile-size", type=int, default=160)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tolerance-px", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    # treat zero-valued pixels as nodata when building masks
    try:
        p.add_argument("--zero-nodata", action=argparse.BooleanOptionalAction, default=True,
                       help="Treat 0 values as nodata when deriving mosaic masks")
    except Exception:
        p.add_argument("--zero-nodata", action="store_true", default=True,
                       help="Treat 0 values as nodata when deriving mosaic masks")
    return p.parse_args()


def parse_filename(stem: str) -> Tuple[str, int]:
    m = re.match(r"^(\d{6})_(\d{4})", stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_tiles(input_dir: Path):
    import rasterio

    tiles = []
    for tif in sorted(input_dir.glob("*.tif")):
        code, year = parse_filename(tif.stem)
        if code is None:
            continue
        with rasterio.open(tif) as src:
            bounds = src.bounds
            res = abs(src.res[0])
            tiles.append({
                "path": tif,
                "code": code,
                "year": year,
                "minx": bounds.left,
                "miny": bounds.bottom,
                "maxx": bounds.right,
                "maxy": bounds.top,
                "width": src.width,
                "height": src.height,
                "res": res,
            })
    return tiles


def build_components(tiles: List[Dict], tolerance_px: float = 1.0) -> Dict[int, List[List[Dict]]]:
    # Group tiles by year first
    by_year = defaultdict(list)
    for t in tiles:
        by_year[t["year"]].append(t)

    comps_by_year = {}
    for year, tlist in by_year.items():
        n = len(tlist)
        adj = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                t1 = tlist[i]
                t2 = tlist[j]
                # Use symmetric tolerance across both tiles
                res = max(float(t1["res"]), float(t2["res"]))
                tol = tolerance_px * res

                def overlap_1d(a0: float, a1: float, b0: float, b1: float, epsilon: float) -> bool:
                    return min(a1, b1) >= (max(a0, b0) - epsilon)

                x_overlap = overlap_1d(t1["minx"], t1["maxx"], t2["minx"], t2["maxx"], tol)
                y_overlap = overlap_1d(t1["miny"], t1["maxy"], t2["miny"], t2["maxy"], tol)

                # Horizontal/vertical adjacency requires overlap on the orthogonal axis.
                horiz_touch = (
                    abs(t1["maxx"] - t2["minx"]) <= tol or abs(t2["maxx"] - t1["minx"]) <= tol
                ) and y_overlap
                vert_touch = (
                    abs(t1["maxy"] - t2["miny"]) <= tol or abs(t2["maxy"] - t1["miny"]) <= tol
                ) and x_overlap
                corner_touch = (
                    (abs(t1["maxx"] - t2["minx"]) <= tol and abs(t1["maxy"] - t2["miny"]) <= tol)
                    or (abs(t1["maxx"] - t2["minx"]) <= tol and abs(t1["miny"] - t2["maxy"]) <= tol)
                    or (abs(t1["minx"] - t2["maxx"]) <= tol and abs(t1["maxy"] - t2["miny"]) <= tol)
                    or (abs(t1["minx"] - t2["maxx"]) <= tol and abs(t1["miny"] - t2["maxy"]) <= tol)
                )
                if horiz_touch or vert_touch or corner_touch:
                    adj[i].add(j)
                    adj[j].add(i)

        # BFS to get connected components
        seen = set()
        comps = []
        for i in range(n):
            if i in seen:
                continue
            q = deque([i])
            comp_idx = []
            while q:
                u = q.popleft()
                if u in seen:
                    continue
                seen.add(u)
                comp_idx.append(u)
                for v in adj[u]:
                    if v not in seen:
                        q.append(v)
            comps.append([tlist[k] for k in comp_idx])
        comps_by_year[year] = comps
    return comps_by_year


def write_mosaic(members: List[Dict], out_path: Path, zero_nodata: bool = True):
    import rasterio
    from rasterio.merge import merge
    import numpy as np

    srcs = [rasterio.open(m["path"]) for m in members]
    try:
        mosaic, out_trans = merge(srcs, method="first")
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw",
            "tiled": True,
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # write merged chm
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic)
            try:
                dst.build_overviews([2, 4, 8, 16], resampling=rasterio.enums.Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")
            except Exception:
                pass

        # derive mask from mosaic
        band = mosaic[0]
        try:
            if zero_nodata:
                # treat zeros as nodata
                mask_bool = np.isfinite(band) & (band != 0)
            else:
                mask_bool = np.isfinite(band)
        except Exception:
            # fallback: non-zero
            mask_bool = band != 0

        mask_arr = (mask_bool.astype("uint8") * 255)[None, ...]
        mask_meta = out_meta.copy()
        # mask is a single-band uint8 image (0 or 255); ensure metadata matches
        mask_meta.update({"dtype": "uint8", "count": 1})
        # remove nodata from the mosaic metadata if present, it may be incompatible with uint8
        mask_meta.pop("nodata", None)
        mask_path = out_path.with_name(out_path.stem + "_mask.tif")
        with rasterio.open(mask_path, "w", **mask_meta) as mdst:
            mdst.write(mask_arr)
        print(f"Wrote mosaic: {out_path}")
        print(f"Wrote mosaic mask: {mask_path}")
    finally:
        for s in srcs:
            s.close()


def run_label_tiler(mosaic_path: Path, out_dir: Path, tile_size: int, overlap: float):
    # reuse scripts/rescale_tile_chm.py
    cmd = ["python", "scripts/rescale_tile_chm.py", "--input", str(mosaic_path), "--out-dir", str(out_dir), "--tile-size", str(tile_size), "--overlap", str(overlap)]
    subprocess.check_call(cmd)


def consolidate_manifest(out_dir: Path, year: int, source_name: str, label_year_dir: Path):
    # rescale_tile_chm writes metadata CSV at out_dir/{stem}/tile_metadata.csv
    stem = Path(source_name).stem
    meta_csv = out_dir / f"{stem}" / "tile_metadata.csv"
    images_dir = out_dir / f"{stem}" / "images"
    target_tiles_dir = label_year_dir / "tiles"
    target_tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    if not meta_csv.exists():
        return manifest
    with open(meta_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # image_path is relative to out_dir/{stem}
            img_rel = row["image_path"]
            # resolve image path: try relative to out_dir, then relative to out_dir/stem
            candidate1 = out_dir / img_rel
            candidate2 = out_dir / f"{stem}" / img_rel
            src_img = None
            if candidate1.exists():
                src_img = candidate1
            elif candidate2.exists():
                src_img = candidate2
            else:
                # if image_path accidentally includes the stem twice, try stripping it
                prefix = f"{stem}/"
                if img_rel.startswith(prefix):
                    stripped = img_rel[len(prefix):]
                    candidate3 = out_dir / f"{stem}" / stripped
                    if candidate3.exists():
                        src_img = candidate3
            if src_img is None:
                print(f"Warning: image not found for row: {img_rel}")
                continue
            # prefix filename with source_name to avoid collisions
            pref = f"{Path(source_name).stem}_"
            dst_name = pref + Path(src_img).name
            dst_path = target_tiles_dir / dst_name
            try:
                shutil.move(str(src_img), str(dst_path))
            except FileNotFoundError:
                print(f"Warning: failed to move missing file {src_img}")
                continue
            # copy row and augment
            minx = row.get("minx")
            manifest.append({
                "tile_filename": str(dst_path.relative_to(label_year_dir)),
                "minx": minx,
                "miny": row.get("miny"),
                "maxx": row.get("maxx"),
                "maxy": row.get("maxy"),
                "year": str(year),
                "mosaic_source": source_name,
                "row_off": row.get("row_off"),
                "col_off": row.get("col_off"),
            })
    # remove the now-empty images dir if present
    try:
        shutil.rmtree(out_dir / f"{stem}")
    except Exception:
        pass
    return manifest


def main():
    args = parse_args()
    tiles = load_tiles(args.input_dir)
    if not tiles:
        print("No tiles found in", args.input_dir)
        return 1

    comps_by_year = build_components(tiles, tolerance_px=args.tolerance_px)

    # For naming clusters, we will enumerate components per year
    for year, comps in sorted(comps_by_year.items()):
        print(f"Year {year}: {len(comps)} component(s)")
        label_year_dir = args.label_out / str(year)
        manifests = []
        for ci, comp in enumerate(comps, start=1):
            codes = sorted({m["code"] for m in comp})
            source_name = f"year{year}_comp{ci}_{'-'.join(codes)}"
            mosaic_path = args.mosaic_out / str(year) / f"{source_name}.tif"
            print(f"  Component {ci}: {codes} -> {mosaic_path}")
            if args.dry_run:
                continue
            # write mosaic
            write_mosaic(comp, mosaic_path, zero_nodata=args.zero_nodata)
            # run tiler
            run_label_tiler(mosaic_path, args.label_out / str(year), args.tile_size, args.overlap)
            # consolidate manifest entries and move tiles to label_year_dir/tiles
            mf = consolidate_manifest(args.label_out / str(year), year, source_name, label_year_dir)
            manifests.extend(mf)

        # write combined manifest for the year
        if not args.dry_run:
            label_year_dir.mkdir(parents=True, exist_ok=True)
            manifest_csv = label_year_dir / "manifest.csv"
            with open(manifest_csv, "w", newline="") as f:
                fieldnames = ["tile_filename", "minx", "miny", "maxx", "maxy", "year", "mosaic_source", "row_off", "col_off"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in manifests:
                    writer.writerow(r)
            print(f"Wrote manifest: {manifest_csv} (tiles: {len(manifests)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
