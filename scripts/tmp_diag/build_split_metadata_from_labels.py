#!/usr/bin/env python3
"""Build tile_metadata-style CSV for split leakage checks from label CSVs + test split keys."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import rasterio


def _load_test_keys(path: Path) -> set[tuple[str, int, int]]:
    data = json.loads(path.read_text())
    keys = data.get("keys", []) if isinstance(data, dict) else []
    out: set[tuple[str, int, int]] = set()
    for k in keys:
        if not isinstance(k, list) or len(k) != 3:
            continue
        out.add((str(k[0]), int(k[1]), int(k[2])))
    return out


def _label_rows(labels_dir: Path):
    for p in sorted(labels_dir.glob("*_labels.csv")):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                label = str(row.get("label", "")).strip()
                if label not in ("cdw", "no_cdw"):
                    continue
                raster = str(row.get("raster", "")).strip()
                if not raster:
                    continue
                yield raster, int(row.get("row_off", 0)), int(row.get("col_off", 0)), label


def _tile_bounds(transform, row_off: int, col_off: int, chunk_size: int):
    # Affine coefficients:
    # x = a*col + b*row + c
    # y = d*col + e*row + f
    x0, y0 = transform * (col_off, row_off)
    x1, y1 = transform * (col_off + chunk_size, row_off + chunk_size)
    minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
    miny, maxy = (y0, y1) if y0 <= y1 else (y1, y0)
    return minx, miny, maxx, maxy


def main() -> int:
    p = argparse.ArgumentParser(description="Build split metadata CSV from labels + test split")
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--chm-dir", type=Path, required=True)
    p.add_argument("--test-split", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.test_split.exists():
        raise FileNotFoundError(args.test_split)

    test_keys = _load_test_keys(args.test_split)

    # Cache raster transforms
    transforms: dict[str, object] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        fields = [
            "tile",
            "split",
            "raster",
            "row_off",
            "col_off",
            "minx",
            "miny",
            "maxx",
            "maxy",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        n = 0
        for raster, ro, co, _label in _label_rows(args.labels_dir):
            if raster not in transforms:
                rp = args.chm_dir / raster
                with rasterio.open(rp) as src:
                    transforms[raster] = src.transform
            tr = transforms[raster]
            minx, miny, maxx, maxy = _tile_bounds(tr, ro, co, args.chunk_size)
            split = "test" if (raster, ro, co) in test_keys else "train"
            tile = f"{Path(raster).stem}_r{ro}_c{co}"
            w.writerow(
                {
                    "tile": tile,
                    "split": split,
                    "raster": raster,
                    "row_off": ro,
                    "col_off": co,
                    "minx": minx,
                    "miny": miny,
                    "maxx": maxx,
                    "maxy": maxy,
                }
            )
            n += 1

    print(f"rows={n}")
    print(f"test_keys={len(test_keys)}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
