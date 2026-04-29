#!/usr/bin/env python3
"""Count points with HAG in a given range and report classification/return stats.

Default target is tile 436646 2018 using the fused surface produced earlier.
Writes JSON and CSV summaries into `analysis/cdw_experiments_436646/`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import laspy
import numpy as np
import rasterio


def _read_chunk_arrays(points):
    d = {
        "x": np.asarray(points.x, dtype=np.float64),
        "y": np.asarray(points.y, dtype=np.float64),
        "z": np.asarray(points.z, dtype=np.float32),
        "classification": None,
        "return_number": None,
        "number_of_returns": None,
    }
    try:
        d["classification"] = np.asarray(points.classification, dtype=np.int16)
    except Exception:
        pass
    try:
        d["return_number"] = np.asarray(points.return_number, dtype=np.int16)
        d["number_of_returns"] = np.asarray(points.number_of_returns, dtype=np.int16)
    except Exception:
        pass
    return d


def main() -> int:
    p = argparse.ArgumentParser(description="Count HAG-range points and report class/return stats")
    p.add_argument("--laz", type=Path, default=Path("data/lamapuit/laz/436646_2018_madal.laz"))
    p.add_argument(
        "--fused",
        type=Path,
        default=Path("data/lamapuit/cdw_experiments_436646_2018_ground_last_vs_last2/median_ground/436646_2018_madal_fused_surface_median_ground_20cm.tif"),
    )
    p.add_argument("--hag-min", type=float, default=0.0)
    p.add_argument("--hag-max", type=float, default=1.3)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("analysis/cdw_experiments_436646/counts_hag_0_1p3_2018.json"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("analysis/cdw_experiments_436646/counts_hag_0_1p3_2018.csv"),
    )
    args = p.parse_args()

    if not args.laz.exists():
        print("LAZ missing:", args.laz)
        return 2
    if not args.fused.exists():
        print("Fused surface missing:", args.fused)
        return 3

    with rasterio.open(args.fused) as src:
        fused = src.read(1)
        origin_x = float(src.transform.c)
        max_y = float(src.transform.f)
        res = float(abs(src.transform.a))
        height = src.height
        width = src.width
        fused_nodata = src.nodata if src.nodata is not None else -9999.0

    total_points_seen = 0
    total_mapped = 0
    total_in_hag = 0

    class_counts = {}
    class6_count = 0
    class9_count = 0

    return_counts = {"last": 0, "second_last": 0, "third_last": 0, "unknown": 0}
    return_counts_after_class_filter = {k: 0 for k in return_counts}

    with laspy.open(str(args.laz)) as fh:
        for points in fh.chunk_iterator(args.chunk_size):
            d = _read_chunk_arrays(points)
            x = d["x"]
            y = d["y"]
            z = d["z"]
            n = len(x)
            if n == 0:
                continue

            total_points_seen += int(n)

            col = ((x - origin_x) / res).astype(np.int64)
            row = ((max_y - y) / res).astype(np.int64)

            valid_rc = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            if not valid_rc.any():
                continue

            row = row[valid_rc]
            col = col[valid_rc]
            z = z[valid_rc]
            cls = d["classification"][valid_rc] if d["classification"] is not None else None
            rn = d["return_number"][valid_rc] if d["return_number"] is not None else None
            nr = d["number_of_returns"][valid_rc] if d["number_of_returns"] is not None else None

            total_mapped += int(row.size)

            ground = fused[row, col]
            valid_ground = (ground != fused_nodata) & np.isfinite(ground)
            if not valid_ground.any():
                continue

            row = row[valid_ground]
            col = col[valid_ground]
            z = z[valid_ground]
            ground = ground[valid_ground]
            if cls is not None:
                cls = cls[valid_ground]
            if rn is not None:
                rn = rn[valid_ground]
            if nr is not None:
                nr = nr[valid_ground]

            hag = z - ground
            in_hag = (hag >= args.hag_min) & (hag <= args.hag_max)
            if not in_hag.any():
                continue

            total_in_hag += int(in_hag.sum())

            # classification counts
            if cls is not None:
                cls_sel = cls[in_hag]
                unique, counts = np.unique(cls_sel, return_counts=True)
                for k, c in zip(unique.tolist(), counts.tolist()):
                    class_counts[int(k)] = class_counts.get(int(k), 0) + int(c)
                class6_count += int((cls_sel == 6).sum())
                class9_count += int((cls_sel == 9).sum())

            # return-mode counts
            if rn is None or nr is None:
                return_counts["unknown"] += int(in_hag.sum())
                # after filter: if no cls info, assume not filtered
                return_counts_after_class_filter["unknown"] += int(in_hag.sum())
            else:
                rn_sel = rn[in_hag]
                nr_sel = nr[in_hag]
                last_mask = rn_sel == nr_sel
                second_mask = (nr_sel >= 2) & (rn_sel == (nr_sel - 1))
                third_mask = (nr_sel >= 3) & (rn_sel == (nr_sel - 2))
                other_mask = ~(last_mask | second_mask | third_mask)

                return_counts["last"] += int(last_mask.sum())
                return_counts["second_last"] += int(second_mask.sum())
                return_counts["third_last"] += int(third_mask.sum())
                return_counts["unknown"] += int(other_mask.sum())

                # after class filter
                if cls is None:
                    # no classification information -> treat as kept
                    return_counts_after_class_filter["last"] += int(last_mask.sum())
                    return_counts_after_class_filter["second_last"] += int(second_mask.sum())
                    return_counts_after_class_filter["third_last"] += int(third_mask.sum())
                    return_counts_after_class_filter["unknown"] += int(other_mask.sum())
                else:
                    cls_sel = cls[in_hag]
                    keep_mask = (cls_sel != 6) & (cls_sel != 9)
                    if keep_mask.any():
                        return_counts_after_class_filter["last"] += int((last_mask & keep_mask).sum())
                        return_counts_after_class_filter["second_last"] += int((second_mask & keep_mask).sum())
                        return_counts_after_class_filter["third_last"] += int((third_mask & keep_mask).sum())
                        return_counts_after_class_filter["unknown"] += int((other_mask & keep_mask).sum())

    summary = {
        "laz": str(args.laz),
        "fused_surface": str(args.fused),
        "hag_min": args.hag_min,
        "hag_max": args.hag_max,
        "total_points_seen": int(total_points_seen),
        "total_mapped_to_grid": int(total_mapped),
        "total_in_hag_range": int(total_in_hag),
        "class_counts_in_hag_range": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "class6_count_in_hag_range": int(class6_count),
        "class9_count_in_hag_range": int(class9_count),
        "return_counts_in_hag_range": return_counts,
        "return_counts_after_excluding_classes_6_9": return_counts_after_class_filter,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"]) 
        w.writerow(["laz", summary["laz"]])
        w.writerow(["fused_surface", summary["fused_surface"]])
        w.writerow(["hag_min", summary["hag_min"]])
        w.writerow(["hag_max", summary["hag_max"]])
        w.writerow(["total_points_seen", summary["total_points_seen"]])
        w.writerow(["total_mapped_to_grid", summary["total_mapped_to_grid"]])
        w.writerow(["total_in_hag_range", summary["total_in_hag_range"]])
        w.writerow(["class6_count_in_hag_range", summary["class6_count_in_hag_range"]])
        w.writerow(["class9_count_in_hag_range", summary["class9_count_in_hag_range"]])
        for k, v in summary["class_counts_in_hag_range"].items():
            w.writerow([f"class_{k}_count_in_hag_range", v])
        for k, v in summary["return_counts_in_hag_range"].items():
            w.writerow([f"return_{k}_count_in_hag_range", v])
        for k, v in summary["return_counts_after_excluding_classes_6_9"].items():
            w.writerow([f"return_{k}_count_after_excluding_6_9", v])

    print("Summary written:", args.out_json, args.out_csv)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
