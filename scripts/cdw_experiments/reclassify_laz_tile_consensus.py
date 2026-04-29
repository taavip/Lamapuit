#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import laspy
import numpy as np


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _accumulate_grid_votes(
    source_laz_paths: list[Path],
    classes_to_vote: list[int],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    grid_res: float,
    chunk_size: int,
) -> tuple[np.ndarray, int, int]:
    width = int(np.ceil((xmax - xmin) / grid_res)) + 1
    height = int(np.ceil((ymax - ymin) / grid_res)) + 1
    n_cells = width * height

    class_to_idx = {c: i for i, c in enumerate(classes_to_vote)}
    votes = np.zeros((len(classes_to_vote), n_cells), dtype=np.int32)

    for src in source_laz_paths:
        print(f"Accumulating votes from {src}")
        with laspy.open(str(src)) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                cls = np.asarray(chunk.classification, dtype=np.uint8)
                xs = np.asarray(chunk.x, dtype=np.float64)
                ys = np.asarray(chunk.y, dtype=np.float64)

                ix = np.floor((xs - xmin) / grid_res).astype(np.int32)
                iy = np.floor((ys - ymin) / grid_res).astype(np.int32)

                in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
                if not np.any(in_bounds):
                    continue

                flat_all = iy[in_bounds] * width + ix[in_bounds]
                cls_all = cls[in_bounds]

                for c in classes_to_vote:
                    m = cls_all == c
                    if not np.any(m):
                        continue
                    flat = flat_all[m]
                    binc = np.bincount(flat, minlength=n_cells)
                    votes[class_to_idx[c]] += binc.astype(np.int32, copy=False)

    return votes, width, height


def main() -> int:
    p = argparse.ArgumentParser(
        description="Reclassify target LAZ using temporal class consensus (no model training)"
    )
    p.add_argument("--input-dir", type=Path, default=Path("data/lamapuit/laz"))
    p.add_argument("--tile-id", default="436646")
    p.add_argument("--target-year", type=int, default=2018)
    p.add_argument("--source-years", default="2020,2022,2024")
    p.add_argument("--update-classes", default="2,6", help="Classes to update via consensus")
    p.add_argument("--preserve-classes", default="9", help="Classes to keep unchanged")
    p.add_argument("--grid-res", type=float, default=0.5)
    p.add_argument("--min-votes", type=int, default=5)
    p.add_argument("--min-ratio", type=float, default=0.6)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--out-dir", type=Path, default=Path("output/laz_reclassified_consensus"))
    args = p.parse_args()

    source_years = [int(x.strip()) for x in args.source_years.split(",") if x.strip()]
    update_classes = _parse_int_list(args.update_classes)
    preserve_classes = set(_parse_int_list(args.preserve_classes))

    if sorted(update_classes) != [2, 6]:
        raise ValueError("This script currently supports update classes 2 and 6")

    target_laz = args.input_dir / f"{args.tile_id}_{args.target_year}_madal.laz"
    source_laz = [args.input_dir / f"{args.tile_id}_{y}_madal.laz" for y in source_years]

    if not target_laz.exists():
        raise FileNotFoundError(f"Target LAZ not found: {target_laz}")
    for src in source_laz:
        if not src.exists():
            raise FileNotFoundError(f"Source LAZ not found: {src}")

    with laspy.open(str(target_laz)) as reader:
        xmin, ymin, _ = reader.header.mins
        xmax, ymax, _ = reader.header.maxs

    votes, width, height = _accumulate_grid_votes(
        source_laz_paths=source_laz,
        classes_to_vote=update_classes,
        xmin=float(xmin),
        ymin=float(ymin),
        xmax=float(xmax),
        ymax=float(ymax),
        grid_res=args.grid_res,
        chunk_size=args.chunk_size,
    )

    votes2 = votes[update_classes.index(2)]
    votes6 = votes[update_classes.index(6)]

    las = laspy.read(str(target_laz))
    original_cls = np.asarray(las.classification, dtype=np.uint8).copy()
    new_cls = original_cls.copy()

    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    ix = np.floor((xs - float(xmin)) / args.grid_res).astype(np.int32)
    iy = np.floor((ys - float(ymin)) / args.grid_res).astype(np.int32)

    in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    flat = np.zeros_like(ix, dtype=np.int64)
    flat[in_bounds] = iy[in_bounds].astype(np.int64) * width + ix[in_bounds].astype(np.int64)

    update_mask = np.isin(original_cls, np.array(update_classes, dtype=np.uint8))
    keep_mask = np.zeros_like(update_mask)
    for cls_keep in preserve_classes:
        keep_mask |= original_cls == cls_keep

    active = in_bounds & update_mask & (~keep_mask)

    p2 = np.zeros_like(xs, dtype=np.int32)
    p6 = np.zeros_like(xs, dtype=np.int32)
    p2[active] = votes2[flat[active]]
    p6[active] = votes6[flat[active]]

    total = p2 + p6
    max_votes = np.maximum(p2, p6)
    ratio = np.zeros_like(xs, dtype=np.float32)
    nz = total > 0
    ratio[nz] = max_votes[nz] / total[nz]

    pred = np.where(p6 > p2, 6, 2).astype(np.uint8)
    confident = active & (total >= args.min_votes) & (ratio >= args.min_ratio) & (p2 != p6)
    new_cls[confident] = pred[confident]

    for cls_keep in preserve_classes:
        keep = original_cls == cls_keep
        new_cls[keep] = original_cls[keep]

    las.classification = new_cls

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_laz = args.out_dir / f"{args.tile_id}_{args.target_year}_madal_reclassified_consensus.laz"
    las.write(str(out_laz))

    changed = original_cls != new_cls
    changed_count = int(np.count_nonzero(changed))

    transitions = Counter(
        zip(original_cls[changed].tolist(), new_cls[changed].tolist())
    )

    summary = {
        "target_laz": str(target_laz),
        "source_laz": [str(p) for p in source_laz],
        "output_laz": str(out_laz),
        "method": "temporal_consensus_grid",
        "params": {
            "grid_res": args.grid_res,
            "min_votes": args.min_votes,
            "min_ratio": args.min_ratio,
            "update_classes": update_classes,
            "preserve_classes": sorted(preserve_classes),
            "chunk_size": args.chunk_size,
        },
        "n_points": int(len(new_cls)),
        "original_class_counts": {str(k): int(v) for k, v in sorted(Counter(original_cls.tolist()).items())},
        "predicted_class_counts": {str(k): int(v) for k, v in sorted(Counter(new_cls.tolist()).items())},
        "changed_points": changed_count,
        "changed_pct": float(changed_count * 100.0 / len(new_cls)),
        "top_transitions": [
            {"from": int(src), "to": int(dst), "count": int(n)}
            for (src, dst), n in transitions.most_common(20)
        ],
        "preserve_class_9_unchanged": bool(np.array_equal(new_cls[original_cls == 9], original_cls[original_cls == 9])),
    }

    summary_path = args.out_dir / f"{args.tile_id}_{args.target_year}_madal_reclassified_consensus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
