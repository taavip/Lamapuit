#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import laspy
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _counts(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(values.tolist()).items())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reclassify tile LAZ files with Myria3D RandLA-Net pretrained model"
    )
    p.add_argument("--input-dir", type=Path, default=Path("data/lamapuit/laz"))
    p.add_argument("--tile-id", default="436646")
    p.add_argument("--years", default="2018,2020,2022,2024")
    p.add_argument("--myria3d-root", type=Path, default=Path("myria3d_tmp/myria3d"))
    p.add_argument("--model", default="urban_")
    p.add_argument("--epsg", default="3301")
    p.add_argument("--gpus", default="[0]")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument(
        "--max-randla-points",
        type=int,
        default=50000000,
        help="If input exceeds this size, run RandLA on a sampled subset and remap classes to the full cloud",
    )
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("output/laz_reclassified_randla"))
    p.add_argument("--tmp-dir", type=Path, default=Path("output/laz_reclassified_randla/.tmp"))
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("output/laz_reclassified_randla/summary_436646_randla_myria3d.json"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    if not years:
        raise ValueError("No years provided")

    input_dir = (ROOT / args.input_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = (ROOT / args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    myria3d_root = (ROOT / args.myria3d_root).resolve()
    runner = myria3d_root / "run_inference.py"
    if not runner.exists():
        raise FileNotFoundError(f"run_inference.py not found: {runner}")

    results: list[dict[str, object]] = []

    for year in years:
        in_laz = input_dir / f"{args.tile_id}_{year}_madal.laz"
        if not in_laz.exists():
            raise FileNotFoundError(f"Missing input LAZ: {in_laz}")

        out_laz = out_dir / f"{in_laz.stem}_reclassified_randla.laz"
        if out_laz.exists():
            out_laz.unlink()

        src = laspy.read(str(in_laz))
        src_cls = np.asarray(src.classification, dtype=np.uint8)

        sampled_for_inference = src_cls.shape[0] > int(args.max_randla_points)
        class_map: dict[int, int] = {}
        n_points_inference = int(src_cls.shape[0])
        pred_counts_inference: dict[str, int] = {}

        print(f"\n[RandLA] Processing {in_laz.name}")
        if sampled_for_inference:
            rng = np.random.default_rng(int(args.sample_seed) + int(year))
            infer_indices = np.sort(
                rng.choice(src_cls.shape[0], size=int(args.max_randla_points), replace=False)
            )
            n_points_inference = int(infer_indices.shape[0])

            sampled_in = tmp_dir / f"{in_laz.stem}_sampled_for_randla.laz"
            sampled_out = tmp_dir / f"{in_laz.stem}_sampled_for_randla_pred.laz"
            if sampled_in.exists():
                sampled_in.unlink()
            if sampled_out.exists():
                sampled_out.unlink()

            sampled_las = laspy.LasData(src.header)
            sampled_las.points = src.points[infer_indices]
            sampled_las.write(str(sampled_in))

            cmd = [
                sys.executable,
                str(runner),
                "--input",
                str(sampled_in),
                "--output",
                str(sampled_out),
                "--model",
                args.model,
                "--epsg",
                str(args.epsg),
                "--gpus",
                str(args.gpus),
                "--batch-size",
                str(args.batch_size),
            ]
            _run(cmd, cwd=myria3d_root)

            pred_sampled = laspy.read(str(sampled_out))
            src_cls_infer = np.asarray(sampled_las.classification, dtype=np.uint8)
            pred_cls_infer = np.asarray(pred_sampled.classification, dtype=np.uint8)
            if src_cls_infer.shape[0] != pred_cls_infer.shape[0]:
                raise RuntimeError(
                    f"Sampled prediction mismatch for {in_laz.name}: "
                    f"src={src_cls_infer.shape[0]} pred={pred_cls_infer.shape[0]}"
                )

            pred_counts_inference = _counts(pred_cls_infer)
            pred_cls = src_cls.copy()
            pred_cls[infer_indices] = pred_cls_infer

            src.classification = pred_cls
            src.write(str(out_laz))

            if sampled_in.exists():
                sampled_in.unlink()
            if sampled_out.exists():
                sampled_out.unlink()
        else:
            cmd = [
                sys.executable,
                str(runner),
                "--input",
                str(in_laz),
                "--output",
                str(out_laz),
                "--model",
                args.model,
                "--epsg",
                str(args.epsg),
                "--gpus",
                str(args.gpus),
                "--batch-size",
                str(args.batch_size),
            ]
            _run(cmd, cwd=myria3d_root)

            pred = laspy.read(str(out_laz))
            pred_cls = np.asarray(pred.classification, dtype=np.uint8)
            pred_counts_inference = _counts(pred_cls)

        if src_cls.shape[0] != pred_cls.shape[0]:
            raise RuntimeError(
                f"Point count mismatch for {in_laz.name}: "
                f"src={src_cls.shape[0]} pred={pred_cls.shape[0]}"
            )

        changed_mask = src_cls != pred_cls
        run_summary = {
            "input_laz": str(in_laz),
            "output_laz": str(out_laz),
            "n_points": int(src_cls.shape[0]),
            "n_points_inference": int(n_points_inference),
            "sampled_for_inference": bool(sampled_for_inference),
            "sample_assignment": "direct_point_indices" if sampled_for_inference else "full_cloud",
            "class_map_from_sample": {str(k): int(v) for k, v in sorted(class_map.items())},
            "n_changed": int(np.count_nonzero(changed_mask)),
            "changed_ratio": float(np.count_nonzero(changed_mask) / max(src_cls.shape[0], 1)),
            "original_class_counts": _counts(src_cls),
            "randla_pred_counts_inference": pred_counts_inference,
            "new_class_counts": _counts(pred_cls),
        }
        results.append(run_summary)
        print(json.dumps(run_summary, indent=2))

    summary = {
        "tile_id": args.tile_id,
        "years": years,
        "model": args.model,
        "epsg": str(args.epsg),
        "gpus": str(args.gpus),
        "batch_size": int(args.batch_size),
        "results": results,
    }

    summary_path = (ROOT / args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())