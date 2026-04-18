#!/usr/bin/env python3
"""Focused RandLA CHM experiment with strict class-2 DTM, last returns, and HAG 0-1.3 m.

This experiment is isolated from broader CDW experiments and is meant for quick
verification over LAZ files in output/laz_reclassified_randla.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import rasterio

# Import the canonical CHM builder from scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from process_laz_to_chm_improved import compute_hag_raster_streamed


def parse_class_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def counter_to_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter)}


def collect_filter_stats(
    laz_path: Path,
    exclude_classes: list[int],
    return_mode: str,
    chunk_size: int,
) -> dict[str, Any]:
    exclude_set = set(exclude_classes)

    class_counts_total: Counter[int] = Counter()
    class_counts_after_filters: Counter[int] = Counter()
    return_counts_after_filters: Counter[str] = Counter()

    total_points = 0
    class2_points = 0
    points_after_class_filter = 0
    points_after_filters = 0
    return_attrs_present = True

    with laspy.open(str(laz_path)) as fh:
        for points in fh.chunk_iterator(chunk_size):
            npts = len(points.x)
            total_points += int(npts)

            cls = None
            try:
                cls = np.asarray(points.classification, dtype=np.int32)
            except Exception:
                cls = None

            if cls is not None:
                vals, cnts = np.unique(cls, return_counts=True)
                class_counts_total.update({int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())})
                class2_points += int(np.sum(cls == 2))

            keep = np.ones(npts, dtype=bool)
            if cls is not None and exclude_set:
                keep &= ~np.isin(cls, np.array(sorted(exclude_set), dtype=np.int32))
            points_after_class_filter += int(np.sum(keep))

            rn = None
            nr = None
            try:
                rn = np.asarray(points.return_number, dtype=np.int16)
                nr = np.asarray(points.number_of_returns, dtype=np.int16)
            except Exception:
                rn = None
                nr = None

            if return_mode != "all":
                if rn is None or nr is None:
                    return_attrs_present = False
                    keep &= False
                elif return_mode == "last":
                    keep &= rn == nr
                elif return_mode == "last2":
                    cutoff = np.maximum(1, nr - 1)
                    keep &= rn >= cutoff
                else:
                    raise ValueError(f"Unsupported return mode: {return_mode}")

            points_after_filters += int(np.sum(keep))

            if np.any(keep) and cls is not None:
                vals, cnts = np.unique(cls[keep], return_counts=True)
                class_counts_after_filters.update(
                    {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}
                )

            if np.any(keep):
                if rn is not None and nr is not None:
                    rnk = rn[keep]
                    nrk = nr[keep]
                    return_counts_after_filters["all"] += int(rnk.size)
                    return_counts_after_filters["last"] += int(np.sum(rnk == nrk))
                    cutoff = np.maximum(1, nrk - 1)
                    return_counts_after_filters["last2"] += int(np.sum(rnk >= cutoff))
                else:
                    return_counts_after_filters["unknown"] += int(np.sum(keep))

    residual_excluded_classes = [
        cls for cls in exclude_classes if int(class_counts_after_filters.get(cls, 0)) > 0
    ]

    return {
        "total_points": int(total_points),
        "class2_points": int(class2_points),
        "points_after_class_filter": int(points_after_class_filter),
        "points_after_filters": int(points_after_filters),
        "return_attrs_present": bool(return_attrs_present),
        "class_counts_total": counter_to_dict(class_counts_total),
        "class_counts_after_filters": counter_to_dict(class_counts_after_filters),
        "return_counts_after_filters": {
            k: int(v) for k, v in sorted(return_counts_after_filters.items(), key=lambda kv: kv[0])
        },
        "residual_excluded_classes": residual_excluded_classes,
    }


def summarize_tif(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        vals = arr.compressed().astype(np.float32)
        return {
            "chm_width": int(src.width),
            "chm_height": int(src.height),
            "chm_nodata": float(src.nodata) if src.nodata is not None else None,
            "chm_valid_px": int(vals.size),
            "chm_min": float(vals.min()) if vals.size else None,
            "chm_max": float(vals.max()) if vals.size else None,
            "chm_mean": float(vals.mean()) if vals.size else None,
            "chm_std": float(vals.std()) if vals.size else None,
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "status",
        "input_laz",
        "single_chm_tif",
        "resolution",
        "hag_min",
        "hag_max",
        "return_mode",
        "exclude_classes",
        "class2_points",
        "total_points",
        "points_after_class_filter",
        "points_after_filters",
        "return_attrs_present",
        "residual_excluded_classes",
        "chm_valid_px",
        "chm_min",
        "chm_max",
        "chm_mean",
        "chm_std",
        "error",
        "class_counts_total",
        "class_counts_after_filters",
        "return_counts_after_filters",
    ]

    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for row in rows:
            out = {}
            for key in ordered:
                val = row.get(key, "")
                if isinstance(val, (dict, list)):
                    out[key] = json.dumps(val, sort_keys=True)
                else:
                    out[key] = val
            w.writerow(out)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("rows", [])
    lines: list[str] = []
    lines.append("# RandLA CHM Experiment Report")
    lines.append("")
    lines.append(f"Generated: {payload.get('generated_at', '')}")
    lines.append("")
    lines.append("## Configuration")
    cfg = payload.get("config", {})
    lines.append(f"- Input dir: {cfg.get('input_dir', '')}")
    lines.append(f"- Pattern: {cfg.get('pattern', '')}")
    lines.append(f"- Return mode: {cfg.get('return_mode', '')}")
    lines.append(f"- Excluded classes: {cfg.get('exclude_classes', '')}")
    lines.append(f"- Resolution: {cfg.get('resolution', '')}")
    lines.append(f"- HAG range: {cfg.get('hag_min', 0.0)} to {cfg.get('hag_max', '')} m")
    lines.append(f"- Strict class-2 DTM: {cfg.get('strict_class2', True)}")
    lines.append("")
    lines.append("## Outcome")
    lines.append(f"- Total rows: {payload.get('total_rows', 0)}")
    lines.append(f"- OK rows: {payload.get('ok_rows', 0)}")
    lines.append(f"- Error rows: {payload.get('error_rows', 0)}")
    lines.append("")
    lines.append("## Per File")
    for row in rows:
        lines.append(
            "- "
            + f"{row.get('status', '')} | "
            + f"{Path(str(row.get('input_laz', ''))).name} | "
            + f"class2={row.get('class2_points', '')} | "
            + f"kept={row.get('points_after_filters', '')} | "
            + f"valid_px={row.get('chm_valid_px', '')}"
        )
        if row.get("residual_excluded_classes"):
            lines.append(f"  residual excluded classes after filters: {row['residual_excluded_classes']}")
        if row.get("error"):
            lines.append(f"  error: {row['error']}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    default_root = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description=(
            "Run focused RandLA CHM experiment with class-2 DTM, class exclusion, "
            "last-return filtering, and strict HAG drop mode."
        )
    )
    p.add_argument("--input-dir", type=Path, default=Path("output/laz_reclassified_randla"))
    p.add_argument("--pattern", default="*_reclassified_randla.laz")
    p.add_argument("--results-dir", type=Path, default=default_root / "results")
    p.add_argument("--analysis-dir", type=Path, default=default_root / "analysis")
    p.add_argument(
        "--exclude-classes",
        default="6,17",
        help="Comma-separated classes to remove (default removes core man-made classes: 6,17)",
    )
    p.add_argument("--return-mode", choices=["all", "last", "last2"], default="last")
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--hag-max", type=float, default=1.3)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument(
        "--allow-ground-fallback",
        action="store_true",
        help="Allow processing when a file has zero class-2 points (fallback may use all points).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    strict_class2 = not args.allow_ground_fallback
    exclude_classes = parse_class_list(args.exclude_classes)

    laz_files = sorted(args.input_dir.glob(args.pattern))
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found with pattern {args.pattern} in {args.input_dir}")

    chm_dir = args.results_dir / "chm"
    chm_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    res_cm = int(round(args.resolution * 100.0))

    for laz_path in laz_files:
        row: dict[str, Any] = {
            "status": "pending",
            "input_laz": str(laz_path),
            "single_chm_tif": "",
            "resolution": float(args.resolution),
            "hag_min": 0.0,
            "hag_max": float(args.hag_max),
            "return_mode": args.return_mode,
            "exclude_classes": ",".join(str(x) for x in exclude_classes),
            "error": "",
        }

        try:
            stats = collect_filter_stats(
                laz_path=laz_path,
                exclude_classes=exclude_classes,
                return_mode=args.return_mode,
                chunk_size=args.chunk_size,
            )
            row.update(stats)

            if strict_class2 and int(stats["class2_points"]) == 0:
                raise RuntimeError("No class-2 points found; strict class-2 DTM requires class 2")

            if args.return_mode != "all" and not bool(stats["return_attrs_present"]):
                raise RuntimeError(
                    "Return attributes missing; cannot enforce last/last2 return filtering"
                )

            out_tif = chm_dir / f"{laz_path.stem}_chm_max_hag_{res_cm}cm_{args.return_mode}.tif"
            row["single_chm_tif"] = str(out_tif)

            if args.dry_run:
                row["status"] = "dry_run"
            else:
                if out_tif.exists() and not args.overwrite:
                    row["status"] = "skipped_existing"
                else:
                    compute_hag_raster_streamed(
                        laz_path=laz_path,
                        out_tif=out_tif,
                        resolution=args.resolution,
                        hag_max=args.hag_max,
                        nodata=-9999.0,
                        chunk_size=args.chunk_size,
                        cog=False,
                        drop_above_hag_max=True,
                        exclude_classes=exclude_classes,
                        return_mode=args.return_mode,
                    )
                    row["status"] = "ok"

                if out_tif.exists():
                    row.update(summarize_tif(out_tif))

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input_dir": str(args.input_dir),
            "pattern": args.pattern,
            "results_dir": str(args.results_dir),
            "analysis_dir": str(args.analysis_dir),
            "exclude_classes": ",".join(str(x) for x in exclude_classes),
            "return_mode": args.return_mode,
            "resolution": float(args.resolution),
            "hag_min": 0.0,
            "hag_max": float(args.hag_max),
            "strict_class2": strict_class2,
            "chunk_size": int(args.chunk_size),
            "overwrite": bool(args.overwrite),
            "dry_run": bool(args.dry_run),
        },
        "rows": rows,
        "total_rows": len(rows),
        "ok_rows": sum(1 for r in rows if r.get("status") in {"ok", "skipped_existing", "dry_run"}),
        "error_rows": sum(1 for r in rows if r.get("status") == "error"),
    }

    report_json = args.analysis_dir / "experiment_report.json"
    report_csv = args.analysis_dir / "experiment_report.csv"
    report_md = args.analysis_dir / "experiment_report.md"

    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(report_csv, rows)
    write_markdown(report_md, payload)

    print(f"Wrote JSON report: {report_json}")
    print(f"Wrote CSV report:  {report_csv}")
    print(f"Wrote MD report:   {report_md}")

    if payload["error_rows"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
