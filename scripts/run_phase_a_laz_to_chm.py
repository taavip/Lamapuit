#!/usr/bin/env python3
"""Phase A wrapper: LAZ -> CHM conversion with per-raster statistics report.

This script wraps `process_laz_to_chm_improved.py` conversion logic and writes:
- a machine-readable JSON report
- a per-raster CSV report

Default settings follow onboarding plan:
- hag_max = 1.3
- resolution = 0.2

Usage examples:
  python scripts/run_phase_a_laz_to_chm.py \
    --input-dir data/lamapuit/laz \
    --out data/lamapuit/chm_max_hag_13

  python scripts/run_phase_a_laz_to_chm.py \
    --input-dir data/lamapuit/laz \
    --pattern "*.laz" \
    --out data/lamapuit/chm_max_hag_13 \
    --workers 4 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import rasterio

from process_laz_to_chm_improved import compute_hag_raster_streamed, find_inputs, setup_logger


def _pct(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _summarize_raster(chm_path: Path) -> dict:
    with rasterio.open(chm_path) as src:
        arr = src.read(1, masked=True)
        valid = arr.compressed().astype(np.float32)
        total_px = int(src.width * src.height)
        valid_px = int(valid.size)
        nodata_px = total_px - valid_px
        tags = src.tags() or {}

        return {
            "output_chm": str(chm_path),
            "status": "ok",
            "width": int(src.width),
            "height": int(src.height),
            "resolution_x": float(abs(src.transform.a)),
            "resolution_y": float(abs(src.transform.e)),
            "crs": str(src.crs) if src.crs is not None else "",
            "nodata": float(src.nodata) if src.nodata is not None else None,
            "total_pixels": total_px,
            "valid_pixels": valid_px,
            "nodata_pixels": nodata_px,
            "nodata_pct": (100.0 * nodata_px / total_px) if total_px else 0.0,
            "hag_min": float(valid.min()) if valid_px else None,
            "hag_max_observed": float(valid.max()) if valid_px else None,
            "hag_mean": float(valid.mean()) if valid_px else None,
            "hag_std": float(valid.std()) if valid_px else None,
            "hag_p50": _pct(valid, 50.0),
            "hag_p95": _pct(valid, 95.0),
            "hag_p99": _pct(valid, 99.0),
            "tag_hag_max": tags.get("HAG_MAX", ""),
            "tag_filter_mode": tags.get("FILTER_MODE", ""),
            "tag_resolution": tags.get("RESOLUTION", ""),
            "tag_source_laz": tags.get("SOURCE_LAZ", ""),
            "tag_proc_date": tags.get("PROC_DATE", ""),
        }


def _out_tif_for_laz(laz_path: Path, out_dir: Path, resolution: float) -> Path:
    return out_dir / f"{laz_path.stem}_chm_max_hag_{int(resolution * 100)}cm.tif"


def _process_one(
    laz_path: Path,
    out_dir: Path,
    resolution: float,
    hag_max: float,
    drop_above_hag_max: bool,
    nodata: float,
    chunk_size: int,
    skip_existing: bool,
    cog: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    out_tif = _out_tif_for_laz(laz_path, out_dir, resolution)
    base = {
        "input_laz": str(laz_path),
        "output_chm": str(out_tif),
        "hag_max_requested": hag_max,
        "filter_mode_requested": "drop" if drop_above_hag_max else "clip",
        "resolution_requested": resolution,
        "status": "error",
        "error": "",
    }

    try:
        if dry_run:
            base["status"] = "dry_run"
            base["error"] = ""
            return base

        if skip_existing and out_tif.exists():
            stats = _summarize_raster(out_tif)
            stats.update(base)
            stats["status"] = "skipped_existing"
            stats["error"] = ""
            return stats

        compute_hag_raster_streamed(
            laz_path=laz_path,
            out_tif=out_tif,
            resolution=resolution,
            hag_max=hag_max,
            drop_above_hag_max=drop_above_hag_max,
            nodata=nodata,
            chunk_size=chunk_size,
            cog=cog,
            logger=logger,
        )

        stats = _summarize_raster(out_tif)
        stats.update(base)
        stats["status"] = "ok"
        stats["error"] = ""
        return stats
    except Exception as exc:
        base["status"] = "error"
        base["error"] = str(exc)
        return base


def _write_reports(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "total": len(rows),
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "skipped_existing": sum(1 for r in rows if r.get("status") == "skipped_existing"),
        "dry_run": sum(1 for r in rows if r.get("status") == "dry_run"),
        "error": sum(1 for r in rows if r.get("status") == "error"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    # stable field order for easier diffs
    fields = [
        "input_laz",
        "output_chm",
        "status",
        "error",
        "hag_max_requested",
        "filter_mode_requested",
        "resolution_requested",
        "tag_hag_max",
        "tag_filter_mode",
        "tag_resolution",
        "tag_source_laz",
        "tag_proc_date",
        "width",
        "height",
        "resolution_x",
        "resolution_y",
        "crs",
        "nodata",
        "total_pixels",
        "valid_pixels",
        "nodata_pixels",
        "nodata_pct",
        "hag_min",
        "hag_max_observed",
        "hag_mean",
        "hag_std",
        "hag_p50",
        "hag_p95",
        "hag_p99",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    p = argparse.ArgumentParser(description="Phase A LAZ->CHM wrapper with detailed reporting")
    p.add_argument("--input-dir", type=Path, default=Path("data/lamapuit/laz"))
    p.add_argument("--pattern", default="*.laz")
    p.add_argument("--out", type=Path, default=Path("data/lamapuit/chm_max_hag_13"))
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--hag-max", type=float, default=1.3)
    p.add_argument(
        "--drop-above-hag-max",
        action="store_true",
        help="Discard points with HAG > hag-max before rasterization (strict low-height CHM).",
    )
    p.add_argument("--nodata", type=float, default=-9999.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Plan conversion and write report only")
    p.add_argument("--cog", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--report-json",
        type=Path,
        default=Path("analysis/onboarding_new_laz/phase_a_laz_to_chm_report.json"),
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        default=Path("analysis/onboarding_new_laz/phase_a_laz_to_chm_report.csv"),
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out / "phase_a_laz_to_chm.log", args.verbose)

    laz_files = find_inputs(args.input_dir, args.pattern)
    if not laz_files:
        logger.error("No LAZ files found in %s with pattern %s", args.input_dir, args.pattern)
        return 2

    logger.info("Phase A starting: files=%d hag_max=%.3f resolution=%.3f", len(laz_files), args.hag_max, args.resolution)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {
            ex.submit(
                _process_one,
                laz,
                args.out,
                args.resolution,
                args.hag_max,
                args.drop_above_hag_max,
                args.nodata,
                args.chunk_size,
                args.skip_existing,
                args.cog,
                args.dry_run,
                logger,
            ): laz
            for laz in laz_files
        }

        for fut in as_completed(future_map):
            laz = future_map[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "input_laz": str(laz),
                    "output_chm": str(_out_tif_for_laz(laz, args.out, args.resolution)),
                    "status": "error",
                    "error": str(exc),
                    "hag_max_requested": args.hag_max,
                    "filter_mode_requested": "drop" if args.drop_above_hag_max else "clip",
                    "resolution_requested": args.resolution,
                }
            rows.append(row)
            logger.info("%s -> %s [%s]", laz.name, Path(row.get("output_chm", "")).name, row.get("status"))

    # deterministic row order in reports
    rows.sort(key=lambda r: str(r.get("input_laz", "")))
    _write_reports(rows, args.report_json, args.report_csv)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    skipped = sum(1 for r in rows if r.get("status") == "skipped_existing")
    dry_run = sum(1 for r in rows if r.get("status") == "dry_run")
    err = sum(1 for r in rows if r.get("status") == "error")

    logger.info("Phase A complete: ok=%d skipped=%d dry_run=%d error=%d", ok, skipped, dry_run, err)
    logger.info("Reports: %s | %s", args.report_json, args.report_csv)
    return 0 if err == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
