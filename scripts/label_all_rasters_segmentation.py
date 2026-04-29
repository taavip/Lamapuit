#!/usr/bin/env python3
"""Batch orchestrator for segmentation labeling across CHM rasters.

This is a segmentation-first counterpart to scripts/label_all_rasters.py.
It builds a tile queue CSV for scripts/brush_mask_labeler.py and optionally
attaches pre-segmentation seeds (mask/cam) from one or more manifests.

Examples
--------
# Build queue and launch brush labeler for all 20cm rasters
python scripts/label_all_rasters_segmentation.py \
  --chm-dir chm_max_hag \
  --output-dir output/manual_masks_all_rasters

# Use Grad-CAM manifest as pre-seg seeds
python scripts/label_all_rasters_segmentation.py \
  --chm-dir output/chm_dataset_harmonized_0p8m_raw_gauss/chm_gauss \
  --pattern "*.tif" \
  --preseg-manifest output/cam_masks_gradcam_lineaware_smallsplit/manifest.csv \
  --output-dir output/manual_masks_from_preseg

# Only build queue CSV (do not open UI)
python scripts/label_all_rasters_segmentation.py \
  --chm-dir chm_max_hag \
  --build-only \
  --output-dir output/manual_masks_all_rasters
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HARMONIZED_RASTER_RE = re.compile(
    r"^(?P<sample>.+)_harmonized_dem_last_(?P<variant>[A-Za-z0-9]+)_chm$",
    re.IGNORECASE,
)


@dataclass
class SeedEntry:
    mask_path: Optional[Path]
    cam_path: Optional[Path]
    hotspot_path: Optional[Path]
    source: str


def _safe_stem(text: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", str(text)).strip("_")
    return cleaned or "tile"


def _parse_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _norm_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)


def _resolve_path(raw_value: str, base_dir: Path, preseg_root: Optional[Path]) -> Optional[Path]:
    value = str(raw_value).strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path

    candidates: list[Path] = [base_dir / path]
    if preseg_root is not None:
        candidates.insert(0, preseg_root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return ""


def _parse_tile_id_fields(tile_id: str) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """Parse tile id of form <sample_id>:<variant>:<row_off>:<col_off>."""
    raw = str(tile_id).strip()
    if not raw:
        return None, None, None, None

    parts = raw.split(":")
    if len(parts) < 4:
        return None, None, None, None

    sample_id = ":".join(parts[:-3]).strip() if len(parts) > 4 else parts[0].strip()
    variant = str(parts[-3]).strip().lower() if parts[-3] else None
    row_off = _parse_int(parts[-2])
    col_off = _parse_int(parts[-1])
    if not sample_id:
        sample_id = None
    return sample_id, variant, row_off, col_off


def _sample_variant_from_raster_path(raster_path: Path) -> tuple[str, Optional[str]]:
    """Infer (sample_id, variant) from CHM raster filename."""
    stem = raster_path.stem
    m = _HARMONIZED_RASTER_RE.match(stem)
    if m is not None:
        sample = str(m.group("sample")).strip()
        variant = str(m.group("variant")).strip().lower()
        return sample, variant or None

    if "_chm_" in stem:
        sample = stem.split("_chm_", 1)[0].strip()
        if sample:
            return sample, None

    if stem.endswith("_chm"):
        sample = stem[: -len("_chm")].strip("_")
        if sample:
            return sample, None

    return stem, None


def _compute_nodata_pct(path: Path) -> float | None:
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata_val = src.nodata
            if nodata_val is None:
                mask = np.isnan(arr)
            else:
                mask = np.isnan(arr) | (arr == nodata_val)
            return float(100.0 * np.sum(mask) / arr.size)
    except Exception:
        return None


def _iter_chunks(height: int, width: int, chunk_size: int, overlap: float) -> list[tuple[int, int]]:
    """Return list of (row_off, col_off) chunk origins.

    Mirrors scripts/label_tiles.py behavior so queue coverage remains consistent
    with existing classification runs.
    """
    stride = max(1, int(chunk_size * (1.0 - overlap)))
    min_gap = chunk_size // 4

    def make_offsets(size: int) -> list[int]:
        if size <= chunk_size:
            return [0]
        offsets = list(range(0, size - chunk_size + 1, stride))
        last_border = size - chunk_size
        gap = size - (offsets[-1] + chunk_size)
        if gap > min_gap and last_border not in offsets:
            offsets.append(last_border)
        return offsets

    rows = make_offsets(height)
    cols = make_offsets(width)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for row_off in rows:
        for col_off in cols:
            key = (row_off, col_off)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _load_preseg_index(
    manifest_paths: list[Path],
    preseg_root: Optional[Path],
) -> tuple[
    dict[tuple[str, int, int], SeedEntry],
    dict[tuple[str, int, int], SeedEntry],
    dict[tuple[str, str, int, int], SeedEntry],
    dict[tuple[str, int, int], SeedEntry],
]:
    """Load pre-seg seed index from one or more manifests.

        Returns:
            - by_full_path[(resolved_raster_path, row_off, col_off)]
            - by_raster_name[(raster_name, row_off, col_off)]
            - by_sample_variant[(sample_id, variant, row_off, col_off)]
            - by_sample[(sample_id, row_off, col_off)]
    """
    by_full_path: dict[tuple[str, int, int], SeedEntry] = {}
    by_raster_name: dict[tuple[str, int, int], SeedEntry] = {}
    by_sample_variant: dict[tuple[str, str, int, int], SeedEntry] = {}
    by_sample: dict[tuple[str, int, int], SeedEntry] = {}

    for manifest in manifest_paths:
        if not manifest.exists():
            print(f"[preseg] manifest missing, skipping: {manifest}")
            continue

        loaded = 0
        with manifest.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raster_raw = _first_nonempty(row, ["raster_path", "raster", "chm_path", "image_path"])
                row_off = _parse_int(row.get("row_off", ""))
                col_off = _parse_int(row.get("col_off", ""))
                tile_id = _first_nonempty(row, ["tile_id"])
                tile_sample, tile_variant, tile_row, tile_col = _parse_tile_id_fields(tile_id)

                if row_off is None:
                    row_off = tile_row
                if col_off is None:
                    col_off = tile_col
                if row_off is None or col_off is None:
                    continue

                mask_raw = _first_nonempty(row, ["mask_path", "init_mask", "mask_file"])
                cam_raw = _first_nonempty(row, ["cam_path", "init_cam", "cam_file"])
                hotspot_raw = _first_nonempty(row, ["hotspot_path", "hotspot_file"])

                mask_path = _resolve_path(mask_raw, manifest.parent, preseg_root) if mask_raw else None
                cam_path = _resolve_path(cam_raw, manifest.parent, preseg_root) if cam_raw else None
                hotspot_path = _resolve_path(hotspot_raw, manifest.parent, preseg_root) if hotspot_raw else cam_path

                if mask_path is None and cam_path is None:
                    continue

                source = f"{manifest.name}:{loaded + 1}"
                entry = SeedEntry(mask_path=mask_path, cam_path=cam_path, hotspot_path=hotspot_path, source=source)

                if raster_raw:
                    raster_path = _resolve_path(raster_raw, manifest.parent, preseg_root)
                    if raster_path is not None:
                        full_key = (_norm_path(raster_path), int(row_off), int(col_off))
                        name_key = (Path(raster_path).name, int(row_off), int(col_off))
                        by_full_path[full_key] = entry
                        by_raster_name[name_key] = entry

                        inferred_sample, inferred_variant = _sample_variant_from_raster_path(raster_path)
                        if inferred_sample:
                            by_sample[(inferred_sample, int(row_off), int(col_off))] = entry
                            if inferred_variant:
                                by_sample_variant[
                                    (inferred_sample, inferred_variant, int(row_off), int(col_off))
                                ] = entry

                sample_id = _first_nonempty(row, ["sample_id", "sample", "mapsheet"]).strip()
                if not sample_id and tile_sample:
                    sample_id = tile_sample

                variant = _first_nonempty(row, ["variant", "mode", "kind"]).strip().lower()
                if not variant and tile_variant:
                    variant = tile_variant

                if sample_id:
                    by_sample[(sample_id, int(row_off), int(col_off))] = entry
                    if variant:
                        by_sample_variant[(sample_id, variant, int(row_off), int(col_off))] = entry

                loaded += 1

        print(f"[preseg] indexed {loaded} seed row(s) from {manifest}")

    return by_full_path, by_raster_name, by_sample_variant, by_sample


def _lookup_seed(
    raster_path: Path,
    row_off: int,
    col_off: int,
    by_full_path: dict[tuple[str, int, int], SeedEntry],
    by_raster_name: dict[tuple[str, int, int], SeedEntry],
    by_sample_variant: dict[tuple[str, str, int, int], SeedEntry],
    by_sample: dict[tuple[str, int, int], SeedEntry],
) -> Optional[SeedEntry]:
    full_key = (_norm_path(raster_path), int(row_off), int(col_off))
    entry = by_full_path.get(full_key)
    if entry is not None:
        return entry

    name_key = (raster_path.name, int(row_off), int(col_off))
    entry = by_raster_name.get(name_key)
    if entry is not None:
        return entry

    sample_id, variant = _sample_variant_from_raster_path(raster_path)
    if sample_id and variant:
        sv_key = (sample_id, str(variant).lower(), int(row_off), int(col_off))
        entry = by_sample_variant.get(sv_key)
        if entry is not None:
            return entry

    if sample_id:
        s_key = (sample_id, int(row_off), int(col_off))
        entry = by_sample.get(s_key)
        if entry is not None:
            return entry

    return None


def _build_queue_rows(
    rasters: list[Path],
    *,
    chunk_size: int,
    overlap: float,
    max_nodata_pct: float,
    force_include: bool,
    max_tiles: int,
    by_full_path: dict[tuple[str, int, int], SeedEntry],
    by_raster_name: dict[tuple[str, int, int], SeedEntry],
    by_sample_variant: dict[tuple[str, str, int, int], SeedEntry],
    by_sample: dict[tuple[str, int, int], SeedEntry],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    skipped_nodata = 0
    total_tiles = 0
    seeded_tiles = 0

    for i, raster_path in enumerate(rasters, start=1):
        nodata_pct = _compute_nodata_pct(raster_path)
        if nodata_pct is None:
            print(f"[{i:3d}/{len(rasters)}] skip read-error: {raster_path.name}")
            continue
        if (not force_include) and nodata_pct >= max_nodata_pct:
            skipped_nodata += 1
            print(f"[{i:3d}/{len(rasters)}] skip high nodata ({nodata_pct:.1f}%): {raster_path.name}")
            continue

        with rasterio.open(raster_path) as src:
            height = int(src.height)
            width = int(src.width)

        chunks = _iter_chunks(height, width, int(chunk_size), float(overlap))
        print(
            f"[{i:3d}/{len(rasters)}] queue {len(chunks):,} tile(s): {raster_path.name} "
            f"(nodata={nodata_pct:.1f}%)"
        )

        for row_off, col_off in chunks:
            tile_id = f"{raster_path.stem}:{int(row_off)}:{int(col_off)}"
            output_stem = _safe_stem(f"{raster_path.stem}_{int(row_off)}_{int(col_off)}")
            row = {
                "tile_id": tile_id,
                "output_stem": output_stem,
                "raster_path": str(raster_path),
                "row_off": str(int(row_off)),
                "col_off": str(int(col_off)),
                "chunk_size": str(int(chunk_size)),
                "init_mask": "",
                "init_cam": "",
                "hotspot_path": "",
                "preseg_source": "",
            }

            seed = _lookup_seed(
                raster_path,
                row_off,
                col_off,
                by_full_path,
                by_raster_name,
                by_sample_variant,
                by_sample,
            )
            if seed is not None:
                if seed.mask_path is not None and seed.mask_path.exists():
                    row["init_mask"] = str(seed.mask_path)
                if seed.cam_path is not None and seed.cam_path.exists():
                    row["init_cam"] = str(seed.cam_path)
                if seed.hotspot_path is not None and seed.hotspot_path.exists():
                    row["hotspot_path"] = str(seed.hotspot_path)
                if row["init_mask"] or row["init_cam"]:
                    row["preseg_source"] = seed.source
                    seeded_tiles += 1

            rows.append(row)
            total_tiles += 1
            if max_tiles > 0 and total_tiles >= max_tiles:
                print(f"[queue] reached --max-tiles={max_tiles}")
                break

        if max_tiles > 0 and total_tiles >= max_tiles:
            break

    print(
        f"[queue] built {len(rows):,} row(s), seeded {seeded_tiles:,} row(s), "
        f"skipped {skipped_nodata} raster(s) by nodata threshold"
    )
    return rows


def _write_queue_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_id",
        "output_stem",
        "raster_path",
        "row_off",
        "col_off",
        "chunk_size",
        "init_mask",
        "init_cam",
        "hotspot_path",
        "preseg_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _launch_brush_labeler(args: argparse.Namespace, queue_csv: Path, output_dir: Path) -> int:
    brush_script = Path(__file__).with_name("brush_mask_labeler.py")
    if not brush_script.exists():
        raise FileNotFoundError(f"brush labeler script missing: {brush_script}")

    tile_root = Path(args.tile_root) if args.tile_root else Path.cwd()

    cmd: list[str] = [
        sys.executable,
        str(brush_script),
        "--tile-csv",
        str(queue_csv),
        "--tile-root",
        str(tile_root),
        "--output-dir",
        str(output_dir),
        "--window-name",
        str(args.window_name),
        "--brush-radius",
        str(int(args.brush_radius)),
        "--wms-timeout",
        str(float(args.wms_timeout)),
    ]

    if args.show_hotspot:
        cmd.append("--show-hotspot")
    if args.show_orthophoto:
        cmd.append("--show-orthophoto")
    if args.no_autosave_nav:
        cmd.append("--no-autosave-nav")

    print("[run] launching brush labeler")
    print("[run] " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch segmentation labeler orchestrator for all CHM rasters",
    )
    parser.add_argument("--chm-dir", default="chm_max_hag", help="Directory containing CHM GeoTIFFs")
    parser.add_argument(
        "--pattern",
        default="*20cm.tif",
        help="Raster glob pattern under --chm-dir (default: *20cm.tif)",
    )
    parser.add_argument("--output-dir", default="output/manual_masks_all_rasters")
    parser.add_argument(
        "--queue-csv",
        default="",
        help="Optional explicit queue CSV path. Default: <output-dir>/tile_queue.csv",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-nodata-pct", type=float, default=75.0)
    parser.add_argument("--force-include", action="store_true", help="Ignore nodata filter")
    parser.add_argument(
        "--preseg-manifest",
        action="append",
        default=[],
        help=(
            "Manifest CSV with raster_path,row_off,col_off and mask/cam columns "
            "(repeat flag to provide multiple manifests)"
        ),
    )
    parser.add_argument(
        "--preseg-root",
        default="",
        help="Optional base dir to resolve relative pre-seg paths",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Optional hard cap on queued tiles for pilot runs (0 = no cap)",
    )
    parser.add_argument(
        "--resume-queue",
        action="store_true",
        help="Reuse existing queue CSV if it exists instead of rebuilding",
    )
    parser.add_argument("--build-only", action="store_true", help="Only build queue CSV and exit")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --build-only")

    # Pass-through UI options for brush labeler.
    parser.add_argument("--tile-root", default="", help="Base path for resolving relative queue paths")
    parser.add_argument("--window-name", default="CWD Brush Browser (All Rasters)")
    parser.add_argument("--brush-radius", type=int, default=7)
    parser.add_argument("--show-hotspot", action="store_true")
    parser.add_argument("--show-orthophoto", action="store_true")
    parser.add_argument("--no-autosave-nav", action="store_true")
    parser.add_argument("--wms-timeout", type=float, default=8.0)
    args = parser.parse_args()

    chm_dir = Path(args.chm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_csv = Path(args.queue_csv) if str(args.queue_csv).strip() else (output_dir / "tile_queue.csv")

    if args.resume_queue and queue_csv.exists():
        print(f"[queue] reuse existing queue: {queue_csv}")
    else:
        rasters = sorted(chm_dir.glob(args.pattern))
        if not rasters:
            raise FileNotFoundError(f"No rasters matched '{args.pattern}' in {chm_dir}")

        preseg_root = Path(args.preseg_root) if str(args.preseg_root).strip() else None
        manifest_paths = [Path(p) for p in args.preseg_manifest]
        by_full_path, by_raster_name, by_sample_variant, by_sample = _load_preseg_index(
            manifest_paths,
            preseg_root,
        )

        rows = _build_queue_rows(
            rasters,
            chunk_size=int(args.chunk_size),
            overlap=float(args.overlap),
            max_nodata_pct=float(args.max_nodata_pct),
            force_include=bool(args.force_include),
            max_tiles=max(0, int(args.max_tiles)),
            by_full_path=by_full_path,
            by_raster_name=by_raster_name,
            by_sample_variant=by_sample_variant,
            by_sample=by_sample,
        )
        if not rows:
            raise RuntimeError("Queue is empty after filtering; nothing to label")
        _write_queue_csv(queue_csv, rows)
        print(f"[queue] wrote {len(rows):,} row(s) to {queue_csv}")

    if args.build_only or args.dry_run:
        print("[done] queue build complete (UI launch skipped)")
        return

    rc = _launch_brush_labeler(args, queue_csv=queue_csv, output_dir=output_dir)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
