#!/usr/bin/env python3
"""Standardize onboarding label CSVs for all CHM variants.

What this script does:
1. Reads all ``*_labels.csv`` files from a source folder.
2. Preserves all input columns losslessly in an event table.
3. Builds latest label per tile keyed by (map_sheet, year, row_off, col_off).
4. Matches labels to CHM variant rasters by 6-digit map sheet + 4-digit year.
5. Writes variant-ready label CSVs with remapped raster naming.

The script defaults to onboarding v2 labels and the CHM variants registry.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MAP_YEAR_RE = re.compile(r"(?P<mapsheet>\d{6})_(?P<year>\d{4})")


def parse_map_year(text: str) -> Optional[Tuple[str, str]]:
    m = MAP_YEAR_RE.search(text)
    if not m:
        return None
    return m.group("mapsheet"), m.group("year")


def try_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def to_repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def discover_label_files(source_dir: Path) -> List[Path]:
    return sorted(p for p in source_dir.glob("*_labels.csv") if p.is_file())


def collect_source_columns(source_files: Iterable[Path]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for csv_path in source_files:
        with csv_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for name in reader.fieldnames or []:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
    return ordered


def load_manifest_paths(manifest_path: Optional[Path], repo_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not manifest_path or not manifest_path.exists():
        return out

    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to read manifest.yaml. Install with `pip install pyyaml`."
        ) from exc

    with manifest_path.open("r", encoding="utf-8") as fp:
        doc = yaml.safe_load(fp) or {}

    for section in ("datasets", "variants"):
        block = doc.get(section, {})
        if not isinstance(block, dict):
            continue
        for name, meta in block.items():
            raw_path = meta.get("path") if isinstance(meta, dict) else None
            if not raw_path:
                continue
            p = Path(str(raw_path))
            if not p.is_absolute():
                p = repo_root / p
            out[str(name)] = p
    return out


def load_variant_paths(
    variants_root: Path,
    manifest_path: Optional[Path],
    repo_root: Path,
    include_manifest_only: bool,
) -> Dict[str, Path]:
    variants: Dict[str, Path] = {}

    # First: explicit dirs/symlinks in data/chm_variants
    if variants_root.exists():
        for child in sorted(variants_root.iterdir()):
            if child.name.startswith("."):
                continue
            if child.name in {"README.md", "manifest.yaml"}:
                continue
            variants[child.name] = child

    manifest_variants = load_manifest_paths(manifest_path, repo_root)

    # Override paths for variants that exist in data/chm_variants.
    for name in list(variants.keys()):
        if name in manifest_variants:
            variants[name] = manifest_variants[name]

    # Optionally include extra manifest-only datasets.
    if include_manifest_only:
        for name, p in manifest_variants.items():
            variants.setdefault(name, p)

    return variants


def resolve_workspace_target(path: Path, repo_root: Path) -> Path:
    text = str(path)
    if text.startswith("/workspace/"):
        mapped = repo_root / text[len("/workspace/") :]
        return mapped
    return path


def resolve_variant_scan_path(variant_path: Path, repo_root: Path) -> Path:
    # Direct path first.
    direct = variant_path
    if not direct.is_absolute():
        direct = repo_root / direct
    if direct.exists() and direct.is_dir():
        return direct

    # Broken symlink in this repo often points to /workspace/... from container paths.
    if variant_path.is_symlink():
        try:
            target = Path(os.readlink(variant_path))
        except OSError:
            target = Path()
        if target:
            if not target.is_absolute():
                target = (variant_path.parent / target).resolve()
            mapped = resolve_workspace_target(target, repo_root)
            if mapped.exists() and mapped.is_dir():
                return mapped

    mapped = resolve_workspace_target(direct, repo_root)
    if mapped.exists() and mapped.is_dir():
        return mapped

    return direct


def index_variant_rasters(variant_path: Path, repo_root: Path) -> Tuple[Path, Dict[Tuple[str, str], List[Path]], int]:
    key_to_rasters: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    total_tifs = 0
    scan_path = resolve_variant_scan_path(variant_path, repo_root)
    if not scan_path.exists() or not scan_path.is_dir():
        return scan_path, key_to_rasters, total_tifs

    for dirpath, _dirnames, filenames in os.walk(scan_path, followlinks=True):
        base = Path(dirpath)
        for name in filenames:
            if not name.lower().endswith(".tif"):
                continue
            total_tifs += 1
            my = parse_map_year(name)
            if not my:
                continue
            key_to_rasters[my].append(base / name)

    for key in key_to_rasters:
        key_to_rasters[key] = sorted(key_to_rasters[key])
    return scan_path, key_to_rasters, total_tifs


def write_csv(path: Path, headers: List[str], rows: Iterable[List[str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Lossless label standardization for CHM variants")
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=Path("output/onboarding_labels_v2_drop13"),
        help="Directory containing *_labels.csv files",
    )
    ap.add_argument(
        "--variants-root",
        type=Path,
        default=Path("data/chm_variants"),
        help="Root folder that defines CHM variants",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/chm_variants/manifest.yaml"),
        help="Variant manifest with dataset paths",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("labels_standardized_onboarding_v2_drop13"),
        help="Output directory",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into non-empty output folder",
    )
    ap.add_argument(
        "--include-manifest-only",
        action="store_true",
        help="Also export datasets that exist only in manifest.yaml and not as entries in data/chm_variants",
    )
    args = ap.parse_args()

    repo_root = Path.cwd()
    source_dir = args.source_dir
    out_dir = args.out_dir

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory is not empty: {out_dir}. Use --overwrite to continue."
        )

    source_files = discover_label_files(source_dir)
    if not source_files:
        raise SystemExit(f"No *_labels.csv files found in {source_dir}")

    source_cols = collect_source_columns(source_files)

    prov_cols = [
        "source_csv",
        "source_row_number",
        "event_index",
        "raster_original",
        "map_sheet",
        "year",
        "map_sheet_year",
        "is_addressable_tile",
        "prov_row_json",
    ]
    event_headers = source_cols + prov_cols
    index_by_col = {c: i for i, c in enumerate(event_headers)}

    all_events_path = out_dir / "all_events.csv"
    latest_path = out_dir / "latest_per_tile.csv"
    by_variant_root = out_dir / "by_variant"

    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows_total = 0
    addressable_rows_total = 0
    event_index = 0

    latest_by_tile: Dict[Tuple[str, str, int, int], Tuple[int, List[str]]] = {}

    with all_events_path.open("w", newline="", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp)
        writer.writerow(event_headers)

        for source_csv in source_files:
            source_my = parse_map_year(source_csv.name)
            with source_csv.open("r", newline="", encoding="utf-8") as in_fp:
                reader = csv.DictReader(in_fp)
                for line_no, row in enumerate(reader, start=2):
                    source_rows_total += 1
                    event_index += 1

                    values = [row.get(c, "") for c in source_cols]
                    raster_original = row.get("raster", "")

                    row_my = parse_map_year(raster_original)
                    map_sheet = ""
                    year = ""
                    if row_my:
                        map_sheet, year = row_my
                    elif source_my:
                        map_sheet, year = source_my

                    row_off = try_int(row.get("row_off", ""))
                    col_off = try_int(row.get("col_off", ""))
                    is_addressable = bool(map_sheet and year and row_off is not None and col_off is not None)
                    if is_addressable:
                        addressable_rows_total += 1

                    values.extend(
                        [
                            to_repo_rel(source_csv, repo_root),
                            str(line_no),
                            str(event_index),
                            raster_original,
                            map_sheet,
                            year,
                            f"{map_sheet}_{year}" if map_sheet and year else "",
                            "1" if is_addressable else "0",
                            json.dumps(row, ensure_ascii=True, sort_keys=True),
                        ]
                    )
                    writer.writerow(values)

                    if is_addressable and row_off is not None and col_off is not None:
                        key = (map_sheet, year, row_off, col_off)
                        prev = latest_by_tile.get(key)
                        if prev is None or event_index >= prev[0]:
                            latest_by_tile[key] = (event_index, values)

    latest_rows = [pair[1] for pair in sorted(latest_by_tile.values(), key=lambda x: x[0])]
    write_csv(latest_path, event_headers, latest_rows)

    # Group latest rows by map-sheet/year for fast variant matching.
    latest_by_map_year: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)
    for row in latest_rows:
        ms = row[index_by_col["map_sheet"]]
        yr = row[index_by_col["year"]]
        if ms and yr:
            latest_by_map_year[(ms, yr)].append(row)

    variants = load_variant_paths(
        args.variants_root,
        args.manifest,
        repo_root,
        include_manifest_only=args.include_manifest_only,
    )

    variant_headers = event_headers + ["variant_name", "variant_raster_relpath"]
    raster_col_idx = index_by_col.get("raster")

    summary = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "source_files": len(source_files),
        "source_rows_total": source_rows_total,
        "addressable_rows_total": addressable_rows_total,
        "latest_per_tile": len(latest_rows),
        "source_columns": source_cols,
        "variants": {},
    }

    for variant_name, variant_path in sorted(variants.items()):
        scan_path, key_to_rasters, indexed_tifs = index_variant_rasters(variant_path, repo_root)
        variant_dir = by_variant_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        all_rows_for_variant: List[List[str]] = []
        matched_rasters = 0
        matched_latest_rows = 0

        keys_with_labels = set(latest_by_map_year.keys())
        keys_with_rasters = set(key_to_rasters.keys())
        matched_keys = keys_with_labels & keys_with_rasters

        for key in sorted(matched_keys):
            source_rows = latest_by_map_year[key]
            rasters = key_to_rasters.get(key, [])
            for raster_path in rasters:
                out_rows: List[List[str]] = []
                rel_raster = to_repo_rel(raster_path, repo_root)
                raster_name = raster_path.name
                for base_row in source_rows:
                    row_out = list(base_row)
                    if raster_col_idx is not None:
                        row_out[raster_col_idx] = raster_name
                    row_out.extend([variant_name, rel_raster])
                    out_rows.append(row_out)
                    all_rows_for_variant.append(row_out)

                out_file = variant_dir / f"{raster_path.stem}_labels.csv"
                write_csv(out_file, variant_headers, out_rows)
                matched_rasters += 1

            matched_latest_rows += len(source_rows)

        write_csv(variant_dir / "all_labels.csv", variant_headers, all_rows_for_variant)

        summary["variants"][variant_name] = {
            "variant_path": str(variant_path),
            "scan_path": str(scan_path),
            "indexed_tifs": indexed_tifs,
            "indexed_keys": len(keys_with_rasters),
            "matched_keys": len(matched_keys),
            "matched_latest_rows": matched_latest_rows,
            "matched_rasters": matched_rasters,
            "unmatched_latest_rows": len(latest_rows) - matched_latest_rows,
        }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "Standardized Labels (Lossless, All Sources)",
                "==========================================",
                "",
                "Generated by:",
                "- scripts/standardize_labels_for_chm_variants.py",
                "",
                "Files:",
                "- all_events.csv: all input events with all source columns + provenance",
                "- latest_per_tile.csv: latest event per (map_sheet, year, row_off, col_off)",
                "- by_variant/<variant>/*_labels.csv: variant-ready labels with remapped raster names",
                "- by_variant/<variant>/all_labels.csv: concatenated labels per variant",
                "- summary.json: conversion stats and matching coverage",
                "",
                "Matching rule:",
                "- 6-digit map sheet + 4-digit year parsed from raster filename.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_dir}")
    print(f"Source files: {len(source_files)}")
    print(f"Source rows: {source_rows_total}")
    print(f"Addressable rows: {addressable_rows_total}")
    print(f"Latest per tile: {len(latest_rows)}")
    print(f"Variants exported: {len(summary['variants'])}")


if __name__ == "__main__":
    main()
