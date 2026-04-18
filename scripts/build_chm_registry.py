#!/usr/bin/env python3
"""Build unified CHM dataset registries from existing outputs.

This script keeps all existing data in place and creates structured registry files
for training/evaluation pipelines:
1) Updates local eligible LAZ manifest to unique local input files.
2) Writes a wide sample table with shared label path across CHM variants.
3) Writes a long variant table (raw/gauss/baseline) for flexible loaders.
4) Writes SQLite tables for reproducible querying.
5) Writes LAZ folder inventory with copy/delete candidates.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

INPUT_LAZ_RE = re.compile(r"^(\d+)_(\d{4})_(.+)\.laz$")
RAW_RE = re.compile(r"^(\d+)_(\d{4})_(.+?)_harmonized_dem_.+_raw_chm\.tif$")
GAUSS_RE = re.compile(r"^(\d+)_(\d{4})_(.+?)_harmonized_dem_.+_gauss_chm\.tif$")
LABEL_RE = re.compile(r"^(\d+)_(\d{4})_(.+)_chm_max_hag_20cm_labels\.csv$")
BASELINE_RE = re.compile(r"^(\d+)_(\d{4})_(.+)_chm_max_hag_20cm\.tif$")


@dataclass(frozen=True)
class Key:
    mapsheet: str
    year: int
    campaign: str

    @property
    def sample_id(self) -> str:
        return f"{self.mapsheet}_{self.year}_{self.campaign}"


def _to_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _parse_key(name: str, pattern: re.Pattern[str]) -> Optional[Key]:
    match = pattern.match(name)
    if match is None:
        return None
    return Key(mapsheet=match.group(1), year=int(match.group(2)), campaign=match.group(3))


def _index_by_key(path: Path, pattern: re.Pattern[str]) -> Dict[Key, Path]:
    out: Dict[Key, Path] = {}
    if not path.exists():
        return out

    for file_path in sorted(path.glob("*")):
        if not file_path.is_file():
            continue
        key = _parse_key(file_path.name, pattern)
        if key is None:
            continue
        out[key] = file_path
    return out


def _discover_local_laz(laz_dir: Path) -> List[Tuple[Key, Path]]:
    entries: List[Tuple[Key, Path]] = []
    for laz_path in sorted(laz_dir.glob("*.laz")):
        key = _parse_key(laz_path.name, INPUT_LAZ_RE)
        if key is None:
            continue
        entries.append((key, laz_path))
    return entries


def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_folder_inventory(workspace_root: Path, primary_laz_dir: Path) -> List[dict]:
    summary: Dict[str, dict] = {}

    def _onerror(_: OSError) -> None:
        # Ignore permission-restricted directories.
        return

    for root, _, files in os.walk(workspace_root, onerror=_onerror):
        laz_names = [name for name in files if name.lower().endswith(".laz")]
        if not laz_names:
            continue

        folder = Path(root)
        rel_folder = _to_rel(folder, workspace_root)
        bucket = summary.setdefault(
            rel_folder,
            {
                "folder": rel_folder,
                "laz_file_count": 0,
                "primary_source_laz_count": 0,
                "copy_like_laz_count": 0,
                "delete_candidate": 0,
            },
        )

        for name in laz_names:
            bucket["laz_file_count"] += 1
            parsed = _parse_key(name, INPUT_LAZ_RE)
            in_primary = folder.resolve() == primary_laz_dir.resolve()
            is_primary_source = parsed is not None and in_primary

            if is_primary_source:
                bucket["primary_source_laz_count"] += 1
            else:
                bucket["copy_like_laz_count"] += 1

        if bucket["copy_like_laz_count"] > 0 and bucket["primary_source_laz_count"] == 0:
            bucket["delete_candidate"] = 1

    rows = list(summary.values())
    rows.sort(key=lambda r: r["folder"])
    return rows


def _write_sqlite(path: Path, samples: List[dict], variants: List[dict], folder_inventory: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS samples;
            DROP TABLE IF EXISTS variants;
            DROP TABLE IF EXISTS laz_folders;

            CREATE TABLE samples (
                sample_id TEXT PRIMARY KEY,
                mapsheet TEXT NOT NULL,
                year INTEGER NOT NULL,
                campaign TEXT NOT NULL,
                laz_input_path TEXT NOT NULL,
                label_path TEXT,
                chm_raw_path TEXT,
                chm_gauss_path TEXT,
                chm_baseline_path TEXT,
                has_label INTEGER NOT NULL,
                has_raw INTEGER NOT NULL,
                has_gauss INTEGER NOT NULL,
                has_baseline INTEGER NOT NULL
            );

            CREATE TABLE variants (
                sample_id TEXT NOT NULL,
                mapsheet TEXT NOT NULL,
                year INTEGER NOT NULL,
                campaign TEXT NOT NULL,
                variant TEXT NOT NULL,
                tif_path TEXT,
                tif_exists INTEGER NOT NULL,
                label_path TEXT,
                label_exists INTEGER NOT NULL,
                trainable INTEGER NOT NULL,
                PRIMARY KEY (sample_id, variant)
            );

            CREATE TABLE laz_folders (
                folder TEXT PRIMARY KEY,
                laz_file_count INTEGER NOT NULL,
                primary_source_laz_count INTEGER NOT NULL,
                copy_like_laz_count INTEGER NOT NULL,
                delete_candidate INTEGER NOT NULL
            );

            CREATE INDEX idx_samples_tile_year ON samples (mapsheet, year);
            CREATE INDEX idx_samples_campaign ON samples (campaign);
            CREATE INDEX idx_variants_trainable ON variants (trainable);
            """
        )

        cur.executemany(
            """
            INSERT INTO samples (
                sample_id, mapsheet, year, campaign, laz_input_path,
                label_path, chm_raw_path, chm_gauss_path, chm_baseline_path,
                has_label, has_raw, has_gauss, has_baseline
            ) VALUES (
                :sample_id, :mapsheet, :year, :campaign, :laz_input_path,
                :label_path, :chm_raw_path, :chm_gauss_path, :chm_baseline_path,
                :has_label, :has_raw, :has_gauss, :has_baseline
            )
            """,
            samples,
        )

        cur.executemany(
            """
            INSERT INTO variants (
                sample_id, mapsheet, year, campaign, variant,
                tif_path, tif_exists, label_path, label_exists, trainable
            ) VALUES (
                :sample_id, :mapsheet, :year, :campaign, :variant,
                :tif_path, :tif_exists, :label_path, :label_exists, :trainable
            )
            """,
            variants,
        )

        cur.executemany(
            """
            INSERT INTO laz_folders (
                folder, laz_file_count, primary_source_laz_count,
                copy_like_laz_count, delete_candidate
            ) VALUES (
                :folder, :laz_file_count, :primary_source_laz_count,
                :copy_like_laz_count, :delete_candidate
            )
            """,
            folder_inventory,
        )

        conn.commit()
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified CHM registries from generated outputs")
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--laz-dir", type=Path, default=Path("data/lamapuit/laz"))
    parser.add_argument("--labels-dir", type=Path, default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss/labels"))
    parser.add_argument("--raw-dir", type=Path, default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss/chm_raw"))
    parser.add_argument("--gauss-dir", type=Path, default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss/chm_gauss"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))

    parser.add_argument(
        "--eligible-manifest",
        type=Path,
        default=Path("data/lamapuit/laz/eligible_manifest.csv"),
        help="Unique local LAZ input list (updated in-place).",
    )
    parser.add_argument(
        "--eligible-manifest-fallback",
        type=Path,
        default=Path("registry/chm_dataset_harmonized_0p8m_raw_gauss/eligible_manifest.local.csv"),
        help="Writable fallback path if eligible manifest is read-only.",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path("registry/chm_dataset_harmonized_0p8m_raw_gauss/ml_samples.csv"),
    )
    parser.add_argument(
        "--variants-csv",
        type=Path,
        default=Path("registry/chm_dataset_harmonized_0p8m_raw_gauss/ml_variants.csv"),
    )
    parser.add_argument(
        "--folder-inventory-csv",
        type=Path,
        default=Path("registry/chm_dataset_harmonized_0p8m_raw_gauss/laz_folder_inventory.csv"),
    )
    parser.add_argument(
        "--registry-db",
        type=Path,
        default=Path("registry/chm_dataset_harmonized_0p8m_raw_gauss/ml_registry.sqlite"),
    )

    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    laz_dir = (workspace_root / args.laz_dir).resolve()
    labels_dir = (workspace_root / args.labels_dir).resolve()
    raw_dir = (workspace_root / args.raw_dir).resolve()
    gauss_dir = (workspace_root / args.gauss_dir).resolve()
    baseline_dir = (workspace_root / args.baseline_dir).resolve()

    local_laz = _discover_local_laz(laz_dir)
    if not local_laz:
        raise RuntimeError(f"No local input LAZ files found in: {laz_dir}")

    raw_by_key = _index_by_key(raw_dir, RAW_RE)
    gauss_by_key = _index_by_key(gauss_dir, GAUSS_RE)
    label_by_key = _index_by_key(labels_dir, LABEL_RE)
    baseline_by_key = _index_by_key(baseline_dir, BASELINE_RE)

    existing_manifest_rows = _read_csv_rows(workspace_root / args.eligible_manifest)
    existing_by_filename = {
        row.get("filename", ""): row
        for row in existing_manifest_rows
        if row.get("filename")
    }

    eligible_rows: List[dict] = []
    samples_rows: List[dict] = []
    variants_rows: List[dict] = []

    for key, laz_path in sorted(local_laz, key=lambda t: (t[0].mapsheet, t[0].year, t[0].campaign)):
        filename = laz_path.name
        previous = existing_by_filename.get(filename, {})

        raw_path = raw_by_key.get(key)
        gauss_path = gauss_by_key.get(key)
        baseline_path = baseline_by_key.get(key)
        label_path = label_by_key.get(key)

        has_raw = int(raw_path is not None and raw_path.exists())
        has_gauss = int(gauss_path is not None and gauss_path.exists())
        has_baseline = int(baseline_path is not None and baseline_path.exists())
        has_label = int(label_path is not None and label_path.exists())

        eligible_rows.append(
            {
                "kaardiruut": previous.get("kaardiruut", key.mapsheet),
                "filename": filename,
                "url": previous.get("url", ""),
                "size_bytes": previous.get("size_bytes", ""),
                "mapsheet": key.mapsheet,
                "year": key.year,
                "campaign": key.campaign,
                "sample_id": key.sample_id,
                "laz_path": _to_rel(laz_path, workspace_root),
            }
        )

        sample_row = {
            "sample_id": key.sample_id,
            "mapsheet": key.mapsheet,
            "year": key.year,
            "campaign": key.campaign,
            "laz_input_path": _to_rel(laz_path, workspace_root),
            "label_path": _to_rel(label_path, workspace_root) if label_path else "",
            "chm_raw_path": _to_rel(raw_path, workspace_root) if raw_path else "",
            "chm_gauss_path": _to_rel(gauss_path, workspace_root) if gauss_path else "",
            "chm_baseline_path": _to_rel(baseline_path, workspace_root) if baseline_path else "",
            "has_label": has_label,
            "has_raw": has_raw,
            "has_gauss": has_gauss,
            "has_baseline": has_baseline,
        }
        samples_rows.append(sample_row)

        for variant_name, tif_path, tif_exists in (
            ("raw", raw_path, has_raw),
            ("gauss", gauss_path, has_gauss),
            ("baseline", baseline_path, has_baseline),
        ):
            variants_rows.append(
                {
                    "sample_id": key.sample_id,
                    "mapsheet": key.mapsheet,
                    "year": key.year,
                    "campaign": key.campaign,
                    "variant": variant_name,
                    "tif_path": _to_rel(tif_path, workspace_root) if tif_path else "",
                    "tif_exists": tif_exists,
                    "label_path": _to_rel(label_path, workspace_root) if label_path else "",
                    "label_exists": has_label,
                    "trainable": int(bool(tif_exists and has_label)),
                }
            )

    folder_inventory_rows = _build_folder_inventory(workspace_root, laz_dir)

    eligible_fieldnames = [
        "kaardiruut",
        "filename",
        "url",
        "size_bytes",
        "mapsheet",
        "year",
        "campaign",
        "sample_id",
        "laz_path",
    ]
    target_manifest = workspace_root / args.eligible_manifest
    try:
        _write_csv(target_manifest, eligible_fieldnames, eligible_rows)
        print(f"eligible_manifest_updated={_to_rel(target_manifest, workspace_root)}")
    except PermissionError:
        fallback_manifest = workspace_root / args.eligible_manifest_fallback
        _write_csv(fallback_manifest, eligible_fieldnames, eligible_rows)
        print(
            "eligible_manifest_read_only="
            f"{_to_rel(target_manifest, workspace_root)}"
        )
        print(
            "eligible_manifest_fallback_written="
            f"{_to_rel(fallback_manifest, workspace_root)}"
        )

    _write_csv(
        workspace_root / args.samples_csv,
        [
            "sample_id",
            "mapsheet",
            "year",
            "campaign",
            "laz_input_path",
            "label_path",
            "chm_raw_path",
            "chm_gauss_path",
            "chm_baseline_path",
            "has_label",
            "has_raw",
            "has_gauss",
            "has_baseline",
        ],
        samples_rows,
    )
    _write_csv(
        workspace_root / args.variants_csv,
        [
            "sample_id",
            "mapsheet",
            "year",
            "campaign",
            "variant",
            "tif_path",
            "tif_exists",
            "label_path",
            "label_exists",
            "trainable",
        ],
        variants_rows,
    )
    _write_csv(
        workspace_root / args.folder_inventory_csv,
        [
            "folder",
            "laz_file_count",
            "primary_source_laz_count",
            "copy_like_laz_count",
            "delete_candidate",
        ],
        folder_inventory_rows,
    )

    _write_sqlite(workspace_root / args.registry_db, samples_rows, variants_rows, folder_inventory_rows)

    print(f"local_input_laz={len(samples_rows)}")
    print(f"samples_with_labels={sum(int(r['has_label']) for r in samples_rows)}")
    print(f"raw_trainable={sum(1 for r in variants_rows if r['variant'] == 'raw' and int(r['trainable']) == 1)}")
    print(f"gauss_trainable={sum(1 for r in variants_rows if r['variant'] == 'gauss' and int(r['trainable']) == 1)}")
    print(f"baseline_trainable={sum(1 for r in variants_rows if r['variant'] == 'baseline' and int(r['trainable']) == 1)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
