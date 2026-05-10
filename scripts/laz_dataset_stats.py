#!/usr/bin/env python3
"""
LAZ dataset statistics for thesis — reads LAZ headers only (no point loading).
Computes area by mapsheet and by (mapsheet × year) for LaTeX tables.
CRS: L-EST97 EPSG:3301 (metres).

Usage:
  conda activate cwd-detect && python scripts/laz_dataset_stats.py

Or via Docker:
  docker-compose -f docker-compose.benchmark.yml run --rm chm-benchmark \\
    bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate cwd-detect && \\
    python scripts/laz_dataset_stats.py"
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import laspy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LAZ_DIR = Path("/workspace/data/lamapuit/laz") if Path("/workspace").exists() else Path("data/lamapuit/laz")
MANIFEST = LAZ_DIR / "eligible_manifest.csv"
CSV_OUT = LAZ_DIR / "stats_per_file.csv"


def read_laz_header_stats(laz_path):
    """
    Read LAZ header (no point loading). Returns dict with bounds, point count, area.
    Falls back to data-based bounds if header unavailable.
    Pattern from process_laz_to_chm_improved.py lines ~182–189.
    """
    try:
        with laspy.open(str(laz_path)) as fh:
            header = fh.header
            try:
                min_x, max_x = float(header.mins[0]), float(header.maxs[0])
                min_y, max_y = float(header.mins[1]), float(header.maxs[1])
            except Exception:
                # Fallback: read first chunk to compute bounds
                points_list = []
                for chunk in fh.chunk_iterator(50_000):
                    points_list.append(chunk)
                    if len(points_list) >= 1:
                        break
                if points_list:
                    p = points_list[0]
                    min_x, max_x = float(p.x.min()), float(p.x.max())
                    min_y, max_y = float(p.y.min()), float(p.y.max())
                else:
                    return None

            n_pts = int(header.point_count)

            # If LAS 1.4 edge case: count via chunk iterator
            if n_pts == 0:
                n_pts = 0
                for chunk in fh.chunk_iterator(50_000):
                    n_pts += len(chunk.x)

        area_m2 = (max_x - min_x) * (max_y - min_y)
        if area_m2 <= 0:
            logger.warning(f"  Non-positive area for {laz_path.name}: {area_m2:.1f} m²")
            return None

        area_km2 = area_m2 / 1e6
        density_pts_m2 = n_pts / area_m2 if area_m2 > 0 else 0

        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "area_m2": area_m2,
            "area_km2": area_km2,
            "point_count": n_pts,
            "density_pts_m2": density_pts_m2,
        }
    except Exception as e:
        logger.warning(f"  Error reading {laz_path.name}: {e}")
        return None


def parse_filename(stem):
    """Parse 'kaardiruut', 'year', 'campaign' from stem like '436646_2018_madal'."""
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            return int(parts[0]), int(parts[1]), parts[2]
        except ValueError:
            pass
    return None, None, None


def main():
    # Load manifest to join size_bytes and validate scope
    logger.info(f"Reading manifest: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    logger.info(f"Manifest has {len(manifest)} files")

    # Discover all .laz files
    laz_files = sorted(LAZ_DIR.glob("*.laz"))
    logger.info(f"Found {len(laz_files)} LAZ files in {LAZ_DIR}")

    records = []
    logger.info("\nReading LAZ headers...")

    for i, laz_path in enumerate(laz_files, 1):
        print(f"\r  {i:3d}/{len(laz_files)}  {laz_path.name:<45}", end="", file=sys.stderr)
        sys.stderr.flush()

        kaardiruut, year, campaign = parse_filename(laz_path.stem)
        if kaardiruut is None:
            logger.warning(f"  Could not parse: {laz_path.name}")
            continue

        stats = read_laz_header_stats(laz_path)
        if stats is None:
            continue

        record = {
            "filename": laz_path.name,
            "kaardiruut": kaardiruut,
            "year": year,
            "campaign": campaign,
            "min_x": stats["min_x"],
            "max_x": stats["max_x"],
            "min_y": stats["min_y"],
            "max_y": stats["max_y"],
            "area_m2": stats["area_m2"],
            "area_km2": stats["area_km2"],
            "point_count": stats["point_count"],
            "density_pts_m2": stats["density_pts_m2"],
        }

        records.append(record)

    print("\n", file=sys.stderr)
    logger.info(f"Successfully read {len(records)} files\n")

    # Build DataFrame and join manifest for size_bytes
    df = pd.DataFrame(records)
    manifest_sub = manifest[["filename", "size_bytes"]].copy()
    df = df.merge(manifest_sub, on="filename", how="left")

    # Save per-file CSV
    df.to_csv(CSV_OUT, index=False, float_format="%.4f")
    logger.info(f"Saved per-file stats: {CSV_OUT}\n")

    # =========================================================================
    # AGGREGATIONS
    # =========================================================================

    # Per-mapsheet: area is fixed by grid, use median to be robust
    ms = (
        df.groupby("kaardiruut")
        .agg(
            area_km2=("area_km2", "median"),
            n_surveys=("year", "nunique"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            total_points=("point_count", "sum"),
            mean_density=("density_pts_m2", "mean"),
            min_density=("density_pts_m2", "min"),
            max_density=("density_pts_m2", "max"),
            n_files=("filename", "count"),
        )
        .reset_index()
    )
    ms["year_range"] = ms["year_min"].astype(str) + "–" + ms["year_max"].astype(str)
    ms["mean_pts_per_file"] = (ms["total_points"] / ms["n_files"]).astype(int)

    # Per-year
    yr = (
        df.groupby("year")
        .agg(
            n_mapsheets=("kaardiruut", "nunique"),
            n_files=("filename", "count"),
            total_area_km2=("area_km2", "sum"),
            total_points=("point_count", "sum"),
            mean_density=("density_pts_m2", "mean"),
            min_density=("density_pts_m2", "min"),
            max_density=("density_pts_m2", "max"),
        )
        .reset_index()
    )

    # Overall scalars
    total_unique_area_km2 = ms["area_km2"].sum()
    total_scan_area_km2 = df["area_km2"].sum()
    total_points = df["point_count"].sum()
    mean_density = df["density_pts_m2"].mean()
    median_density = df["density_pts_m2"].median()
    min_density = df["density_pts_m2"].min()
    max_density = df["density_pts_m2"].max()
    total_size_gb = manifest["size_bytes"].sum() / 1e9
    n_tava = (df["campaign"] == "tava").sum()
    n_madal = (df["campaign"] == "madal").sum()

    # =========================================================================
    # LATEX TABLES (stdout)
    # =========================================================================

    print()
    print("% ========================================================================")
    print("% TABLE A: OVERALL DATASET SUMMARY")
    print("% ========================================================================")
    print()
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{ll}")
    print(r"\toprule")
    print(r"Statistic & Value \\")
    print(r"\midrule")
    print(rf"Total LAZ files & {len(df):,} \\")
    print(rf"Unique mapsheets & {df['kaardiruut'].nunique()} \\")
    print(rf"Survey years & {df['year'].min():.0f}–{df['year'].max():.0f} ({df['year'].nunique()} years) \\")
    print(rf"Campaign type & \SI{{{n_madal}}}{{}} madal, \SI{{{n_tava}}}{{}} tava \\")
    print(rf"Unique area (mapsheet extent) & \SI{{{total_unique_area_km2:.0f}}}{{\kilo\meter\squared}} \\")
    print(rf"Total scan area (all years) & \SI{{{total_scan_area_km2:.0f}}}{{\kilo\meter\squared}} \\")
    print(rf"Total point count & \SI{{{total_points/1e9:.2f}}}{{\billion}} \\")
    print(rf"Mean point density & \SI{{{mean_density:.2f}}}{{pts/\meter\squared}} \\")
    print(rf"Median point density & \SI{{{median_density:.2f}}}{{pts/\meter\squared}} \\")
    print(rf"Density range (min–max) & \SI{{{min_density:.2f}}}{{}} – \SI{{{max_density:.2f}}}{{pts/\meter\squared}} \\")
    print(rf"Total raw data size & $\sim$\SI{{{total_size_gb:.0f}}}{{\giga\byte}} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Dataset extent and properties.}")
    print(r"\label{tbl:laz_dataset_overview}")
    print(r"\end{table}")
    print()

    # =========================================================================
    print("% ========================================================================")
    print("% TABLE B: PER-YEAR BREAKDOWN")
    print("% ========================================================================")
    print()
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(rf"Year & Files & Mapsheets & Area (\si{{\kilo\meter\squared}}) & Points (M) & Mean density (\si{{pts/\meter\squared}}) \\")
    print(r"\midrule")
    for _, row in yr.iterrows():
        print(
            rf"{int(row['year'])} & {int(row['n_files'])} & {int(row['n_mapsheets'])} & "
            rf"{row['total_area_km2']:.0f} & {row['total_points']/1e6:.1f} & {row['mean_density']:.2f} \\"
        )
    print(r"\midrule")
    print(
        rf"Total & {df.shape[0]} & — & {total_scan_area_km2:.0f} & {total_points/1e6:.1f} & "
        rf"{mean_density:.2f} \\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Per-year breakdown of coverage, files, and density.}")
    print(r"\label{tbl:laz_per_year}")
    print(r"\end{table}")
    print()

    # =========================================================================
    print("% ========================================================================")
    print("% TABLE C: PER-MAPSHEET SUMMARY")
    print("% ========================================================================")
    print()
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{lrrlrr}")
    print(r"\toprule")
    print(
        rf"Kaardiruut & Area (\si{{\kilo\meter\squared}}) & Surveys & "
        rf"Years & Points (M) & Mean density (\si{{pts/\meter\squared}}) \\"
    )
    print(r"\midrule")
    for _, row in ms.sort_values("kaardiruut").iterrows():
        print(
            rf"{int(row['kaardiruut'])} & {row['area_km2']:.1f} & {int(row['n_surveys'])} & "
            rf"{row['year_range']} & {row['total_points']/1e6:.1f} & {row['mean_density']:.2f} \\"
        )
    print(r"\midrule")
    print(
        rf"Total & {total_unique_area_km2:.0f} & — & — & {total_points/1e6:.1f} & "
        rf"{mean_density:.2f} \\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(
        r"\caption{Per-mapsheet summary. Area computed from LAS bounding box (L-EST97, EPSG:3301). "
        r"Unique area (sum of distinct mapsheets) represents geographic footprint; total points reflect cumulative acquisitions across years.}"
    )
    print(r"\label{tbl:laz_per_mapsheet}")
    print(r"\end{table}")
    print()

    logger.info("LaTeX tables written to stdout. Copy-paste into thesis.")


if __name__ == "__main__":
    main()
