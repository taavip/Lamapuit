#!/usr/bin/env python3
"""Generate a compact 1.0/0.6/0.2 comparison from experiment_report.json files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def main() -> int:
    base = Path("experiments/dtm_hag_436646_reproduce_single_2018/dem_resolution_fasttest_2018_436646")
    configs = [
        ("1.0", base / "r1p0" / "experiment_report.json"),
        ("0.6", base / "r0p6" / "experiment_report.json"),
        ("0.2", base / "r0p2" / "experiment_report.json"),
    ]

    rows = []
    for res, report_path in configs:
        if not report_path.exists():
            raise FileNotFoundError(f"Missing report: {report_path}")

        data = json.loads(report_path.read_text())
        agg = data.get("evaluation_aggregate", {})
        harm_raw = agg.get("harmonized_dem_raw", {})
        harm_gauss = agg.get("harmonized_dem_gauss", {})
        baseline = agg.get("baseline_idw3_drop13", {})

        rows.append(
            {
                "dem_resolution_m": float(res),
                "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
                "best_method": data.get("best_method"),
                "harm_raw_auc_tile_max": _get(harm_raw, "auc_tile_max"),
                "harm_raw_j_tile_max": _get(harm_raw, "best_youden_tile_max", "youden_j"),
                "harm_raw_cdw_detect_rate_15cm": _get(harm_raw, "cdw_detect_rate_15cm"),
                "harm_raw_no_false_high_rate_15cm": _get(harm_raw, "no_false_high_rate_15cm"),
                "harm_gauss_auc_tile_max": _get(harm_gauss, "auc_tile_max"),
                "harm_gauss_j_tile_max": _get(harm_gauss, "best_youden_tile_max", "youden_j"),
                "baseline_auc_tile_max": _get(baseline, "auc_tile_max"),
                "baseline_j_tile_max": _get(baseline, "best_youden_tile_max", "youden_j"),
            }
        )

    rows.sort(key=lambda r: r["dem_resolution_m"], reverse=True)

    csv_path = base / "comparison_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = base / "comparison_report.md"
    lines = []
    lines.append("# DEM Resolution Fast Comparison (tile 436646, year 2018)")
    lines.append("")
    lines.append("- Mode: harmonized raw+gauss only (`--only-raw-gauss`) with baseline comparison")
    lines.append("- HAG filter: drop, 0.0 <= HAG <= 1.3 m")
    lines.append("- Gaussian sigma: 0.3")
    lines.append("")
    lines.append("## Runtime")
    for row in rows:
        lines.append(f"- {row['dem_resolution_m']:.1f} m: {row['elapsed_seconds']:.2f} s")
    lines.append("")
    lines.append("## Best Method By Run")
    for row in rows:
        lines.append(f"- {row['dem_resolution_m']:.1f} m: {row['best_method']}")
    lines.append("")
    lines.append("## Harmonized Raw vs Baseline (AUC/J on tile_max)")
    for row in rows:
        lines.append(
            f"- {row['dem_resolution_m']:.1f} m: "
            f"harm_raw AUC={row['harm_raw_auc_tile_max']}, J={row['harm_raw_j_tile_max']} | "
            f"baseline AUC={row['baseline_auc_tile_max']}, J={row['baseline_j_tile_max']}"
        )
    lines.append("")
    lines.append("## Note")
    lines.append("- All-NaN harmonization warnings were removed in the runner by warning-safe reducers.")
    lines.append("- Raw+gauss mode now skips IDW/TIN/TPS interpolation, which is why runtime dropped substantially.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote_csv={csv_path}")
    print(f"wrote_md={md_path}")
    for row in rows:
        print(
            f"res={row['dem_resolution_m']:.1f} "
            f"elapsed={row['elapsed_seconds']:.2f}s "
            f"best={row['best_method']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
