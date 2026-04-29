"""Run spatial split comparison experiments and write RESULTS.md.

Usage (from repo root, with cwd-detect env active):
    python scripts/spatial_split_experiments/run_experiments.py \\
        --labels-dir output/model_search_v4/prepared/labels_main_budget \\
        --output-dir output/spatial_split_experiments \\
        [--seeds 2026,2027,2028,2029,2030]

The script:
1. Loads all label CSVs from labels-dir (same data used by model_search_v4).
2. Runs experiments E01–E05 (see METHODOLOGY.md) over multiple seeds.
3. Writes per-experiment JSON splits + GeoJSON to output-dir.
4. Writes a summary CSV and updates RESULTS.md.

Does NOT modify any existing runs or split files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root as well as from the package
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.model_search_v4._labels import parse_raster_identity  # noqa: E402
from scripts.model_search_v4._splits import write_spatial_block_test_split  # noqa: E402
from scripts.spatial_split_experiments._split_v2 import write_spatial_distance_split  # noqa: E402
from scripts.spatial_split_experiments._split_v3_chips import write_intra_raster_chip_split  # noqa: E402


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_label_candidates(labels_dir: Path) -> list[dict[str, Any]]:
    """Read all *_labels.csv files and return the all_candidates list."""
    candidates: list[dict[str, Any]] = []
    for f in sorted(labels_dir.glob("*_labels.csv")):
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raster = str(row["raster"])
                row_off = int(row["row_off"])
                col_off = int(row["col_off"])
                label = str(row.get("label", ""))
                source = str(row.get("source", ""))
                reason = "manual_or_reviewed" if ("manual" in source.lower() or source == "auto_reviewed") else "threshold_gate"
                candidates.append(
                    {
                        "key": (raster, row_off, col_off),
                        "raster": raster,
                        "label": label,
                        "source": source,
                        "reason": reason,
                    }
                )
    return candidates


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "id": "E01_v1_block2_nbr1",
        "description": "V1 baseline: block_size=2, neighbor_buffer=1 (~4 km effective)",
        "method": "v1",
        "kwargs": {"split_block_size_places": 2, "neighbor_buffer_blocks": 1, "test_fraction": 0.20},
    },
    {
        "id": "E02_v1_block1_nbr1",
        "description": "V1 with block_size=1, neighbor_buffer=1 (1 km buffer, simple fix)",
        "method": "v1",
        "kwargs": {"split_block_size_places": 1, "neighbor_buffer_blocks": 1, "test_fraction": 0.20},
    },
    {
        "id": "E03_v2_buf1_no_val",
        "description": "V2 distance buffer=1, no val set, no stratification",
        "method": "v2",
        "kwargs": {
            "test_fraction": 0.20, "val_fraction": 0.0, "buffer_tiles": 1,
            "stratify_regions": False,
        },
    },
    {
        "id": "E04_v2_buf1_val_no_strat",
        "description": "V2 distance buffer=1, val=10%, no stratification",
        "method": "v2",
        "kwargs": {
            "test_fraction": 0.20, "val_fraction": 0.10, "buffer_tiles": 1,
            "stratify_regions": False,
        },
    },
    {
        "id": "E05_v2_buf1_val_strat",
        "description": "V2 distance buffer=1, val=10%, with regional stratification (RECOMMENDED)",
        "method": "v2",
        "kwargs": {
            "test_fraction": 0.20, "val_fraction": 0.10, "buffer_tiles": 1,
            "stratify_regions": True,
        },
    },
    {
        "id": "E06_v2_buf2_val_strat",
        "description": "V2 distance buffer=2 (2 km), val=10%, with stratification",
        "method": "v2",
        "kwargs": {
            "test_fraction": 0.20, "val_fraction": 0.10, "buffer_tiles": 2,
            "stratify_regions": True,
        },
    },
    {
        "id": "E07_v3_blocks3_buf2",
        "description": "V3 intra-raster chip split: 3×3 blocks, buffer=2 chips (51.2 m)",
        "method": "v3",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 3, "buffer_chips": 2},
    },
    {
        "id": "E08_v3_blocks4_buf2",
        "description": "V3 intra-raster chip split: 4×4 blocks, buffer=2 chips (51.2 m)",
        "method": "v3",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 4, "buffer_chips": 2},
    },
    {
        "id": "E09_v3_blocks3_buf1",
        "description": "V3 intra-raster chip split: 3×3 blocks, buffer=1 chip (25.6 m)",
        "method": "v3",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 3, "buffer_chips": 1},
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(
    exp: dict[str, Any],
    candidates: list[dict[str, Any]],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a single experiment configuration and return a result dict."""
    exp_id = exp["id"]
    method = exp["method"]
    kwargs = dict(exp["kwargs"])
    kwargs["seed"] = seed

    split_path = output_dir / exp_id / f"split_seed{seed}.json"
    geojson_path = output_dir / exp_id / f"split_seed{seed}.geojson"

    if method == "v1":
        meta = write_spatial_block_test_split(
            all_candidates=candidates,
            output_test_split=split_path,
            **kwargs,
        )
    elif method == "v3":
        meta = write_intra_raster_chip_split(
            all_candidates=candidates,
            output_test_split=split_path,
            output_geojson=geojson_path,
            **kwargs,
        )
    else:
        meta = write_spatial_distance_split(
            all_candidates=candidates,
            output_test_split=split_path,
            output_geojson=geojson_path,
            **kwargs,
        )

    total = meta.get("total_rows", 0)
    buf_rows = meta.get("buffer_rows", 0)
    train_rows = meta.get("train_rows", meta.get("train_rows_estimate", 0))
    test_rows = meta.get("test_rows", 0)
    val_rows = meta.get("val_rows", 0)

    return {
        "exp_id": exp_id,
        "description": exp["description"],
        "seed": seed,
        "method": method,
        "buffer_tiles": kwargs.get("buffer_tiles", kwargs.get("neighbor_buffer_blocks", "?")),
        "total_rows": total,
        "test_rows": test_rows,
        "val_rows": val_rows,
        "train_rows": train_rows,
        "buffer_rows": buf_rows,
        "buffer_pct": round(100.0 * buf_rows / total, 1) if total else 0.0,
        "train_pct": round(100.0 * train_rows / total, 1) if total else 0.0,
        "test_pct": round(100.0 * test_rows / total, 1) if total else 0.0,
        "val_pct": round(100.0 * val_rows / total, 1) if total else 0.0,
        "n_places_test": meta.get("n_places_test", 0),
        "n_places_val": meta.get("n_places_val", 0),
        "n_places_train": meta.get("n_places_train", 0),
        "n_places_buffer": meta.get("n_places_buffer", 0),
        "n_regions_in_test": meta.get("n_regions_in_test", "n/a"),
        "test_cdw_ratio": round(
            meta.get("test_cdw_rows", 0) / test_rows, 3
        ) if test_rows else 0.0,
        "place_overlap": meta.get("place_overlap_train_vs_test", 0),
    }


def write_results_md(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Write RESULTS.md with per-experiment summary table."""
    from collections import defaultdict
    import statistics

    # Group by exp_id, average across seeds
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_exp[r["exp_id"]].append(r)

    lines = [
        "# Spatial Split Experiments — Results",
        "",
        "Generated by `run_experiments.py`. See METHODOLOGY.md for experiment design.",
        "",
        "## Summary Table (averaged across seeds)",
        "",
        "| Exp ID | Method | Buffer (km) | buffer% | train% | test% | val% | regions_in_test | CDW ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    best_train_pct = 0.0
    best_exp_id = ""

    for exp_id, rows in sorted(by_exp.items()):
        desc_short = rows[0]["description"].split(":")[0].strip()
        method = rows[0]["method"]
        buf = rows[0]["buffer_tiles"]
        buf_pct = statistics.mean(r["buffer_pct"] for r in rows)
        train_pct = statistics.mean(r["train_pct"] for r in rows)
        test_pct = statistics.mean(r["test_pct"] for r in rows)
        val_pct = statistics.mean(r["val_pct"] for r in rows)
        n_reg = rows[0]["n_regions_in_test"]
        cdw_ratio = statistics.mean(r["test_cdw_ratio"] for r in rows)
        lines.append(
            f"| {exp_id} | {method} | {buf} | {buf_pct:.1f}% | {train_pct:.1f}% "
            f"| {test_pct:.1f}% | {val_pct:.1f}% | {n_reg} | {cdw_ratio:.3f} |"
        )
        if train_pct > best_train_pct:
            best_train_pct = train_pct
            best_exp_id = exp_id

    lines += [
        "",
        "## Per-seed Stability",
        "",
        "| Exp ID | seed | buffer% | train% | test% | val% |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x["exp_id"], x["seed"])):
        lines.append(
            f"| {r['exp_id']} | {r['seed']} | {r['buffer_pct']} | {r['train_pct']} "
            f"| {r['test_pct']} | {r['val_pct']} |"
        )

    # Find recommended experiment description
    best_desc = next(
        (e["description"] for e in EXPERIMENTS if e["id"] == best_exp_id), best_exp_id
    )
    best_kwargs = next(
        (e["kwargs"] for e in EXPERIMENTS if e["id"] == best_exp_id), {}
    )

    lines += [
        "",
        "## Recommendation",
        "",
        f"Best experiment by training set size: **{best_exp_id}** — {best_desc}",
        "",
        "Call arguments:",
        "```",
        f"# model_search_v4 with V1-compatible parameters:",
        f"#   --split-block-size-places 1 --neighbor-buffer-blocks 1",
        f"",
        f"# V2 (scripts/spatial_split_experiments/_split_v2.py):",
        f"#   buffer_tiles={best_kwargs.get('buffer_tiles', 1)}",
        f"#   test_fraction={best_kwargs.get('test_fraction', 0.20)}",
        f"#   val_fraction={best_kwargs.get('val_fraction', 0.10)}",
        f"#   stratify_regions={best_kwargs.get('stratify_regions', True)}",
        "```",
        "",
        "### Academic justification",
        "",
        "- **Roberts et al. 2017**: spatial blocking necessary; no standard random CV.",
        "- **Le Rest et al. 2014**: buffer = variogram range (~50–300 m for forests).",
        "  1-tile Chebyshev (1 000 m) is conservative but not wasteful.",
        "- **Milà et al. 2022**: NNDM LOO-CV is ideal but requires ≥20 places;",
        "  distance-buffer + stratification is a practical approximation for n = 22.",
        "- **Ploton et al. 2020**: spatial CV reveals real performance; random CV",
        "  inflates metrics by 20–50 pp — always report spatial CV results.",
    ]

    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run spatial split comparison experiments.")
    parser.add_argument(
        "--labels-dir",
        default="output/model_search_v4/prepared/labels_curated_v4",
        help="Directory with *_labels.csv files (default: output/model_search_v4/prepared/labels_curated_v4)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/spatial_split_experiments",
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030",
        help="Comma-separated list of seeds (default: 2026,2027,2028,2029,2030)",
    )
    parser.add_argument(
        "--experiments",
        default="",
        help="Comma-separated subset of experiment IDs to run (default: all)",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    exp_filter = set(args.experiments.split(",")) if args.experiments else None

    print(f"Loading labels from {labels_dir} ...")
    candidates = load_label_candidates(labels_dir)
    print(f"  {len(candidates)} label rows loaded.")

    exps = [e for e in EXPERIMENTS if exp_filter is None or e["id"] in exp_filter]
    print(f"Running {len(exps)} experiments × {len(seeds)} seeds = {len(exps)*len(seeds)} runs")

    all_results: list[dict[str, Any]] = []
    for exp in exps:
        for seed in seeds:
            print(f"  {exp['id']} seed={seed} ...", end=" ", flush=True)
            try:
                result = run_one(exp, candidates, seed, output_dir)
                all_results.append(result)
                print(
                    f"buffer={result['buffer_pct']}% train={result['train_pct']}% "
                    f"test={result['test_pct']}% val={result['val_pct']}%"
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                import traceback
                traceback.print_exc()

    # Write summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment_summary.csv"
    if all_results:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary CSV: {csv_path}")

    write_results_md(all_results, output_dir)
    print(f"Results: {output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
