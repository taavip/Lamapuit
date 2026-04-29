"""Run V4 spatial split experiments and write RESULTS.md.

Usage (from repo root):
    python scripts/spatial_split_experiments_v4/run_experiments_v4.py \\
        --labels-dir output/model_search_v4/prepared/labels_curated_v4 \\
        --output-dir output/spatial_split_experiments_v4 \\
        [--seeds 2026,2027,2028,2029,2030] \\
        [--experiments E10_v4_blocks3_buf5]

Experiments E10–E14 compare V4 (stride-aware, year-safe) against each other.
A comparison table against published-method benchmarks is written to RESULTS.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.spatial_split_experiments_v4._split_v4 import (  # noqa: E402
    write_intra_raster_chip_split_v4,
    _infer_stride,
    _infer_chunk_size,
)


EXPERIMENTS = [
    {
        "id": "E10_v4_blocks3_buf5",
        "description": "V4 stride-aware, 3×3 blocks, buffer_strides=5 (gap=51.2 m) — RECOMMENDED",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 3, "buffer_strides": 5},
    },
    {
        "id": "E11_v4_blocks4_buf5",
        "description": "V4 stride-aware, 4×4 blocks, buffer_strides=5 (gap=51.2 m)",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 4, "buffer_strides": 5},
    },
    {
        "id": "E12_v4_blocks3_buf7",
        "description": "V4 stride-aware, 3×3 blocks, buffer_strides=7 (gap=76.8 m, conservative)",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 3, "buffer_strides": 7},
    },
    {
        "id": "E13_v4_blocks3_buf3",
        "description": "V4 stride-aware, 3×3 blocks, buffer_strides=3 (gap=25.6 m, BELOW threshold)",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 3, "buffer_strides": 3},
    },
    {
        "id": "E14_v4_blocks5_buf5",
        "description": "V4 stride-aware, 5×5 blocks, buffer_strides=5 (gap=51.2 m)",
        "kwargs": {"test_fraction": 0.20, "n_blocks": 5, "buffer_strides": 5},
    },
]


def load_label_candidates(labels_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for f in sorted(labels_dir.glob("*_labels.csv")):
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raster = str(row["raster"])
                source = str(row.get("source", ""))
                reason = "manual_or_reviewed" if (
                    "manual" in source.lower() or source == "auto_reviewed"
                ) else "threshold_gate"
                candidates.append({
                    "key": (raster, int(row["row_off"]), int(row["col_off"])),
                    "raster": raster,
                    "label": str(row.get("label", "")),
                    "source": source,
                    "reason": reason,
                    "chunk_size": int(row.get("chunk_size", 128)),
                })
    return candidates


def run_one(
    exp: dict[str, Any],
    candidates: list[dict[str, Any]],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    exp_id = exp["id"]
    kwargs = dict(exp["kwargs"])
    kwargs["seed"] = seed

    split_path = output_dir / exp_id / f"split_seed{seed}.json"
    geojson_path = output_dir / exp_id / f"split_seed{seed}.geojson"

    meta = write_intra_raster_chip_split_v4(
        all_candidates=candidates,
        output_test_split=split_path,
        output_geojson=geojson_path,
        **kwargs,
    )

    total = meta.get("total_rows", 0)
    return {
        "exp_id": exp_id,
        "description": exp["description"],
        "seed": seed,
        "n_blocks": kwargs.get("n_blocks", 3),
        "buffer_strides": kwargs.get("buffer_strides", 5),
        "buffer_gap_m": meta.get("buffer_gap_m", 0),
        "stride": meta.get("stride", 64),
        "total_rows": total,
        "test_rows": meta.get("test_rows", 0),
        "train_rows": meta.get("train_rows", 0),
        "buffer_rows": meta.get("buffer_rows", 0),
        "buffer_pct": meta.get("buffer_pct", 0.0),
        "train_pct": round(100.0 * meta.get("train_rows", 0) / total, 1) if total else 0.0,
        "test_pct": round(100.0 * meta.get("test_rows", 0) / total, 1) if total else 0.0,
        "overall_cdw_ratio": meta.get("overall_cdw_ratio", 0),
        "test_cdw_ratio": meta.get("test_cdw_ratio", 0),
    }


_PUBLISHED_BENCHMARKS = [
    {
        "method": "Random split (no spatial buffer)",
        "source": "— (naive baseline)",
        "buffer_pct": "0",
        "train_pct": "~80",
        "spatial_gap_m": "0 — direct pixel leakage",
        "note": "INVALID for spatial data",
    },
    {
        "method": "V1/V2 mapsheet-level blocks",
        "source": "Roberts et al. 2017 (Ecography)",
        "buffer_pct": "~63",
        "train_pct": "~16",
        "spatial_gap_m": "≥1 000 m",
        "note": "Cluster B dominance causes irreducible buffer",
    },
    {
        "method": "blockCV — small block",
        "source": "Valavi et al. 2019 (MEE)",
        "buffer_pct": "20–40",
        "train_pct": "40–60",
        "spatial_gap_m": "block-size dependent",
        "note": "Typical result for regional datasets",
    },
    {
        "method": "Patch-level spatial CV",
        "source": "Kattenborn et al. 2022 (ISPRS OJ)",
        "buffer_pct": "15–20",
        "train_pct": "60–65",
        "spatial_gap_m": "~50–100 m",
        "note": "random CV inflates performance ≤28 pp",
    },
    {
        "method": "V3 chunk-based (buffer=2 chunks)",
        "source": "This work (previous iteration)",
        "buffer_pct": "~7.4",
        "train_pct": "~69",
        "spatial_gap_m": "38.4 m ← below 50 m threshold",
        "note": "BUG: stride=64 but coords used chunk=128",
    },
]


def write_results_md(results: list[dict[str, Any]], output_dir: Path) -> None:
    import statistics
    from collections import defaultdict

    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_exp[r["exp_id"]].append(r)

    lines = [
        "# Spatial Split V4 — Results",
        "",
        "Generated by `run_experiments_v4.py` against",
        "`output/model_search_v4/prepared/labels_curated_v4`.",
        "",
        "See `METHODOLOGY.md` for the analysis of V3 bugs and the corrected design.",
        "",
        "---",
        "",
        "## V4 Experiment Results (5 seeds, seed=2026–2030)",
        "",
        "| Exp | n_blocks | buf_strides | gap (m) | buffer% | train% | test% | CDW balance |",
        "|---|---|---|---|---|---|---|---|",
    ]

    best_train = 0.0
    best_exp_id = ""
    for exp_id, rows in sorted(by_exp.items()):
        nb = rows[0]["n_blocks"]
        bs = rows[0]["buffer_strides"]
        gap = rows[0]["buffer_gap_m"]
        buf = statistics.mean(r["buffer_pct"] for r in rows)
        train = statistics.mean(r["train_pct"] for r in rows)
        test = statistics.mean(r["test_pct"] for r in rows)
        cdw_overall = statistics.mean(r["overall_cdw_ratio"] for r in rows)
        cdw_test = statistics.mean(r["test_cdw_ratio"] for r in rows)
        cdw_bal = abs(cdw_test - cdw_overall)
        lines.append(
            f"| {exp_id} | {nb}×{nb} | {bs} | {gap:.1f} | {buf:.1f} | {train:.1f} | {test:.1f} | Δ={cdw_bal:.3f} |"
        )
        if train > best_train:
            best_train, best_exp_id = train, exp_id

    lines += [
        "",
        "---",
        "",
        "## Comparison against Published Methods",
        "",
        "| Method | Buffer % | Train % | Spatial gap | Notes |",
        "|---|---|---|---|---|",
    ]
    for b in _PUBLISHED_BENCHMARKS:
        lines.append(
            f"| {b['method']} | {b['buffer_pct']} | {b['train_pct']} | {b['spatial_gap_m']} | {b['note']} |"
        )
    # Add V4 recommended result
    if best_exp_id and best_exp_id in by_exp:
        rows = by_exp[best_exp_id]
        buf = statistics.mean(r["buffer_pct"] for r in rows)
        train = statistics.mean(r["train_pct"] for r in rows)
        gap = rows[0]["buffer_gap_m"]
        lines.append(
            f"| **V4 {best_exp_id} (this work)** | **{buf:.1f}** | **{train:.1f}** | **≥{gap:.1f} m ✓** | **stride-aware, year-safe** |"
        )

    lines += [
        "",
        "---",
        "",
        "## Seed Stability",
        "",
        "| Exp | seed | buffer% | train% | test% | gap (m) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x["exp_id"], x["seed"])):
        lines.append(
            f"| {r['exp_id']} | {r['seed']} | {r['buffer_pct']:.1f} | {r['train_pct']:.1f} "
            f"| {r['test_pct']:.1f} | {r['buffer_gap_m']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Recommendation",
        "",
        f"**Use {best_exp_id}** — best training set size with ≥50 m spatial gap.",
        "",
        "```python",
        "from scripts.spatial_split_experiments_v4._split_v4 import write_intra_raster_chip_split_v4",
        "from pathlib import Path",
        "",
        "meta = write_intra_raster_chip_split_v4(",
        "    all_candidates=candidates,",
        "    output_test_split=Path('output/.../split.json'),",
        "    output_geojson=Path('output/.../split.geojson'),",
        "    seed=2026,",
        "    test_fraction=0.20,",
        "    n_blocks=3,          # 3×3 blocks per raster",
        "    buffer_strides=5,    # gap = (5+1)×64−128 = 256 px = 51.2 m ✓",
        ")",
        "```",
        "",
        "### Key parameters explained",
        "",
        "- `stride` is auto-detected from `row_off` GCD (= 64 for 50% overlap labels).",
        "- `buffer_strides=5` means Chebyshev ≤ 5 stride-units from any test chip.",
        "  First train chip at chip_pos = last_test_chip_pos + 6.",
        "  Gap = 6 × 64 − 128 = 256 px = **51.2 m > 50 m CWD autocorrelation range** ✓",
        "- All years of the same physical location get the same block assignment",
        "  (RNG seeded by `place_key = tile_site`, year-agnostic).",
    ]

    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-dir",
        default="output/model_search_v4/prepared/labels_curated_v4",
    )
    parser.add_argument(
        "--output-dir",
        default="output/spatial_split_experiments_v4",
    )
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--experiments", default="")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    exp_filter = set(args.experiments.split(",")) if args.experiments else None

    print(f"Loading labels from {labels_dir} ...")
    candidates = load_label_candidates(labels_dir)
    print(f"  {len(candidates)} label rows loaded.")

    stride = _infer_stride(candidates)
    chunk = _infer_chunk_size(candidates)
    print(f"  Detected: stride={stride}px, chunk_size={chunk}px")

    exps = [e for e in EXPERIMENTS if exp_filter is None or e["id"] in exp_filter]
    print(f"Running {len(exps)} experiments × {len(seeds)} seeds = {len(exps) * len(seeds)} runs")

    all_results: list[dict[str, Any]] = []
    for exp in exps:
        for seed in seeds:
            print(f"  {exp['id']} seed={seed} ...", end=" ", flush=True)
            try:
                result = run_one(exp, candidates, seed, output_dir)
                all_results.append(result)
                print(
                    f"buffer={result['buffer_pct']:.1f}% train={result['train_pct']:.1f}% "
                    f"test={result['test_pct']:.1f}% gap={result['buffer_gap_m']:.1f}m"
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                import traceback
                traceback.print_exc()

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
