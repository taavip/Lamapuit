#!/usr/bin/env python3
"""
Batch orchestrator for labeling all CHM rasters sequentially.

Iterates every GeoTIFF in a directory, launches the interactive labeling
session for each one, and tracks completion in a progress JSON file.
Supports stopping mid-raster and resuming where you left off.

Usage
-----
python scripts/label_all_rasters.py --chm-dir chm_max_hag --output output/tile_labels

# Resume from last raster / last chunk:
python scripts/label_all_rasters.py --chm-dir chm_max_hag --output output/tile_labels --resume

# Only 20cm resolution rasters:
python scripts/label_all_rasters.py --chm-dir chm_max_hag --output output/tile_labels --pattern "*20cm.tif"
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import rasterio
import numpy as np

_MANUAL_SOURCES = {"manual", "auto_reviewed"}

try:
    from label_tiles import QuitAllException, run_labeling_session
except ModuleNotFoundError:
    from scripts.label_tiles import QuitAllException, run_labeling_session

PROGRESS_FILE = "progress.json"


def _load_progress(output_dir: Path) -> dict:
    pf = output_dir / PROGRESS_FILE
    if pf.exists():
        return json.loads(pf.read_text())
    return {"completed": [], "started": [], "last_updated": None}


def _save_progress(output_dir: Path, progress: dict) -> None:
    progress["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (output_dir / PROGRESS_FILE).write_text(json.dumps(progress, indent=2))


def _find_latest_manual_tile_in_queue(
    output_dir: Path,
    queue_rows: list[dict],
) -> tuple[str, int, int, datetime] | None:
    """Return newest manual tile event that is present in the queue."""
    if not queue_rows:
        return None

    queue_tiles = {
        (str(r["raster"]), int(r["row_off"]), int(r["col_off"]))
        for r in queue_rows
    }
    queue_rasters = {str(r["raster"]) for r in queue_rows}

    latest: tuple[str, int, int, datetime] | None = None
    for raster_name in queue_rasters:
        csv_path = output_dir / f"{Path(raster_name).stem}_labels.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                src = str(row.get("source", "")).strip().lower()
                if src not in _MANUAL_SOURCES:
                    continue
                ts_str = str(row.get("timestamp", "")).strip()
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    continue
                key = (
                    str(row.get("raster", "")),
                    int(row.get("row_off", 0)),
                    int(row.get("col_off", 0)),
                )
                if key not in queue_tiles:
                    continue
                cand = (key[0], key[1], key[2], ts)
                if latest is None or cand[3] > latest[3]:
                    latest = cand
    return latest


def _trim_queue_after_tile(
    queue_rows: list[dict],
    tile_key: tuple[str, int, int],
) -> list[dict]:
    """Trim queue rows so iteration starts *after* tile_key in queue order."""
    last_idx = -1
    for i, row in enumerate(queue_rows):
        key = (str(row["raster"]), int(row["row_off"]), int(row["col_off"]))
        if key == tile_key:
            last_idx = i
    if last_idx < 0:
        return queue_rows
    return queue_rows[last_idx + 1 :]


def _latest_source_by_tile(csv_path: Path) -> dict[tuple[int, int], str]:
    """Return latest source per tile key (row_off, col_off), last-row-wins."""
    out: dict[tuple[int, int], str] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row.get("row_off", 0)), int(row.get("col_off", 0)))
            except Exception:
                continue
            out[key] = str(row.get("source", "")).strip().lower()
    return out


def _count_pending_manual_in_queue(
    output_dir: Path,
    raster_name: str,
    queued_tiles: list[tuple[int, int]],
) -> tuple[int, int]:
    """Return (pending, total) queued tiles lacking a manual/latest reviewed source."""
    total = len(queued_tiles)
    if total == 0:
        return 0, 0
    csv_path = output_dir / f"{Path(raster_name).stem}_labels.csv"
    latest_src = _latest_source_by_tile(csv_path)
    pending = 0
    for key in queued_tiles:
        src = latest_src.get(key, "")
        if src not in _MANUAL_SOURCES:
            pending += 1
    return pending, total


def _can_trim_queue_after_pointer(
    output_dir: Path,
    queue_rows: list[dict],
    pointer: tuple[str, int, int],
) -> bool:
    """Only trim if every tile before pointer is already manually completed."""
    idx = -1
    for i, row in enumerate(queue_rows):
        key = (str(row["raster"]), int(row["row_off"]), int(row["col_off"]))
        if key == pointer:
            idx = i
    if idx < 0:
        return False

    by_raster: dict[str, list[tuple[int, int]]] = {}
    for row in queue_rows[: idx + 1]:
        rn = str(row["raster"])
        by_raster.setdefault(rn, []).append((int(row["row_off"]), int(row["col_off"])))

    for rn, tiles in by_raster.items():
        pending, _total = _count_pending_manual_in_queue(output_dir, rn, tiles)
        if pending > 0:
            return False
    return True


def _count_labels(csv_path: Path) -> dict[str, int]:
    """Count label types in a per-raster CSV, deduplicating by (row_off, col_off).

    Uses last-row-wins to match _load_existing semantics — re-labeled tiles
    (which have two rows in the CSV) are counted only once with their latest label.
    """
    counts = {"cdw": 0, "no_cdw": 0, "unknown": 0}
    if not csv_path.exists():
        return counts
    import csv

    deduped: dict[tuple, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("row_off", ""), row.get("col_off", ""))
            deduped[key] = row.get("label", "")
    for lbl in deduped.values():
        if lbl in counts:
            counts[lbl] += 1
    return counts


def _compute_nodata_pct(p: Path) -> float | None:
    """Return the percentage of nodata pixels in a raster, or None on read error."""
    try:
        with rasterio.open(p) as src:
            arr = src.read(1)
            nodata_val = src.nodata
            if nodata_val is None:
                try:
                    mask = np.isnan(arr)
                except Exception:
                    mask = np.zeros(arr.shape, dtype=bool)
            else:
                mask = np.isnan(arr) | (arr == nodata_val)
            return 100.0 * np.sum(mask) / arr.size
    except Exception:
        return None


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _analyze_spotcheck_sessions(output_dir: Path) -> dict | None:
    """Aggregate QA metrics from session_*.json files.

    Produces stratified metrics so random-spotcheck and low-confidence queues
    are not conflated in one potentially biased headline number.
    """
    session_files = sorted(output_dir.glob("session_*.json"))
    if not session_files:
        return None

    section_map = {
        "legacy_spotcheck": "spotcheck",
        "queue_spotcheck": "queue_spotcheck",
        "queue_low_confidence": "queue_low_confidence",
        "queue_all_auto": "queue_all_auto",
    }
    count_keys = (
        "candidate_count",
        "reviewed_count",
        "evaluated_count",
        "unknown_count",
        "unknown_proposal_count",
        "agreement_count",
        "tp",
        "tn",
        "fp",
        "fn",
    )

    def _zeros() -> dict[str, int]:
        return {k: 0 for k in count_keys}

    aggregates: dict[str, dict] = {
        name: {
            "session_count": 0,
            "overall": _zeros(),
            "per_raster": {},
        }
        for name in section_map
    }

    for sf in session_files:
        try:
            data = json.loads(sf.read_text())
        except Exception:
            continue

        raster = str(data.get("raster", "unknown"))
        for section_name, section_key in section_map.items():
            sc = data.get(section_key)
            if not isinstance(sc, dict):
                continue
            ag = aggregates[section_name]
            ag["session_count"] += 1

            for key in count_keys:
                try:
                    ag["overall"][key] += int(sc.get(key, 0) or 0)
                except Exception:
                    pass

            per_raster = ag["per_raster"]
            if raster not in per_raster:
                per_raster[raster] = _zeros()
            for key in count_keys:
                try:
                    per_raster[raster][key] += int(sc.get(key, 0) or 0)
                except Exception:
                    pass

    if not any(aggregates[name]["session_count"] > 0 for name in aggregates):
        return None

    def _finalize(counts: dict[str, int]) -> dict:
        evaluated = counts.get("evaluated_count", 0)
        agreement = counts.get("agreement_count", 0)
        tp = counts.get("tp", 0)
        tn = counts.get("tn", 0)
        fp = counts.get("fp", 0)
        fn = counts.get("fn", 0)
        return {
            **counts,
            "agreement_rate": (agreement / evaluated) if evaluated else 0.0,
            **_binary_metrics(tp, tn, fp, fn),
        }

    sections: dict[str, dict] = {}
    stratified_rows: list[dict] = []
    for section_name, ag in aggregates.items():
        if ag["session_count"] == 0:
            continue
        overall_metrics = _finalize(ag["overall"])
        per_raster_rows = []
        for raster, vals in sorted(ag["per_raster"].items()):
            row = {
                "raster": raster,
                **_finalize(vals),
            }
            per_raster_rows.append(row)
            stratified_rows.append({"section": section_name, **row})
        sections[section_name] = {
            "session_count": ag["session_count"],
            "overall": overall_metrics,
            "per_raster": per_raster_rows,
        }

    # Statistically meaningful headline: random queue spot-check if available,
    # otherwise fall back to legacy online spot-check.
    headline_source = "queue_spotcheck"
    headline = sections.get("queue_spotcheck", {}).get("overall")
    if not headline or headline.get("evaluated_count", 0) == 0:
        headline_source = "legacy_spotcheck"
        headline = sections.get("legacy_spotcheck", {}).get("overall", {})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_metric_source": headline_source,
        "notes": {
            "headline_interpretation": (
                "Use queue_spotcheck as primary estimate of auto-label quality "
                "because it is sampled from auto remainder."
            ),
            "bias_warning": (
                "queue_low_confidence is intentionally hard-case enriched and "
                "must not be pooled with spotcheck for a headline model KPI."
            ),
        },
        "sections": sections,
    }

    out_json = output_dir / "spotcheck_metrics_summary.json"
    out_csv = output_dir / "spotcheck_metrics_by_raster.csv"
    out_strat_csv = output_dir / "spotcheck_metrics_stratified_by_raster.csv"
    out_json.write_text(json.dumps(report, indent=2))

    import csv

    # Backward-compatible legacy CSV output.
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "raster",
            "candidate_count",
            "reviewed_count",
            "evaluated_count",
            "unknown_count",
            "unknown_proposal_count",
            "agreement_count",
            "agreement_rate",
            "tp",
            "tn",
            "fp",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sections.get("legacy_spotcheck", {}).get("per_raster", []):
            w.writerow(row)

    # Stratified CSV output with explicit section label.
    with open(out_strat_csv, "w", newline="") as f:
        fieldnames = [
            "section",
            "raster",
            "candidate_count",
            "reviewed_count",
            "evaluated_count",
            "unknown_count",
            "unknown_proposal_count",
            "agreement_count",
            "agreement_rate",
            "tp",
            "tn",
            "fp",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in stratified_rows:
            w.writerow(row)

    print("\n[spot-check] Summary written:")
    print(f"  - {out_json}")
    print(f"  - {out_csv}")
    print(f"  - {out_strat_csv}")
    if headline:
        print(
            f"  [headline:{headline_source}] "
            f"reviewed={headline.get('reviewed_count', 0)} "
            f"eval={headline.get('evaluated_count', 0)} "
            f"agree={headline.get('agreement_rate', 0.0)*100:.1f}% "
            f"F1={headline.get('f1', 0.0):.3f} "
            f"Acc={headline.get('accuracy', 0.0):.3f}"
        )
    low_conf = sections.get("queue_low_confidence", {}).get("overall", {})
    if low_conf.get("evaluated_count", 0) > 0:
        print(
            "  [note] low_confidence metrics are stress-test diagnostics and "
            "should not be pooled with random spotcheck for model KPI reporting."
        )

    return report


def _discover_headwide_checkpoint(
    model_search_csv: Path,
    checkpoints_root: Path,
    prefer_tta: int = 1,
) -> Path | None:
    """Return best deep_cnn_attn_headwide checkpoint from model_search outputs.

    Selection rule is deterministic:
    1) model_name == deep_cnn_attn_headwide
    2) prefer experiment_source containing `tta_{prefer_tta}`
    3) highest test_f1
    4) lexical tie-break by experiment_source
    """
    if not model_search_csv.exists() or not checkpoints_root.exists():
        return None

    rows: list[dict] = []
    with open(model_search_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model_name") != "deep_cnn_attn_headwide":
                continue
            try:
                row["_test_f1"] = float(row.get("test_f1", "0") or 0.0)
            except Exception:
                row["_test_f1"] = 0.0
            rows.append(row)

    if not rows:
        return None

    pref = [r for r in rows if f"tta_{prefer_tta}" in str(r.get("experiment_source", ""))]
    candidates = pref if pref else rows
    candidates.sort(key=lambda r: (r.get("_test_f1", 0.0), str(r.get("experiment_source", ""))), reverse=True)

    for row in candidates:
        exp = str(row.get("experiment_source", "")).strip()
        if not exp:
            continue
        ckpt_dir = checkpoints_root / exp
        if not ckpt_dir.exists():
            continue
        # Deterministic fold preference: fold1 -> fold2 -> ...
        for i in range(1, 6):
            fp = ckpt_dir / f"fold{i}.pt"
            if fp.exists():
                return fp
        any_pts = sorted(ckpt_dir.glob("*.pt"))
        if any_pts:
            return any_pts[0]
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch CDW tile labeling across all CHM rasters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Per-raster controls:\n"
            "  → CDW   ← No CDW   ↑ Unknown   w Newer year   s Older year   Esc/q save & advance\n"
            "  Archive years are view-only (red border, labeling locked).\n\n"
            "Labels CSV is saved after every labeled chunk; progress.json is updated per raster."
        ),
    )
    p.add_argument("--chm-dir", default="chm_max_hag", help="Directory containing CHM GeoTIFFs")
    p.add_argument("--output", default="output/tile_labels", help="Output directory for label CSVs")
    p.add_argument(
        "--pattern",
        default="*20cm.tif",
        help="Glob pattern for raster files (default: *20cm.tif). "
        "Use '*.tif' for all resolutions.",
    )
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--auto-skip-threshold", type=float, default=0.15)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip fully-completed rasters, resume partially-labeled ones",
    )
    p.add_argument("--scale", type=int, default=3, help="Display upscale factor (default: 3)")
    p.add_argument(
        "--auto-advance",
        type=float,
        default=0.0,
        help="CNN confidence threshold for auto-labeling without UI "
        "(0=off, e.g. 0.97 = auto-label when CNN is "
        ">=97%% or <=3%% confident). Requires CNN model loaded.",
    )
    p.add_argument(
        "--review-pct",
        type=float,
        default=0.05,
        help="Fraction of auto-labeled tiles sent to GUI for manual spot-check "
        "(default: 0.05 = 5%%). Use 0 to disable.",
    )
    p.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip the background CNN fine-tune at session start (faster cold start)",
    )
    p.add_argument(
        "--start-finetune",
        action="store_true",
        help="Explicitly start background CNN fine-tune at session start (opt-in)."
        " Use this to opt into finetuning; by default finetune is disabled.",
    )
    p.add_argument(
        "--annotator",
        default="",
        help="Annotator name or ID stamped into every label row (default: empty)",
    )
    p.add_argument(
        "--tile-list",
        default=None,
        help="Path to audit_review_queue.csv (or any CSV with raster/row_off/col_off). "
        "Only rasters present in this file are processed; only flagged tiles "
        "are shown in the GUI. Use with --resume to fix audit findings.",
    )
    p.add_argument(
        "--max-nodata-pct",
        type=float,
        default=75.0,
        help="Skip rasters with >= this percent nodata (default: 75)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not launch UI; only print which rasters would be processed/skipped",
    )
    p.add_argument(
        "--force-include", action="store_true", help="Ignore nodata filter and include all rasters"
    )
    p.add_argument(
        "--recompute-nodata",
        action="store_true",
        help="Recompute nodata percentages even if cached in progress.json",
    )
    p.add_argument(
        "--model-path",
        default="",
        help="Explicit CNN checkpoint (.pt) for auto-labeling. If empty, auto-discovery is used.",
    )
    p.add_argument(
        "--auto-discover-headwide",
        action="store_true",
        help="Auto-discover deep_cnn_attn_headwide checkpoint from output/model_search (enabled by default if --model-path is empty).",
    )
    p.add_argument(
        "--model-search-csv",
        default="output/model_search/final_test_results.csv",
        help="Model-search final results CSV used for checkpoint auto-discovery.",
    )
    p.add_argument(
        "--model-search-checkpoints",
        default="output/model_search/checkpoints",
        help="Root folder with model-search checkpoint experiment directories.",
    )
    p.add_argument(
        "--preferred-tta",
        type=int,
        default=1,
        help="Preferred TTA variant during headwide checkpoint auto-discovery (default: 1).",
    )
    p.add_argument(
        "--analyze-spotcheck",
        dest="analyze_spotcheck",
        action="store_true",
        help="Analyze 5%% spot-check QA metrics from session JSON files after batch run",
    )
    p.add_argument(
        "--no-analyze-spotcheck",
        dest="analyze_spotcheck",
        action="store_false",
        help="Disable post-run spot-check analysis",
    )
    p.set_defaults(analyze_spotcheck=True)
    args = p.parse_args()

    chm_dir = Path(args.chm_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path: Path | None = Path(args.model_path) if args.model_path else None
    if model_path is None:
        # Preserve reproducibility by deterministic auto-discovery from model_search outputs.
        discovered = _discover_headwide_checkpoint(
            Path(args.model_search_csv),
            Path(args.model_search_checkpoints),
            prefer_tta=args.preferred_tta,
        )
        if discovered is not None:
            model_path = discovered
            print(f"[model] Auto-discovered deep_cnn_attn_headwide checkpoint: {model_path}")
        elif args.auto_advance > 0.0:
            print("[model] Warning: no headwide checkpoint discovered; auto-advance may be disabled.")
    elif not model_path.exists():
        raise FileNotFoundError(f"--model-path not found: {model_path}")

    # ── Optional audit tile-list ──────────────────────────────────────────────
    import csv as _csv

    tile_list: dict[str, list[tuple[int, int]]] | None = None
    tile_meta: dict[str, dict[tuple[int, int], dict]] | None = None
    queue_rows: list[dict] = []
    if args.tile_list:
        with open(args.tile_list, newline="") as _f:
            for _row in _csv.DictReader(_f):
                queue_rows.append(
                    {
                        "raster": _row["raster"],
                        "row_off": int(_row["row_off"]),
                        "col_off": int(_row["col_off"]),
                        "reason": _row.get("reason", ""),
                        "last_source": _row.get("last_source", _row.get("source", "")),
                        "last_model_prob": _row.get("last_model_prob", _row.get("model_prob", "")),
                        "last_label": _row.get("last_label", ""),
                    }
                )

        if args.resume:
            latest_manual = _find_latest_manual_tile_in_queue(output_dir, queue_rows)
            if latest_manual is not None:
                _raster, _row_off, _col_off, _ts = latest_manual
                if _can_trim_queue_after_pointer(
                    output_dir,
                    queue_rows,
                    (_raster, _row_off, _col_off),
                ):
                    queue_rows = _trim_queue_after_tile(queue_rows, (_raster, _row_off, _col_off))
                    print(
                        "[tile-list] Resume pointer from latest manual tile: "
                        f"{_raster} ({_row_off},{_col_off}) at {_ts.isoformat(timespec='seconds')}"
                    )
                else:
                    print(
                        "[tile-list] Resume pointer found but queue was NOT trimmed: "
                        "earlier queued tiles are still pending manual review."
                    )

        from collections import defaultdict as _dd

        _tl: dict[str, list[tuple[int, int]]] = _dd(list)
        _tm: dict[str, dict[tuple[int, int], dict]] = _dd(dict)
        for _row in queue_rows:
            _raster = _row["raster"]
            _key = (_row["row_off"], _row["col_off"])
            _tl[_raster].append(_key)
            _tm[_raster][_key] = {
                "reason": _row.get("reason", ""),
                "last_source": _row.get("last_source", ""),
                "last_model_prob": _row.get("last_model_prob", ""),
                "last_label": _row.get("last_label", ""),
            }
        tile_list = dict(_tl)
        tile_meta = dict(_tm)
        print(
            f"[tile-list] Loaded {sum(len(v) for v in tile_list.values()):,} tiles "
            f"across {len(tile_list)} raster(s) from {args.tile_list}"
        )

    rasters = sorted(chm_dir.glob(args.pattern))
    if not rasters:
        print(f"No rasters found matching '{args.pattern}' in {chm_dir}")
        return

    # In tile-list workflows, keep raster traversal in queue-file order.
    if tile_list is not None:
        by_name = {p.name: p for p in rasters}
        ordered_names: list[str] = []
        seen: set[str] = set()
        for _row in queue_rows:
            nm = str(_row["raster"])
            if nm in by_name and nm not in seen:
                ordered_names.append(nm)
                seen.add(nm)
        rasters = [by_name[nm] for nm in ordered_names]
        if not rasters:
            print("[tile-list] No queued rasters matched --pattern in --chm-dir")
            return

    progress = _load_progress(output_dir)
    # Ensure progress contains nodata tracking structures
    progress.setdefault("nodata_pct", {})
    progress.setdefault("skipped", {})

    print(f"\nFound {len(rasters)} raster(s) matching '{args.pattern}'")
    print(f"Output  : {output_dir}")
    print(f"Progress: {output_dir / PROGRESS_FILE}\n")

    quit_all = False
    for i, chm_path in enumerate(rasters, 1):
        if quit_all:
            break
        stem = chm_path.stem

        # Skip fully completed rasters in resume mode
        if args.resume and stem in progress["completed"]:
            # For tile-list mode, validate completion against queued manual tiles.
            if tile_list is not None and chm_path.name in tile_list:
                pending, total = _count_pending_manual_in_queue(
                    output_dir,
                    chm_path.name,
                    tile_list[chm_path.name],
                )
                if pending == 0:
                    csv_path = output_dir / f"{stem}_labels.csv"
                    counts = _count_labels(csv_path)
                    print(
                        f"[{i:3d}/{len(rasters)}] ✓ (done)  {stem}  "
                        f"queued={total}/{total}  CDW={counts['cdw']}  "
                        f"No={counts['no_cdw']}  Unk={counts['unknown']}"
                    )
                    continue
                progress["completed"] = [s for s in progress["completed"] if s != stem]
                _save_progress(output_dir, progress)
                print(
                    f"[{i:3d}/{len(rasters)}] ↺ stale-complete removed for {stem}: "
                    f"{pending}/{total} queued tiles still pending"
                )
            else:
                csv_path = output_dir / f"{stem}_labels.csv"
                counts = _count_labels(csv_path)
                print(
                    f"[{i:3d}/{len(rasters)}] ✓ (done)  {stem}  "
                    f"CDW={counts['cdw']}  No={counts['no_cdw']}  Unk={counts['unknown']}"
                )
                continue

        print(f"\n{'='*70}")
        print(f"[{i:3d}/{len(rasters)}]  {stem}")
        if stem not in progress["started"]:
            progress["started"].append(stem)
        _save_progress(output_dir, progress)

        # --- nodata check & decision logic ---------------------------------
        nodata_cached = progress.get("nodata_pct", {}).get(stem)
        nodata_pct = None
        if nodata_cached is None or args.recompute_nodata:
            nodata_pct = _compute_nodata_pct(chm_path)
            progress.setdefault("nodata_pct", {})[stem] = nodata_pct
            _save_progress(output_dir, progress)
        else:
            nodata_pct = nodata_cached

        decision = "PROCESS"
        reason = ""
        if nodata_pct is None:
            decision = "SKIP"
            reason = "read_error"
            progress.setdefault("skipped", {})[stem] = reason
        elif (not args.force_include) and (nodata_pct >= args.max_nodata_pct):
            decision = "SKIP"
            reason = f"high_nodata_{nodata_pct:.1f}%"
            progress.setdefault("skipped", {})[stem] = reason

        # Report decision
        print(
            f"  nodata: {nodata_pct if nodata_pct is not None else 'ERR'}% -> {decision} {'' if reason=='' else '('+reason+')'}"
        )
        _save_progress(output_dir, progress)

        if args.dry_run:
            # Do not launch UI; continue to next raster
            continue

        if decision == "SKIP":
            print(f"  Skipping raster {stem} (reason: {reason})")
            continue

        try:
            # Decide finetune behavior: explicit --no-finetune disables; otherwise opt-in via --start-finetune
            run_labeling_session(
                chm_path=chm_path,
                output_dir=output_dir,
                chm_dir=chm_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                auto_skip_threshold=args.auto_skip_threshold,
                resume=args.resume,
                display_scale=args.scale,
                auto_advance_thresh=args.auto_advance,
                review_pct=args.review_pct,
                no_finetune=(args.no_finetune or not args.start_finetune),
                annotator=args.annotator,
                model_path=model_path,
                tile_list=tile_list,
                tile_meta=tile_meta,
            )
        except QuitAllException:
            print(f"\nUser quit — stopping all rasters. Progress saved to {output_dir}")
            _save_progress(output_dir, progress)
            quit_all = True
            break
        except KeyboardInterrupt:
            print(f"\nInterrupted — progress saved to {output_dir}")
            _save_progress(output_dir, progress)
            return

        # Mark complete only if CSV covers all chunks
        csv_path = output_dir / f"{stem}_labels.csv"
        counts = _count_labels(csv_path)
        total_labels = sum(counts.values())
        print(
            f"  Labeled {total_labels} chunks  "
            f"CDW={counts['cdw']}  No={counts['no_cdw']}  Unk={counts['unknown']}"
        )

        mark_complete = True
        if tile_list is not None and chm_path.name in tile_list:
            pending, total = _count_pending_manual_in_queue(
                output_dir,
                chm_path.name,
                tile_list[chm_path.name],
            )
            mark_complete = pending == 0
            if mark_complete:
                print(f"  [tile-list] Queue completion: {total}/{total} manually reviewed")
            else:
                print(f"  [tile-list] Queue completion: {total-pending}/{total} manually reviewed")

        if mark_complete and stem not in progress["completed"]:
            progress["completed"].append(stem)
        if (not mark_complete) and stem in progress["completed"]:
            progress["completed"] = [s for s in progress["completed"] if s != stem]
        _save_progress(output_dir, progress)

    print(f"\n{'='*70}")
    print(f"All rasters processed. Labels in: {output_dir}")
    _print_summary(output_dir, rasters)
    if args.analyze_spotcheck:
        _analyze_spotcheck_sessions(output_dir)


def _print_summary(output_dir: Path, rasters: list[Path]) -> None:
    total_cdw = total_no = total_unk = 0
    import csv as _csv

    for chm in rasters:
        csv_path = output_dir / f"{chm.stem}_labels.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                lbl = row.get("label", "")
                if lbl == "cdw":
                    total_cdw += 1
                elif lbl == "no_cdw":
                    total_no += 1
                elif lbl == "unknown":
                    total_unk += 1

    print(f"\nGrand total  CDW: {total_cdw}  No CDW: {total_no}  Unknown: {total_unk}")
    print(f"Positive rate: {100*total_cdw/(total_cdw+total_no+1e-9):.1f}%")


if __name__ == "__main__":
    main()
