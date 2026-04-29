#!/usr/bin/env python3
"""
Model Search V2 for CDW classification.

Goals:
- Reuse the proven model_search pipeline without modifying existing files.
- Build a curated training pool that combines:
  1) Existing baseline labels.
  2) Additional drop13 labels, keeping only:
     - manual/auto_reviewed rows, or
     - high-confidence pseudo-labels with strict thresholds.
- Augment test split with a small stratified sample from the new drop13 pool.
- Restrict model search to top-N models from previous results (default: 12).
- Build a diverse top-3 ensemble from final test models (different architecture families).
- Save CSV outputs and a run analysis markdown.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


DEFAULT_FALLBACK_MODELS = [
    "deep_cnn_attn",
    "deep_cnn_attn_headlight",
    "deep_cnn_attn_dropout_tuned",
    "deep_cnn_attn_headwide",
    "convnext_small",
    "convnext_tiny",
    "efficientnet_b2",
    "densenet121",
    "efficientnet_b0",
    "swin_t",
    "resnet34",
    "resnet50",
]


_RASTER_ID_RE = re.compile(
    r"^(?P<tile>\d+)_(?P<year>\d{4})_(?P<site>.+?)_chm_max_hag_20cm$"
)


def _parse_raster_identity(raster_name: str) -> dict[str, Any]:
    stem = Path(str(raster_name)).stem
    m = _RASTER_ID_RE.match(stem)
    if not m:
        return {
            "stem": stem,
            "tile": None,
            "year": "unknown",
            "site": "unknown",
            "place_key": stem,
            "grid_x": None,
            "grid_y": None,
        }

    tile = str(m.group("tile"))
    year = str(m.group("year"))
    site = str(m.group("site"))
    grid_x = None
    grid_y = None

    if tile.isdigit() and len(tile) >= 6:
        # Parse 6-digit map-tile id into coarse grid coordinates (e.g. 601546 -> x=601, y=546).
        try:
            grid_x = int(tile[:3])
            grid_y = int(tile[3:6])
        except Exception:
            grid_x = None
            grid_y = None

    return {
        "stem": stem,
        "tile": tile,
        "year": year,
        "site": site,
        "place_key": f"{tile}_{site}",
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


def _import_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register module in sys.modules so decorators (e.g., dataclasses)
    # that inspect the module via sys.modules[...] work correctly.
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Clean up partial registration on failure
        sys.modules.pop(module_name, None)
        raise
    return mod


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _manual_source(source: str) -> bool:
    s = (source or "").strip().lower()
    return ("manual" in s) or (s == "auto_reviewed")


def _row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["raster"]), int(row["row_off"]), int(row["col_off"]))


def _row_priority(source: str, reason: str) -> int:
    if _manual_source(source):
        return 30
    if reason == "threshold_gate":
        return 20
    return 10


def _timestamp_score(row: dict[str, Any]) -> str:
    # lexical ordering is enough for ISO-like timestamps; missing timestamps remain stable.
    return str(row.get("timestamp") or "")


def _normalize_row(row: dict[str, Any], default_raster: str) -> dict[str, str]:
    headers = [
        "raster",
        "row_off",
        "col_off",
        "chunk_size",
        "label",
        "source",
        "annotator",
        "model_name",
        "model_prob",
        "timestamp",
    ]
    out = {k: "" for k in headers}
    for k in headers:
        if k in row and row[k] is not None:
            out[k] = str(row[k])
    if not out["raster"]:
        out["raster"] = default_raster
    if not out["chunk_size"]:
        out["chunk_size"] = "128"
    return out


def _iter_label_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            label = (row.get("label") or "").strip()
            if label not in ("cdw", "no_cdw"):
                continue
            if row.get("row_off") in (None, "") or row.get("col_off") in (None, ""):
                continue
            yield row


def _include_drop13_row(row: dict[str, Any], t_high: float, t_low: float) -> tuple[bool, str]:
    source = row.get("source") or ""
    label = (row.get("label") or "").strip()

    if _manual_source(source):
        return True, "manual_or_reviewed"

    mp = _safe_float(row.get("model_prob"))
    if mp is None:
        return False, "no_prob"

    if label == "cdw" and mp >= t_high:
        return True, "threshold_gate"
    if label == "no_cdw" and mp <= t_low:
        return True, "threshold_gate"
    return False, "below_threshold"


def _top_models_from_prior(summary_csv: Path, n_models: int) -> list[str]:
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            needed = {"model_name", "mean_cv_f1", "std_cv_f1"}
            if needed.issubset(set(df.columns)):
                df = df.sort_values(["mean_cv_f1", "std_cv_f1"], ascending=[False, True])
                ordered = []
                seen = set()
                for model_name in df["model_name"].astype(str).tolist():
                    if model_name in seen:
                        continue
                    seen.add(model_name)
                    ordered.append(model_name)
                    if len(ordered) >= n_models:
                        break
                if ordered:
                    if len(ordered) < n_models:
                        for m in DEFAULT_FALLBACK_MODELS:
                            if m not in seen:
                                ordered.append(m)
                                seen.add(m)
                            if len(ordered) >= n_models:
                                break
                    return ordered[:n_models]
        except Exception:
            pass

    return DEFAULT_FALLBACK_MODELS[:n_models]


def _write_curated_labels(
    base_labels_dir: Path,
    drop_labels_dir: Path,
    curated_labels_dir: Path,
    t_high: float,
    t_low: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    curated_labels_dir.mkdir(parents=True, exist_ok=True)
    for stale in curated_labels_dir.glob("*_labels.csv"):
        stale.unlink()

    chosen: dict[tuple[str, int, int], dict[str, Any]] = {}

    stats = {
        "base_labeled_rows": 0,
        "drop_labeled_rows": 0,
        "drop_kept_manual_or_reviewed": 0,
        "drop_kept_threshold_gate": 0,
        "drop_kept_total_before_dedup": 0,
    }

    # 1) Baseline rows (same pool as previous search)
    for csv_path in sorted(base_labels_dir.glob("*_labels.csv")):
        for row in _iter_label_rows(csv_path):
            stats["base_labeled_rows"] += 1
            norm = _normalize_row(row, default_raster=f"{csv_path.stem.replace('_labels', '')}.tif")
            key = _row_key(norm)
            source = norm.get("source", "")
            chosen[key] = {
                "row": norm,
                "priority": _row_priority(source, "baseline"),
                "ts": _timestamp_score(norm),
                "origin": "base",
                "reason": "baseline",
            }

    # 2) Additional drop13 rows with strict inclusion
    for csv_path in sorted(drop_labels_dir.glob("*_labels.csv")):
        for row in _iter_label_rows(csv_path):
            stats["drop_labeled_rows"] += 1
            include, reason = _include_drop13_row(row, t_high=t_high, t_low=t_low)
            if not include:
                continue

            norm = _normalize_row(row, default_raster=f"{csv_path.stem.replace('_labels', '')}.tif")
            if reason == "threshold_gate" and not _manual_source(norm.get("source", "")):
                norm["source"] = "auto_threshold_gate_v2"

            if reason == "manual_or_reviewed":
                stats["drop_kept_manual_or_reviewed"] += 1
            elif reason == "threshold_gate":
                stats["drop_kept_threshold_gate"] += 1
            stats["drop_kept_total_before_dedup"] += 1

            key = _row_key(norm)
            candidate = {
                "row": norm,
                "priority": _row_priority(norm.get("source", ""), reason),
                "ts": _timestamp_score(norm),
                "origin": "drop13",
                "reason": reason,
            }

            prev = chosen.get(key)
            if prev is None:
                chosen[key] = candidate
            else:
                prev_rank = (int(prev["priority"]), str(prev["ts"]))
                new_rank = (int(candidate["priority"]), str(candidate["ts"]))
                if new_rank >= prev_rank:
                    chosen[key] = candidate

    by_raster: dict[str, list[dict[str, str]]] = defaultdict(list)
    drop_candidates_after_dedup: list[dict[str, Any]] = []
    all_candidates_after_dedup: list[dict[str, Any]] = []

    for key, payload in chosen.items():
        row = payload["row"]
        by_raster[row["raster"]].append(row)

        all_candidates_after_dedup.append(
            {
                "key": key,
                "label": row["label"],
                "raster": row["raster"],
                "source": row.get("source", ""),
                "origin": payload["origin"],
                "reason": payload["reason"],
            }
        )

        if payload["origin"] == "drop13":
            drop_candidates_after_dedup.append(
                {
                    "key": key,
                    "label": row["label"],
                    "raster": row["raster"],
                    "reason": payload["reason"],
                }
            )

    headers = [
        "raster",
        "row_off",
        "col_off",
        "chunk_size",
        "label",
        "source",
        "annotator",
        "model_name",
        "model_prob",
        "timestamp",
    ]

    n_written = 0
    for raster, rows in sorted(by_raster.items()):
        rows.sort(key=lambda r: (int(r["row_off"]), int(r["col_off"])))
        out_csv = curated_labels_dir / f"{Path(raster).stem}_labels.csv"
        with out_csv.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=headers)
            wr.writeheader()
            for row in rows:
                wr.writerow({k: row.get(k, "") for k in headers})
                n_written += 1

    stats["curated_rows_after_dedup"] = n_written
    stats["curated_rasters"] = len(by_raster)
    stats["drop_rows_after_dedup"] = len(drop_candidates_after_dedup)
    return stats, drop_candidates_after_dedup, all_candidates_after_dedup


def _load_test_keys(test_split_path: Path) -> set[tuple[str, int, int]]:
    if not test_split_path.exists():
        return set()
    data = json.loads(test_split_path.read_text())
    return {(str(r), int(ro), int(co)) for r, ro, co in data.get("keys", [])}


def _write_augmented_test_split(
    base_test_split: Path,
    drop_candidates: list[dict[str, Any]],
    output_test_split: Path,
    seed: int,
    extra_fraction: float,
    max_extra: int,
    min_group_size: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    base_keys = _load_test_keys(base_test_split)

    grouped: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for item in drop_candidates:
        key = item["key"]
        if key in base_keys:
            continue
        grouped[(item["label"], item["raster"])].append(key)

    extra_keys: list[tuple[str, int, int]] = []
    for (_label, _raster), keys in grouped.items():
        keys = list(keys)
        rng.shuffle(keys)
        n_pick = int(round(len(keys) * extra_fraction))
        if n_pick <= 0 and len(keys) >= min_group_size:
            n_pick = 1
        if n_pick > 0:
            extra_keys.extend(keys[: min(n_pick, len(keys))])

    if len(extra_keys) > max_extra:
        rng.shuffle(extra_keys)
        extra_keys = extra_keys[:max_extra]

    merged = set(base_keys)
    merged.update(extra_keys)

    keys_sorted = sorted(list(merged), key=lambda x: (x[0], x[1], x[2]))
    payload = {
        "keys": [[r, ro, co] for (r, ro, co) in keys_sorted],
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_test_keys": len(base_keys),
            "extra_test_keys": len(extra_keys),
            "total_test_keys": len(keys_sorted),
            "extra_fraction": extra_fraction,
            "max_extra": max_extra,
            "min_group_size": min_group_size,
        },
    }
    output_test_split.parent.mkdir(parents=True, exist_ok=True)
    output_test_split.write_text(json.dumps(payload, indent=2))
    return payload["meta"]


def _write_spatial_block_test_split(
    all_candidates: list[dict[str, Any]],
    output_test_split: Path,
    seed: int,
    test_fraction: float,
    split_block_size_places: int,
) -> dict[str, Any]:
    rng = random.Random(seed)

    # place_key -> aggregate across all years for strict anti-leak grouping.
    by_place: dict[str, dict[str, Any]] = {}
    for item in all_candidates:
        key = item["key"]
        raster = str(item["raster"])
        ident = _parse_raster_identity(raster)
        place_key = str(ident["place_key"])

        entry = by_place.get(place_key)
        if entry is None:
            gx = ident["grid_x"]
            gy = ident["grid_y"]
            if gx is not None and gy is not None:
                bsize = max(int(split_block_size_places), 1)
                block = (int(gx) // bsize, int(gy) // bsize)
            else:
                # Deterministic fallback when tile id cannot be parsed.
                h = int(zlib.crc32(place_key.encode("utf-8")))
                block = (h % 10000, (h // 10000) % 10000)

            entry = {
                "place_key": place_key,
                "block": block,
                "keys": [],
                "years": set(),
                "n_cdw": 0,
                "n_no_cdw": 0,
            }
            by_place[place_key] = entry

        entry["keys"].append(key)
        if ident["year"] != "unknown":
            entry["years"].add(str(ident["year"]))
        if str(item.get("label", "")).strip() == "cdw":
            entry["n_cdw"] += 1
        else:
            entry["n_no_cdw"] += 1

    if not by_place:
        payload = {
            "keys": [],
            "meta": {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "split_mode": "spatial_blocks",
                "test_fraction_target": float(test_fraction),
                "test_fraction_actual": 0.0,
                "n_places_total": 0,
                "n_places_test": 0,
                "n_blocks_total": 0,
                "n_blocks_test": 0,
                "place_overlap_train_vs_test": 0,
            },
        }
        output_test_split.parent.mkdir(parents=True, exist_ok=True)
        output_test_split.write_text(json.dumps(payload, indent=2))
        return payload["meta"]

    block_to_places: dict[tuple[int, int], list[str]] = defaultdict(list)
    block_row_count: dict[tuple[int, int], int] = defaultdict(int)
    total_rows = 0
    for place_key, info in by_place.items():
        block = info["block"]
        block_to_places[block].append(place_key)
        n_rows = len(info["keys"])
        block_row_count[block] += n_rows
        total_rows += n_rows

    target_test_rows = max(1, int(round(total_rows * float(test_fraction))))

    blocks = list(block_to_places.keys())
    rng.shuffle(blocks)

    selected_blocks: set[tuple[int, int]] = set()
    selected_rows = 0
    while blocks and (selected_rows < target_test_rows or not selected_blocks):
        best_block = None
        best_next_rows = None
        best_score = None

        for block in blocks:
            nxt = selected_rows + int(block_row_count.get(block, 0))
            score = abs(target_test_rows - nxt)
            if (
                best_score is None
                or score < best_score
                or (score == best_score and (best_next_rows is None or nxt > best_next_rows))
            ):
                best_score = score
                best_next_rows = nxt
                best_block = block

        if best_block is None:
            break

        selected_blocks.add(best_block)
        selected_rows = int(best_next_rows or selected_rows)
        blocks.remove(best_block)

    test_places: set[str] = set()
    for block in selected_blocks:
        for place_key in block_to_places.get(block, []):
            test_places.add(place_key)

    test_keys: set[tuple[str, int, int]] = set()
    for place_key in test_places:
        for key in by_place[place_key]["keys"]:
            test_keys.add((str(key[0]), int(key[1]), int(key[2])))

    train_places = set(by_place.keys()) - test_places
    place_overlap = len(train_places.intersection(test_places))

    keys_sorted = sorted(list(test_keys), key=lambda x: (x[0], x[1], x[2]))
    payload = {
        "keys": [[r, ro, co] for (r, ro, co) in keys_sorted],
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split_mode": "spatial_blocks",
            "test_fraction_target": float(test_fraction),
            "test_fraction_actual": (float(len(test_keys)) / float(total_rows)) if total_rows else 0.0,
            "total_rows": int(total_rows),
            "test_rows": int(len(test_keys)),
            "train_rows_estimate": int(total_rows - len(test_keys)),
            "n_places_total": int(len(by_place)),
            "n_places_test": int(len(test_places)),
            "n_places_train": int(len(train_places)),
            "n_blocks_total": int(len(block_to_places)),
            "n_blocks_test": int(len(selected_blocks)),
            "split_block_size_places": int(max(split_block_size_places, 1)),
            "place_overlap_train_vs_test": int(place_overlap),
            "places_with_multi_year": int(sum(1 for v in by_place.values() if len(v["years"]) > 1)),
            "seed": int(seed),
        },
    }

    output_test_split.parent.mkdir(parents=True, exist_ok=True)
    output_test_split.write_text(json.dumps(payload, indent=2))
    return payload["meta"]


def _prepare_chm_merged_dir(chm_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    missing_dirs = []

    for src_dir in chm_dirs:
        if not src_dir.exists():
            missing_dirs.append(str(src_dir))
            continue
        for tif in sorted(src_dir.glob("*.tif")):
            dst = out_dir / tif.name
            if dst.exists():
                skipped += 1
                continue
            try:
                dst.symlink_to(tif.resolve())
                linked += 1
            except FileExistsError:
                skipped += 1

    return {
        "merged_dir": str(out_dir),
        "linked_files": linked,
        "skipped_existing": skipped,
        "missing_dirs": missing_dirs,
    }


def _run_base_model_search(
    base_script: Path,
    curated_labels_dir: Path,
    merged_chm_dir: Path,
    test_split_path: Path,
    output_dir: Path,
    top_models: list[str],
    args: argparse.Namespace,
) -> None:
    base = _import_module(base_script, "model_search_base")

    original_select = base._select_models_after_analysis
    original_deprioritize = base._is_deprioritized_model

    def _force_top_models(_summary: dict[str, Any], _smoke_test: bool) -> list[str]:
        return list(top_models)

    try:
        base._select_models_after_analysis = _force_top_models
        base._is_deprioritized_model = lambda _name: False

        old_argv = sys.argv[:]
        sys.argv = [
            str(base_script),
            "--labels",
            str(curated_labels_dir),
            "--chm-dir",
            str(merged_chm_dir),
            "--test-split",
            str(test_split_path),
            "--output",
            str(output_dir),
            "--seed",
            str(args.seed),
            "--n-folds",
            str(args.n_folds),
            "--batch-size",
            str(args.batch_size),
            "--epochs-scratch",
            str(args.epochs_scratch),
            "--epochs-pretrained",
            str(args.epochs_pretrained),
            "--patience",
            str(args.patience),
            "--cv-group-block-size",
            str(args.cv_group_block_size),
            "--spatial-fence-m",
            str(args.spatial_fence_m),
            "--cv-spatial-block-m",
            str(args.cv_spatial_block_m),
            "--cv-block-candidates-m",
            str(args.cv_block_candidates_m),
            "--top-k-expand",
            str(min(args.top_k_expand, len(top_models))),
            "--top-k-final",
            str(min(args.top_k_final, len(top_models))),
            "--max-extended",
            str(args.max_extended),
            "--augment-random-nodata-frac",
            str(args.augment_random_nodata_frac),
            "--augment-pattern-nodata-frac",
            str(args.augment_pattern_nodata_frac),
            "--stage2-keep-models",
            ",".join(top_models),
            "--stage2-strategies",
            args.stage2_strategies,
            "--stage2-epochs",
            str(args.stage2_epochs),
            "--stage2-patience",
            str(args.stage2_patience),
        ]
        if args.smoke_test:
            sys.argv.append("--smoke-test")
        if args.auto_cv_block_size:
            sys.argv.append("--auto-cv-block-size")
        if args.stage2_pilot:
            sys.argv.append("--stage2-pilot")
            sys.argv.extend(["--stage2-pilot-top-models", str(args.stage2_pilot_top_models)])

        base.main()
    finally:
        base._select_models_after_analysis = original_select
        base._is_deprioritized_model = original_deprioritize
        sys.argv = old_argv


def _model_family(model_name: str) -> str:
    m = str(model_name).lower()
    if m.startswith("deep_cnn_attn"):
        return "deep_cnn"
    if m.startswith("convnext"):
        return "convnext"
    if m.startswith("efficientnet"):
        return "efficientnet"
    if m.startswith("densenet"):
        return "densenet"
    if m.startswith("resnet"):
        return "resnet"
    if m.startswith("swin"):
        return "swin"
    if m.startswith("regnet"):
        return "regnet"
    if m.startswith("mobilenet"):
        return "mobilenet"
    return m.split("_")[0] if "_" in m else m


def _build_diverse_ensemble_top3(
    base_script: Path,
    curated_labels_dir: Path,
    merged_chm_dir: Path,
    test_split_path: Path,
    output_dir: Path,
) -> dict[str, Any] | None:
    final_csv = output_dir / "final_test_results.csv"
    if not final_csv.exists():
        return None

    df = pd.read_csv(final_csv)
    if df.empty:
        return None

    singles = df[
        (~df["model_name"].astype(str).isin(["soft_vote", "stacking_logreg"]))
        & (~df["model_id"].astype(str).isin(["soft_vote_top5", "stacking_top5", "diverse_top3"]))
    ].copy()
    if singles.empty:
        return None

    singles = singles.sort_values("test_f1", ascending=False)
    singles = singles.drop_duplicates(subset=["model_id"], keep="first")
    singles["family"] = singles["model_name"].map(_model_family)

    selected_rows = []
    used_families = set()
    for _, row in singles.iterrows():
        fam = row["family"]
        if fam in used_families:
            continue
        selected_rows.append(row)
        used_families.add(fam)
        if len(selected_rows) == 3:
            break

    if len(selected_rows) < 3:
        existing_ids = {str(r["model_id"]) for r in selected_rows}
        for _, row in singles.iterrows():
            mid = str(row["model_id"])
            if mid in existing_ids:
                continue
            selected_rows.append(row)
            existing_ids.add(mid)
            if len(selected_rows) == 3:
                break

    if len(selected_rows) < 3:
        return None

    base = _import_module(base_script, "model_search_base_ensemble")

    # Build test array exactly like base search.
    source_weights = {
        "manual": 1.00,
        "auto_reviewed": 0.85,
        "": 0.75,
        "auto": 0.60,
        "auto_skip": 0.30,
        "auto_threshold_gate_v2": 0.70,
    }

    records = base._load_records_with_probs(curated_labels_dir, source_weights)
    test_keys = base._load_test_keys(test_split_path)
    rec_test = [r for r in records if r.key in test_keys]
    if not rec_test:
        return None

    helpers = base._import_fine_tune_helpers(base_script.parent / "fine_tune_cnn.py")
    build_deep_cnn = helpers["build_deep_cnn"]
    norm_tile = helpers["norm_tile"]

    X_test, y_test, _w_test, _meta_test = base._build_arrays_with_meta(rec_test, merged_chm_dir, norm_tile)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs_list = []
    thresholds = []
    members = []

    for row in selected_rows:
        model_name = str(row["model_name"])
        model_id = str(row["model_id"])
        ckpt = output_dir / "final_models" / f"{model_id}.pt"
        if not ckpt.exists():
            continue

        net, _ = base._build_model(model_name, build_deep_cnn)
        net = net.to(device)

        state = torch.load(ckpt, map_location=device, weights_only=False)
        cleaned = base._clean_state_dict(state.get("state_dict", {}))
        net.load_state_dict(cleaned, strict=False)

        ev = base._evaluate_classifier(net, X_test, y_test, device=device, tta=True)
        probs_list.append(ev["probs"])

        thr = _safe_float(row.get("threshold_from_val"))
        thresholds.append(float(thr) if thr is not None else 0.5)
        members.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "family": _model_family(model_name),
                "threshold_from_val": float(thresholds[-1]),
                "test_f1": float(row.get("test_f1", 0.0)),
            }
        )

    if len(probs_list) < 2:
        return None

    probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    thr_ens = float(np.mean(thresholds)) if thresholds else 0.5
    pred = (probs >= thr_ens).astype(int)

    if len(np.unique(y_test)) > 1:
        auc = float(roc_auc_score(y_test, probs))
    else:
        auc = 0.5

    f1 = float(f1_score(y_test, pred, zero_division=0))
    precision = float(precision_score(y_test, pred, zero_division=0))
    recall = float(recall_score(y_test, pred, zero_division=0))
    cm = confusion_matrix(y_test, pred).tolist()

    row = {
        "model_id": "diverse_top3",
        "experiment_source": "ensemble_diverse",
        "model_name": "diverse_soft_vote_top3",
        "data_strategy": "mixed",
        "loss_name": "mixed",
        "regularization": "mixed",
        "test_auc": auc,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "threshold_from_val": thr_ens,
        "confusion_matrix": cm,
    }

    out_csv = output_dir / "diverse_ensemble_top3.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "members": members,
        "metrics": row,
        "n_test_samples": int(len(y_test)),
        "n_test_cdw": int(np.sum(y_test == 1)),
        "n_test_no_cdw": int(np.sum(y_test == 0)),
    }
    (output_dir / "diverse_ensemble_top3.json").write_text(json.dumps(payload, indent=2))

    # Append to final_test_results if not present.
    final_df = pd.read_csv(final_csv)
    if "model_id" not in final_df.columns or "diverse_top3" not in set(final_df["model_id"].astype(str)):
        final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
        final_df.to_csv(final_csv, index=False)

    return payload


def _write_run_docs(
    output_dir: Path,
    curated_stats: dict[str, Any],
    test_stats: dict[str, Any],
    chm_stats: dict[str, Any],
    top_models: list[str],
    diverse_payload: dict[str, Any] | None,
) -> None:
    final_csv = output_dir / "final_test_results.csv"
    summary_csv = output_dir / "experiment_summary.csv"

    lines = []
    lines.append("# Model Search V2 - Results Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Data Curation")
    lines.append(f"- Base labeled rows: {curated_stats.get('base_labeled_rows', 0)}")
    lines.append(f"- Drop13 labeled rows: {curated_stats.get('drop_labeled_rows', 0)}")
    lines.append(
        f"- Drop13 kept manual/reviewed: {curated_stats.get('drop_kept_manual_or_reviewed', 0)}"
    )
    lines.append(
        f"- Drop13 kept by thresholds: {curated_stats.get('drop_kept_threshold_gate', 0)}"
    )
    lines.append(f"- Curated rows after dedup: {curated_stats.get('curated_rows_after_dedup', 0)}")
    lines.append(f"- Curated rasters: {curated_stats.get('curated_rasters', 0)}")
    lines.append("")
    lines.append("## Test Split")
    lines.append(f"- Base test keys: {test_stats.get('base_test_keys', 0)}")
    lines.append(f"- Extra test keys (drop13): {test_stats.get('extra_test_keys', 0)}")
    lines.append(f"- Total test keys: {test_stats.get('total_test_keys', 0)}")
    lines.append("")
    lines.append("## CHM Merge")
    lines.append(f"- Linked files: {chm_stats.get('linked_files', 0)}")
    lines.append(f"- Existing skipped: {chm_stats.get('skipped_existing', 0)}")
    missing = chm_stats.get("missing_dirs", []) or []
    lines.append(f"- Missing dirs: {', '.join(missing) if missing else 'none'}")
    lines.append("")
    lines.append("## Selected Model Pool (Top-N from previous search)")
    for m in top_models:
        lines.append(f"- {m}")
    lines.append("")

    if summary_csv.exists():
        lines.append("## Experiment Summary CSV")
        lines.append(f"- Path: {summary_csv}")
        lines.append("")

    if final_csv.exists():
        lines.append("## Final Test Results CSV")
        lines.append(f"- Path: {final_csv}")
        try:
            df = pd.read_csv(final_csv)
            if not df.empty and {"model_id", "test_f1", "test_auc"}.issubset(df.columns):
                top = df.sort_values("test_f1", ascending=False).head(8)
                lines.append("")
                lines.append(top.to_markdown(index=False))
        except Exception:
            pass
        lines.append("")

    lines.append("## Diverse Ensemble (Top-3 different families)")
    if diverse_payload:
        lines.append(
            f"- Metrics: F1={diverse_payload['metrics']['test_f1']:.4f}, "
            f"AUC={diverse_payload['metrics']['test_auc']:.4f}, "
            f"P={diverse_payload['metrics']['test_precision']:.4f}, "
            f"R={diverse_payload['metrics']['test_recall']:.4f}"
        )
        lines.append("- Members:")
        for m in diverse_payload.get("members", []):
            lines.append(
                f"  - {m['model_name']} ({m['family']}) "
                f"thr={m['threshold_from_val']:.3f} f1={m['test_f1']:.4f}"
            )
    else:
        lines.append("- Not available (insufficient final models or missing artifacts).")

    (output_dir / "RESULTS_ANALYSIS_V2.md").write_text("\n".join(lines))


def _write_prepared_summary(
    output_dir: Path,
    curated_stats: dict[str, Any],
    test_stats: dict[str, Any],
    chm_stats: dict[str, Any],
    top_models: list[str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "thresholds": {
            "t_high": args.t_high,
            "t_low": args.t_low,
        },
        "top_models": top_models,
        "curated_stats": curated_stats,
        "test_stats": test_stats,
        "chm_stats": chm_stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prepared_dataset_summary.json").write_text(json.dumps(payload, indent=2))

    rows = [
        {"metric": "base_labeled_rows", "value": curated_stats.get("base_labeled_rows", 0)},
        {"metric": "drop_labeled_rows", "value": curated_stats.get("drop_labeled_rows", 0)},
        {
            "metric": "drop_kept_manual_or_reviewed",
            "value": curated_stats.get("drop_kept_manual_or_reviewed", 0),
        },
        {
            "metric": "drop_kept_threshold_gate",
            "value": curated_stats.get("drop_kept_threshold_gate", 0),
        },
        {
            "metric": "curated_rows_after_dedup",
            "value": curated_stats.get("curated_rows_after_dedup", 0),
        },
        {"metric": "base_test_keys", "value": test_stats.get("base_test_keys", 0)},
        {"metric": "extra_test_keys", "value": test_stats.get("extra_test_keys", 0)},
        {"metric": "total_test_keys", "value": test_stats.get("total_test_keys", 0)},
        {"metric": "linked_chm_files", "value": chm_stats.get("linked_files", 0)},
        {"metric": "selected_model_count", "value": len(top_models)},
    ]
    pd.DataFrame(rows).to_csv(output_dir / "prepared_dataset_counts.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Search V2: curated data + top-N model pool + diverse ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    root = Path(__file__).resolve().parents[2]

    parser.add_argument("--base-labels", default=str(root / "output/tile_labels"))
    parser.add_argument("--drop-labels", default=str(root / "output/onboarding_labels_v2_drop13"))
    parser.add_argument("--base-test-split", default=str(root / "output/tile_labels/cnn_test_split.json"))
    parser.add_argument("--base-chm-dir", default=str(root / "chm_max_hag"))
    parser.add_argument("--drop-chm-dir", default=str(root / "data/lamapuit/chm_max_hag_13_drop"))

    parser.add_argument("--prior-summary", default=str(root / "output/model_search/experiment_summary.csv"))
    parser.add_argument("--base-search-script", default=str(root / "scripts/model_search.py"))

    parser.add_argument("--output", default=str(root / "output/model_search_v2"))

    parser.add_argument("--n-models", type=int, default=12)
    parser.add_argument(
        "--force-models",
        default="",
        help="Comma-separated model names to force as top model pool (overrides --n-models/prior ranking)",
    )
    parser.add_argument("--t-high", type=float, default=0.9995)
    parser.add_argument("--t-low", type=float, default=0.0698)

    parser.add_argument(
        "--split-mode",
        choices=["legacy", "spatial_blocks"],
        default="legacy",
        help="Test split construction mode. Use spatial_blocks to keep same-place multi-year tiles together.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Target test-set fraction when split-mode=spatial_blocks",
    )
    parser.add_argument(
        "--split-block-size-places",
        type=int,
        default=2,
        help="Spatial block size in coarse place-grid cells for split-mode=spatial_blocks",
    )

    parser.add_argument("--extra-test-fraction", type=float, default=0.10)
    parser.add_argument("--max-extra-test", type=int, default=1500)
    parser.add_argument("--min-extra-group-size", type=int, default=15)

    parser.add_argument("--seed", type=int, default=2026)

    # Forwarded to base model_search.
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-scratch", type=int, default=60)
    parser.add_argument("--epochs-pretrained", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--top-k-expand", type=int, default=12)
    parser.add_argument("--top-k-final", type=int, default=12)
    parser.add_argument("--max-extended", type=int, default=36)
    parser.add_argument("--cv-group-block-size", type=int, default=128)
    parser.add_argument("--spatial-fence-m", type=float, default=0.0)
    parser.add_argument(
        "--cv-spatial-block-m",
        type=float,
        default=0.0,
        help="CV grouping block size in meters. If 0, defaults to --spatial-fence-m",
    )
    parser.add_argument(
        "--auto-cv-block-size",
        action="store_true",
        help="Probe candidate spatial block sizes and choose the best-balanced CV setting",
    )
    parser.add_argument(
        "--cv-block-candidates-m",
        default="26,39,52,78,104",
        help="Comma-separated candidate CV block sizes in meters used when --auto-cv-block-size is set",
    )
    parser.add_argument(
        "--augment-random-nodata-frac",
        type=float,
        default=0.0,
        help="Random per-pixel nodata fraction used in training augmentation (0..1)",
    )
    parser.add_argument(
        "--augment-pattern-nodata-frac",
        type=float,
        default=0.0,
        help="Repeating-pattern nodata fraction used in training augmentation (0..1)",
    )
    parser.add_argument("--stage2-strategies", default="full,focal,mixup_swa,tta")
    parser.add_argument("--stage2-epochs", type=int, default=60)
    parser.add_argument("--stage2-patience", type=int, default=10)
    parser.add_argument("--stage2-pilot", action="store_true")
    parser.add_argument("--stage2-pilot-top-models", type=int, default=6)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    prepared_dir = output_dir / "prepared"
    curated_labels_dir = prepared_dir / "labels_curated_v2"
    augmented_test_split = prepared_dir / "cnn_test_split_v2.json"
    merged_chm_dir = prepared_dir / "chm_merged"

    base_labels_dir = Path(args.base_labels)
    drop_labels_dir = Path(args.drop_labels)
    base_test_split = Path(args.base_test_split)
    base_chm_dir = Path(args.base_chm_dir)
    drop_chm_dir = Path(args.drop_chm_dir)
    prior_summary = Path(args.prior_summary)
    base_search_script = Path(args.base_search_script)

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    if args.force_models.strip():
        seen = set()
        top_models = []
        for raw in args.force_models.split(","):
            m = raw.strip().lower()
            if not m or m in seen:
                continue
            seen.add(m)
            top_models.append(m)
    else:
        top_models = _top_models_from_prior(prior_summary, n_models=args.n_models)

    curated_stats, drop_candidates, all_candidates = _write_curated_labels(
        base_labels_dir=base_labels_dir,
        drop_labels_dir=drop_labels_dir,
        curated_labels_dir=curated_labels_dir,
        t_high=args.t_high,
        t_low=args.t_low,
    )

    if args.split_mode == "spatial_blocks":
        test_stats = _write_spatial_block_test_split(
            all_candidates=all_candidates,
            output_test_split=augmented_test_split,
            seed=args.seed,
            test_fraction=args.test_fraction,
            split_block_size_places=args.split_block_size_places,
        )
    else:
        test_stats = _write_augmented_test_split(
            base_test_split=base_test_split,
            drop_candidates=drop_candidates,
            output_test_split=augmented_test_split,
            seed=args.seed,
            extra_fraction=args.extra_test_fraction,
            max_extra=args.max_extra_test,
            min_group_size=args.min_extra_group_size,
        )

    chm_stats = _prepare_chm_merged_dir(
        chm_dirs=[base_chm_dir, drop_chm_dir],
        out_dir=merged_chm_dir,
    )

    _write_prepared_summary(
        output_dir=output_dir,
        curated_stats=curated_stats,
        test_stats=test_stats,
        chm_stats=chm_stats,
        top_models=top_models,
        args=args,
    )

    print("Prepared dataset and model list for Model Search V2")
    print(json.dumps({"top_models": top_models, "curated_stats": curated_stats, "test_stats": test_stats}, indent=2))

    if args.prepare_only:
        _write_run_docs(
            output_dir=output_dir,
            curated_stats=curated_stats,
            test_stats=test_stats,
            chm_stats=chm_stats,
            top_models=top_models,
            diverse_payload=None,
        )
        print("prepare-only mode complete")
        return

    _run_base_model_search(
        base_script=base_search_script,
        curated_labels_dir=curated_labels_dir,
        merged_chm_dir=merged_chm_dir,
        test_split_path=augmented_test_split,
        output_dir=output_dir,
        top_models=top_models,
        args=args,
    )

    diverse_payload = _build_diverse_ensemble_top3(
        base_script=base_search_script,
        curated_labels_dir=curated_labels_dir,
        merged_chm_dir=merged_chm_dir,
        test_split_path=augmented_test_split,
        output_dir=output_dir,
    )

    _write_run_docs(
        output_dir=output_dir,
        curated_stats=curated_stats,
        test_stats=test_stats,
        chm_stats=chm_stats,
        top_models=top_models,
        diverse_payload=diverse_payload,
    )

    print("Model Search V2 completed")
    print(f"Results: {output_dir / 'final_test_results.csv'}")
    print(f"Analysis: {output_dir / 'RESULTS_ANALYSIS_V2.md'}")


if __name__ == "__main__":
    main()
