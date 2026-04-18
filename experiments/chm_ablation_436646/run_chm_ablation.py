#!/usr/bin/env python3
"""CHM ablation for tile classification using best model_search_v2/v3 models.

This script discovers CHM rasters for one tile ID across years, selects top
classification checkpoints from model_search_v2 and model_search_v3 rankings,
and evaluates each selected model on each CHM using year-matched tile labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("chm_ablation")


@dataclass
class LabelRow:
    row_off: int
    col_off: int
    chunk_size: int
    y: int


@dataclass
class ModelCandidate:
    source: str
    experiment_id: str
    model_name: str
    score_f1: float
    checkpoint_path: Path


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _split_csv_arg(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_years(raw: str) -> list[int]:
    years: list[int] = []
    for token in _split_csv_arg(raw):
        try:
            years.append(int(token))
        except ValueError:
            LOGGER.warning("Ignoring invalid year token: %s", token)
    return sorted(set(years))


def _extract_year(name: str) -> int | None:
    m = re.search(r"_(20\d{2})_", name)
    if not m:
        return None
    return int(m.group(1))


def _variant_id(tile_id: str, year: int, stem: str) -> str:
    prefix = f"{tile_id}_{year}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    parts = stem.split("_")
    if len(parts) > 2 and parts[0] == tile_id and parts[1] == str(year):
        return "_".join(parts[2:])
    return stem


def _source_context(path: Path, roots: list[Path]) -> tuple[str, str]:
    for root in roots:
        try:
            rel = path.relative_to(root)
            rel_parent = "." if str(rel.parent) == "." else str(rel.parent)
            return root.name, rel_parent
        except ValueError:
            continue
    return path.parent.name, str(path.parent)


def _load_label_rows(labels_csv: Path, allowed_sources: set[str]) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = str(row.get("source", "")).strip().lower()
            if allowed_sources and src not in allowed_sources:
                continue

            label = str(row.get("label", "")).strip().lower()
            if label == "cdw":
                y = 1
            elif label == "no_cdw":
                y = 0
            else:
                continue

            try:
                rows.append(
                    LabelRow(
                        row_off=int(row["row_off"]),
                        col_off=int(row["col_off"]),
                        chunk_size=int(row.get("chunk_size") or 128),
                        y=y,
                    )
                )
            except Exception:
                continue
    return rows


def discover_labels(
    tile_id: str,
    years: list[int],
    label_dirs: list[Path],
    allowed_sources: set[str],
) -> tuple[dict[int, Path], dict[int, list[LabelRow]]]:
    labels_by_year_path: dict[int, Path] = {}
    labels_by_year_rows: dict[int, list[LabelRow]] = {}

    for year in years:
        selected: Path | None = None
        for label_dir in label_dirs:
            if not label_dir.exists():
                continue
            candidates = sorted(label_dir.glob(f"{tile_id}_{year}_*_labels.csv"))
            if candidates:
                selected = candidates[0]
                break

        if selected is None:
            LOGGER.warning("Missing labels for year %s in %s", year, label_dirs)
            continue

        rows = _load_label_rows(selected, allowed_sources)
        if not rows:
            LOGGER.warning("No usable labels in %s", selected)
            continue

        labels_by_year_path[year] = selected
        labels_by_year_rows[year] = rows

    return labels_by_year_path, labels_by_year_rows


def discover_chms(
    tile_id: str,
    years: list[int],
    search_roots: list[Path],
    max_chms: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in search_roots:
        if not root.exists():
            LOGGER.warning("Search root does not exist: %s", root)
            continue
        for path in root.rglob(f"*{tile_id}*.tif"):
            year = _extract_year(path.name)
            if year is None or year not in years:
                continue

            full_path = str(path.resolve())
            if full_path in seen:
                continue
            seen.add(full_path)

            source_root, rel_parent = _source_context(path.resolve(), search_roots)
            records.append(
                {
                    "year": year,
                    "chm_path": full_path,
                    "chm_name": path.name,
                    "chm_stem": path.stem,
                    "variant_id": _variant_id(tile_id, year, path.stem),
                    "source_root": source_root,
                    "relative_parent": rel_parent,
                }
            )

    records.sort(key=lambda r: (r["year"], r["chm_name"], r["chm_path"]))
    if max_chms > 0:
        records = records[:max_chms]
    return records


def _first_checkpoint_for_experiment(checkpoint_root: Path, experiment_id: str) -> Path | None:
    exp_dir = checkpoint_root / experiment_id
    if not exp_dir.exists():
        return None
    fold1 = exp_dir / "fold1.pt"
    if fold1.exists():
        return fold1
    folds = sorted(exp_dir.glob("fold*.pt"))
    if folds:
        return folds[0]
    return None


def _read_ranked_models(
    ranking_csv: Path,
    checkpoint_root: Path,
    source_tag: str,
    score_column: str = "mean_cv_f1",
) -> list[ModelCandidate]:
    if not ranking_csv.exists():
        LOGGER.warning("Ranking CSV not found: %s", ranking_csv)
        return []

    out: list[ModelCandidate] = []
    with ranking_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exp_id = str(row.get("experiment_id", "")).strip()
            model_name = str(row.get("model_name", "")).strip() or "unknown"
            if not exp_id:
                continue
            try:
                score_f1 = float(row.get(score_column, ""))
            except Exception:
                continue

            ckpt = _first_checkpoint_for_experiment(checkpoint_root, exp_id)
            if ckpt is None:
                continue

            out.append(
                ModelCandidate(
                    source=source_tag,
                    experiment_id=exp_id,
                    model_name=model_name,
                    score_f1=score_f1,
                    checkpoint_path=ckpt,
                )
            )

    out.sort(key=lambda m: m.score_f1, reverse=True)
    return out


def select_models(
    v2_candidates: list[ModelCandidate],
    v3_candidates: list[ModelCandidate],
    top_k: int,
    min_per_source: int,
) -> list[ModelCandidate]:
    top_k = max(1, top_k)

    per_source: dict[str, list[ModelCandidate]] = {
        "v2": v2_candidates,
        "v3": v3_candidates,
    }
    active_sources = [src for src, cands in per_source.items() if cands]
    if not active_sources:
        return []

    safe_min_per_source = max(0, min_per_source)
    if safe_min_per_source * len(active_sources) > top_k:
        safe_min_per_source = max(1, top_k // len(active_sources))

    selected: list[ModelCandidate] = []
    selected_ckpts: set[str] = set()

    for src in active_sources:
        taken = 0
        for cand in per_source[src]:
            key = str(cand.checkpoint_path)
            if key in selected_ckpts:
                continue
            selected.append(cand)
            selected_ckpts.add(key)
            taken += 1
            if taken >= safe_min_per_source:
                break

    all_candidates = sorted(v2_candidates + v3_candidates, key=lambda m: m.score_f1, reverse=True)
    for cand in all_candidates:
        if len(selected) >= top_k:
            break
        key = str(cand.checkpoint_path)
        if key in selected_ckpts:
            continue
        selected.append(cand)
        selected_ckpts.add(key)

    selected.sort(key=lambda m: m.score_f1, reverse=True)
    return selected


def _load_label_tiles_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "label_tiles.py"
    spec = importlib.util.spec_from_file_location("label_tiles_mod", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CheckpointPredictor:
    """Minimal predictor wrapper for architecture-aware checkpoint inference."""

    def __init__(self, net: Any, device: Any) -> None:
        self._net = net
        self._device = device

    def predict_proba_cdw(self, tile: np.ndarray) -> float | None:
        try:
            import importlib

            torch = importlib.import_module("torch")

            tile_norm = np.clip(tile, 0.0, 20.0) / 20.0
            x = torch.tensor(tile_norm[np.newaxis, np.newaxis], dtype=torch.float32).to(self._device)
            with torch.no_grad():
                prob = float(torch.softmax(self._net(x), dim=1)[0, 1].cpu())
            return prob
        except Exception:
            return None


def _load_checkpoint_predictor(
    label_tiles_mod: Any,
    candidate: ModelCandidate,
) -> tuple[CheckpointPredictor | None, float, str]:
    """Load one model checkpoint using build_fn_name/model_name fallback logic."""
    try:
        import importlib

        torch = importlib.import_module("torch")

        ckpt = torch.load(candidate.checkpoint_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

        build_fn_name = str(ckpt.get("build_fn_name", "") or meta.get("build_fn_name", "")).strip()
        model_name = str(
            ckpt.get("model_name", "")
            or meta.get("model_name", "")
            or candidate.model_name
        ).strip().lower()

        build_fn = label_tiles_mod._get_build_fn(build_fn_name) if build_fn_name else None
        if build_fn is None and model_name:
            build_fn = label_tiles_mod._get_build_fn_for_model_name(model_name)
        if build_fn is None and model_name.startswith("deep_cnn"):
            build_fn = label_tiles_mod._get_build_fn("_build_deep_cnn_attn")
        if build_fn is None:
            return None, 0.5, f"No build function for model_name={model_name} build_fn={build_fn_name}"

        net = label_tiles_mod._instantiate_model_from_build_fn(build_fn)

        # Accept multiple checkpoint formats (plain, DataParallel, SWA).
        raw_state = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "ema_state_dict"):
                val = ckpt.get(key)
                if isinstance(val, dict):
                    raw_state = val
                    break
            if raw_state is None and isinstance(ckpt.get("model"), dict):
                raw_state = ckpt["model"]

        if not isinstance(raw_state, dict):
            return None, 0.5, "Checkpoint has no usable state dict"

        cleaned_state: dict[str, Any] = {}
        for key, value in raw_state.items():
            if key in {"n_averaged", "module.n_averaged"}:
                continue
            new_key = key[7:] if key.startswith("module.") else key
            cleaned_state[new_key] = value

        try:
            net.load_state_dict(cleaned_state)
        except RuntimeError:
            net.load_state_dict(cleaned_state, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        net.eval()

        thresh = float(meta.get("best_thresh", meta.get("threshold", 0.5)))
        return CheckpointPredictor(net, device), thresh, "ok"
    except Exception as exc:
        return None, 0.5, str(exc)


def _extract_tiles(chm_path: Path, label_rows: list[LabelRow]) -> np.ndarray:
    tiles: list[np.ndarray] = []
    with rasterio.open(chm_path) as src:
        for row in label_rows:
            arr = src.read(
                1,
                window=Window(row.col_off, row.row_off, row.chunk_size, row.chunk_size),
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
            tiles.append(arr)
    return np.stack(tiles, axis=0)


def _auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    y = y_true.astype(np.int32)
    s = y_prob.astype(np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_pos = float(ranks[pos].sum())
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    n = int(y_true.size)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    auc = _auc_score(y_true, y_prob)

    return {
        "n_tiles": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
    }


def _safe_prob(prob: float | None) -> float:
    if prob is None:
        return 0.5
    try:
        val = float(prob)
    except Exception:
        return 0.5
    if not np.isfinite(val):
        return 0.5
    return float(min(1.0, max(0.0, val)))


def _evaluate(
    models: list[ModelCandidate],
    chm_records: list[dict[str, Any]],
    labels_by_year: dict[int, list[LabelRow]],
    dry_run: bool,
) -> pd.DataFrame:
    if dry_run:
        rows: list[dict[str, Any]] = []
        for chm in chm_records:
            year = int(chm["year"])
            n_tiles = len(labels_by_year.get(year, []))
            for model in models:
                rows.append(
                    {
                        "model_source": model.source,
                        "model_experiment_id": model.experiment_id,
                        "model_name": model.model_name,
                        "model_score_f1": model.score_f1,
                        "model_checkpoint": str(model.checkpoint_path),
                        "threshold": np.nan,
                        "year": year,
                        "chm_name": chm["chm_name"],
                        "chm_path": chm["chm_path"],
                        "variant_id": chm["variant_id"],
                        "source_root": chm["source_root"],
                        "relative_parent": chm["relative_parent"],
                        "n_tiles": n_tiles,
                        "tp": np.nan,
                        "fp": np.nan,
                        "tn": np.nan,
                        "fn": np.nan,
                        "accuracy": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "auc": np.nan,
                        "status": "dry_run",
                    }
                )
        return pd.DataFrame(rows)

    label_tiles_mod = _load_label_tiles_module()
    loaded: list[tuple[ModelCandidate, Any, float]] = []
    for model in models:
        predictor, threshold, status = _load_checkpoint_predictor(label_tiles_mod, model)
        if predictor is None:
            LOGGER.warning("Skipping model (failed to load): %s (%s)", model.checkpoint_path, status)
            continue
        loaded.append((model, predictor, threshold))

    if not loaded:
        raise RuntimeError("No selected model could be loaded.")

    rows = []
    for idx, chm in enumerate(chm_records, 1):
        year = int(chm["year"])
        labels = labels_by_year.get(year, [])
        if not labels:
            LOGGER.warning("Skipping CHM without labels for year %s: %s", year, chm["chm_name"])
            continue

        chm_path = Path(chm["chm_path"])
        LOGGER.info("[%d/%d] Evaluating CHM %s", idx, len(chm_records), chm_path.name)
        try:
            tiles = _extract_tiles(chm_path, labels)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable CHM %s: %s", chm_path, exc)
            continue

        y_true = np.array([row.y for row in labels], dtype=np.int32)
        for model, predictor, threshold in loaded:
            probs = [_safe_prob(predictor.predict_proba_cdw(tile)) for tile in tiles]
            y_prob = np.array(probs, dtype=np.float64)
            metrics = _compute_metrics(y_true, y_prob, threshold)

            row = {
                "model_source": model.source,
                "model_experiment_id": model.experiment_id,
                "model_name": model.model_name,
                "model_score_f1": model.score_f1,
                "model_checkpoint": str(model.checkpoint_path),
                "threshold": threshold,
                "year": year,
                "chm_name": chm["chm_name"],
                "chm_path": chm["chm_path"],
                "variant_id": chm["variant_id"],
                "source_root": chm["source_root"],
                "relative_parent": chm["relative_parent"],
                "status": "ok",
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def _write_markdown_report(
    path: Path,
    tile_id: str,
    years: list[int],
    label_paths: dict[int, Path],
    selected_models_df: pd.DataFrame,
    discovered_chms_df: pd.DataFrame,
    summary_chm_df: pd.DataFrame,
    summary_variant_df: pd.DataFrame,
    summary_model_df: pd.DataFrame,
    dry_run: bool,
) -> None:
    lines: list[str] = []
    lines.append("# CHM Ablation Report")
    lines.append("")
    lines.append(f"- Tile ID: {tile_id}")
    lines.append(f"- Years: {', '.join(str(y) for y in years)}")
    lines.append(f"- Dry run: {dry_run}")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## Labels")
    lines.append("")
    for year in years:
        if year in label_paths:
            lines.append(f"- {year}: {label_paths[year]}")
        else:
            lines.append(f"- {year}: MISSING")
    lines.append("")

    lines.append("## Selected Models")
    lines.append("")
    lines.append(f"- Total selected: {len(selected_models_df)}")
    lines.append("")
    lines.append("| source | experiment_id | model_name | mean_cv_f1 | checkpoint |")
    lines.append("|---|---|---|---:|---|")
    for _, row in selected_models_df.iterrows():
        lines.append(
            f"| {row['source']} | {row['experiment_id']} | {row['model_name']} | {row['score_f1']:.6f} | {row['checkpoint_path']} |"
        )
    lines.append("")

    lines.append("## CHM Discovery")
    lines.append("")
    lines.append(f"- Total CHMs discovered: {len(discovered_chms_df)}")
    lines.append("")

    if dry_run:
        lines.append("Dry run completed. Metrics are not computed in dry-run mode.")
        lines.append("")
    else:
        lines.append("## Top CHMs (Mean F1 Across Selected Models)")
        lines.append("")
        lines.append("| rank | year | chm_name | mean_f1 | mean_recall | mean_precision | n_models |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|")
        top_chm = summary_chm_df.head(20).reset_index(drop=True)
        for idx, row in top_chm.iterrows():
            lines.append(
                "| {rank} | {year} | {name} | {f1:.4f} | {recall:.4f} | {prec:.4f} | {n_models} |".format(
                    rank=idx + 1,
                    year=int(row["year"]),
                    name=row["chm_name"],
                    f1=float(row["mean_f1"]),
                    recall=float(row["mean_recall"]),
                    prec=float(row["mean_precision"]),
                    n_models=int(row["n_models"]),
                )
            )
        lines.append("")

        lines.append("## Top Variants (Across Years)")
        lines.append("")
        lines.append("| rank | variant_id | mean_f1 | mean_recall | n_years | n_model_chm_cases |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        top_variant = summary_variant_df.head(20).reset_index(drop=True)
        for idx, row in top_variant.iterrows():
            lines.append(
                "| {rank} | {variant} | {f1:.4f} | {recall:.4f} | {years} | {cases} |".format(
                    rank=idx + 1,
                    variant=row["variant_id"],
                    f1=float(row["mean_f1"]),
                    recall=float(row["mean_recall"]),
                    years=int(row["n_years"]),
                    cases=int(row["n_cases"]),
                )
            )
        lines.append("")

        lines.append("## Model Robustness Across CHMs")
        lines.append("")
        lines.append("| source | experiment_id | model_name | mean_f1 | mean_recall | mean_precision | n_chms |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for _, row in summary_model_df.iterrows():
            lines.append(
                "| {source} | {exp} | {name} | {f1:.4f} | {recall:.4f} | {prec:.4f} | {n_chms} |".format(
                    source=row["model_source"],
                    exp=row["model_experiment_id"],
                    name=row["model_name"],
                    f1=float(row["mean_f1"]),
                    recall=float(row["mean_recall"]),
                    prec=float(row["mean_precision"]),
                    n_chms=int(row["n_chms"]),
                )
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CHM ablation for tile classification.")
    parser.add_argument("--tile-id", default="436646", help="Tile ID to evaluate.")
    parser.add_argument("--years", default="2018,2020,2022,2024", help="Comma-separated years.")

    parser.add_argument("--top-k", type=int, default=5, help="Number of top models to evaluate.")
    parser.add_argument(
        "--min-per-source",
        type=int,
        default=2,
        help="Minimum number of selected models from each source (v2 and v3).",
    )

    parser.add_argument(
        "--search-roots",
        default="data/lamapuit,output",
        help="Comma-separated directories where CHMs are searched.",
    )
    parser.add_argument(
        "--labels-dirs",
        default=(
            "output/model_search_v3_academic_leakage26/prepared/labels_curated_v2,"
            "output/model_search_v2/prepared/labels_curated_v2"
        ),
        help="Comma-separated label directories (first match per year is used).",
    )
    parser.add_argument(
        "--label-sources",
        default="",
        help="Optional comma-separated source filter (for example: manual,reviewed). Empty means all.",
    )

    parser.add_argument("--v2-ranking", default="analysis/model_search_v2_ranked_all.csv")
    parser.add_argument("--v2-checkpoints", default="output/model_search_v2/checkpoints")
    parser.add_argument(
        "--v3-ranking",
        default="output/model_search_v3_academic_leakage26/experiment_summary.csv",
    )
    parser.add_argument(
        "--v3-checkpoints",
        default="output/model_search_v3_academic_leakage26/checkpoints",
    )

    parser.add_argument(
        "--output-dir",
        default="experiments/chm_ablation_436646/results",
        help="Directory where run artifacts are stored.",
    )
    parser.add_argument("--max-chms", type=int, default=0, help="Optional CHM cap for smoke runs.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, no CNN inference.")
    return parser.parse_args()


def run_experiment() -> None:
    args = parse_args()
    years = _parse_years(args.years)
    if not years:
        raise ValueError("At least one valid year is required.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.info("Starting CHM ablation for tile %s", args.tile_id)

    output_root = _resolve_path(args.output_dir)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run directory: %s", run_dir)

    search_roots = [_resolve_path(x) for x in _split_csv_arg(args.search_roots)]
    label_dirs = [_resolve_path(x) for x in _split_csv_arg(args.labels_dirs)]
    label_sources = {s.lower() for s in _split_csv_arg(args.label_sources)}

    labels_by_year_path, labels_by_year_rows = discover_labels(
        tile_id=args.tile_id,
        years=years,
        label_dirs=label_dirs,
        allowed_sources=label_sources,
    )
    if not labels_by_year_rows:
        raise RuntimeError("No usable labels found for selected years.")

    discovered_chms = discover_chms(
        tile_id=args.tile_id,
        years=years,
        search_roots=search_roots,
        max_chms=args.max_chms,
    )
    if not discovered_chms:
        raise RuntimeError("No CHM rasters discovered for requested tile and years.")

    v2_candidates = _read_ranked_models(
        ranking_csv=_resolve_path(args.v2_ranking),
        checkpoint_root=_resolve_path(args.v2_checkpoints),
        source_tag="v2",
    )
    v3_candidates = _read_ranked_models(
        ranking_csv=_resolve_path(args.v3_ranking),
        checkpoint_root=_resolve_path(args.v3_checkpoints),
        source_tag="v3",
    )

    selected_models = select_models(
        v2_candidates=v2_candidates,
        v3_candidates=v3_candidates,
        top_k=args.top_k,
        min_per_source=args.min_per_source,
    )
    if not selected_models:
        raise RuntimeError("No ranked model checkpoint was selected from v2/v3.")

    selected_models_df = pd.DataFrame([asdict(m) for m in selected_models])
    discovered_chms_df = pd.DataFrame(discovered_chms)
    labels_rows_df = pd.DataFrame(
        [
            {
                "year": year,
                "labels_csv": str(path),
                "n_tiles": len(labels_by_year_rows.get(year, [])),
            }
            for year, path in sorted(labels_by_year_path.items())
        ]
    )

    selected_models_df.to_csv(run_dir / "selected_models.csv", index=False)
    discovered_chms_df.to_csv(run_dir / "discovered_chms.csv", index=False)
    labels_rows_df.to_csv(run_dir / "labels_by_year.csv", index=False)

    metrics_df = _evaluate(
        models=selected_models,
        chm_records=discovered_chms,
        labels_by_year=labels_by_year_rows,
        dry_run=args.dry_run,
    )
    metrics_df.to_csv(run_dir / "metrics_detailed.csv", index=False)

    if args.dry_run:
        summary_chm_df = pd.DataFrame()
        summary_variant_df = pd.DataFrame()
        summary_model_df = pd.DataFrame()
    else:
        if metrics_df.empty:
            raise RuntimeError("Inference finished but no metric rows were produced.")

        summary_chm_df = (
            metrics_df.groupby(
                ["year", "chm_name", "chm_path", "variant_id", "source_root", "relative_parent"],
                as_index=False,
            )
            .agg(
                mean_f1=("f1", "mean"),
                mean_auc=("auc", "mean"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_accuracy=("accuracy", "mean"),
                n_models=("model_experiment_id", "nunique"),
            )
            .sort_values(["mean_f1", "mean_recall", "mean_precision"], ascending=[False, False, False])
            .reset_index(drop=True)
        )

        summary_variant_df = (
            metrics_df.groupby("variant_id", as_index=False)
            .agg(
                mean_f1=("f1", "mean"),
                mean_auc=("auc", "mean"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_accuracy=("accuracy", "mean"),
                n_years=("year", "nunique"),
                n_cases=("chm_path", "count"),
            )
            .sort_values(["mean_f1", "mean_recall"], ascending=[False, False])
            .reset_index(drop=True)
        )

        summary_model_df = (
            metrics_df.groupby(["model_source", "model_experiment_id", "model_name"], as_index=False)
            .agg(
                mean_f1=("f1", "mean"),
                mean_auc=("auc", "mean"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_accuracy=("accuracy", "mean"),
                n_chms=("chm_path", "nunique"),
            )
            .sort_values(["mean_f1", "mean_recall"], ascending=[False, False])
            .reset_index(drop=True)
        )

        summary_chm_df.to_csv(run_dir / "summary_chm.csv", index=False)
        summary_variant_df.to_csv(run_dir / "summary_variant.csv", index=False)
        summary_model_df.to_csv(run_dir / "summary_model.csv", index=False)

    report_path = run_dir / "experiment_report.md"
    _write_markdown_report(
        path=report_path,
        tile_id=args.tile_id,
        years=years,
        label_paths=labels_by_year_path,
        selected_models_df=selected_models_df,
        discovered_chms_df=discovered_chms_df,
        summary_chm_df=summary_chm_df,
        summary_variant_df=summary_variant_df,
        summary_model_df=summary_model_df,
        dry_run=args.dry_run,
    )

    payload = {
        "tile_id": args.tile_id,
        "years": years,
        "dry_run": bool(args.dry_run),
        "run_dir": str(run_dir),
        "n_selected_models": int(len(selected_models_df)),
        "n_discovered_chms": int(len(discovered_chms_df)),
        "n_label_years": int(len(labels_by_year_path)),
        "n_metric_rows": int(len(metrics_df)),
        "label_sources_filter": sorted(label_sources),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info("Run completed. Artifacts are under: %s", run_dir)


if __name__ == "__main__":
    run_experiment()
