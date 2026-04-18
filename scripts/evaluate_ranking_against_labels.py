#!/usr/bin/env python3
"""Evaluate tile ranking quality against existing labeled tiles.

This script joins a ranked tile CSV (from rank_tiles_for_manual_masks.py) with
label CSV files and reports ranking quality metrics such as Precision@K,
Recall@K, Lift@K, NDCG@K, AP, and ROC-AUC.

Default behavior prioritizes manual labels only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TileKey:
    sample_id: str
    row_off: int
    col_off: int
    chunk_size: int


@dataclass
class LabelRecord:
    label: str
    source: str
    timestamp: str


@dataclass
class RankedRecord:
    key: TileKey
    rank: int
    score: float
    tile_id: str


def _parse_int(text: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(text).strip())
    except Exception:
        return default


def _parse_float(text: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(str(text).strip())
    except Exception:
        return default


def _split_csv_set(text: str) -> set[str]:
    return {part.strip().lower() for part in str(text).split(",") if part.strip()}


def _sample_id_from_raster_name(raster_name: str) -> str:
    stem = Path(str(raster_name).strip()).stem
    if "_chm_" in stem:
        return stem.split("_chm_", 1)[0]
    return stem


def _sample_id_from_ranked_row(row: dict[str, str]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id

    tile_id = str(row.get("tile_id", "")).strip()
    if tile_id and ":" in tile_id:
        return tile_id.split(":", 1)[0]
    return tile_id


def _row_col_from_tile_id(tile_id: str) -> tuple[Optional[int], Optional[int]]:
    parts = str(tile_id).split(":")
    if len(parts) < 4:
        return None, None
    return _parse_int(parts[-2]), _parse_int(parts[-1])


def _source_priority(source: str) -> int:
    src = str(source).strip().lower()
    if src == "manual":
        return 3
    if src == "auto":
        return 2
    if src == "auto_skip":
        return 1
    return 0


def _resolve_topk(topk_text: str, n: int) -> list[int]:
    parsed: list[int] = []
    for part in str(topk_text).split(","):
        try:
            value = int(part.strip())
        except Exception:
            continue
        if value > 0:
            parsed.append(value)
    if not parsed:
        parsed = [10, 25, 50, 100, 250, 500]
    out = sorted({k for k in parsed if k <= max(1, n)})
    if not out:
        out = [min(max(1, n), parsed[0])]
    return out


def _load_labels(
    labels_glob: str,
    *,
    allowed_sources: set[str],
    positive_labels: set[str],
    negative_labels: set[str],
) -> tuple[dict[TileKey, LabelRecord], dict[str, int]]:
    label_map: dict[TileKey, LabelRecord] = {}
    stats = {
        "label_files": 0,
        "label_rows_total": 0,
        "label_rows_source_filtered": 0,
        "label_rows_label_filtered": 0,
        "label_rows_key_invalid": 0,
        "label_rows_kept": 0,
        "label_rows_replaced": 0,
    }

    paths = sorted(glob.glob(labels_glob))
    for path in paths:
        stats["label_files"] += 1
        with Path(path).open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                stats["label_rows_total"] += 1

                source = str(row.get("source", "")).strip().lower()
                if allowed_sources and source not in allowed_sources:
                    stats["label_rows_source_filtered"] += 1
                    continue

                label = str(row.get("label", "")).strip().lower()
                if label not in positive_labels and label not in negative_labels:
                    stats["label_rows_label_filtered"] += 1
                    continue

                sample_id = _sample_id_from_raster_name(str(row.get("raster", "")).strip())
                row_off = _parse_int(str(row.get("row_off", "")).strip())
                col_off = _parse_int(str(row.get("col_off", "")).strip())
                chunk_size = _parse_int(str(row.get("chunk_size", "")).strip())
                if not sample_id or row_off is None or col_off is None or chunk_size is None:
                    stats["label_rows_key_invalid"] += 1
                    continue

                key = TileKey(sample_id=sample_id, row_off=row_off, col_off=col_off, chunk_size=chunk_size)
                rec = LabelRecord(
                    label=label,
                    source=source,
                    timestamp=str(row.get("timestamp", "")).strip(),
                )

                existing = label_map.get(key)
                if existing is None:
                    label_map[key] = rec
                    stats["label_rows_kept"] += 1
                    continue

                old_pri = _source_priority(existing.source)
                new_pri = _source_priority(rec.source)
                replace = new_pri > old_pri
                if not replace and new_pri == old_pri and rec.timestamp and rec.timestamp > existing.timestamp:
                    replace = True

                if replace:
                    label_map[key] = rec
                    stats["label_rows_replaced"] += 1

    return label_map, stats


def _load_ranked(
    ranked_csv: Path,
    *,
    default_chunk_size: int,
    dedupe_by_key: bool,
    max_ranked: int,
) -> tuple[list[RankedRecord], dict[str, int]]:
    records: list[RankedRecord] = []
    stats = {
        "rank_rows_total": 0,
        "rank_rows_key_invalid": 0,
        "rank_rows_kept": 0,
        "rank_rows_dedup_dropped": 0,
    }

    with ranked_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=1):
            stats["rank_rows_total"] += 1

            rank_val = _parse_int(str(row.get("rank", "")).strip(), default=i)
            rank = int(rank_val) if rank_val is not None else i

            score = _parse_float(str(row.get("rank_score", "")).strip())
            if score is None:
                score = -float(rank)

            sample_id = _sample_id_from_ranked_row(row)
            row_off = _parse_int(str(row.get("row_off", "")).strip())
            col_off = _parse_int(str(row.get("col_off", "")).strip())
            chunk_size = _parse_int(str(row.get("chunk_size", "")).strip(), default=default_chunk_size)

            tile_id = str(row.get("tile_id", "")).strip()
            if (row_off is None or col_off is None) and tile_id:
                parsed_row, parsed_col = _row_col_from_tile_id(tile_id)
                if row_off is None:
                    row_off = parsed_row
                if col_off is None:
                    col_off = parsed_col

            if not sample_id or row_off is None or col_off is None or chunk_size is None:
                stats["rank_rows_key_invalid"] += 1
                continue

            records.append(
                RankedRecord(
                    key=TileKey(sample_id=sample_id, row_off=row_off, col_off=col_off, chunk_size=chunk_size),
                    rank=rank,
                    score=float(score),
                    tile_id=tile_id,
                )
            )

    records.sort(key=lambda r: (r.rank, -r.score))

    if dedupe_by_key:
        deduped: list[RankedRecord] = []
        seen: set[TileKey] = set()
        for rec in records:
            if rec.key in seen:
                stats["rank_rows_dedup_dropped"] += 1
                continue
            deduped.append(rec)
            seen.add(rec.key)
        records = deduped

    if max_ranked > 0 and len(records) > max_ranked:
        records = records[:max_ranked]

    stats["rank_rows_kept"] = len(records)
    return records, stats


def _precision_at_k(binary_rel: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    kk = min(k, len(binary_rel))
    if kk <= 0:
        return 0.0
    return float(sum(binary_rel[:kk])) / float(kk)


def _recall_at_k(binary_rel: list[int], k: int, total_pos: int) -> float:
    if total_pos <= 0:
        return float("nan")
    kk = min(k, len(binary_rel))
    return float(sum(binary_rel[:kk])) / float(total_pos)


def _ndcg_at_k(binary_rel: list[int], k: int) -> float:
    kk = min(max(0, k), len(binary_rel))
    if kk == 0:
        return 0.0

    dcg = 0.0
    for idx, rel in enumerate(binary_rel[:kk], start=1):
        if rel > 0:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(sum(binary_rel), kk)
    if ideal_hits <= 0:
        return 0.0

    idcg = 0.0
    for idx in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(idx + 1)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _average_precision(binary_rel: list[int], total_pos: int) -> float:
    if total_pos <= 0:
        return float("nan")

    hits = 0
    sum_prec = 0.0
    for idx, rel in enumerate(binary_rel, start=1):
        if rel > 0:
            hits += 1
            sum_prec += float(hits) / float(idx)
    return float(sum_prec / float(total_pos))


def _roc_auc(y_true: list[int], y_score: list[float]) -> float:
    n = len(y_true)
    if n == 0 or n != len(y_score):
        return float("nan")

    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs = sorted((float(s), int(y)) for y, s in zip(y_true, y_score))
    sum_ranks_pos = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1

        avg_rank = (i + 1 + j) / 2.0
        pos_in_tie = sum(1 for k in range(i, j) if pairs[k][1] == 1)
        sum_ranks_pos += avg_rank * pos_in_tie
        i = j

    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _fmt_metric(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked tiles against labeled tiles")
    parser.add_argument("--ranked-csv", type=str, required=True, help="Ranked CSV from rank_tiles_for_manual_masks.py")
    parser.add_argument(
        "--labels-glob",
        type=str,
        default="output/onboarding_labels_v2_drop13/*_labels.csv",
        help="Glob for label CSV files",
    )
    parser.add_argument("--allowed-sources", type=str, default="manual", help="Comma-separated label sources")
    parser.add_argument("--positive-labels", type=str, default="cdw", help="Comma-separated positive labels")
    parser.add_argument("--negative-labels", type=str, default="no_cdw", help="Comma-separated negative labels")
    parser.add_argument("--default-chunk-size", type=int, default=128)
    parser.add_argument("--dedupe-by-key", action="store_true", help="Deduplicate ranked rows by spatial key")
    parser.add_argument("--max-ranked", type=int, default=0, help="Optional cap on ranked rows to evaluate (0=all)")
    parser.add_argument("--topk", type=str, default="10,25,50,100,250,500")
    parser.add_argument("--output-json", type=str, default="", help="Optional metrics JSON output path")
    parser.add_argument("--output-csv", type=str, default="", help="Optional top-k metrics CSV output path")
    args = parser.parse_args()

    ranked_csv = Path(args.ranked_csv)
    if not ranked_csv.exists():
        raise FileNotFoundError(f"Ranked CSV not found: {ranked_csv}")

    positive_labels = _split_csv_set(args.positive_labels)
    negative_labels = _split_csv_set(args.negative_labels)
    allowed_sources = _split_csv_set(args.allowed_sources)

    if not positive_labels:
        raise ValueError("positive-labels must contain at least one value")
    if not negative_labels:
        raise ValueError("negative-labels must contain at least one value")

    label_map, label_stats = _load_labels(
        labels_glob=str(args.labels_glob),
        allowed_sources=allowed_sources,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
    )

    ranked, rank_stats = _load_ranked(
        ranked_csv=ranked_csv,
        default_chunk_size=int(args.default_chunk_size),
        dedupe_by_key=bool(args.dedupe_by_key),
        max_ranked=int(args.max_ranked),
    )

    matched: list[tuple[RankedRecord, LabelRecord]] = []
    unmatched = 0
    for rec in ranked:
        label_rec = label_map.get(rec.key)
        if label_rec is None:
            unmatched += 1
            continue
        matched.append((rec, label_rec))

    matched_binary: list[int] = []
    matched_scores: list[float] = []
    for rec, label_rec in matched:
        matched_binary.append(1 if label_rec.label in positive_labels else 0)
        matched_scores.append(rec.score)

    n_eval = len(matched_binary)
    n_pos = sum(matched_binary)
    n_neg = n_eval - n_pos
    prevalence = (float(n_pos) / float(n_eval)) if n_eval > 0 else float("nan")

    topk_values = _resolve_topk(args.topk, n_eval if n_eval > 0 else 1)

    metrics_topk: list[dict[str, float]] = []
    for k in topk_values:
        p_k = _precision_at_k(matched_binary, k)
        r_k = _recall_at_k(matched_binary, k, total_pos=n_pos)
        ndcg_k = _ndcg_at_k(matched_binary, k)
        lift_k = p_k / prevalence if prevalence == prevalence and prevalence > 0 else float("nan")
        metrics_topk.append(
            {
                "k": float(k),
                "precision_at_k": p_k,
                "recall_at_k": r_k,
                "lift_at_k": lift_k,
                "ndcg_at_k": ndcg_k,
            }
        )

    ap = _average_precision(matched_binary, total_pos=n_pos)
    auc = _roc_auc(matched_binary, matched_scores)

    summary = {
        "ranked_csv": str(ranked_csv),
        "labels_glob": str(args.labels_glob),
        "allowed_sources": sorted(allowed_sources),
        "positive_labels": sorted(positive_labels),
        "negative_labels": sorted(negative_labels),
        "dedupe_by_key": bool(args.dedupe_by_key),
        "max_ranked": int(args.max_ranked),
        "label_stats": label_stats,
        "rank_stats": rank_stats,
        "matched_count": n_eval,
        "unmatched_count": unmatched,
        "coverage": (float(n_eval) / float(len(ranked))) if ranked else 0.0,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "prevalence": prevalence,
        "average_precision": ap,
        "roc_auc": auc,
        "topk_metrics": metrics_topk,
    }

    print("=== Ranking Evaluation ===")
    print(f"ranked_csv: {ranked_csv}")
    print(f"labels_glob: {args.labels_glob}")
    print(f"allowed_sources: {','.join(sorted(allowed_sources)) if allowed_sources else '(all)'}")
    print(f"rank_rows_kept: {rank_stats['rank_rows_kept']}")
    print(f"matched_count: {n_eval}")
    print(f"unmatched_count: {unmatched}")
    print(f"coverage: {_fmt_metric(summary['coverage'])}")
    print(f"positive_count: {n_pos}")
    print(f"negative_count: {n_neg}")
    print(f"prevalence: {_fmt_metric(prevalence)}")
    print(f"average_precision: {_fmt_metric(ap)}")
    print(f"roc_auc: {_fmt_metric(auc)}")

    print("top-k metrics:")
    for row in metrics_topk:
        k = int(row["k"])
        print(
            "  "
            f"k={k:4d} "
            f"P@k={_fmt_metric(row['precision_at_k'])} "
            f"R@k={_fmt_metric(row['recall_at_k'])} "
            f"Lift@k={_fmt_metric(row['lift_at_k'])} "
            f"NDCG@k={_fmt_metric(row['ndcg_at_k'])}"
        )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote_json: {out_json}")

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["k", "precision_at_k", "recall_at_k", "lift_at_k", "ndcg_at_k"],
            )
            writer.writeheader()
            for row in metrics_topk:
                writer.writerow(
                    {
                        "k": int(row["k"]),
                        "precision_at_k": _fmt_metric(float(row["precision_at_k"])),
                        "recall_at_k": _fmt_metric(float(row["recall_at_k"])),
                        "lift_at_k": _fmt_metric(float(row["lift_at_k"])),
                        "ndcg_at_k": _fmt_metric(float(row["ndcg_at_k"])),
                    }
                )
        print(f"wrote_csv: {out_csv}")


if __name__ == "__main__":
    main()