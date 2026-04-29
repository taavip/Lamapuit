"""Label curation and deduplication for Model Search V4.

Pure-logic module: no filesystem side effects beyond reading/writing CSVs passed
in by the caller. All sorting and tie-breaking is deterministic so that two
runs with identical inputs produce byte-identical curated label files.

Rationale (for the thesis methods section):
    The drop13 label pool contains three mutually exclusive provenance
    categories — `manual` (human-reviewed), `auto_reviewed` (human-audited
    model output), and `auto` (pure model output with a calibrated
    probability). We deliberately do *not* discard the `auto` rows: removing
    them leaves ~2.1% of the dataset (12 177 manual rows) which is too small
    to train deep models. Instead we apply a **confidence gate** (t_high /
    t_low) that admits only auto rows where the prior V3 ensemble had an
    ensemble probability ≥ t_high (for `cdw`) or ≤ t_low (for `no_cdw`).
    Surviving auto rows are renamed to `auto_threshold_gate_v4` so that
    downstream code can weight them below manual rows. Duplicates by
    (raster, row_off, col_off) are resolved in favour of the highest-priority
    provenance; timestamps break priority ties to prefer the most recent
    decision.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

LABEL_HEADERS: list[str] = [
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

_RASTER_ID_RE = re.compile(r"^(?P<tile>\d+)_(?P<year>\d{4})_(?P<site>.+?)_chm_max_hag_20cm$")
_LEGACY_RASTER_RE = re.compile(r"^(?P<sample>.+?)_chm_max_hag_20cm(?:\.tif)?$")

# Provenance priority for dedup: larger wins.
PRIORITY_MANUAL = 30
PRIORITY_THRESHOLD_GATE = 20
PRIORITY_AUTO = 10


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def is_manual_source(source: str) -> bool:
    s = (source or "").strip().lower()
    return ("manual" in s) or (s == "auto_reviewed")


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["raster"]), int(row["row_off"]), int(row["col_off"]))


def row_priority(source: str, reason: str) -> int:
    if is_manual_source(source):
        return PRIORITY_MANUAL
    if reason == "threshold_gate":
        return PRIORITY_THRESHOLD_GATE
    return PRIORITY_AUTO


def legacy_sample_id(raster_name: str) -> str | None:
    """Strip the ``_chm_max_hag_20cm`` suffix to get the logical sample key."""
    stem = Path(str(raster_name)).stem
    m = _LEGACY_RASTER_RE.match(stem)
    return None if m is None else str(m.group("sample"))


def parse_raster_identity(raster_name: str) -> dict[str, Any]:
    """Parse tile/year/site fields from a raster filename.

    Returns a dict with ``stem``, ``tile``, ``year``, ``site``, ``place_key``
    (= ``tile_site``), ``grid_x``, ``grid_y``. For tiles that don't match the
    expected pattern, year/site are ``"unknown"`` and grid coordinates are
    ``None``; ``place_key`` defaults to the stem so grouping stays stable.
    """
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
    grid_x = grid_y = None
    if tile.isdigit() and len(tile) >= 6:
        try:
            grid_x = int(tile[:3])
            grid_y = int(tile[3:6])
        except Exception:
            grid_x = grid_y = None

    return {
        "stem": stem,
        "tile": tile,
        "year": year,
        "site": site,
        "place_key": f"{tile}_{site}",
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


def iter_label_rows(csv_path: Path) -> Iterator[dict[str, Any]]:
    with csv_path.open(newline="") as fp:
        rd = csv.DictReader(fp)
        for row in rd:
            label = (row.get("label") or "").strip()
            if label not in ("cdw", "no_cdw"):
                continue
            if row.get("row_off") in (None, "") or row.get("col_off") in (None, ""):
                continue
            yield row


def include_drop_row(row: dict[str, Any], t_high: float, t_low: float) -> tuple[bool, str]:
    """Decide whether a drop13 row is admitted and why.

    Returns a ``(include, reason)`` pair. Reasons:
        - ``"manual_or_reviewed"`` — human-in-loop provenance; always kept.
        - ``"threshold_gate"``   — auto row with model_prob above/below gates.
        - ``"no_prob"``          — auto row without a probability; dropped.
        - ``"below_threshold"``  — auto row that failed the gate; dropped.
    """
    source = row.get("source") or ""
    label = (row.get("label") or "").strip()

    if is_manual_source(source):
        return True, "manual_or_reviewed"

    mp = safe_float(row.get("model_prob"))
    if mp is None:
        return False, "no_prob"

    if label == "cdw" and mp >= t_high:
        return True, "threshold_gate"
    if label == "no_cdw" and mp <= t_low:
        return True, "threshold_gate"
    return False, "below_threshold"


def normalize_row(row: dict[str, Any], default_raster: str) -> dict[str, str]:
    out = {k: "" for k in LABEL_HEADERS}
    for k in LABEL_HEADERS:
        if k in row and row[k] is not None:
            out[k] = str(row[k])
    if not out["raster"]:
        out["raster"] = default_raster
    if not out["chunk_size"]:
        out["chunk_size"] = "128"
    return out


def write_curated_labels_drop_only(
    drop_labels_dir: Path,
    curated_labels_dir: Path,
    t_high: float,
    t_low: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the confidence gate, dedup by (raster,row,col), and write per-raster CSVs.

    Returns ``(stats, all_candidates)``. ``stats`` reports row counts at each
    stage for the methodology table. ``all_candidates`` is the post-dedup row
    set used to build spatial splits.
    """
    curated_labels_dir.mkdir(parents=True, exist_ok=True)
    for stale in curated_labels_dir.glob("*_labels.csv"):
        stale.unlink()

    chosen: dict[tuple[str, int, int], dict[str, Any]] = {}

    stats = {
        "drop_labeled_rows": 0,
        "drop_kept_manual_or_reviewed": 0,
        "drop_kept_threshold_gate": 0,
        "drop_kept_total_before_dedup": 0,
        "drop_rejected_no_prob": 0,
        "drop_rejected_below_threshold": 0,
    }

    for csv_path in sorted(drop_labels_dir.glob("*_labels.csv")):
        for row in iter_label_rows(csv_path):
            stats["drop_labeled_rows"] += 1
            include, reason = include_drop_row(row, t_high=t_high, t_low=t_low)
            if not include:
                if reason == "no_prob":
                    stats["drop_rejected_no_prob"] += 1
                elif reason == "below_threshold":
                    stats["drop_rejected_below_threshold"] += 1
                continue

            norm = normalize_row(row, default_raster=f"{csv_path.stem.replace('_labels', '')}.tif")
            if reason == "threshold_gate" and not is_manual_source(norm.get("source", "")):
                norm["source"] = "auto_threshold_gate_v4"

            if reason == "manual_or_reviewed":
                stats["drop_kept_manual_or_reviewed"] += 1
            elif reason == "threshold_gate":
                stats["drop_kept_threshold_gate"] += 1
            stats["drop_kept_total_before_dedup"] += 1

            key = row_key(norm)
            candidate = {
                "row": norm,
                "priority": row_priority(norm.get("source", ""), reason),
                "ts": str(norm.get("timestamp") or ""),
                "reason": reason,
            }

            prev = chosen.get(key)
            if prev is None or (candidate["priority"], candidate["ts"]) >= (prev["priority"], prev["ts"]):
                chosen[key] = candidate

    by_raster: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_candidates: list[dict[str, Any]] = []
    for key, payload in chosen.items():
        row = payload["row"]
        by_raster[row["raster"]].append(row)
        all_candidates.append(
            {
                "key": key,
                "label": row["label"],
                "raster": row["raster"],
                "source": row.get("source", ""),
                "reason": payload["reason"],
            }
        )

    n_written = 0
    for raster, rows in sorted(by_raster.items()):
        rows.sort(key=lambda r: (int(r["row_off"]), int(r["col_off"])))
        out_csv = curated_labels_dir / f"{Path(raster).stem}_labels.csv"
        with out_csv.open("w", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=LABEL_HEADERS)
            wr.writeheader()
            for row in rows:
                wr.writerow({k: row.get(k, "") for k in LABEL_HEADERS})
                n_written += 1

    stats["curated_rows_after_dedup"] = n_written
    stats["curated_rasters"] = len(by_raster)
    return stats, all_candidates
