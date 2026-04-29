"""Build a ranked queue of tiles most likely to contain CWD.

Base score: CHM-geometry heuristic (CWD-height-band + local edge strength).
Optional refinement: blend in model confidence from the current best checkpoint
for top candidates, so the queue reflects both structure and model belief.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .tile_index import TileChip


HAG_LO = 0.10
HAG_HI = 1.50
GRAD_THRESHOLD = 0.15  # m / pixel; local |grad| above this counts as an edge
MIN_VALID_FRAC = 0.20  # skip tiles with too much nodata


def _blend_score(geom_score: float, model_score: float, model_boost: float) -> float:
        """Blend geometric and model confidence scores.

        `model_boost` is interpreted as a weight in [0, 1]:
            0.0 => geometry-only, 1.0 => model-only.
        """

        w = float(np.clip(model_boost, 0.0, 1.0))
        return float((1.0 - w) * float(geom_score) + w * float(model_score))


@dataclass
class TileScore:
    tile_id: str
    score: float
    geom_score: float
    model_score: float
    valid_frac: float
    cwd_frac: float
    edge_frac: float


def _chips_fingerprint(chips: list[TileChip]) -> str:
    """Hash tile identity + raster file stats so cache invalidates on data changes."""

    h = hashlib.sha1()
    for c in chips:
        p = Path(c.raster_path)
        try:
            st = p.stat()
            stamp = f"{c.tile_id}|{p}|{st.st_size}|{st.st_mtime_ns}\n"
        except OSError:
            stamp = f"{c.tile_id}|{p}|missing\n"
        h.update(stamp.encode("utf-8"))
    return h.hexdigest()


def _model_stamp(model_path: str | Path | None) -> str:
    if not model_path:
        return "none"
    p = Path(model_path)
    try:
        st = p.stat()
        return f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return f"{p}|missing"


def _file_stamp(path: str | Path | None) -> str:
    if not path:
        return "none"
    p = Path(path)
    try:
        st = p.stat()
        return f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return f"{p}|missing"


def _parse_float(v: object) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pick_score_value(row: dict[str, str]) -> float | None:
    # Preference order: explicit probabilities first, then ranking score fallbacks.
    for key in ("prob", "model_prob", "legacy_prob", "base_score", "rank_score", "score"):
        v = _parse_float(row.get(key))
        if v is not None:
            return float(max(0.0, min(1.0, v)))
    return None


def _row_to_parent_tile_key(row: dict[str, str], target_chip_size: int) -> tuple[str, int, int] | None:
    row_off_f = _parse_float(row.get("row_off"))
    col_off_f = _parse_float(row.get("col_off"))
    if row_off_f is None or col_off_f is None:
        return None

    row_off = int(row_off_f)
    col_off = int(col_off_f)
    parent_row = (row_off // target_chip_size) * target_chip_size
    parent_col = (col_off // target_chip_size) * target_chip_size

    stem = ""
    raster_path = (row.get("raster_path") or "").strip()
    raster_stem = (row.get("raster_stem") or "").strip()
    raster = (row.get("raster") or "").strip()
    tile_id = (row.get("tile_id") or "").strip()

    if raster_path:
        stem = Path(raster_path).stem
    elif raster_stem:
        stem = raster_stem
    elif raster:
        stem = Path(raster).stem
    elif tile_id:
        # Common ranking format: "<sample>:<product>:<row>:<col>".
        if ":" in tile_id:
            stem = tile_id.split(":", 1)[0]
        elif "__r" in tile_id:
            stem = tile_id.split("__r", 1)[0]

    if not stem:
        return None
    return stem, parent_row, parent_col


def _load_external_model_scores(
    score_csv_path: str | Path | None,
    *,
    target_chip_size: int,
) -> dict[tuple[str, int, int], float]:
    if not score_csv_path:
        return {}

    score_path = Path(score_csv_path)
    if not score_path.exists():
        return {}

    csv_files: list[Path]
    if score_path.is_dir():
        csv_files = sorted(score_path.rglob("*.csv"))
    else:
        csv_files = [score_path]

    accum: dict[tuple[str, int, int], tuple[float, int]] = {}
    try:
        for csv_path in csv_files:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = _row_to_parent_tile_key(row, target_chip_size)
                    if key is None:
                        continue
                    val = _pick_score_value(row)
                    if val is None:
                        continue
                    total, n = accum.get(key, (0.0, 0))
                    accum[key] = (total + float(val), n + 1)
    except OSError:
        return {}

    return {k: (total / n) for k, (total, n) in accum.items() if n > 0}


def score_chip(path: str, row_off: int, col_off: int, chip_size: int) -> TileScore | None:
    try:
        with rasterio.open(path) as src:
            nodata = src.nodata
            full = Window(0, 0, src.width, src.height)
            win = Window(col_off, row_off, chip_size, chip_size).intersection(full)
            w, h = int(win.width), int(win.height)
            if w <= 0 or h <= 0:
                return None
            data = src.read(1, window=win).astype("float32")
    except Exception:
        return None

    if data.size == 0:
        return None

    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    valid &= data >= 0
    total = data.size
    n_valid = int(valid.sum())
    valid_frac = n_valid / total

    cwd_band = valid & (data >= HAG_LO) & (data <= HAG_HI)
    cwd_frac = cwd_band.sum() / max(1, n_valid)

    # Local gradient magnitude — log edges show up as high |grad| inside the band.
    filled = np.where(valid, data, 0.0)
    dy, dx = np.gradient(filled)
    grad = np.hypot(dx, dy)
    edges = cwd_band & (grad > GRAD_THRESHOLD)
    edge_frac = edges.sum() / max(1, n_valid)

    # Score: favour tiles with many CWD-band *edges*, penalise saturated / mostly-nodata tiles.
    if valid_frac < MIN_VALID_FRAC:
        score = 0.0
    else:
        # Saturation factor: penalise when cwd_frac > 0.4 (likely dense understory).
        sat = 1.0 if cwd_frac < 0.4 else max(0.0, 1.0 - (cwd_frac - 0.4) / 0.4)
        score = float(edge_frac * sat * valid_frac)

    return TileScore(
        tile_id=f"{Path(path).stem}__r{row_off}_c{col_off}",
        score=score,
        geom_score=score,
        model_score=0.0,
        valid_frac=valid_frac,
        cwd_frac=float(cwd_frac),
        edge_frac=float(edge_frac),
    )


def build_queue(
    chips: list[TileChip],
    cache_path: Path,
    *,
    top_n: int = 2000,
    min_score: float = 0.01,
    model_path: str | Path | None = None,
    model_score_csv: str | Path | None = None,
    model_refine_k: int = 300,
    model_boost: float = 1.0,
    model_confidence: float = 0.15,
    force: bool = False,
) -> list[dict]:
    """Score every chip and return the top N as the labeling queue.

    Cached to `cache_path` as JSON keyed by (chip count + CHM dir fingerprint)
    so subsequent boots skip the expensive scan.
    """
    cache_path = Path(cache_path)
    base_fingerprint = _chips_fingerprint(chips)
    model_sig = _model_stamp(model_path)
    csv_sig = _file_stamp(model_score_csv)
    fingerprint = hashlib.sha1(
        (
            f"{base_fingerprint}|model={model_sig}|"
            f"score_csv={csv_sig}|"
            f"refine={model_refine_k}|boost={model_boost}|conf={model_confidence}"
        ).encode("utf-8")
    ).hexdigest()
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("fingerprint") == fingerprint
                and float(cached.get("min_score", min_score)) == float(min_score)
                and cached.get("top_n") >= top_n
                and int(cached.get("model_refine_k", model_refine_k)) == int(model_refine_k)
                and float(cached.get("model_boost", model_boost)) == float(model_boost)
                and float(cached.get("model_confidence", model_confidence))
                == float(model_confidence)
                and str(cached.get("model_score_csv", model_score_csv or ""))
                == str(model_score_csv or "")
            ):
                return cached["queue"][:top_n]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    rows: list[dict] = []
    scanned = 0
    kept = 0
    for c in chips:
        scanned += 1
        try:
            s = score_chip(c.raster_path, c.row_off, c.col_off, c.chip_size)
        except Exception:
            s = None
        if s is None or s.score < min_score:
            continue
        kept += 1
        rows.append(
            {
                "tile_id": c.tile_id,
                "raster_stem": c.raster_stem,
                "year": c.year,
                "row_off": c.row_off,
                "col_off": c.col_off,
                "score": s.score,
                "geom_score": s.geom_score,
                "model_score": s.model_score,
                "valid_frac": s.valid_frac,
                "cwd_frac": s.cwd_frac,
                "edge_frac": s.edge_frac,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)

    # Optional external model scores (CSV) get blended first.
    target_chip_size = chips[0].chip_size if chips else 256
    external_scores = _load_external_model_scores(
        model_score_csv,
        target_chip_size=target_chip_size,
    )
    if external_scores:
        for r in rows:
            m = external_scores.get((r["raster_stem"], r["row_off"], r["col_off"]))
            if m is None:
                continue
            r["model_score"] = float(m)
            r["score"] = _blend_score(r["geom_score"], r["model_score"], model_boost)
        rows.sort(key=lambda r: r["score"], reverse=True)
        print(f"[queue] applied external model scores from {model_score_csv} to {len(external_scores)} keys")

    refined = 0
    refine_n = min(max(0, int(model_refine_k)), len(rows))
    if model_path and refine_n > 0:
        try:
            from .predictor import predict_chip_score

            by_id = {c.tile_id: c for c in chips}
            print(
                f"[queue] refining top {refine_n} with model score "
                f"(boost={model_boost:.3f}, conf={model_confidence:.3f})"
            )
            for r in rows[:refine_n]:
                chip = by_id.get(r["tile_id"])
                if chip is None:
                    continue
                mscore = predict_chip_score(
                    chip.raster_path,
                    chip.row_off,
                    chip.col_off,
                    chip.chip_size,
                    model_path,
                    confidence=model_confidence,
                )
                r["model_score"] = max(float(r.get("model_score", 0.0)), float(mscore))
                r["score"] = _blend_score(r["geom_score"], r["model_score"], model_boost)
                refined += 1
            rows.sort(key=lambda r: r["score"], reverse=True)
        except Exception as e:  # noqa: BLE001 — queue should still work without model refine
            print(f"[queue] model refinement skipped: {e}")

    top = rows[:top_n]
    print(
        f"[queue] scanned {scanned} chips, kept {kept} above score {min_score}, "
        f"refined {refined}, returning top {len(top)}"
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "top_n": top_n,
                "min_score": min_score,
                "model_refine_k": model_refine_k,
                "model_boost": model_boost,
                "model_confidence": model_confidence,
                "model_path": str(model_path) if model_path else None,
                "model_score_csv": str(model_score_csv) if model_score_csv else None,
                "queue": top,
            },
            indent=2,
        )
    )
    return top
