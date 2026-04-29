"""FastAPI app for the CHM brush labeling tool."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .orthophoto import fetch_orthophoto, fetch_orthophoto_context
from .predictor import predict_chip, predict_chip_score
from .queue import build_queue
from .renderer import read_chip, render_chm_heatmap, render_hillshade, render_png, smooth_chip
from .temporal import product_views_for_tile, year_views_for_tile
from .tile_index import (
    TileChip,
    candidate_label_keys,
    load_or_build,
    neighbors,
    preferred_label_key,
    status,
)

CHM_DIR = Path(os.environ.get("CHM_DIR", "chm_max_hag"))
MASK_DIR = Path(os.environ.get("MASK_DIR", "output/manual_masks"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "runs/cdw_detect/train/weights/best.pt"))
CLASSIFIER_MODEL_PATH = Path(
    os.environ.get("CLASSIFIER_MODEL_PATH", "output/tile_labels/ensemble_model.pt")
)
QUEUE_MODEL_PATH = Path(os.environ.get("QUEUE_MODEL_PATH", str(CLASSIFIER_MODEL_PATH)))
CHIP_SIZE = int(os.environ.get("CHIP_SIZE", "256"))
QUEUE_TOP_N = int(os.environ.get("QUEUE_TOP_N", "2000"))
QUEUE_MODEL_SCORE_CSV = Path(
    os.environ.get("QUEUE_MODEL_SCORE_CSV", "/workspace/ranked_for_manual_masks_lineaware.csv")
)
QUEUE_MODEL_REFINE_K = int(os.environ.get("QUEUE_MODEL_REFINE_K", "2000"))
QUEUE_MODEL_BOOST = float(os.environ.get("QUEUE_MODEL_BOOST", "1.0"))
QUEUE_MODEL_CONFIDENCE = float(os.environ.get("QUEUE_MODEL_CONFIDENCE", "0.15"))
CACHE_PATH = MASK_DIR / ".tile_index.json"
QUEUE_CACHE = MASK_DIR / ".tile_queue.json"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="CDW Brush Labeler", version="0.2.0")

_index: list[TileChip] = []
_by_id: dict[str, TileChip] = {}
_queue: list[dict] = []


@app.on_event("startup")
def _startup() -> None:
    global _index, _by_id, _queue
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    if not CHM_DIR.exists():
        print(f"[labeler] WARNING: CHM_DIR does not exist: {CHM_DIR}")
        return
    _index = load_or_build(CACHE_PATH, CHM_DIR, chip_size=CHIP_SIZE)
    _by_id = {c.tile_id: c for c in _index}
    print(f"[labeler] Loaded {len(_index)} chips from {CHM_DIR}")

    print(f"[labeler] Building CWD score queue (top {QUEUE_TOP_N})…")
    try:
        _queue = build_queue(
            _index,
            QUEUE_CACHE,
            top_n=QUEUE_TOP_N,
            model_path=str(QUEUE_MODEL_PATH) if QUEUE_MODEL_PATH.exists() else None,
            model_score_csv=str(QUEUE_MODEL_SCORE_CSV) if QUEUE_MODEL_SCORE_CSV.exists() else None,
            model_refine_k=QUEUE_MODEL_REFINE_K,
            model_boost=QUEUE_MODEL_BOOST,
            model_confidence=QUEUE_MODEL_CONFIDENCE,
        )
        print(f"[labeler] Queue: {len(_queue)} scored tiles cached at {QUEUE_CACHE}")
    except Exception as e:  # noqa: BLE001 — index still usable without queue
        print(f"[labeler] Queue build failed: {e}")
        _queue = []


def _get_chip(tile_id: str) -> TileChip:
    chip = _by_id.get(tile_id)
    if chip is None:
        raise HTTPException(404, f"tile_id not found: {tile_id}")
    return chip


def _label_keys(tile_id: str) -> list[str]:
    return candidate_label_keys(tile_id, _by_id)


def _preferred_paths(tile_id: str) -> tuple[str, Path, Path]:
    key = preferred_label_key(tile_id, _by_id)
    return key, MASK_DIR / f"{key}_mask.png", MASK_DIR / f"{key}_meta.json"


def _existing_mask_path(tile_id: str) -> Path | None:
    for key in _label_keys(tile_id):
        path = MASK_DIR / f"{key}_mask.png"
        if path.exists():
            return path
    return None


def _existing_meta(tile_id: str) -> tuple[dict, str | None]:
    for key in _label_keys(tile_id):
        path = MASK_DIR / f"{key}_meta.json"
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text()), key
        except (json.JSONDecodeError, OSError):
            continue
    return {}, None


def _gauss_path_for(raw_path: str) -> str | None:
    """Map `.../chm_raw/<stem>_raw_chm.tif` → `.../chm_gauss/<stem>_gauss_chm.tif` if it exists."""
    p = Path(raw_path)
    if "_raw_chm" not in p.name:
        return None
    sibling = p.parent.parent / "chm_gauss" / p.name.replace("_raw_chm", "_gauss_chm")
    return str(sibling) if sibling.exists() else None


def _read_and_maybe_smooth(chip: TileChip, smooth: bool):
    """Prefer the paired `_gauss_chm.tif` when smooth=1; fall back to on-the-fly blur."""
    path = chip.raster_path
    if smooth:
        paired = _gauss_path_for(path)
        if paired:
            return read_chip(paired, chip.row_off, chip.col_off, chip.chip_size)
    data = read_chip(path, chip.row_off, chip.col_off, chip.chip_size)
    return smooth_chip(data) if smooth else data


# -------------------------------------------------------------------- routes --


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def api_config() -> dict:
    return {
        "chip_size": CHIP_SIZE,
        "chm_dir": str(CHM_DIR),
        "mask_dir": str(MASK_DIR),
        "label_scope": "location-year-source",
        "model_path": str(MODEL_PATH),
        "model_available": MODEL_PATH.exists(),
        "classifier_model_path": str(CLASSIFIER_MODEL_PATH),
        "classifier_model_available": CLASSIFIER_MODEL_PATH.exists(),
        "queue_model_path": str(QUEUE_MODEL_PATH),
        "queue_model_available": QUEUE_MODEL_PATH.exists(),
        "queue_model_refine_k": QUEUE_MODEL_REFINE_K,
        "queue_model_boost": QUEUE_MODEL_BOOST,
        "queue_model_confidence": QUEUE_MODEL_CONFIDENCE,
        "queue_model_score_csv": str(QUEUE_MODEL_SCORE_CSV),
        "queue_model_score_csv_exists": QUEUE_MODEL_SCORE_CSV.exists(),
        "queue_size": len(_queue),
        "total_chips": len(_index),
    }


@app.get("/api/tiles")
def list_tiles(
    raster: str | None = None,
    status_filter: str | None = None,
    year: str | None = None,
    grid: str | None = None,
    source: str | None = None,
    product: str | None = None,
    limit: int = 500,
) -> list[dict]:
    out: list[dict] = []
    for c in _index:
        if raster and c.raster_stem != raster:
            continue
        if year and c.year != str(year):
            continue
        if grid and c.grid != str(grid):
            continue
        if source and c.source != str(source):
            continue
        if product and c.product != str(product):
            continue
        st = status(c.tile_id, MASK_DIR, _by_id)
        if status_filter and st != status_filter:
            continue
        out.append({**c.as_dict(), "status": st})
        if len(out) >= limit:
            break
    return out


@app.get("/api/queue")
def api_queue(
    status_filter: str | None = None,
    year: str | None = None,
    grid: str | None = None,
    source: str | None = None,
    product: str | None = None,
    limit: int = 2000,
    offset: int = 0,
) -> dict:
    """Return the ranked labeling queue with mask status and ordering info."""
    rows: list[dict] = []
    for i, q in enumerate(_queue):
        chip = _by_id.get(q["tile_id"])
        if chip is None:
            continue
        if year and chip.year != str(year):
            continue
        if grid and chip.grid != str(grid):
            continue
        if source and chip.source != str(source):
            continue
        if product and chip.product != str(product):
            continue
        st = status(q["tile_id"], MASK_DIR, _by_id)
        if status_filter and st != status_filter:
            continue
        rows.append({**q, "status": st, "rank": i + 1, **chip.as_dict()})
    total = len(rows)
    labeled = sum(1 for r in rows if r["status"] == "labeled")
    needs = sum(1 for r in rows if r["status"] == "needs-review")
    return {
        "total": total,
        "labeled": labeled,
        "needs_review": needs,
        "unlabeled": total - labeled - needs,
        "items": rows[offset : offset + limit],
    }


@app.get("/api/catalog")
def api_catalog() -> dict:
    grids = sorted({c.grid for c in _index if c.grid})
    years = sorted({c.year for c in _index if c.year})
    sources = sorted({c.source for c in _index if c.source})
    products = sorted({c.product for c in _index if c.product})
    return {
        "grids": grids,
        "years": years,
        "sources": sources,
        "products": products,
    }


@app.get("/api/rasters")
def list_rasters() -> list[dict]:
    by_stem: dict[str, dict] = {}
    for c in _index:
        if c.raster_stem not in by_stem:
            by_stem[c.raster_stem] = {
                "stem": c.raster_stem,
                "grid": c.grid,
                "year": c.year,
                "source": c.source,
                "product": c.product,
                "location_key": c.location_key,
                "chip_count": 0,
                "labeled_count": 0,
            }
        by_stem[c.raster_stem]["chip_count"] += 1
        if status(c.tile_id, MASK_DIR, _by_id) == "labeled":
            by_stem[c.raster_stem]["labeled_count"] += 1
    return sorted(by_stem.values(), key=lambda r: r["stem"])


# ---- per-tile endpoints ----------------------------------------------------


@app.get("/api/tile/{tile_id}/chm")
def tile_chm(tile_id: str, smooth: int = 0) -> Response:
    chip = _get_chip(tile_id)
    data = _read_and_maybe_smooth(chip, bool(smooth))
    return Response(render_png(data), media_type="image/png")


@app.get("/api/tile/{tile_id}/hillshade")
def tile_hillshade(tile_id: str, smooth: int = 0) -> Response:
    chip = _get_chip(tile_id)
    data = _read_and_maybe_smooth(chip, bool(smooth))
    return Response(render_hillshade(data), media_type="image/png")


@app.get("/api/tile/{tile_id}/mask")
def tile_mask(tile_id: str) -> Response:
    _get_chip(tile_id)
    mask_path = _existing_mask_path(tile_id)
    if mask_path is None or not mask_path.exists():
        raise HTTPException(404, "mask not saved yet")
    return FileResponse(mask_path, media_type="image/png")


@app.post("/api/tile/{tile_id}/mask")
async def save_mask(tile_id: str, request: Request) -> JSONResponse:
    _get_chip(tile_id)
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    if not body.startswith(b"\x89PNG"):
        raise HTTPException(400, "body is not a PNG")

    label_key, mask_path, meta_path = _preferred_paths(tile_id)
    mask_path.write_bytes(body)

    existing_meta, _ = _existing_meta(tile_id)
    meta = {
        **existing_meta,
        "tile_id": tile_id,
        "label_key": label_key,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return JSONResponse(
        {
            "ok": True,
            "tile_id": tile_id,
            "label_key": label_key,
            "bytes": len(body),
        }
    )


@app.get("/api/tile/{tile_id}/predict")
def tile_predict(tile_id: str, mode: str = "IntGrad") -> Response:
    """Return heatmap-style guidance overlay for the focused tile.

    `mode` mirrors label_tiles heatmap mode names for UI consistency.
    """
    chip = _get_chip(tile_id)
    if CLASSIFIER_MODEL_PATH.exists():
        png = predict_chip(
            chip.raster_path,
            chip.row_off,
            chip.col_off,
            chip.chip_size,
            CLASSIFIER_MODEL_PATH,
            confidence=QUEUE_MODEL_CONFIDENCE,
            mode=mode,
        )
        return Response(png, media_type="image/png")

    # Fallback keeps endpoint functional when classifier checkpoint is missing.
    data = read_chip(chip.raster_path, chip.row_off, chip.col_off, chip.chip_size)
    return Response(render_chm_heatmap(data, mode=mode), media_type="image/png")


@app.get("/api/tile/{tile_id}/score")
def tile_score(tile_id: str) -> dict:
    """Return best-classifier P(CDW) for this tile (0.0 if model missing)."""
    chip = _get_chip(tile_id)
    if not CLASSIFIER_MODEL_PATH.exists():
        return {"score": None, "available": False, "model": str(CLASSIFIER_MODEL_PATH)}
    score = predict_chip_score(
        chip.raster_path,
        chip.row_off,
        chip.col_off,
        chip.chip_size,
        CLASSIFIER_MODEL_PATH,
        confidence=QUEUE_MODEL_CONFIDENCE,
    )
    return {"score": float(score), "available": True, "model": str(CLASSIFIER_MODEL_PATH)}


@app.get("/api/tile/{tile_id}/ortho")
def tile_ortho(tile_id: str) -> Response:
    chip = _get_chip(tile_id)
    png = fetch_orthophoto(chip, out_size=chip.chip_size)
    if png is None:
        raise HTTPException(404, "orthophoto unavailable for this raster")
    return Response(png, media_type="image/png")


@app.get("/api/tile/{tile_id}/ortho_context")
def tile_ortho_context(tile_id: str, row_radius: int = 1, col_radius: int = 1) -> Response:
    chip = _get_chip(tile_id)
    row_radius = max(0, min(3, int(row_radius)))
    col_radius = max(0, min(3, int(col_radius)))
    png = fetch_orthophoto_context(
        chip,
        row_radius=row_radius,
        col_radius=col_radius,
        out_chip_size=chip.chip_size,
    )
    if png is None:
        raise HTTPException(404, "orthophoto context unavailable for this raster")
    return Response(png, media_type="image/png")


@app.get("/api/tile/{tile_id}/neighbors")
def tile_neighbors(tile_id: str, row_radius: int = 1, col_radius: int = 1) -> dict:
    chip = _get_chip(tile_id)
    row_radius = max(1, min(3, int(row_radius)))
    col_radius = max(1, min(3, int(col_radius)))
    return {
        "center": tile_id,
        "neighbors": neighbors(chip, _index, row_radius=row_radius, col_radius=col_radius),
    }


@app.get("/api/tile/{tile_id}/years")
def tile_years(tile_id: str) -> list[dict]:
    _get_chip(tile_id)
    return year_views_for_tile(tile_id, _by_id)


@app.get("/api/tile/{tile_id}/products")
def tile_products(tile_id: str) -> list[dict]:
    _get_chip(tile_id)
    return product_views_for_tile(tile_id, _by_id)


@app.get("/api/tile/{tile_id}/meta")
def tile_meta(tile_id: str) -> dict:
    chip = _get_chip(tile_id)
    label_key = preferred_label_key(tile_id, _by_id)
    meta, meta_key = _existing_meta(tile_id)
    # Attach queue score if the tile is in the queue.
    score = next((q for q in _queue if q["tile_id"] == tile_id), None)
    return {
        **chip.as_dict(),
        "label_key": label_key,
        "label_keys": _label_keys(tile_id),
        "meta_key": meta_key,
        "status": status(tile_id, MASK_DIR, _by_id),
        "meta": meta,
        "queue": score,
    }


@app.post("/api/tile/{tile_id}/meta")
async def save_meta(tile_id: str, request: Request) -> dict:
    _get_chip(tile_id)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")

    label_key, _mask_path, meta_path = _preferred_paths(tile_id)
    existing, _meta_key = _existing_meta(tile_id)
    merged = {
        **existing,
        **body,
        "tile_id": tile_id,
        "label_key": label_key,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(merged, indent=2))
    return {"ok": True, "meta": merged}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
