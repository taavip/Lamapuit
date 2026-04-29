"""Unit tests for labeler/ backend (tile_index, renderer, main API)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from labeler import renderer, tile_index  # noqa: E402


def _write_chm(path: Path, width: int = 1300, height: int = 1300) -> None:
    data = np.random.rand(height, width).astype("float32") * 1.2
    data[0:50, 0:50] = -9999.0  # nodata
    transform = from_bounds(0, 0, width * 0.2, height * 0.2, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:3301",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def chm_dir(tmp_path: Path) -> Path:
    d = tmp_path / "chm"
    d.mkdir()
    _write_chm(d / "406455_2021_tava_chm_max_hag_20cm.tif")
    return d


@pytest.fixture
def mask_dir(tmp_path: Path) -> Path:
    d = tmp_path / "masks"
    d.mkdir()
    return d


# ---------------- tile_index -------------------------------------------------


def test_scan_produces_chips(chm_dir):
    chips = tile_index.scan(chm_dir, chip_size=640)
    # 1300 / 640 -> ceil 3 per axis => 9 chips
    assert len(chips) == 9
    assert all(c.chip_size == 640 for c in chips)
    assert {c.row_off for c in chips} == {0, 640, 1280}
    assert chips[0].grid == "406455"
    assert chips[0].year == "2021"
    assert chips[0].source == "tava"


def test_status_transitions(chm_dir, mask_dir):
    chips = tile_index.scan(chm_dir, chip_size=640)
    c = chips[0]
    assert tile_index.status(c.tile_id, mask_dir) == "unlabeled"

    (mask_dir / f"{c.tile_id}_mask.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
    assert tile_index.status(c.tile_id, mask_dir) == "labeled"

    (mask_dir / f"{c.tile_id}_meta.json").write_text(json.dumps({"review_flag": True}))
    assert tile_index.status(c.tile_id, mask_dir) == "needs-review"


def test_status_shared_label_keys_across_products(tmp_path: Path):
    chm = tmp_path / "chm"
    chm.mkdir()
    _write_chm(chm / "406455_2021_tava_harmonized_dem_last_raw_chm.tif")
    _write_chm(chm / "406455_2021_tava_harmonized_dem_last_gauss_chm.tif")

    chips = tile_index.scan(chm, chip_size=640)
    by_id = {c.tile_id: c for c in chips}
    raw_chip = next(
        c for c in chips if c.product == "harmonized_raw" and c.row_off == 0 and c.col_off == 0
    )
    gauss_chip = next(
        c for c in chips if c.product == "harmonized_gauss" and c.row_off == 0 and c.col_off == 0
    )
    assert raw_chip.label_key == gauss_chip.label_key

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / f"{raw_chip.label_key}_mask.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    assert tile_index.status(gauss_chip.tile_id, mask_dir, by_id) == "labeled"


def test_load_or_build_caches(chm_dir, tmp_path):
    cache = tmp_path / "idx.json"
    chips1 = tile_index.load_or_build(cache, chm_dir, chip_size=640)
    assert cache.exists()
    # Corrupt raster dir; load from cache must still succeed.
    (chm_dir / "406455_2021_tava_chm_max_hag_20cm.tif").unlink()
    chips2 = tile_index.load_or_build(cache, chm_dir, chip_size=640)
    assert len(chips1) == len(chips2)


def test_scan_recurses_harmonized_product_dirs(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "chm_raw").mkdir(parents=True)
    (root / "chm_gauss").mkdir(parents=True)
    _write_chm(root / "chm_raw" / "406455_2021_tava_harmonized_dem_last_raw_chm.tif")
    _write_chm(root / "chm_gauss" / "406455_2021_tava_harmonized_dem_last_gauss_chm.tif")

    chips = tile_index.scan(root, chip_size=640)
    assert len(chips) == 18
    products = {c.product for c in chips}
    assert products == {"harmonized_raw", "harmonized_gauss"}


def test_load_or_build_rebuilds_when_chm_dir_changes(tmp_path: Path):
    d1 = tmp_path / "d1"
    d1.mkdir()
    _write_chm(d1 / "406455_2021_tava_chm_max_hag_20cm.tif")

    d2 = tmp_path / "d2"
    d2.mkdir()
    _write_chm(d2 / "406455_2021_tava_chm_max_hag_20cm.tif")
    _write_chm(d2 / "406455_2022_tava_chm_max_hag_20cm.tif")

    cache = tmp_path / "idx.json"
    chips1 = tile_index.load_or_build(cache, d1, chip_size=640)
    chips2 = tile_index.load_or_build(cache, d2, chip_size=640)

    assert len(chips1) == 9
    assert len(chips2) == 18


def test_neighbors_layout(chm_dir):
    chips = tile_index.scan(chm_dir, chip_size=640)
    center = next(c for c in chips if c.row_off == 640 and c.col_off == 640)
    nb = tile_index.neighbors(center, chips)
    # Middle chip has 8 populated neighbors.
    assert sum(1 for v in nb.values() if v is not None) == 8


# ---------------- renderer ---------------------------------------------------


def test_read_chip_pads_with_nan(chm_dir):
    tif = next(chm_dir.glob("*.tif"))
    chip = renderer.read_chip(str(tif), row_off=1000, col_off=1000, chip_size=640)
    assert chip.shape == (640, 640)
    # Region past raster bounds must be NaN (raster is 1300px -> 300px of pad).
    assert np.isnan(chip[400:, 400:]).all()


def test_render_png_is_valid_image(chm_dir):
    tif = next(chm_dir.glob("*.tif"))
    chip = renderer.read_chip(str(tif), 0, 0, 640)
    png = renderer.render_png(chip)
    img = Image.open(io.BytesIO(png))
    assert img.size == (640, 640)
    assert img.mode == "RGBA"


def test_render_hillshade_handles_nan(chm_dir):
    tif = next(chm_dir.glob("*.tif"))
    chip = renderer.read_chip(str(tif), 0, 0, 640)
    png = renderer.render_hillshade(chip)
    img = Image.open(io.BytesIO(png))
    assert img.size == (640, 640)


# ---------------- main app ---------------------------------------------------


@pytest.fixture
def client(chm_dir, mask_dir, monkeypatch):
    monkeypatch.setenv("CHM_DIR", str(chm_dir))
    monkeypatch.setenv("MASK_DIR", str(mask_dir))
    monkeypatch.setenv("CHIP_SIZE", "640")
    # Reimport to pick up env vars.
    import importlib

    import labeler.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def test_api_list_tiles(client):
    r = client.get("/api/tiles")
    assert r.status_code == 200
    tiles = r.json()
    assert len(tiles) == 9
    assert {"tile_id", "raster_path", "row_off", "col_off", "status"} <= set(tiles[0])


def test_api_rasters(client):
    r = client.get("/api/rasters")
    assert r.status_code == 200
    rs = r.json()
    assert len(rs) == 1
    assert rs[0]["chip_count"] == 9
    assert rs[0]["labeled_count"] == 0


def test_api_catalog(client):
    r = client.get("/api/catalog")
    assert r.status_code == 200
    cat = r.json()
    assert "grids" in cat
    assert "years" in cat
    assert "products" in cat
    assert "406455" in cat["grids"]
    assert "2021" in cat["years"]


def test_api_chm_png(client):
    tiles = client.get("/api/tiles").json()
    tid = tiles[0]["tile_id"]
    r = client.get(f"/api/tile/{tid}/chm")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")


def test_api_products(client):
    tiles = client.get("/api/tiles").json()
    tid = tiles[0]["tile_id"]
    r = client.get(f"/api/tile/{tid}/products")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert rows
    assert "tile_id" in rows[0]
    assert "product" in rows[0]


def test_api_mask_roundtrip(client, mask_dir):
    tiles = client.get("/api/tiles").json()
    tid = tiles[0]["tile_id"]

    # 404 when not yet saved
    assert client.get(f"/api/tile/{tid}/mask").status_code == 404

    # Build a 640×640 single-channel PNG payload
    buf = io.BytesIO()
    Image.new("L", (640, 640), color=255).save(buf, format="PNG")
    body = buf.getvalue()

    r = client.post(f"/api/tile/{tid}/mask", content=body, headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    label_key = r.json().get("label_key", tid)
    assert (mask_dir / f"{label_key}_mask.png").exists()

    r2 = client.get(f"/api/tile/{tid}/mask")
    assert r2.status_code == 200
    assert r2.content.startswith(b"\x89PNG")


def test_api_mask_rejects_non_png(client):
    tiles = client.get("/api/tiles").json()
    tid = tiles[0]["tile_id"]
    r = client.post(f"/api/tile/{tid}/mask", content=b"not a png", headers={"Content-Type": "image/png"})
    assert r.status_code == 400


def test_api_meta_roundtrip(client):
    tiles = client.get("/api/tiles").json()
    tid = tiles[0]["tile_id"]
    r = client.post(f"/api/tile/{tid}/meta", json={"review_flag": True, "note": "log cluster"})
    assert r.status_code == 200

    r2 = client.get(f"/api/tile/{tid}/meta")
    payload = r2.json()
    meta = payload["meta"]
    assert meta["review_flag"] is True
    assert meta["note"] == "log cluster"
    assert "updated_at" in meta
    assert "label_key" in payload
    assert payload["label_key"] in payload["label_keys"]


def test_unknown_tile_404(client):
    assert client.get("/api/tile/does-not-exist/chm").status_code == 404
    assert client.get("/api/tile/does-not-exist/meta").status_code == 404
