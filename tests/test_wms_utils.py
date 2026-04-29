from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cdw_detect.wms_utils import (
    WMS_CRS,
    build_wms_getmap_params,
    build_wms_layer_name,
    fetch_wms_for_bbox,
    map_campaign_token,
    parse_chm_filename,
)


def test_parse_chm_filename_valid() -> None:
    assert parse_chm_filename("436647_2024_madal_chm_max_hag_20cm.tif") == (2024, "madal")
    assert parse_chm_filename("464663_2019_tava_chm_max_hag_20cm.tif") == (2019, "tava")


def test_parse_chm_filename_invalid() -> None:
    assert parse_chm_filename("invalid_name.tif") is None
    assert parse_chm_filename("464663_xxxx_tava_chm_max_hag_20cm.tif") is None


def test_map_campaign_token_strict() -> None:
    assert map_campaign_token("madal") == "asulad"
    assert map_campaign_token("tava") == "aeropildistamine"
    assert map_campaign_token("mets") is None


def test_build_wms_layer_name() -> None:
    assert build_wms_layer_name("436647_2024_madal_chm_max_hag_20cm.tif") == "of2024asulad"
    assert (
        build_wms_layer_name("464663_2019_tava_chm_max_hag_20cm.tif")
        == "of2019aeropildistamine"
    )
    assert build_wms_layer_name("464663_2019_mets_chm_max_hag_20cm.tif") is None


def test_build_wms_getmap_params() -> None:
    params = build_wms_getmap_params(
        layer="of2024asulad",
        bbox=(663000.0, 6464000.0, 663512.5, 6464512.5),
        width=640,
        height=256,
    )
    assert params["service"] == "WMS"
    assert params["request"] == "GetMap"
    assert params["layers"] == "of2024asulad"
    assert params["srs"] == WMS_CRS
    assert params["width"] == "640"
    assert params["height"] == "256"
    assert params["bbox"] == "663000.000,6464000.000,663512.500,6464512.500"


def test_fetch_wms_for_bbox_success(monkeypatch) -> None:
    img = Image.fromarray(np.full((6, 8, 3), 127, dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    payload = buf.getvalue()

    class _Resp:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self) -> None:
            return

    def _fake_get(url, params, timeout):
        assert "bbox" in params
        assert params["srs"] == "EPSG:3301"
        return _Resp(payload)

    monkeypatch.setattr("cdw_detect.wms_utils.requests.get", _fake_get)

    out = fetch_wms_for_bbox(
        layer="of2024asulad",
        bbox=(1.0, 2.0, 3.0, 4.0),
        width=8,
        height=6,
        timeout=1.0,
    )

    assert out is not None
    assert out.shape == (6, 8, 3)
    assert out.dtype == np.uint8


def test_fetch_wms_for_bbox_failure(monkeypatch) -> None:
    def _fake_get(url, params, timeout):
        raise RuntimeError("network down")

    monkeypatch.setattr("cdw_detect.wms_utils.requests.get", _fake_get)

    out = fetch_wms_for_bbox(
        layer="of2024asulad",
        bbox=(1.0, 2.0, 3.0, 4.0),
        width=8,
        height=6,
    )
    assert out is None
