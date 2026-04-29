"""Tests for scripts/spatial_split_experiments_v4 — stride-aware V4 split.

Run with:
    pytest tests/test_spatial_split_experiments_v4.py -v
    pytest tests/test_spatial_split_experiments_v4.py -v -m "not slow"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.spatial_split_experiments_v4._split_v4 import (
    _chip_in_buffer,
    _chip_pos,
    _block_coord,
    _get_place_key,
    _infer_stride,
    _infer_chunk_size,
    _total_chips_per_dim,
    write_intra_raster_chip_split_v4,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cand(raster: str, row_off: int, col_off: int, label: str = "cdw") -> dict[str, Any]:
    return {
        "key": (raster, row_off, col_off),
        "raster": raster,
        "label": label,
        "source": "manual",
        "reason": "manual_or_reviewed",
        "chunk_size": 128,
    }


def _stride64_candidates(raster: str, n: int = 100, cdw_frac: float = 0.4) -> list[dict[str, Any]]:
    """Generate chips with stride=64 (50% overlap) spread across a 5000px raster."""
    n_cdw = round(n * cdw_frac)
    out = []
    for i in range(n):
        row_off = (i % 77) * 64    # stride=64, up to chip_pos=76
        col_off = (i // 77) * 64
        label = "cdw" if i < n_cdw else "no_cdw"
        out.append(_cand(raster, row_off, col_off, label))
    return out


def _stride64_grid(raster: str, n_per_block: int = 8) -> list[dict[str, Any]]:
    """Generate chips covering all 9 blocks with stride=64.

    chips_per_block = (77 // 3) = 25, block starts at chip_pos 0, 25, 50.
    row_off at block start = 0, 25*64=1600, 50*64=3200.
    """
    block_starts_chip = [0, 25, 50]
    out = []
    for br in range(3):
        for bc in range(3):
            for i in range(n_per_block):
                row_off = (block_starts_chip[br] + i) * 64
                col_off = (block_starts_chip[bc] + i) * 64
                label = "cdw" if i % 2 == 0 else "no_cdw"
                out.append(_cand(raster, row_off, col_off, label))
    return out


RASTER_A = "436646_2020_madal_chm_max_hag_20cm.tif"
RASTER_A_2018 = "436646_2018_madal_chm_max_hag_20cm.tif"
RASTER_B = "436647_2020_madal_chm_max_hag_20cm.tif"


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestStrideInference:
    def test_stride64_detected(self) -> None:
        # row_offs at multiples of 64 (including odd multiples like 64, 192 ≡ 64 mod 128)
        cands = [_cand("r.tif", r, 0) for r in [0, 64, 128, 192, 256, 320]]
        assert _infer_stride(cands) == 64

    def test_stride128_detected(self) -> None:
        cands = [_cand("r.tif", r, 0) for r in [0, 128, 256, 384, 512]]
        assert _infer_stride(cands) == 128

    def test_empty_returns_default(self) -> None:
        assert _infer_stride([]) == 64

    def test_total_chips_per_dim_stride64(self) -> None:
        # range(0, 5000-128+1, 64) = range(0, 4873, 64) → 0,64,...,4864 = 77 values
        assert _total_chips_per_dim(5000, 128, 64) == 77

    def test_total_chips_per_dim_stride128(self) -> None:
        # range(0, 4873, 128) → 0,128,...,4864 → but 4864/128=38, so 39 values
        assert _total_chips_per_dim(5000, 128, 128) == 39


class TestBufferGapFormula:
    """Verify gap = (buffer_strides+1)×stride − chunk_size."""

    def test_buffer5_gives_51m_gap(self) -> None:
        gap_px = (5 + 1) * 64 - 128
        assert gap_px == 256
        gap_m = gap_px * 0.2
        assert gap_m == pytest.approx(51.2)

    def test_buffer5_exceeds_50m_threshold(self) -> None:
        gap_px = (5 + 1) * 64 - 128
        assert gap_px >= 250  # 50 m at 0.2 m/px

    def test_buffer4_below_50m(self) -> None:
        gap_px = (4 + 1) * 64 - 128
        assert gap_px == 192
        assert gap_px < 250  # 38.4 m < 50 m

    def test_buffer7_gives_76m_gap(self) -> None:
        gap_px = (7 + 1) * 64 - 128
        assert gap_px == 384
        assert gap_px * 0.2 == pytest.approx(76.8)


class TestPlaceKey:
    def test_strips_year(self) -> None:
        pk = _get_place_key("436646_2020_madal_chm_max_hag_20cm")
        assert pk == "436646_madal"

    def test_same_for_different_years(self) -> None:
        assert _get_place_key("436646_2018_madal_chm_max_hag_20cm") == \
               _get_place_key("436646_2020_madal_chm_max_hag_20cm")

    def test_fallback_for_unknown_pattern(self) -> None:
        pk = _get_place_key("some_weird_raster_name")
        assert pk == "some_weird_raster_name"


class TestChipPosStride:
    def test_row0_col0(self) -> None:
        assert _chip_pos(0, 0, 64) == (0, 0)

    def test_row64_col0(self) -> None:
        # stride=64: row_off=64 → chip_row=1 (NOT 0 like in V3's chunk-based coords)
        assert _chip_pos(64, 0, 64) == (1, 0)

    def test_row128(self) -> None:
        assert _chip_pos(128, 0, 64) == (2, 0)

    def test_large_offset(self) -> None:
        assert _chip_pos(1600, 3200, 64) == (25, 50)

    def test_buffer_stride5_correct_chip(self) -> None:
        """Test chip at chip_pos 30 is NOT in buffer when test chip is at 24, buffer=5."""
        test_pos = {(24, 0)}
        # Distance from (30,0) to (24,0) = 6 > 5 → NOT in buffer
        assert not _chip_in_buffer(30, 0, test_pos, buffer_strides=5)

    def test_buffer_stride5_boundary_chip(self) -> None:
        """Chip at chip_pos 29 IS in buffer (distance=5=buffer_strides)."""
        test_pos = {(24, 0)}
        assert _chip_in_buffer(29, 0, test_pos, buffer_strides=5)


# ---------------------------------------------------------------------------
# Integration tests: write_intra_raster_chip_split_v4
# ---------------------------------------------------------------------------

class TestWriteV4Split:
    def test_rows_add_up(self, tmp_path: Path) -> None:
        candidates = _stride64_candidates(RASTER_A, 100)
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026,
        )
        assert meta["train_rows"] + meta["test_rows"] + meta["buffer_rows"] == meta["total_rows"]

    def test_rows_add_up_multi_raster(self, tmp_path: Path) -> None:
        candidates = _stride64_candidates(RASTER_A, 100) + _stride64_candidates(RASTER_B, 80)
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026,
        )
        total = meta["total_rows"]
        assert meta["train_rows"] + meta["test_rows"] + meta["buffer_rows"] == total

    def test_stride_detected_as_64(self, tmp_path: Path) -> None:
        candidates = _stride64_candidates(RASTER_A, 100)
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026,
        )
        assert meta["stride"] == 64

    def test_buffer_gap_51m(self, tmp_path: Path) -> None:
        """Default buffer_strides=5 gives gap=51.2m."""
        candidates = _stride64_candidates(RASTER_A, 100)
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026, buffer_strides=5,
        )
        assert meta["buffer_gap_px"] == 256
        assert meta["buffer_gap_m"] == pytest.approx(51.2)

    def test_deterministic(self, tmp_path: Path) -> None:
        candidates = _stride64_candidates(RASTER_A, 100)
        out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
        write_intra_raster_chip_split_v4(candidates, out1, seed=42)
        write_intra_raster_chip_split_v4(candidates, out2, seed=42)
        assert json.loads(out1.read_text())["keys"] == json.loads(out2.read_text())["keys"]

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        # Use grid data with many chips per block so different seeds can pick different blocks
        rasters = [
            "436646_2020_madal_chm_max_hag_20cm.tif",
            "436647_2020_madal_chm_max_hag_20cm.tif",
            "437646_2020_madal_chm_max_hag_20cm.tif",
            "580535_2021_laat_chm_max_hag_20cm.tif",
        ]
        candidates = []
        for r in rasters:
            candidates += _stride64_grid(r, n_per_block=10)
        splits = set()
        for seed in range(2026, 2036):
            out = tmp_path / f"s{seed}.json"
            write_intra_raster_chip_split_v4(candidates, out, seed=seed)
            splits.add(tuple(tuple(k) for k in json.loads(out.read_text())["keys"]))
        assert len(splits) > 1, "All 10 seeds produced identical splits — year-safe seeding too rigid"

    def test_year_leakage_prevention(self, tmp_path: Path) -> None:
        """Same physical location across years must get the SAME test/train roles."""
        # Two years of the same place (same tile+site, different year in raster name)
        cands_2018 = _stride64_grid(RASTER_A_2018, n_per_block=6)
        cands_2020 = _stride64_grid(RASTER_A, n_per_block=6)
        candidates = cands_2018 + cands_2020

        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026, n_blocks=3, buffer_strides=5,
        )

        payload = json.loads((tmp_path / "s.json").read_text())
        test_keys = {(k[0], k[1], k[2]) for k in payload["keys"]}

        # Find which block is test for 2018 — the same block should be test for 2020
        test_2018 = {(r, ro, co) for r, ro, co in test_keys if "2018" in r}
        test_2020 = {(r, ro, co) for r, ro, co in test_keys if "2020" in r}

        # Both years should have test chips (year-safe means same blocks → both years have test)
        assert len(test_2018) > 0, "2018 raster has no test chips"
        assert len(test_2020) > 0, "2020 raster has no test chips"

        # The (row_off, col_off) positions in test should be the SAME for both years
        pos_2018 = {(ro, co) for (_, ro, co) in test_2018}
        pos_2020 = {(ro, co) for (_, ro, co) in test_2020}
        assert pos_2018 == pos_2020, (
            f"Year leakage: 2018 test positions {pos_2018} != 2020 test positions {pos_2020}"
        )

    def test_buffer_fraction_below_20pct(self, tmp_path: Path) -> None:
        """V4 buffer% should be well below V2's ~63%, target < 20%."""
        cands = []
        for tile in ["436646", "436647", "437646"]:
            cands += _stride64_candidates(f"{tile}_2020_madal_chm_max_hag_20cm.tif", 80)
        meta = write_intra_raster_chip_split_v4(
            cands, tmp_path / "s.json", seed=2026, n_blocks=3, buffer_strides=5,
        )
        assert meta["buffer_pct"] < 20.0, f"buffer={meta['buffer_pct']:.1f}% too high"

    def test_geojson_9_blocks_per_raster(self, tmp_path: Path) -> None:
        candidates = _stride64_grid(RASTER_A, n_per_block=8)
        out_j, out_g = tmp_path / "s.json", tmp_path / "s.geojson"
        write_intra_raster_chip_split_v4(
            candidates, out_j, seed=2026, n_blocks=3, output_geojson=out_g
        )
        fc = json.loads(out_g.read_text())
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 9  # 3×3 blocks × 1 raster

    def test_geojson_crs_epsg3301(self, tmp_path: Path) -> None:
        candidates = _stride64_grid(RASTER_A, n_per_block=5)
        out_j, out_g = tmp_path / "s.json", tmp_path / "s.geojson"
        write_intra_raster_chip_split_v4(candidates, out_j, seed=2026, output_geojson=out_g)
        fc = json.loads(out_g.read_text())
        assert "3301" in str(fc.get("crs", ""))

    def test_geojson_block_coords_epsg3301(self, tmp_path: Path) -> None:
        """Blocks should tile the full 1 km × 1 km raster in L-EST97.

        Tile 436646: grid_x=436, grid_y=646.
        northing_top = 6_000_000 + 437×1000 = 6_437_000
        easting_left = 646×1000 = 646_000
        """
        candidates = _stride64_grid("436646_2020_madal_chm_max_hag_20cm.tif", n_per_block=8)
        out_j, out_g = tmp_path / "s.json", tmp_path / "s.geojson"
        write_intra_raster_chip_split_v4(candidates, out_j, seed=2026, n_blocks=3, output_geojson=out_g)
        fc = json.loads(out_g.read_text())
        assert len(fc["features"]) == 9

        all_xs = [c[0] for feat in fc["features"] for c in feat["geometry"]["coordinates"][0]]
        all_ys = [c[1] for feat in fc["features"] for c in feat["geometry"]["coordinates"][0]]

        assert min(all_xs) == pytest.approx(646_000.0)
        assert max(all_xs) == pytest.approx(647_000.0)
        assert min(all_ys) == pytest.approx(6_436_000.0)
        assert max(all_ys) == pytest.approx(6_437_000.0)

    def test_geojson_place_key_property(self, tmp_path: Path) -> None:
        """GeoJSON features should include year-agnostic place_key."""
        candidates = _stride64_grid(RASTER_A, n_per_block=5)
        out_j, out_g = tmp_path / "s.json", tmp_path / "s.geojson"
        write_intra_raster_chip_split_v4(candidates, out_j, seed=2026, output_geojson=out_g)
        fc = json.loads(out_g.read_text())
        for feat in fc["features"]:
            assert feat["properties"]["place_key"] == "436646_madal"

    def test_output_metadata(self, tmp_path: Path) -> None:
        candidates = _stride64_candidates(RASTER_A, 80)
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026, buffer_strides=5,
        )
        assert meta["split_version"] == "v4_stride_aware_year_safe"
        assert meta["stride"] == 64
        assert meta["buffer_strides"] == 5
        assert meta["buffer_gap_px"] == 256
        assert meta["buffer_gap_m"] == pytest.approx(51.2)

    def test_empty_candidates(self, tmp_path: Path) -> None:
        meta = write_intra_raster_chip_split_v4([], tmp_path / "s.json", seed=2026)
        assert meta["total_rows"] == 0


# ---------------------------------------------------------------------------
# Slow / integration tests (require real label data on disk)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestV4WithRealData:
    LABELS_DIR = Path("output/model_search_v4/prepared/labels_curated_v4")

    @pytest.fixture(autouse=True)
    def _require_labels(self) -> None:
        if not self.LABELS_DIR.exists():
            pytest.skip("Real label data not available")

    def _load(self) -> list[dict[str, Any]]:
        import csv
        candidates = []
        for f in sorted(self.LABELS_DIR.glob("*_labels.csv")):
            with open(f, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    source = str(row.get("source", ""))
                    reason = "manual_or_reviewed" if (
                        "manual" in source.lower() or source == "auto_reviewed"
                    ) else "threshold_gate"
                    candidates.append({
                        "key": (str(row["raster"]), int(row["row_off"]), int(row["col_off"])),
                        "raster": str(row["raster"]),
                        "label": str(row.get("label", "")),
                        "source": source,
                        "reason": reason,
                        "chunk_size": int(row.get("chunk_size", 128)),
                    })
        return candidates

    def test_stride_detected_as_64(self) -> None:
        candidates = self._load()
        assert _infer_stride(candidates) == 64

    def test_buffer_pct_below_20(self, tmp_path: Path) -> None:
        """V4 buffer must be < 20% (vs ~63% for V2 mapsheet-level)."""
        candidates = self._load()
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026,
            n_blocks=3, buffer_strides=5,
        )
        assert meta["buffer_pct"] < 20.0, f"buffer={meta['buffer_pct']:.1f}%"
        assert meta["train_rows"] > 8000, f"train_rows={meta['train_rows']}"

    def test_rows_add_up_real(self, tmp_path: Path) -> None:
        candidates = self._load()
        meta = write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026,
        )
        total = meta["total_rows"]
        assert meta["train_rows"] + meta["test_rows"] + meta["buffer_rows"] == total

    def test_year_leakage_prevention_real(self, tmp_path: Path) -> None:
        """For any tile with multiple years, test chip positions must be identical."""
        candidates = self._load()
        out = tmp_path / "s.json"
        write_intra_raster_chip_split_v4(candidates, out, seed=2026, n_blocks=3, buffer_strides=5)
        payload = json.loads(out.read_text())
        test_keys = payload["keys"]

        from collections import defaultdict
        place_test_pos: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for r, ro, co in test_keys:
            pk = _get_place_key(r.replace(".tif", ""))
            place_test_pos[pk].add((ro, co))

        multi_year_places = []
        rasters_by_place: dict[str, set[str]] = defaultdict(set)
        for r, ro, co in test_keys:
            pk = _get_place_key(r.replace(".tif", ""))
            rasters_by_place[pk].add(r)
        multi_year_places = [pk for pk, rs in rasters_by_place.items() if len(rs) > 1]

        for pk in multi_year_places:
            rasters = rasters_by_place[pk]
            pos_sets = [
                {(ro, co) for (r, ro, co) in test_keys if _get_place_key(r.replace(".tif", "")) == pk and r == rstr}
                for rstr in rasters
            ]
            if all(len(p) > 0 for p in pos_sets):
                first = pos_sets[0]
                for p in pos_sets[1:]:
                    assert first == p, f"Year leakage in {pk}: positions differ across years"

    def test_geojson_crs_real(self, tmp_path: Path) -> None:
        candidates = self._load()
        out_g = tmp_path / "s.geojson"
        write_intra_raster_chip_split_v4(
            candidates, tmp_path / "s.json", seed=2026, output_geojson=out_g,
        )
        fc = json.loads(out_g.read_text())
        assert "3301" in str(fc.get("crs", ""))
