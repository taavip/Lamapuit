"""Unit tests for the pure-logic modules under ``scripts/model_search_v4``.

These tests cover the parts of V4 that do not need a GPU or the base search
script:

- ``_labels``: confidence gate, dedup by provenance priority, raster parsing.
- ``_splits``: spatial block split invariants — zero place overlap,
  same-place-multi-year grouping, neighbour buffer exclusion.
- ``_ranking``: LCB ordering, composite score, overwhelming-margin gate.
- ``_features``: output shape, finiteness.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_V4_DIR = Path(__file__).resolve().parents[1] / "scripts" / "model_search_v4"
if str(_V4_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_V4_DIR.parent))

from model_search_v4._audit import breakdown_test_metrics
from model_search_v4._features import N_FEATURES_PER_CHANNEL, tile_feature_matrix
from model_search_v4._labels import (
    LABEL_HEADERS,
    PRIORITY_AUTO,
    PRIORITY_MANUAL,
    PRIORITY_THRESHOLD_GATE,
    include_drop_row,
    is_manual_source,
    parse_raster_identity,
    row_priority,
    write_curated_labels_drop_only,
)
from model_search_v4._ranking import (
    composite_mode_score,
    select_best_mode,
    top_models_from_v3_lcb,
)
from model_search_v4._splits import write_spatial_block_test_split


# ---------- _labels ----------


class TestLabels:
    def test_is_manual_source(self):
        assert is_manual_source("manual")
        assert is_manual_source("auto_reviewed")
        assert is_manual_source("MANUAL_REVIEWED")
        assert not is_manual_source("auto")
        assert not is_manual_source("")
        assert not is_manual_source("auto_threshold_gate_v4")

    def test_row_priority_ordering(self):
        assert row_priority("manual", "manual_or_reviewed") == PRIORITY_MANUAL
        assert row_priority("auto", "threshold_gate") == PRIORITY_THRESHOLD_GATE
        assert row_priority("auto", "below_threshold") == PRIORITY_AUTO
        assert PRIORITY_MANUAL > PRIORITY_THRESHOLD_GATE > PRIORITY_AUTO

    def test_parse_raster_identity_canonical(self):
        ident = parse_raster_identity("123456_2022_pollumaa_chm_max_hag_20cm.tif")
        assert ident["tile"] == "123456"
        assert ident["year"] == "2022"
        assert ident["site"] == "pollumaa"
        assert ident["place_key"] == "123456_pollumaa"
        assert ident["grid_x"] == 123
        assert ident["grid_y"] == 456

    def test_parse_raster_identity_same_place_different_year(self):
        a = parse_raster_identity("654321_2018_lahemaa_chm_max_hag_20cm.tif")
        b = parse_raster_identity("654321_2022_lahemaa_chm_max_hag_20cm.tif")
        assert a["place_key"] == b["place_key"]
        assert a["year"] != b["year"]

    def test_parse_raster_identity_unknown_format(self):
        ident = parse_raster_identity("weird_name.tif")
        assert ident["year"] == "unknown"
        assert ident["site"] == "unknown"
        assert ident["grid_x"] is None
        assert ident["place_key"] == "weird_name"

    def test_include_drop_row_manual_always_kept(self):
        row = {"source": "manual", "label": "cdw", "model_prob": ""}
        ok, reason = include_drop_row(row, t_high=0.9, t_low=0.1)
        assert ok and reason == "manual_or_reviewed"

    def test_include_drop_row_threshold_gate(self):
        hi = {"source": "auto", "label": "cdw", "model_prob": "0.95"}
        assert include_drop_row(hi, t_high=0.9, t_low=0.1) == (True, "threshold_gate")

        lo = {"source": "auto", "label": "no_cdw", "model_prob": "0.02"}
        assert include_drop_row(lo, t_high=0.9, t_low=0.1) == (True, "threshold_gate")

    def test_include_drop_row_rejections(self):
        noprob = {"source": "auto", "label": "cdw", "model_prob": ""}
        assert include_drop_row(noprob, 0.9, 0.1) == (False, "no_prob")

        mid = {"source": "auto", "label": "cdw", "model_prob": "0.5"}
        assert include_drop_row(mid, 0.9, 0.1) == (False, "below_threshold")

    def test_dedup_prefers_manual_over_auto(self, tmp_path: Path):
        drop_dir = tmp_path / "drop"
        drop_dir.mkdir()
        curated = tmp_path / "curated"
        csv_path = drop_dir / "tileA_labels.csv"
        with csv_path.open("w", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=LABEL_HEADERS)
            wr.writeheader()
            # Same (raster,row,col) appearing twice: auto first, manual second
            wr.writerow({
                "raster": "tileA.tif", "row_off": "0", "col_off": "0",
                "chunk_size": "128", "label": "cdw", "source": "auto",
                "annotator": "", "model_name": "m3", "model_prob": "0.95",
                "timestamp": "2024-01-01T00:00:00",
            })
            wr.writerow({
                "raster": "tileA.tif", "row_off": "0", "col_off": "0",
                "chunk_size": "128", "label": "no_cdw", "source": "manual",
                "annotator": "human", "model_name": "", "model_prob": "",
                "timestamp": "2023-01-01T00:00:00",
            })

        stats, candidates = write_curated_labels_drop_only(
            drop_dir, curated, t_high=0.9, t_low=0.1
        )
        assert len(candidates) == 1
        # Manual must win regardless of timestamp because priority is higher.
        # Read back the written CSV to verify.
        out_csv = curated / "tileA_labels.csv"
        rows = list(csv.DictReader(out_csv.open()))
        assert len(rows) == 1
        assert rows[0]["source"].lower() == "manual"
        assert rows[0]["label"] == "no_cdw"

    def test_dedup_threshold_gate_renamed(self, tmp_path: Path):
        drop_dir = tmp_path / "drop"
        drop_dir.mkdir()
        curated = tmp_path / "curated"
        csv_path = drop_dir / "tileB_labels.csv"
        with csv_path.open("w", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=LABEL_HEADERS)
            wr.writeheader()
            wr.writerow({
                "raster": "tileB.tif", "row_off": "5", "col_off": "7",
                "chunk_size": "128", "label": "cdw", "source": "auto",
                "annotator": "", "model_name": "m3", "model_prob": "0.99",
                "timestamp": "2024-01-01T00:00:00",
            })

        write_curated_labels_drop_only(drop_dir, curated, t_high=0.9, t_low=0.1)
        rows = list(csv.DictReader((curated / "tileB_labels.csv").open()))
        assert rows[0]["source"] == "auto_threshold_gate_v4"


# ---------- _splits ----------


def _make_candidate(raster: str, row_off: int, col_off: int, label: str = "cdw", reason: str = "manual_or_reviewed"):
    return {
        "key": (raster, row_off, col_off),
        "label": label,
        "raster": raster,
        "source": "manual" if reason == "manual_or_reviewed" else "auto_threshold_gate_v4",
        "reason": reason,
    }


class TestSplits:
    def test_place_overlap_is_zero(self, tmp_path: Path):
        # 10 tiles, 5 places, 2 years each.
        candidates = []
        for tile_grid in range(1, 11):
            tile = f"{tile_grid * 1000:06d}"  # distinct grid slots
            site = f"site{tile_grid}"
            for year in ("2018", "2022"):
                raster = f"{tile}_{year}_{site}_chm_max_hag_20cm.tif"
                for i in range(20):
                    candidates.append(_make_candidate(raster, i, 0))

        out = tmp_path / "split.json"
        meta = write_spatial_block_test_split(
            candidates,
            output_test_split=out,
            seed=42,
            test_fraction=0.2,
            split_block_size_places=1,
            neighbor_buffer_blocks=0,
        )
        assert meta["place_overlap_train_vs_test"] == 0
        assert meta["n_places_total"] == 10

    def test_same_place_multi_year_stays_together(self, tmp_path: Path):
        # One place, two years. All rows must end up on the same side.
        tile = "100200"
        site = "siteX"
        candidates = []
        for year in ("2018", "2022"):
            raster = f"{tile}_{year}_{site}_chm_max_hag_20cm.tif"
            for i in range(50):
                candidates.append(_make_candidate(raster, i, 0))

        out = tmp_path / "split.json"
        meta = write_spatial_block_test_split(
            candidates,
            output_test_split=out,
            seed=0,
            test_fraction=0.5,
            split_block_size_places=1,
            neighbor_buffer_blocks=0,
        )
        payload = json.loads(out.read_text())
        test_rasters = {k[0] for k in payload["keys"]}
        # Either both years in test, or neither — never a mix.
        rasters_2018 = f"{tile}_2018_{site}_chm_max_hag_20cm.tif"
        rasters_2022 = f"{tile}_2022_{site}_chm_max_hag_20cm.tif"
        assert (rasters_2018 in test_rasters) == (rasters_2022 in test_rasters)
        assert meta["places_with_multi_year"] == 1

    def test_neighbor_buffer_excludes_adjacent(self, tmp_path: Path):
        # A 3x3 grid of single-place blocks. With buffer=1 and one test block,
        # surrounding 8 blocks must land in buffer (not train).
        candidates = []
        for gx in range(3):
            for gy in range(3):
                tile = f"{gx:03d}{gy:03d}00"  # 7 digit? ensure 6+
                # we need parse_raster_identity to recover grid coords
                tile = f"{gx:03d}{gy:03d}"  # 6 digits
                site = f"p_{gx}_{gy}"
                raster = f"{tile}_2020_{site}_chm_max_hag_20cm.tif"
                for i in range(10):
                    candidates.append(_make_candidate(raster, i, 0))

        out = tmp_path / "split.json"
        meta = write_spatial_block_test_split(
            candidates,
            output_test_split=out,
            seed=7,
            test_fraction=1.0 / 9.0,  # aim for one block
            split_block_size_places=1,
            neighbor_buffer_blocks=1,
        )
        # With a 3x3 grid centered, one test block should produce some buffer
        # blocks and non-empty train.
        assert meta["n_blocks_test"] >= 1
        assert meta["n_blocks_buffer"] >= 1
        assert meta["n_places_train"] >= 1
        # Sum invariant.
        assert (
            meta["n_places_train"] + meta["n_places_test"] + meta["n_places_buffer"]
            == meta["n_places_total"]
        )

    def test_empty_candidates(self, tmp_path: Path):
        out = tmp_path / "empty.json"
        meta = write_spatial_block_test_split(
            [], output_test_split=out, seed=0, test_fraction=0.2,
            split_block_size_places=1, neighbor_buffer_blocks=1,
        )
        assert meta["n_places_total"] == 0
        assert meta["place_overlap_train_vs_test"] == 0


# ---------- _ranking ----------


class TestRanking:
    def test_lcb_picks_low_variance_model(self, tmp_path: Path):
        df = pd.DataFrame({
            "model_name": ["convnext_small", "efficientnet_b2", "maxvit_small", "deep_cnn_attn_dropout_tuned"],
            "mean_cv_f1": [0.935, 0.930, 0.925, 0.949],
            "std_cv_f1": [0.002, 0.005, 0.010, 0.067],
        })
        csv_path = tmp_path / "experiment_summary.csv"
        df.to_csv(csv_path, index=False)

        names, audit = top_models_from_v3_lcb(csv_path, n_models=3, lcb_k=1.0)
        # Highest-variance model with mean 0.949 has LCB 0.882 — should lose.
        assert "deep_cnn_attn_dropout_tuned" not in names[:2]
        assert names[0] == "convnext_small"
        assert len(audit) >= 3

    def test_lcb_fallback_on_missing_csv(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.csv"
        names, audit = top_models_from_v3_lcb(missing, n_models=3)
        assert len(names) == 3
        assert audit == []

    def test_composite_mode_score_weighting(self):
        assert composite_mode_score(0.8, 0.6, deep_weight=0.7) == pytest.approx(0.74)
        assert composite_mode_score(0.8, None, deep_weight=0.7) == 0.8
        assert composite_mode_score(None, 0.6, deep_weight=0.7) == 0.6
        assert np.isnan(composite_mode_score(None, None, deep_weight=0.7))

    def test_select_best_mode_overwhelming(self):
        df = pd.DataFrame({
            "mode": ["original", "raw", "gauss"],
            "composite_score": [0.80, 0.72, 0.70],
        })
        best, over, diag = select_best_mode(df, overwhelming_margin=0.05)
        assert best == "original"
        assert over is True
        assert diag["runner_up_mode"] == "raw"
        assert diag["margin"] == pytest.approx(0.08)

    def test_select_best_mode_not_overwhelming(self):
        df = pd.DataFrame({
            "mode": ["original", "raw"],
            "composite_score": [0.80, 0.79],
        })
        best, over, diag = select_best_mode(df, overwhelming_margin=0.05)
        assert best == "original"
        assert over is False


# ---------- _features ----------


class TestFeatures:
    def test_shape_single_channel(self):
        x = np.random.rand(4, 1, 32, 32).astype(np.float32)
        feats = tile_feature_matrix(x)
        assert feats.shape == (4, 1 * N_FEATURES_PER_CHANNEL)
        assert np.isfinite(feats).all()

    def test_shape_three_channel(self):
        x = np.random.rand(3, 3, 16, 16).astype(np.float32)
        feats = tile_feature_matrix(x)
        assert feats.shape == (3, 3 * N_FEATURES_PER_CHANNEL)
        assert np.isfinite(feats).all()

    def test_constant_tile_has_zero_variance(self):
        x = np.full((1, 1, 8, 8), 2.5, dtype=np.float32)
        feats = tile_feature_matrix(x)
        # std should be 0, gradients should be 0
        assert feats[0, 1] == pytest.approx(0.0)  # std

    def test_features_differ_with_content(self):
        a = np.zeros((1, 1, 8, 8), dtype=np.float32)
        b = np.zeros((1, 1, 8, 8), dtype=np.float32)
        b[0, 0, 2:6, 2:6] = 1.0  # a square "object"
        fa = tile_feature_matrix(a)
        fb = tile_feature_matrix(b)
        assert not np.allclose(fa, fb)


# ---------- _audit ----------


class TestAudit:
    def test_manual_only_breakdown(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        probs = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.4])
        meta = [
            {"source": "manual", "raster": "100200_2020_a_chm_max_hag_20cm.tif"},
            {"source": "manual", "raster": "100200_2020_a_chm_max_hag_20cm.tif"},
            {"source": "auto_threshold_gate_v4", "raster": "100201_2020_b_chm_max_hag_20cm.tif"},
            {"source": "auto_threshold_gate_v4", "raster": "100201_2020_b_chm_max_hag_20cm.tif"},
            {"source": "manual", "raster": "100200_2022_a_chm_max_hag_20cm.tif"},
            {"source": "auto_threshold_gate_v4", "raster": "100201_2022_b_chm_max_hag_20cm.tif"},
        ]
        out = breakdown_test_metrics(y, probs, threshold=0.5, meta_test=meta)
        assert out["manual_only"]["n"] == 3
        assert out["threshold_gate_only"]["n"] == 3
        assert set(out["per_year"].keys()) == {"2020", "2022"}
        assert out["per_place_summary"]["n_places"] == 2
