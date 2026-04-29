"""Focused tests for PartialConv CWD pipeline split and loss behaviors."""

from pathlib import Path

import numpy as np
import pytest
import torch

from cdw_detect.cwd_partialconv_pipeline import CHMDataset
from cdw_detect.cwd_partialconv_pipeline import JointLoss
from cdw_detect.cwd_partialconv_pipeline import PseudoLabelStore
from cdw_detect.cwd_partialconv_pipeline import Trainer
from cdw_detect.cwd_partialconv_pipeline import TrainingConfig
from cdw_detect.cwd_partialconv_pipeline import TileRecord
from cdw_detect.cwd_partialconv_pipeline import _boundary_metrics_from_binary_maps
from cdw_detect.cwd_partialconv_pipeline import _best_f1_over_thresholds
from cdw_detect.cwd_partialconv_pipeline import _count_cam_mask_tiles
from cdw_detect.cwd_partialconv_pipeline import _dice_iou_from_confusion
from cdw_detect.cwd_partialconv_pipeline import _f1_score_from_maps
from cdw_detect.cwd_partialconv_pipeline import _hd95_to_score
from cdw_detect.cwd_partialconv_pipeline import _strict_split_tile_records
from cdw_detect.cwd_partialconv_pipeline import _summarize_tile_label_stats
from cdw_detect.cwd_partialconv_pipeline import _tile_level_vectors
from cdw_detect.cwd_partialconv_pipeline import cam_to_binary_mask
from cdw_detect.cwd_partialconv_pipeline import hotspot_to_confidence_map
from cdw_detect.cwd_partialconv_pipeline import tile_id_to_artifact_stem


def _make_tile(
    tile_id: str,
    sample_id: str,
    year: int,
    center_x: float,
    center_y: float,
    place_key: str,
) -> TileRecord:
    """Create a minimal TileRecord for split unit tests.

    Parameters
    ----------
    tile_id : str
        Unique tile identifier.
    sample_id : str
        Parent sample identifier.
    year : int
        Acquisition year.
    center_x : float
        Tile center x coordinate.
    center_y : float
        Tile center y coordinate.
    place_key : str
        Shared spatial identity key.

    Returns
    -------
    TileRecord
        Synthetic tile record.
    """
    return TileRecord(
        tile_id=tile_id,
        sample_id=sample_id,
        mapsheet="111111",
        year=year,
        variant="raw",
        raster_path=Path("synthetic.tif"),
        label_path=None,
        label_raster_name=f"{sample_id}_chm_max_hag_20cm.tif",
        row_off=0,
        col_off=0,
        chunk_size=128,
        center_x=center_x,
        center_y=center_y,
        place_key=place_key,
    )


def test_strict_split_prevents_place_key_leakage_between_train_val_and_test() -> None:
    """Ensure split logic removes place-key overlap across partitions.

    Returns
    -------
    None
        Assertions validate leakage invariants.
    """
    tiles = [
        _make_tile(
            tile_id="a_low_shared",
            sample_id="111111_2022_demo",
            year=2022,
            center_x=49.96,
            center_y=100.0,
            place_key="111111:50.0:100.0",
        ),
        _make_tile(
            tile_id="b_high_shared",
            sample_id="111111_2023_demo",
            year=2023,
            center_x=50.04,
            center_y=100.0,
            place_key="111111:50.0:100.0",
        ),
        _make_tile(
            tile_id="c_low_unique",
            sample_id="111111_2022_demo2",
            year=2022,
            center_x=10.0,
            center_y=100.0,
            place_key="111111:10.0:100.0",
        ),
        _make_tile(
            tile_id="d_high_unique",
            sample_id="111111_2023_demo2",
            year=2023,
            center_x=90.0,
            center_y=100.0,
            place_key="111111:90.0:100.0",
        ),
    ]

    split = _strict_split_tile_records(
        tile_records=tiles,
        test_size=0.25,
        val_size=0.1,
        buffer_meters=0.0,
        seed=7,
    )

    assert len(split.test) > 0
    train_val_place_keys = {tile.place_key for tile in split.train + split.val}
    test_place_keys = {tile.place_key for tile in split.test}

    assert train_val_place_keys.isdisjoint(test_place_keys)
    assert split.metadata["temporal_pruned_from_train"] >= 1


def test_joint_loss_forward_returns_scalar_and_backpropagates() -> None:
    """Validate scalar loss output and gradient flow for joint objective.

    Returns
    -------
    None
        Assertions validate shape and gradient invariants.
    """
    batch_size, height, width = 2, 64, 64
    criterion = JointLoss(recon_weight=1.0, seg_weight=5.0)

    recon_logits = torch.randn(batch_size, 1, height, width, requires_grad=True)
    seg_logits = torch.randn(batch_size, 1, height, width, requires_grad=True)

    recon_pred = torch.sigmoid(recon_logits)
    seg_pred = torch.sigmoid(seg_logits)

    recon_target = torch.rand(batch_size, 1, height, width)
    valid_mask = torch.ones(batch_size, 1, height, width)
    seg_target = (torch.rand(batch_size, 1, height, width) > 0.7).float()
    confidence = torch.rand(batch_size, 1, height, width) * 0.9 + 0.1
    has_label = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    hotspot = torch.rand(batch_size, 1, height, width)

    total_loss, logs = criterion(
        recon_pred=recon_pred,
        recon_target=recon_target,
        valid_mask=valid_mask,
        seg_pred=seg_pred,
        seg_target=seg_target,
        confidence=confidence,
        has_label=has_label,
        hotspot=hotspot,
    )

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert set(logs.keys()) == {"loss_total", "loss_recon", "loss_seg"}

    total_loss.backward()
    assert recon_logits.grad is not None
    assert seg_logits.grad is not None


def test_joint_loss_sets_segmentation_component_to_zero_for_unlabeled_batches() -> None:
    """Ensure segmentation term is zero when no labels are available.

    Returns
    -------
    None
        Assertions validate unlabeled-batch handling.
    """
    criterion = JointLoss(recon_weight=1.0, seg_weight=5.0)

    recon_pred = torch.sigmoid(torch.randn(2, 1, 32, 32, requires_grad=True))
    seg_pred = torch.sigmoid(torch.randn(2, 1, 32, 32, requires_grad=True))
    recon_target = torch.rand(2, 1, 32, 32)
    valid_mask = torch.ones(2, 1, 32, 32)
    seg_target = torch.zeros(2, 1, 32, 32)
    confidence = torch.ones(2, 1, 32, 32)
    has_label = torch.zeros(2, 1)

    total_loss, logs = criterion(
        recon_pred=recon_pred,
        recon_target=recon_target,
        valid_mask=valid_mask,
        seg_pred=seg_pred,
        seg_target=seg_target,
        confidence=confidence,
        has_label=has_label,
        hotspot=None,
    )

    assert torch.isfinite(total_loss)
    assert logs["loss_seg"] == pytest.approx(0.0, abs=1e-8)


def test_threshold_sweep_recovers_nonzero_f1_when_default_threshold_misses_positives() -> None:
    """Ensure threshold sweep captures a better operating point than 0.5.

    Returns
    -------
    None
        Assertions validate threshold-sweep invariants.
    """
    tile_probs = torch.tensor([0.15, 0.22, 0.31, 0.41], dtype=torch.float32)
    tile_targets = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    prob_maps = tile_probs.view(-1, 1, 1, 1).repeat(1, 1, 2, 2)
    target_maps = tile_targets.view(-1, 1, 1, 1).repeat(1, 1, 2, 2)
    has_label = torch.ones((4, 1), dtype=torch.float32)

    f1_at_default = _f1_score_from_maps(prob_maps, target_maps, has_label, threshold=0.5)
    best_f1, best_threshold = _best_f1_over_thresholds(
        tile_prob=tile_probs,
        tile_target=tile_targets,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    )

    assert f1_at_default == pytest.approx(0.0, abs=1e-8)
    assert best_f1 == pytest.approx(1.0, abs=1e-8)
    assert best_threshold == pytest.approx(0.2, abs=1e-8)


def test_threshold_sweep_handles_empty_labeled_vectors() -> None:
    """Ensure threshold sweep remains well-defined for empty labeled inputs.

    Returns
    -------
    None
        Assertions validate empty-input behavior.
    """
    empty = torch.empty(0, dtype=torch.float32)
    best_f1, best_threshold = _best_f1_over_thresholds(
        tile_prob=empty,
        tile_target=empty,
        thresholds=[0.1, 0.3, 0.5],
    )

    assert best_f1 == pytest.approx(0.0, abs=1e-8)
    assert best_threshold == pytest.approx(0.5, abs=1e-8)


def test_summarize_tile_label_stats_reports_coverage_balance_and_confidence() -> None:
    """Ensure label stats summary captures coverage and class balance correctly.

    Returns
    -------
    None
        Assertions validate summary values.
    """
    tile_a = _make_tile(
        tile_id="stats_a",
        sample_id="111111_2022_demo",
        year=2022,
        center_x=10.0,
        center_y=10.0,
        place_key="111111:10.0:10.0",
    )
    tile_b = _make_tile(
        tile_id="stats_b",
        sample_id="111111_2023_demo",
        year=2023,
        center_x=20.0,
        center_y=20.0,
        place_key="111111:20.0:20.0",
    )
    tile_c = _make_tile(
        tile_id="stats_c",
        sample_id="111111_2024_demo",
        year=2024,
        center_x=30.0,
        center_y=30.0,
        place_key="111111:30.0:30.0",
    )

    store = PseudoLabelStore(default_confidence=0.5)
    store.set_tile_label(tile_a, label_value=1.0, confidence=0.9, source="manual")
    store.set_tile_label(tile_b, label_value=0.0, confidence=0.8, source="manual")

    stats = _summarize_tile_label_stats([tile_a, tile_b, tile_c], store)

    assert stats["total_tiles"] == pytest.approx(3.0, abs=1e-8)
    assert stats["labeled_tiles"] == pytest.approx(2.0, abs=1e-8)
    assert stats["unlabeled_tiles"] == pytest.approx(1.0, abs=1e-8)
    assert stats["positive_tiles"] == pytest.approx(1.0, abs=1e-8)
    assert stats["negative_tiles"] == pytest.approx(1.0, abs=1e-8)
    assert stats["coverage"] == pytest.approx(2.0 / 3.0, abs=1e-8)
    assert stats["positive_ratio"] == pytest.approx(0.5, abs=1e-8)
    assert stats["negative_ratio"] == pytest.approx(0.5, abs=1e-8)
    assert stats["mean_confidence"] == pytest.approx(0.85, abs=1e-8)
    assert stats["labeled_samples"] == pytest.approx(2.0, abs=1e-8)


def test_cam_to_binary_mask_otsu_separates_bimodal_map() -> None:
    """Ensure Otsu conversion separates low/high CAM regions."""
    cam = np.full((32, 32), 0.05, dtype=np.float32)
    cam[16:, :] = 0.9

    binary, threshold = cam_to_binary_mask(cam)

    assert 0.0 <= threshold <= 1.0
    assert float(binary[:16, :].mean()) == pytest.approx(0.0, abs=1e-8)
    assert float(binary[16:, :].mean()) == pytest.approx(1.0, abs=1e-8)


def test_hotspot_to_confidence_map_tracks_certainty_not_raw_activation() -> None:
    """Ensure hotspot confidence differs from hotspot intensity around ambiguous values."""
    hotspot = torch.tensor([[[[0.0, 0.25, 0.5, 0.75, 1.0]]]], dtype=torch.float32)

    confidence = hotspot_to_confidence_map(hotspot, min_conf=0.05, gamma=1.0)
    expected = torch.tensor([[[[1.0, 0.525, 0.05, 0.525, 1.0]]]], dtype=torch.float32)

    assert torch.allclose(confidence, expected, atol=1e-6)
    assert float(torch.max(torch.abs(confidence - hotspot)).item()) > 0.4


def test_hotspot_to_confidence_map_gamma_controls_sharpness() -> None:
    """Ensure larger gamma produces sharper (lower mid-certainty) confidence."""
    hotspot = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)

    conf_flat = hotspot_to_confidence_map(hotspot, min_conf=0.05, gamma=0.5)
    conf_sharp = hotspot_to_confidence_map(hotspot, min_conf=0.05, gamma=2.0)

    assert float(conf_flat[0, 0, 0, 0].item()) > float(conf_sharp[0, 0, 0, 0].item())
    assert float(conf_flat[0, 0, 0, 1].item()) > float(conf_sharp[0, 0, 0, 1].item())


def test_dataset_prefers_offline_cam_supervision_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure dataset uses offline CAM/Otsu maps over tile-level scalar labels."""
    tile = _make_tile(
        tile_id="401676_2022_demo:raw:0:0:128",
        sample_id="401676_2022_demo",
        year=2022,
        center_x=10.0,
        center_y=10.0,
        place_key="401676:10.0:10.0",
    )

    stem = tile_id_to_artifact_stem(tile.tile_id)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[:, 64:] = 1.0
    cam = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
    np.save(tmp_path / f"{stem}_mask.npy", mask)
    np.save(tmp_path / f"{stem}_cam.npy", cam)

    store = PseudoLabelStore(default_confidence=0.5)
    store.set_tile_label(tile, label_value=0.0, confidence=0.7, source="manual")

    ds = CHMDataset(
        tiles=[tile],
        label_store=store,
        augment=False,
        cam_mask_dir=tmp_path,
    )

    def _fake_read_tile(_: CHMDataset, __: TileRecord) -> tuple[np.ndarray, np.ndarray]:
        chm = np.zeros((256, 256), dtype=np.float32)
        valid = np.ones((256, 256), dtype=np.float32)
        return chm, valid

    monkeypatch.setattr(CHMDataset, "_read_tile", _fake_read_tile)
    sample = ds[0]

    seg_target = sample["seg_target"].numpy()
    conf = sample["confidence"].numpy()

    assert float(sample["has_label"].item()) == pytest.approx(1.0, abs=1e-8)
    assert float(seg_target.mean()) > 0.05
    assert float(seg_target.mean()) < 0.95
    assert float(conf.max()) <= 1.0
    assert float(conf.min()) >= 0.0


def test_dataset_falls_back_to_mask_when_cam_confidence_file_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure invalid CAM confidence arrays do not disable mask supervision."""
    tile = _make_tile(
        tile_id="401676_2022_invalidcam:raw:0:0:128",
        sample_id="401676_2022_invalidcam",
        year=2022,
        center_x=11.0,
        center_y=11.0,
        place_key="401676:11.0:11.0",
    )

    stem = tile_id_to_artifact_stem(tile.tile_id)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[32:96, 32:96] = 1.0
    np.save(tmp_path / f"{stem}_mask.npy", mask)
    (tmp_path / f"{stem}_cam.npy").write_text("not a valid npy payload", encoding="utf-8")

    ds = CHMDataset(
        tiles=[tile],
        label_store=PseudoLabelStore(default_confidence=0.5),
        augment=False,
        cam_mask_dir=tmp_path,
    )

    def _fake_read_tile(_: CHMDataset, __: TileRecord) -> tuple[np.ndarray, np.ndarray]:
        chm = np.zeros((256, 256), dtype=np.float32)
        valid = np.ones((256, 256), dtype=np.float32)
        return chm, valid

    monkeypatch.setattr(CHMDataset, "_read_tile", _fake_read_tile)
    sample = ds[0]

    seg_target = sample["seg_target"].numpy()
    conf = sample["confidence"].numpy()

    assert float(sample["has_label"].item()) == pytest.approx(1.0, abs=1e-8)
    assert float(seg_target.mean()) > 0.0
    assert float(np.abs(conf - seg_target).max()) == pytest.approx(0.0, abs=1e-8)


def test_count_cam_mask_tiles_accepts_preindexed_stems(tmp_path: Path) -> None:
    """Ensure CAM coverage counting remains correct with pre-indexed mask stems."""
    tile_hit = _make_tile(
        tile_id="401676_2022_hit:raw:0:0:128",
        sample_id="401676_2022_hit",
        year=2022,
        center_x=1.0,
        center_y=1.0,
        place_key="401676:1.0:1.0",
    )
    tile_miss = _make_tile(
        tile_id="401676_2022_miss:raw:0:0:128",
        sample_id="401676_2022_miss",
        year=2022,
        center_x=2.0,
        center_y=2.0,
        place_key="401676:2.0:2.0",
    )

    stem_hit = tile_id_to_artifact_stem(tile_hit.tile_id)
    np.save(tmp_path / f"{stem_hit}_mask.npy", np.zeros((8, 8), dtype=np.float32))

    direct_count = _count_cam_mask_tiles([tile_hit, tile_miss], tmp_path)
    indexed_count = _count_cam_mask_tiles([tile_hit, tile_miss], tmp_path, mask_index={stem_hit})

    assert direct_count == 1
    assert indexed_count == 1


def test_tile_level_vectors_treat_sparse_padded_positives_as_positive() -> None:
    """Ensure tile target extraction is presence-based for sparse positives."""
    prob = torch.zeros((2, 1, 256, 256), dtype=torch.float32)
    prob[0, 0, 100:108, 100:108] = 0.8
    prob[1, 0, :, :] = 0.1

    target = torch.zeros((2, 1, 256, 256), dtype=torch.float32)
    target[0, 0, 64:192, 64:192] = 1.0
    has_label = torch.ones((2, 1), dtype=torch.float32)

    tile_prob, tile_target = _tile_level_vectors(prob, target, has_label)

    assert tile_prob.shape == (2,)
    assert tile_target.shape == (2,)
    assert float(tile_target[0].item()) == pytest.approx(1.0, abs=1e-8)
    assert float(tile_target[1].item()) == pytest.approx(0.0, abs=1e-8)


def test_boundary_metrics_report_perfect_overlap_as_ideal() -> None:
    """Ensure clDice/HD95 become ideal for identical binary maps."""
    pred = torch.zeros((1, 1, 32, 32), dtype=torch.bool)
    target = torch.zeros((1, 1, 32, 32), dtype=torch.bool)
    pred[0, 0, 8:24, 10:22] = True
    target[0, 0, 8:24, 10:22] = True

    cldice_values, hd95_values = _boundary_metrics_from_binary_maps(pred, target)

    assert len(cldice_values) == 1
    assert len(hd95_values) == 1
    assert cldice_values[0] == pytest.approx(1.0, abs=1e-8)
    assert hd95_values[0] == pytest.approx(0.0, abs=1e-8)


def test_dice_iou_and_hd95_score_helpers_have_expected_ranges() -> None:
    """Ensure helper transforms stay in expected numerical bounds."""
    dice, iou = _dice_iou_from_confusion(tp=10.0, fp=5.0, fn=5.0)

    assert 0.0 <= dice <= 1.0
    assert 0.0 <= iou <= 1.0
    assert dice == pytest.approx(20.0 / 30.0, abs=1e-8)
    assert iou == pytest.approx(10.0 / 20.0, abs=1e-8)
    assert _hd95_to_score(0.0) == pytest.approx(1.0, abs=1e-8)
    assert _hd95_to_score(9.0) == pytest.approx(0.1, abs=1e-8)


def test_trainer_early_stops_when_sota_metric_plateaus(tmp_path: Path) -> None:
    """Ensure trainer stops once monitored validation metric stalls."""

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x[:, :1], x[:, :1]

    model = _DummyModel()
    criterion = JointLoss(recon_weight=1.0, seg_weight=5.0)
    config = TrainingConfig(
        epochs=10,
        batch_size=1,
        num_workers=0,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
        monitor_metric="sota_score",
    )
    trainer = Trainer(
        model=model,
        criterion=criterion,
        config=config,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        legacy_model=None,
    )

    val_sota_sequence = iter([0.50, 0.60, 0.59, 0.58, 0.57, 0.56])

    def _fake_run_epoch(
        self: Trainer,
        dataloader: object,
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        _ = dataloader
        _ = epoch
        if training:
            return {
                "loss": 1.0,
                "f1": 0.5,
                "dice": 0.5,
                "iou": 0.5,
                "skipped_batches": 0.0,
            }

        score = next(val_sota_sequence)
        return {
            "loss": 1.0,
            "f1": score,
            "dice": score,
            "iou": score,
            "cldice": score,
            "hd95": 1.0 - score,
            "hd95_score": _hd95_to_score(1.0 - score),
            "sota_score": score,
            "skipped_batches": 0.0,
        }

    trainer._run_epoch = _fake_run_epoch.__get__(trainer, Trainer)
    history = trainer.fit(train_loader=[None], val_loader=[None])

    assert len(history) == 4
    assert history[-1].get("early_stopped", 0.0) == pytest.approx(1.0, abs=1e-8)
    assert history[-1]["val_monitor_metric_name"] == "sota_score"
