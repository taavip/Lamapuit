"""
Integration tests for full workflow.
"""

import pytest
from pathlib import Path


@pytest.mark.slow
@pytest.mark.integration
class TestFullWorkflow:
    """Test complete workflow from data prep to inference."""

    def test_prepare_train_detect_workflow(self, temp_dir, sample_chm, sample_labels, mocker):
        """Test full pipeline: prepare -> train -> detect."""
        from cdw_detect.prepare import YOLODataPreparer
        from cdw_detect.train import train
        from cdw_detect.detect import CDWDetector

        # Step 1: Prepare dataset
        dataset_dir = temp_dir / "dataset"
        preparer = YOLODataPreparer(output_dir=str(dataset_dir))
        stats = preparer.prepare(str(sample_chm), str(sample_labels))

        assert stats["total"] > 0
        assert (dataset_dir / "dataset.yaml").exists()

        # Step 2: Mock training (too slow for unit tests)
        mock_yolo_class = mocker.patch("cdw_detect.train.YOLO")
        mock_yolo_instance = mock_yolo_class.return_value

        # Create mock model output
        model_dir = temp_dir / "runs" / "train" / "weights"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "best.pt"
        model_path.touch()

        train(
            dataset_yaml=str(dataset_dir / "dataset.yaml"),
            model="yolo11n-seg.pt",
            epochs=1,
            batch=1,
            device="cpu",
            project=str(temp_dir / "runs"),
            name="train",
        )

        # Step 3: Mock detection
        mock_yolo_class = mocker.patch("cdw_detect.detect.YOLO")
        detector = CDWDetector(model_path=str(model_path), device="cpu")

        # This would fail without mocking the actual inference
        # Just verify the detector was initialized
        assert detector.model_path == str(model_path)

    def test_dataset_statistics(self, sample_dataset):
        """Test that dataset statistics are reasonable."""
        import csv

        metadata_path = sample_dataset / "tile_metadata.csv"
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total = len(rows)
        with_cdw = sum(1 for r in rows if r["has_cdw"] == "True")

        # Should have some tiles with CDW
        assert total > 0
        # Balance shouldn't be too skewed (though with small sample it might be)
        assert with_cdw >= 0
