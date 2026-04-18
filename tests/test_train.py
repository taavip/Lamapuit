"""
Unit tests for cdw_detect.train module.
"""

import pytest
from pathlib import Path
from cdw_detect.train import train


class TestTrain:
    """Test suite for train function."""

    def test_train_missing_dataset_yaml(self, temp_dir):
        """Test error handling for missing dataset.yaml."""
        with pytest.raises(FileNotFoundError, match="Dataset config not found"):
            train(
                dataset_yaml=str(temp_dir / "nonexistent.yaml"),
                model="yolo11n-seg.pt",
                epochs=1,
                device="cpu",
            )

    def test_train_invalid_yaml_format(self, temp_dir):
        """Test error handling for invalid YAML format."""
        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("not: valid: yaml: [[[")

        with pytest.raises(ValueError, match="Invalid dataset.yaml format"):
            train(dataset_yaml=str(invalid_yaml), model="yolo11n-seg.pt", epochs=1, device="cpu")

    def test_train_missing_required_keys(self, temp_dir):
        """Test error handling for dataset.yaml missing required keys."""
        incomplete_yaml = temp_dir / "incomplete.yaml"
        incomplete_yaml.write_text("path: /some/path\n")

        with pytest.raises(ValueError, match="must contain 'train', 'val', and 'names' keys"):
            train(dataset_yaml=str(incomplete_yaml), model="yolo11n-seg.pt", epochs=1, device="cpu")

    def test_train_invalid_epochs(self, temp_dir):
        """Test error handling for invalid epochs parameter."""
        valid_yaml = temp_dir / "dataset.yaml"
        valid_yaml.write_text("""
path: /some/path
train: images/train
val: images/val
names:
  0: cdw
""")

        with pytest.raises(ValueError, match="epochs must be >= 1"):
            train(dataset_yaml=str(valid_yaml), model="yolo11n-seg.pt", epochs=0, device="cpu")

    def test_train_invalid_batch(self, temp_dir):
        """Test error handling for invalid batch size."""
        valid_yaml = temp_dir / "dataset.yaml"
        valid_yaml.write_text("""
path: /some/path
train: images/train
val: images/val
names:
  0: cdw
""")

        with pytest.raises(ValueError, match="batch must be >= 1"):
            train(
                dataset_yaml=str(valid_yaml),
                model="yolo11n-seg.pt",
                epochs=1,
                batch=0,
                device="cpu",
            )

    def test_train_invalid_imgsz(self, temp_dir):
        """Test error handling for invalid image size."""
        valid_yaml = temp_dir / "dataset.yaml"
        valid_yaml.write_text("""
path: /some/path
train: images/train
val: images/val
names:
  0: cdw
""")

        with pytest.raises(ValueError, match="imgsz must be multiple of 32"):
            train(
                dataset_yaml=str(valid_yaml),
                model="yolo11n-seg.pt",
                epochs=1,
                imgsz=100,
                device="cpu",
            )

    def test_train_invalid_patience(self, temp_dir):
        """Test error handling for invalid patience parameter."""
        valid_yaml = temp_dir / "dataset.yaml"
        valid_yaml.write_text("""
path: /some/path
train: images/train
val: images/val
names:
  0: cdw
""")

        with pytest.raises(ValueError, match="patience must be >= 1"):
            train(
                dataset_yaml=str(valid_yaml),
                model="yolo11n-seg.pt",
                epochs=1,
                patience=0,
                device="cpu",
            )

    @pytest.mark.slow
    def test_train_with_valid_dataset(self, mocker, sample_dataset):
        """Test training with a valid dataset (mocked)."""
        # Mock YOLO to avoid actual training
        mock_yolo_class = mocker.patch("cdw_detect.train.YOLO")
        mock_yolo_instance = mock_yolo_class.return_value
        mock_yolo_instance.train.return_value = None

        # Create output directory structure that would be created by YOLO
        output_dir = Path("runs/cdw_detect/train/weights")
        output_dir.mkdir(parents=True, exist_ok=True)
        best_model = output_dir / "best.pt"
        best_model.touch()

        result = train(
            dataset_yaml=str(sample_dataset / "dataset.yaml"),
            model="yolo11n-seg.pt",
            epochs=1,
            batch=1,
            device="cpu",
            project="runs/cdw_detect",
            name="train",
        )

        # Verify YOLO was called with correct parameters
        mock_yolo_instance.train.assert_called_once()
        call_kwargs = mock_yolo_instance.train.call_args[1]
        assert call_kwargs["epochs"] == 1
        assert call_kwargs["batch"] == 1
        assert call_kwargs["device"] == "cpu"
