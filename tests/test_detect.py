"""
Unit tests for cdw_detect.detect module.
"""

import pytest
import numpy as np
from pathlib import Path
from cdw_detect.detect import CDWDetector


class TestCDWDetector:
    """Test suite for CDWDetector class."""

    def test_initialization_missing_model(self, temp_dir):
        """Test error handling for missing model file."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            CDWDetector(model_path=str(temp_dir / "nonexistent.pt"), device="cpu")

    def test_initialization_invalid_model_format(self, temp_dir):
        """Test error handling for invalid model format."""
        invalid_model = temp_dir / "model.txt"
        invalid_model.write_text("not a model")

        with pytest.raises(ValueError, match="Model must be .pt or .pth format"):
            CDWDetector(model_path=str(invalid_model), device="cpu")

    def test_detect_missing_chm(self, mocker, temp_dir):
        """Test error handling for missing CHM file."""
        # Create a dummy model file
        model_path = temp_dir / "model.pt"
        model_path.touch()

        # Mock the YOLO loading to avoid needing real model
        mocker.patch("cdw_detect.detect.YOLO")

        detector = CDWDetector(model_path=str(model_path), device="cpu")

        with pytest.raises(FileNotFoundError, match="CHM raster not found"):
            detector.detect("/nonexistent/chm.tif")

    def test_detect_invalid_chm_format(self, mocker, temp_dir):
        """Test error handling for invalid CHM format."""
        model_path = temp_dir / "model.pt"
        model_path.touch()

        invalid_chm = temp_dir / "chm.txt"
        invalid_chm.write_text("not a tiff")

        mocker.patch("cdw_detect.detect.YOLO")

        detector = CDWDetector(model_path=str(model_path), device="cpu")

        with pytest.raises(ValueError, match="CHM must be GeoTIFF format"):
            detector.detect(str(invalid_chm))

    def test_detect_raster_too_small(self, mocker, temp_dir, sample_chm):
        """Test error handling for raster smaller than tile size."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create a small raster (smaller than tile_size=640)
        small_chm = temp_dir / "small_chm.tif"
        data = np.random.rand(300, 300) * 5

        with rasterio.open(
            small_chm,
            "w",
            driver="GTiff",
            height=300,
            width=300,
            count=1,
            dtype=data.dtype,
            crs="EPSG:3301",
            transform=from_bounds(0, 0, 30, 30, 300, 300),
        ) as dst:
            dst.write(data, 1)

        model_path = temp_dir / "model.pt"
        model_path.touch()

        mocker.patch("cdw_detect.detect.YOLO")

        detector = CDWDetector(model_path=str(model_path), tile_size=640, device="cpu")

        with pytest.raises(ValueError, match="Raster too small"):
            detector.detect(str(small_chm))

    @pytest.mark.slow
    def test_detect_returns_geodataframe(self, mocker, temp_dir, sample_chm, mock_yolo_model):
        """Test that detect() returns a valid GeoDataFrame."""
        model_path = temp_dir / "model.pt"
        model_path.touch()

        # Mock YOLO to return our mock model
        mocker.patch("cdw_detect.detect.YOLO", return_value=mock_yolo_model)

        detector = CDWDetector(model_path=str(model_path), device="cpu")

        result = detector.detect(str(sample_chm))

        # Check result structure
        assert hasattr(result, "geometry")
        assert "confidence" in result.columns
        assert "area_m2" in result.columns
        assert result.crs is not None

    @pytest.mark.slow
    def test_detect_saves_output(self, mocker, temp_dir, sample_chm, mock_yolo_model):
        """Test that detect() can save output to file."""
        model_path = temp_dir / "model.pt"
        model_path.touch()

        output_path = temp_dir / "detections.gpkg"

        mocker.patch("cdw_detect.detect.YOLO", return_value=mock_yolo_model)

        detector = CDWDetector(model_path=str(model_path), device="cpu")
        result = detector.detect(str(sample_chm), output_path=str(output_path))

        assert output_path.exists()

    def test_detector_parameters(self, mocker, temp_dir):
        """Test that detector parameters are set correctly."""
        model_path = temp_dir / "model.pt"
        model_path.touch()

        mocker.patch("cdw_detect.detect.YOLO")

        detector = CDWDetector(
            model_path=str(model_path),
            tile_size=512,
            stride=384,
            confidence=0.25,
            iou_threshold=0.6,
            min_area_m2=1.0,
            device="cpu",
        )

        assert detector.tile_size == 512
        assert detector.stride == 384
        assert detector.confidence == 0.25
        assert detector.iou_threshold == 0.6
        assert detector.min_area_m2 == 1.0
        assert detector.device == "cpu"


class TestNMS:
    """Test suite for Non-Maximum Suppression."""

    def test_nms_removes_overlaps(self, mocker, temp_dir, sample_chm):
        """Test that NMS properly removes overlapping detections."""
        # This would require actual detection results
        # Marked for future implementation
        pass
