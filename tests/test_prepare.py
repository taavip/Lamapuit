"""
Unit tests for cdw_detect.prepare module.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from cdw_detect.prepare import YOLODataPreparer, augment_with_nodata


class TestYOLODataPreparer:
    """Test suite for YOLODataPreparer class."""

    def test_initialization(self, temp_dir):
        """Test proper initialization of preparer."""
        output_dir = temp_dir / "dataset"
        preparer = YOLODataPreparer(
            output_dir=str(output_dir),
            tile_size=640,
            buffer_width=0.5,
        )

        assert preparer.tile_size == 640
        assert preparer.buffer_width == 0.5
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "labels" / "train").exists()
        assert (output_dir / "labels" / "val").exists()

    def test_prepare_missing_chm(self, temp_dir, sample_labels):
        """Test error handling for missing CHM file."""
        preparer = YOLODataPreparer(output_dir=str(temp_dir / "dataset"))

        with pytest.raises(FileNotFoundError, match="CHM raster not found"):
            preparer.prepare("/nonexistent/chm.tif", str(sample_labels))

    def test_prepare_missing_labels(self, temp_dir, sample_chm):
        """Test error handling for missing labels file."""
        preparer = YOLODataPreparer(output_dir=str(temp_dir / "dataset"))

        with pytest.raises(FileNotFoundError, match="Labels file not found"):
            preparer.prepare(str(sample_chm), "/nonexistent/labels.gpkg")

    def test_prepare_invalid_chm_format(self, temp_dir, sample_labels):
        """Test error handling for invalid CHM format."""
        # Create a non-GeoTIFF file
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("not a tiff")

        preparer = YOLODataPreparer(output_dir=str(temp_dir / "dataset"))

        with pytest.raises(ValueError, match="CHM must be GeoTIFF format"):
            preparer.prepare(str(invalid_file), str(sample_labels))

    def test_prepare_creates_tiles(self, sample_dataset):
        """Test that prepare() creates tiles correctly."""
        train_images = list((sample_dataset / "images" / "train").glob("*.png"))
        val_images = list((sample_dataset / "images" / "val").glob("*.png"))

        # Should have at least some tiles
        assert len(train_images) > 0 or len(val_images) > 0

        # Check tile properties
        if train_images:
            img = cv2.imread(str(train_images[0]), cv2.IMREAD_GRAYSCALE)
            assert img.shape == (640, 640)

    def test_prepare_creates_labels(self, sample_dataset):
        """Test that label files are created."""
        train_labels = list((sample_dataset / "labels" / "train").glob("*.txt"))
        val_labels = list((sample_dataset / "labels" / "val").glob("*.txt"))

        assert len(train_labels) > 0 or len(val_labels) > 0

        # Check label format
        if train_labels:
            with open(train_labels[0]) as f:
                lines = f.readlines()
                if lines:  # May be empty for tiles without CDW
                    parts = lines[0].split()
                    assert parts[0] == "0"  # Class ID
                    assert len(parts) > 3  # Class + at least 1 point (x,y)

    def test_prepare_creates_metadata(self, sample_dataset):
        """Test that metadata CSV is created."""
        metadata_path = sample_dataset / "tile_metadata.csv"
        assert metadata_path.exists()

        # Check metadata content
        import csv

        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0
            assert "tile" in rows[0]
            assert "crs" in rows[0]
            assert "has_cdw" in rows[0]

    def test_prepare_creates_dataset_yaml(self, sample_dataset):
        """Test that dataset.yaml is created correctly."""
        yaml_path = sample_dataset / "dataset.yaml"
        assert yaml_path.exists()

        import yaml

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert "path" in config
        assert "train" in config
        assert "val" in config
        assert "names" in config
        assert config["names"][0] == "cdw"

    def test_prepare_returns_stats(self, temp_dir, sample_chm, sample_labels):
        """Test that prepare() returns statistics."""
        preparer = YOLODataPreparer(output_dir=str(temp_dir / "dataset"))
        stats = preparer.prepare(str(sample_chm), str(sample_labels))

        assert "total" in stats
        assert "with_cdw" in stats
        assert "empty" in stats
        assert "skipped" in stats
        assert stats["total"] == stats["with_cdw"] + stats["empty"]

    def test_prepare_handles_nodata(self, temp_dir, sample_chm, sample_labels):
        """Test that tiles with excessive nodata are skipped."""
        preparer = YOLODataPreparer(output_dir=str(temp_dir / "dataset"))
        stats = preparer.prepare(str(sample_chm), str(sample_labels))

        # Should skip some tiles due to nodata
        assert stats["skipped"] > 0


class TestAugmentation:
    """Test suite for data augmentation functions."""

    def test_augment_with_nodata(self, sample_dataset, temp_dir):
        """Test nodata augmentation."""
        output_dir = temp_dir / "dataset_augmented"

        augment_with_nodata(str(sample_dataset), str(output_dir), nodata_fraction=0.5)

        # Check that augmented dataset exists
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "dataset.yaml").exists()

        # Verify some images were modified
        train_images = list((output_dir / "images" / "train").glob("*.png"))
        assert len(train_images) > 0
