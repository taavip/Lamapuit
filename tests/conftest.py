"""
Test configuration and fixtures for cdw_detect tests.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import LineString, box


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chm(temp_dir):
    """Create a sample CHM raster for testing."""
    width, height = 1000, 1000
    bounds = (0, 0, 100, 100)  # 100x100 meters, 0.1m resolution

    # Create CHM data with some features
    data = np.random.rand(height, width) * 5  # 0-5m heights
    # Add some taller "logs"
    data[200:220, 300:500] = 1.5  # Horizontal log
    data[400:600, 500:520] = 1.8  # Vertical log

    # Add nodata regions
    data[:100, :100] = np.nan
    data[-50:, -50:] = -9999

    chm_path = temp_dir / "test_chm.tif"

    transform = from_bounds(*bounds, width, height)

    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3301",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)

    return chm_path


@pytest.fixture
def sample_labels(temp_dir, sample_chm):
    """Create sample vector labels matching the CHM."""
    # Create some line features
    lines = [
        LineString([(30, 20), (50, 22)]),  # Horizontal log
        LineString([(50, 40), (52, 60)]),  # Vertical log
        LineString([(70, 70), (75, 80)]),  # Diagonal log
    ]

    gdf = gpd.GeoDataFrame({"geometry": lines, "class": ["cdw", "cdw", "cdw"]}, crs="EPSG:3301")

    labels_path = temp_dir / "test_labels.gpkg"
    gdf.to_file(labels_path, driver="GPKG")

    return labels_path


@pytest.fixture
def sample_dataset(temp_dir, sample_chm, sample_labels):
    """Create a minimal YOLO dataset for testing."""
    from cdw_detect.prepare import YOLODataPreparer

    output_dir = temp_dir / "dataset"
    preparer = YOLODataPreparer(
        output_dir=str(output_dir),
        tile_size=640,
        buffer_width=0.5,
        min_log_pixels=30,
        val_split=0.2,
    )

    preparer.prepare(str(sample_chm), str(sample_labels))

    return output_dir


@pytest.fixture
def mock_yolo_model(mocker):
    """Create a mock YOLO model for testing without actual model."""
    mock_model = mocker.Mock()
    mock_result = mocker.Mock()

    # Mock prediction output
    mock_masks = mocker.Mock()
    mock_masks.data.cpu().numpy.return_value = np.array(
        [np.random.rand(640, 640) > 0.5]  # Binary mask
    )

    mock_boxes = mocker.Mock()
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.85])

    mock_result.masks = mock_masks
    mock_result.boxes = mock_boxes

    mock_model.predict.return_value = [mock_result]

    return mock_model
