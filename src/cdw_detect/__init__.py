"""
CDW-Detect: Coarse Woody Debris Detection from LiDAR
====================================================
Deep learning pipeline for detecting fallen logs in LiDAR-derived CHM rasters.
"""

__version__ = "0.1.0"

from .prepare import YOLODataPreparer
from .detect import CDWDetector

__all__ = ["YOLODataPreparer", "CDWDetector", "__version__"]
