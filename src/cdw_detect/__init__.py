"""
CDW-Detect: Coarse Woody Debris Detection from LiDAR
====================================================
Deep learning pipeline for detecting fallen logs in LiDAR-derived CHM rasters.

Submodules:
  - prepare: Data preparation (CHM → tiled dataset)
  - detect: Inference engine (sliding-window detection)
  - chm_variants: CHM variant generation (LAZ → multiple CHM types)
  - laz_classifier: Point-cloud RF classifier
"""

__version__ = "0.1.0"

from .prepare import YOLODataPreparer
from .detect import CDWDetector
from .chm_variants import CHMVariantGenerator, CompositeGenerator, MaskedCHMGenerator

__all__ = [
    "YOLODataPreparer",
    "CDWDetector",
    "CHMVariantGenerator",
    "CompositeGenerator",
    "MaskedCHMGenerator",
    "__version__",
]
