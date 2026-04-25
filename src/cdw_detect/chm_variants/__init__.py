"""
CHM Variant Generation Module

Comprehensive CHM generation from LAZ data with multiple variants:
  - baseline: Original sparse LiDAR
  - raw: Harmonized raw CHM (no smoothing)
  - gaussian: Harmonized Gaussian CHM (0.8m kernel smoothing)
  - composite: 4-band (Gauss+Raw+Base+Mask)
  - masked-raw: 2-band (Raw+Mask)

Clear folder naming to avoid ambiguity:
  baseline_chm_0p2m/
  harmonized_raw_0p2m/
  harmonized_gauss_kernel0p8m_0p2m/  (0p8m = kernel, not resolution)
  composite_4band_raw_base_mask/
  masked_raw_2band_0p2m/
"""

from .composite import CompositeGenerator
from .masked import MaskedCHMGenerator
from .generator import CHMVariantGenerator

__all__ = [
    "CHMVariantGenerator",
    "CompositeGenerator",
    "MaskedCHMGenerator",
]
