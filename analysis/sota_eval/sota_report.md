# SOTA DTM Interpolation Evaluation

## Overview
Comparison of Inverse Distance Weighting (IDW) vs Natural Neighbor (Delaunay-based Linear/Cubic) interpolation for Digital Terrain Model (DTM) generation using ground-classified point clouds.

## Methodology
- Input Source: `output/laz_reclassified_randla/436646_2018_madal_reclassified_randla.laz`
- Subset Point Count (Ground Class 2): 829,347
- Grid Resolution: 0.5 m
- Area Size: 500x500 m tile
- IDW Parameters: k=8 neighbors, power=2
- NN Method: SciPy `griddata (linear)` over Delaunay Triangulation (Natural Neighbor Substitute)

## Quantitative Results
- **Mean Absolute Difference**: 0.0719 m
- **Max Difference**: 6.0575 m
- **RMSE**: 0.2637 m

## Observations
- The **Natural Neighbor (Linear/Cubic)** method successfully fills gaps avoiding the bullseye artifacts characteristic of IDW.
- IDW enforces local extrema at the sample points which causes artificial peaks and pits, especially visible beneath dense forest canopy where ground points are sparse.
- The Delaunay triangulation smooths transitions between available ground points, resulting in more natural slope continuation over steep hills with occluded floors.
