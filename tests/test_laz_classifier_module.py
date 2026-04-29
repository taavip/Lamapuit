from __future__ import annotations

import numpy as np

from cdw_detect.laz_classifier.features import FeatureConfig, build_features


def test_build_features_runs_with_minimal_fields() -> None:
    n = 128
    rng = np.random.default_rng(0)
    fields = {
        "x": rng.uniform(0, 10, n).astype(np.float32),
        "y": rng.uniform(0, 10, n).astype(np.float32),
        "z": rng.uniform(0, 5, n).astype(np.float32),
        "intensity": rng.uniform(0, 65535, n).astype(np.float32),
        "return_number": rng.integers(1, 3, n).astype(np.float32),
        "number_of_returns": rng.integers(1, 3, n).astype(np.float32),
        "classification": rng.integers(1, 4, n).astype(np.float32),
        "scan_angle": rng.uniform(-20, 20, n).astype(np.float32),
        "scan_direction_flag": rng.integers(0, 2, n).astype(np.float32),
        "edge_of_flight_line": rng.integers(0, 2, n).astype(np.float32),
        "user_data": np.zeros(n, dtype=np.float32),
        "point_source_id": rng.integers(1, 20, n).astype(np.float32),
        "gps_time": rng.uniform(1e6, 2e6, n).astype(np.float32),
        "red": rng.uniform(0, 65535, n).astype(np.float32),
        "green": rng.uniform(0, 65535, n).astype(np.float32),
        "blue": rng.uniform(0, 65535, n).astype(np.float32),
        "nir": rng.uniform(0, 65535, n).astype(np.float32),
    }

    X, names = build_features(fields, FeatureConfig(use_neighborhood_features=True, knn=8, radius_m=1.0))
    assert X.shape[0] == n
    assert X.shape[1] == len(names)
    assert X.shape[1] > 10
