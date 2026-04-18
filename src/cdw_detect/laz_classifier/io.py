from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np


@dataclass(frozen=True)
class PointTable:
    fields: dict[str, np.ndarray]
    field_available: dict[str, bool]


def _get_dimension(points: laspy.ScaleAwarePointRecord, name: str) -> np.ndarray | None:
    try:
        arr = getattr(points, name)
        return np.asarray(arr)
    except Exception:
        return None


def _first_present(
    points: laspy.ScaleAwarePointRecord,
    names: list[str],
) -> tuple[np.ndarray | None, str | None]:
    for n in names:
        arr = _get_dimension(points, n)
        if arr is not None:
            return arr, n
    return None, None


def sample_laz_points(
    laz_path: Path,
    max_points: int | None = None,
    chunk_size: int = 2_000_000,
    random_seed: int = 42,
) -> PointTable:
    """Read LAZ points and return a point table using all common LAS dimensions.

    If ``max_points`` is provided, points are sampled from streamed chunks until
    the requested limit is reached.
    """
    rng = np.random.default_rng(random_seed)

    # Canonical feature-friendly names mapped from LAS dimensions.
    dim_map: dict[str, list[str]] = {
        "x": ["x"],
        "y": ["y"],
        "z": ["z"],
        "intensity": ["intensity"],
        "return_number": ["return_number"],
        "number_of_returns": ["number_of_returns"],
        "classification": ["classification"],
        "scan_angle": ["scan_angle", "scan_angle_rank"],
        "scan_direction_flag": ["scan_direction_flag"],
        "edge_of_flight_line": ["edge_of_flight_line"],
        "user_data": ["user_data"],
        "point_source_id": ["point_source_id"],
        "gps_time": ["gps_time"],
        "red": ["red"],
        "green": ["green"],
        "blue": ["blue"],
        "nir": ["nir"],
    }

    collected: dict[str, list[np.ndarray]] = {k: [] for k in dim_map}
    field_available = {k: False for k in dim_map}

    remaining = max_points

    with laspy.open(str(laz_path)) as fh:
        for points in fh.chunk_iterator(chunk_size):
            n = len(points.x)
            if n == 0:
                continue

            if remaining is None:
                keep_idx = np.arange(n, dtype=np.int64)
            else:
                if remaining <= 0:
                    break
                take = min(remaining, n)
                if take == n:
                    keep_idx = np.arange(n, dtype=np.int64)
                else:
                    keep_idx = np.sort(rng.choice(n, size=take, replace=False))
                remaining -= take

            # Resolve each requested field from first available LAS dimension name.
            for out_name, candidates in dim_map.items():
                arr, found = _first_present(points, candidates)
                if arr is None:
                    continue

                vals = np.asarray(arr)[keep_idx]
                collected[out_name].append(vals)
                field_available[out_name] = True

            if remaining is not None and remaining <= 0:
                break

    fields: dict[str, np.ndarray] = {}

    # x/y/z are mandatory; fail early if any is missing.
    for mandatory in ("x", "y", "z"):
        if not collected[mandatory]:
            raise ValueError(f"Missing mandatory LAS dimension: {mandatory}")

    n_total = sum(len(a) for a in collected["x"])
    for name, chunks in collected.items():
        if chunks:
            arr = np.concatenate(chunks)
            # Normalize dtypes for downstream model building.
            if name in {"x", "y", "z", "gps_time", "scan_angle"}:
                arr = arr.astype(np.float64, copy=False)
            else:
                arr = arr.astype(np.float32, copy=False)
            fields[name] = arr
        else:
            # Missing optional dimensions become NaN float arrays to simplify feature logic.
            fields[name] = np.full(n_total, np.nan, dtype=np.float32)

    return PointTable(fields=fields, field_available=field_available)


def point_table_summary(table: PointTable) -> dict[str, Any]:
    n = int(table.fields["x"].shape[0])
    return {
        "n_points": n,
        "available_fields": [k for k, v in table.field_available.items() if v],
        "missing_fields": [k for k, v in table.field_available.items() if not v],
    }
