from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FeatureConfig:
    use_neighborhood_features: bool = True
    knn: int = 16
    radius_m: float = 1.0


def _robust_scale(values: np.ndarray) -> np.ndarray:
    vals = values.astype(np.float64, copy=False)
    finite = np.isfinite(vals)
    out = np.zeros(vals.shape[0], dtype=np.float32)
    if not np.any(finite):
        return out
    v = vals[finite]
    p1 = np.percentile(v, 1.0)
    p99 = np.percentile(v, 99.0)
    span = max(p99 - p1, 1e-6)
    out[finite] = np.clip((vals[finite] - p1) / span, 0.0, 1.0).astype(np.float32)
    return out


def _append_feature(
    x_parts: list[np.ndarray],
    names: list[str],
    arr: np.ndarray,
    name: str,
) -> None:
    x_parts.append(arr.astype(np.float32).reshape(-1, 1))
    names.append(name)


def _compute_neighborhood_features(
    xyz: np.ndarray,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    n = xyz.shape[0]
    if n < 8:
        return np.zeros((n, 0), dtype=np.float32), []

    k = max(8, min(int(cfg.knn), n - 1))
    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=k + 1)

    nbr = xyz[idx[:, 1:]]  # exclude self
    centered = nbr - np.mean(nbr, axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)

    # eigh returns ascending eigenvalues: lam0 <= lam1 <= lam2
    lam0 = eigvals[:, 0]
    lam1 = eigvals[:, 1]
    lam2 = eigvals[:, 2]
    lam_sum = lam0 + lam1 + lam2

    linearity = (lam2 - lam1) / lam2
    planarity = (lam1 - lam0) / lam2
    scattering = lam0 / lam2
    anisotropy = (lam2 - lam0) / lam2
    omnivariance = (lam0 * lam1 * lam2) ** (1.0 / 3.0)
    eigenentropy = -(lam0 * np.log(lam0) + lam1 * np.log(lam1) + lam2 * np.log(lam2))
    surface_variation = lam0 / lam_sum

    # Verticality from smallest-eigenvalue normal vector.
    normal = eigvecs[:, :, 0]
    verticality = 1.0 - np.abs(normal[:, 2])

    local_z_std = np.std(nbr[:, :, 2], axis=1)
    z_local_min = np.min(nbr[:, :, 2], axis=1)
    height_above_local_min = xyz[:, 2] - z_local_min

    # Density proxy in a fixed XY radius.
    radius = max(float(cfg.radius_m), 1e-3)
    density_count = np.sum(dists[:, 1:] <= radius, axis=1)
    density = density_count / (np.pi * radius * radius)

    # "Skeleton-like" structural proxy: highly linear + vertical neighborhoods.
    skeleton_proxy = linearity * verticality

    out = np.column_stack(
        [
            linearity,
            planarity,
            scattering,
            anisotropy,
            omnivariance,
            eigenentropy,
            surface_variation,
            verticality,
            local_z_std,
            height_above_local_min,
            density,
            skeleton_proxy,
        ]
    ).astype(np.float32)

    names = [
        "linearity",
        "planarity",
        "scattering",
        "anisotropy",
        "omnivariance",
        "eigenentropy",
        "surface_variation",
        "verticality",
        "local_z_std",
        "height_above_local_min",
        "density_radius",
        "skeleton_proxy",
    ]
    return out, names


def build_features(
    fields: dict[str, np.ndarray],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    """Build a feature matrix from LAS fields using all available dimensions."""
    x = fields["x"].astype(np.float64)
    y = fields["y"].astype(np.float64)
    z = fields["z"].astype(np.float64)

    x_parts: list[np.ndarray] = []
    names: list[str] = []

    # Geometry coordinates normalized to [0,1] per file/chunk sample.
    _append_feature(x_parts, names, _robust_scale(x), "x_norm")
    _append_feature(x_parts, names, _robust_scale(y), "y_norm")
    _append_feature(x_parts, names, _robust_scale(z), "z_norm")

    # Core LAS fields
    intensity = fields.get("intensity", np.full_like(x, np.nan))
    _append_feature(x_parts, names, _robust_scale(intensity), "intensity_norm")

    rn = fields.get("return_number", np.full_like(x, np.nan))
    nr = fields.get("number_of_returns", np.full_like(x, np.nan))
    rn_safe = np.nan_to_num(rn, nan=0.0)
    nr_safe = np.nan_to_num(nr, nan=0.0)

    _append_feature(x_parts, names, _robust_scale(rn_safe), "return_number_norm")
    _append_feature(x_parts, names, _robust_scale(nr_safe), "number_of_returns_norm")

    is_last = (rn_safe == nr_safe) & (nr_safe >= 1)
    is_second_last = (nr_safe >= 2) & (rn_safe == (nr_safe - 1))
    is_third_last = (nr_safe >= 3) & (rn_safe == (nr_safe - 2))
    _append_feature(x_parts, names, is_last.astype(np.float32), "is_last_return")
    _append_feature(x_parts, names, is_second_last.astype(np.float32), "is_second_last_return")
    _append_feature(x_parts, names, is_third_last.astype(np.float32), "is_third_last_return")

    scan_angle = fields.get("scan_angle", np.full_like(x, np.nan))
    _append_feature(x_parts, names, _robust_scale(scan_angle), "scan_angle_norm")

    point_source_id = fields.get("point_source_id", np.full_like(x, np.nan))
    _append_feature(x_parts, names, _robust_scale(point_source_id), "point_source_id_norm")

    gps_time = fields.get("gps_time", np.full_like(x, np.nan))
    _append_feature(x_parts, names, _robust_scale(gps_time), "gps_time_norm")

    # Spectral channels (if available in LAZ point format)
    red = fields.get("red", np.full_like(x, np.nan))
    green = fields.get("green", np.full_like(x, np.nan))
    blue = fields.get("blue", np.full_like(x, np.nan))
    nir = fields.get("nir", np.full_like(x, np.nan))

    _append_feature(x_parts, names, _robust_scale(red), "red_norm")
    _append_feature(x_parts, names, _robust_scale(green), "green_norm")
    _append_feature(x_parts, names, _robust_scale(blue), "blue_norm")
    _append_feature(x_parts, names, _robust_scale(nir), "nir_norm")

    # Simple spectral indices often useful for vegetation/wood separation.
    eps = 1e-6
    rg_sum = np.nan_to_num(red + green, nan=0.0)
    nd_rg = np.nan_to_num((red - green) / (rg_sum + eps), nan=0.0)
    nd_nir_red = np.nan_to_num((nir - red) / (nir + red + eps), nan=0.0)
    nd_nir_green = np.nan_to_num((nir - green) / (nir + green + eps), nan=0.0)

    _append_feature(x_parts, names, nd_rg.astype(np.float32), "nd_red_green")
    _append_feature(x_parts, names, nd_nir_red.astype(np.float32), "nd_nir_red")
    _append_feature(x_parts, names, nd_nir_green.astype(np.float32), "nd_nir_green")

    if cfg.use_neighborhood_features:
        xyz = np.column_stack([x, y, z]).astype(np.float64)
        nbr_feats, nbr_names = _compute_neighborhood_features(xyz, cfg)
        if nbr_feats.size > 0:
            x_parts.append(nbr_feats)
            names.extend(nbr_names)

    X = np.hstack(x_parts).astype(np.float32)
    return X, names
