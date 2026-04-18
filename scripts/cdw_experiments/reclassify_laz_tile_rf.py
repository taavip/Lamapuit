#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import joblib
import laspy
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cdw_detect.laz_classifier.features import FeatureConfig, build_features
from cdw_detect.laz_classifier.io import sample_laz_points
from cdw_detect.laz_classifier.rf import RFConfig, train_and_evaluate_rf


def _extract_fields_from_las(las: laspy.LasData) -> dict[str, np.ndarray]:
    pts = las.points
    n = len(las.x)

    def _get(name: str) -> np.ndarray:
        try:
            return np.asarray(getattr(pts, name))
        except Exception:
            return np.full(n, np.nan, dtype=np.float32)

    fields = {
        "x": np.asarray(las.x, dtype=np.float64),
        "y": np.asarray(las.y, dtype=np.float64),
        "z": np.asarray(las.z, dtype=np.float32),
        "intensity": _get("intensity").astype(np.float32, copy=False),
        "return_number": _get("return_number").astype(np.float32, copy=False),
        "number_of_returns": _get("number_of_returns").astype(np.float32, copy=False),
        "classification": _get("classification").astype(np.float32, copy=False),
        "scan_angle": (
            _get("scan_angle") if hasattr(pts, "scan_angle") else _get("scan_angle_rank")
        ).astype(np.float32, copy=False),
        "scan_direction_flag": _get("scan_direction_flag").astype(np.float32, copy=False),
        "edge_of_flight_line": _get("edge_of_flight_line").astype(np.float32, copy=False),
        "user_data": _get("user_data").astype(np.float32, copy=False),
        "point_source_id": _get("point_source_id").astype(np.float32, copy=False),
        "gps_time": _get("gps_time").astype(np.float32, copy=False),
        "red": _get("red").astype(np.float32, copy=False),
        "green": _get("green").astype(np.float32, copy=False),
        "blue": _get("blue").astype(np.float32, copy=False),
        "nir": _get("nir").astype(np.float32, copy=False),
    }
    return fields


def main() -> int:
    p = argparse.ArgumentParser(description="Reclassify target LAZ using RF trained on neighbor years")
    p.add_argument("--input-dir", type=Path, default=Path("data/lamapuit/laz"))
    p.add_argument("--tile-id", default="436646")
    p.add_argument("--target-year", type=int, default=2018)
    p.add_argument("--train-years", default="2020,2022,2024")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--max-train-points-per-file", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-labels", default="0")
    p.add_argument("--n-estimators", type=int, default=120)
    p.add_argument("--max-depth", type=int, default=24)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--predict-batch-size", type=int, default=250000)
    p.add_argument("--out-dir", type=Path, default=Path("output/laz_reclassified"))
    args = p.parse_args()

    train_years = [int(x.strip()) for x in args.train_years.split(",") if x.strip()]
    exclude_labels = {int(x.strip()) for x in args.exclude_labels.split(",") if x.strip()}

    target_laz = args.input_dir / f"{args.tile_id}_{args.target_year}_madal.laz"
    train_laz = [args.input_dir / f"{args.tile_id}_{y}_madal.laz" for y in train_years]

    if not target_laz.exists():
        raise FileNotFoundError(f"Target LAZ not found: {target_laz}")
    if args.model_path is None:
        for pth in train_laz:
            if not pth.exists():
                raise FileNotFoundError(f"Training LAZ not found: {pth}")
    else:
        if not args.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / f"rf_model_{args.tile_id}_{args.target_year}"
    model_dir.mkdir(parents=True, exist_ok=True)

    feat_cfg = FeatureConfig(use_neighborhood_features=False)

    y_train = np.array([], dtype=np.int32)
    metrics: dict = {}

    if args.model_path is None:
        X_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []

        for laz_path in train_laz:
            print(f"Sampling/train-features: {laz_path}")
            table = sample_laz_points(
                laz_path=laz_path,
                max_points=args.max_train_points_per_file,
                random_seed=args.seed,
            )
            y = np.round(table.fields["classification"]).astype(np.int32)
            valid = np.isfinite(table.fields["classification"])
            for lbl in exclude_labels:
                valid &= y != lbl

            fields_sel = {k: v[valid] for k, v in table.fields.items()}
            y_sel = y[valid]
            if y_sel.size == 0:
                continue

            X, _ = build_features(fields_sel, feat_cfg)
            X_all.append(X)
            y_all.append(y_sel)

        if not X_all:
            raise RuntimeError("No training samples after filtering labels")

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)
        print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")

        _, feature_names = build_features({k: v[: min(2, len(v))] for k, v in fields_sel.items()}, feat_cfg)
        rf_cfg = RFConfig(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            cv_folds=args.cv_folds,
        )
        print(
            "RF config:",
            {
                "n_estimators": rf_cfg.n_estimators,
                "max_depth": rf_cfg.max_depth,
                "min_samples_leaf": rf_cfg.min_samples_leaf,
                "cv_folds": rf_cfg.cv_folds,
            },
        )
        metrics = train_and_evaluate_rf(
            X=X_train,
            y=y_train,
            feature_names=feature_names,
            out_dir=model_dir,
            cfg=rf_cfg,
        )

        model = joblib.load(model_dir / "rf_laz_classifier.joblib")
        del X_train
        del X_all
        del y_all
    else:
        print(f"Using existing model without training: {args.model_path}")
        model = joblib.load(args.model_path)

    las = laspy.read(str(target_laz))
    # Copy to keep original labels for summary after we overwrite las.classification.
    original_cls = np.asarray(las.classification, dtype=np.uint8).copy()

    fields_target = _extract_fields_from_las(las)
    n_points = len(las.x)
    pred = np.empty(n_points, dtype=np.uint8)
    print(f"Predicting {n_points} points in batches of {args.predict_batch_size}")
    for start in range(0, n_points, args.predict_batch_size):
        end = min(start + args.predict_batch_size, n_points)
        batch_fields = {k: v[start:end] for k, v in fields_target.items()}
        X_batch, _ = build_features(batch_fields, feat_cfg)
        pred[start:end] = model.predict(X_batch).astype(np.uint8)
        print(f"Predicted points {start}:{end}")

    if pred.shape[0] != len(las.x):
        raise RuntimeError("Prediction length does not match LAS point count")

    las.classification = pred

    out_laz = args.out_dir / f"{args.tile_id}_{args.target_year}_madal_reclassified_rf.laz"
    las.write(str(out_laz))

    summary = {
        "target_laz": str(target_laz),
        "train_laz": [str(pth) for pth in train_laz] if args.model_path is None else [],
        "model_path": str(args.model_path) if args.model_path is not None else str(model_dir / "rf_laz_classifier.joblib"),
        "inference_only": bool(args.model_path is not None),
        "output_laz": str(out_laz),
        "n_points": int(len(las.x)),
        "train_label_counts": {str(k): int(v) for k, v in sorted(Counter(y_train.tolist()).items())},
        "original_class_counts": {str(k): int(v) for k, v in sorted(Counter(original_cls.tolist()).items())},
        "predicted_class_counts": {str(k): int(v) for k, v in sorted(Counter(pred.tolist()).items())},
        "metrics": {
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "cv_macro_f1_mean": metrics.get("cv_macro_f1_mean"),
        },
    }
    summary_path = args.out_dir / f"{args.tile_id}_{args.target_year}_madal_reclassified_rf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
