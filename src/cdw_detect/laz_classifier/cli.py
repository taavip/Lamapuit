from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

import joblib
import numpy as np

from .features import FeatureConfig, build_features
from .io import point_table_summary, sample_laz_points
from .rf import RFConfig, train_and_evaluate_rf


def _parse_int_list(raw: str | None) -> set[int]:
    if raw is None or raw.strip() == "":
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _subset_fields(fields: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in fields.items()}


def cmd_train(args: argparse.Namespace) -> int:
    laz_path = Path(args.laz)
    out_dir = Path(args.out_dir)

    table = sample_laz_points(
        laz_path=laz_path,
        max_points=args.max_points,
        chunk_size=args.chunk_size,
        random_seed=args.seed,
    )

    summary = point_table_summary(table)
    print("Point table summary:", json.dumps(summary, indent=2))

    label_dim = args.label_dim
    if label_dim not in table.fields:
        raise ValueError(f"Label dimension '{label_dim}' not available in sampled point table")

    y_raw = table.fields[label_dim]
    valid = np.isfinite(y_raw)

    include_labels = _parse_int_list(args.include_labels)
    exclude_labels = _parse_int_list(args.exclude_labels)

    y_int = np.zeros(y_raw.shape[0], dtype=np.int32)
    y_int[valid] = np.round(y_raw[valid]).astype(np.int32)

    if include_labels:
        valid &= np.isin(y_int, sorted(include_labels))
    if exclude_labels:
        valid &= ~np.isin(y_int, sorted(exclude_labels))

    if np.count_nonzero(valid) < 100:
        raise ValueError("Too few labeled points after filtering; increase max_points or adjust label filters")

    fields_sel = _subset_fields(table.fields, valid)
    y = y_int[valid]

    feat_cfg = FeatureConfig(
        use_neighborhood_features=args.use_neighborhood_features,
        knn=args.knn,
        radius_m=args.radius_m,
    )
    X, feature_names = build_features(fields_sel, feat_cfg)

    counts = dict(sorted(Counter(y.tolist()).items(), key=lambda kv: kv[0]))
    print("Label counts used for training:", counts)

    rf_cfg = RFConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

    metrics = train_and_evaluate_rf(
        X=X,
        y=y,
        feature_names=feature_names,
        out_dir=out_dir,
        cfg=rf_cfg,
    )

    # Save training metadata for reproducibility.
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "laz": str(laz_path),
        "label_dim": label_dim,
        "max_points": args.max_points,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "include_labels": sorted(include_labels),
        "exclude_labels": sorted(exclude_labels),
        "point_table_summary": summary,
        "n_labeled_points": int(np.count_nonzero(valid)),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "features": feature_names,
        "feature_config": {
            "use_neighborhood_features": feat_cfg.use_neighborhood_features,
            "knn": feat_cfg.knn,
            "radius_m": feat_cfg.radius_m,
        },
        "metrics_path": str(Path(out_dir) / "metrics.json"),
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "cv_macro_f1_mean": metrics["cv_macro_f1_mean"],
    }
    meta_path = Path(out_dir) / "training_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Training completed.")
    print(json.dumps({
        "out_dir": str(out_dir),
        "model": metrics["model_path"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "cv_macro_f1_mean": metrics["cv_macro_f1_mean"],
    }, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    laz_path = Path(args.laz)
    model_path = Path(args.model)

    table = sample_laz_points(
        laz_path=laz_path,
        max_points=args.max_points,
        chunk_size=args.chunk_size,
        random_seed=args.seed,
    )

    feat_cfg = FeatureConfig(
        use_neighborhood_features=args.use_neighborhood_features,
        knn=args.knn,
        radius_m=args.radius_m,
    )
    X, _ = build_features(table.fields, feat_cfg)

    model = joblib.load(model_path)
    pred = model.predict(X)
    counts = dict(sorted(Counter([int(v) for v in pred.tolist()]).items(), key=lambda kv: kv[0]))

    result = {
        "laz": str(laz_path),
        "model": str(model_path),
        "n_points_predicted": int(X.shape[0]),
        "predicted_class_counts": {str(k): int(v) for k, v in counts.items()},
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAZ classifier using all available LAS fields")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train RandomForest classifier from labeled LAZ points")
    train.add_argument("--laz", required=True, help="Input LAZ with label dimension present")
    train.add_argument("--label-dim", default="classification", help="LAS dimension used as training label")
    train.add_argument("--include-labels", default="", help="Comma-separated labels to keep")
    train.add_argument("--exclude-labels", default="", help="Comma-separated labels to remove")
    train.add_argument("--max-points", type=int, default=300_000)
    train.add_argument("--chunk-size", type=int, default=2_000_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out-dir", default="runs/laz_classifier")

    train.add_argument("--use-neighborhood-features", action="store_true")
    train.add_argument("--knn", type=int, default=16)
    train.add_argument("--radius-m", type=float, default=1.0)

    train.add_argument("--n-estimators", type=int, default=400)
    train.add_argument("--max-depth", type=int, default=None)
    train.add_argument("--min-samples-leaf", type=int, default=1)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--cv-folds", type=int, default=3)
    train.set_defaults(func=cmd_train)

    pred = sub.add_parser("predict", help="Run model on LAZ sample and print class counts")
    pred.add_argument("--laz", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--max-points", type=int, default=300_000)
    pred.add_argument("--chunk-size", type=int, default=2_000_000)
    pred.add_argument("--seed", type=int, default=42)
    pred.add_argument("--use-neighborhood-features", action="store_true")
    pred.add_argument("--knn", type=int, default=16)
    pred.add_argument("--radius-m", type=float, default=1.0)
    pred.add_argument("--out-json", default="")
    pred.set_defaults(func=cmd_predict)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
