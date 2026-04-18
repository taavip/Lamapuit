from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_leaf: int = 1
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 3


def build_rf_pipeline(cfg: RFConfig) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    # Missing fields (e.g. no NIR) are imputed using feature medians.
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", clf),
        ]
    )
    return pipe


def train_and_evaluate_rf(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    cfg: RFConfig,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipe = build_rf_pipeline(cfg)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    # Lightweight CV estimate on training data for robustness reporting.
    cv_scores: list[float] = []
    if cfg.cv_folds >= 2:
        skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        for tr_idx, va_idx in skf.split(X_train, y_train):
            fold_model = build_rf_pipeline(cfg)
            fold_model.fit(X_train[tr_idx], y_train[tr_idx])
            va_pred = fold_model.predict(X_train[va_idx])
            cv_scores.append(float(f1_score(y_train[va_idx], va_pred, average="macro")))

    # Save model
    model_path = out_dir / "rf_laz_classifier.joblib"
    joblib.dump(pipe, model_path)

    # Feature importances
    rf_model: RandomForestClassifier = pipe.named_steps["rf"]
    importances = rf_model.feature_importances_
    importance_rows = sorted(
        (
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, importances)
        ),
        key=lambda r: r["importance"],
        reverse=True,
    )

    imp_csv = out_dir / "feature_importances.csv"
    with imp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "importance"])
        w.writeheader()
        for row in importance_rows:
            w.writerow(row)

    metrics = {
        "model_path": str(model_path),
        "feature_importances_csv": str(imp_csv),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": sorted({int(v) for v in y.tolist()}),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "cv_macro_f1_scores": cv_scores,
        "cv_macro_f1_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_macro_f1_std": float(np.std(cv_scores)) if cv_scores else None,
        "config": asdict(cfg),
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
