#!/usr/bin/env python3
"""
Train the CDW tile ensemble classifier from collected label CSVs.

Reads every *_labels.csv in the label directory, reopens the corresponding
CHM raster to extract features for each labeled chunk, then trains a
3-model soft-voting ensemble (LR + RF + GB) and saves it to disk.
The saved model is automatically loaded by label_tiles.py at startup.

Usage
-----
python scripts/train_tile_classifier.py \
    --labels  output/tile_labels \
    --chm-dir chm_max_hag \
    --output  output/tile_labels/ensemble_model.pkl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Feature extraction shared with label_tiles.py
_HEIGHT_BANDS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.30]


def _extract_features(tile: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    valid = tile[np.isfinite(tile)]
    n = max(valid.size, 1)
    above = valid[valid > threshold]
    feats = [
        *(float(np.sum(valid > b)) / n for b in _HEIGHT_BANDS),
        *(
            float(np.percentile(above, q)) if above.size > 0 else 0.0
            for q in [10, 25, 50, 75, 90, 99]
        ),
        float(above.mean()) if above.size > 0 else 0.0,
        float(above.std()) if above.size > 0 else 0.0,
        float(above.max()) if above.size > 0 else 0.0,
        float(above.size) / n,
        float(np.sum(np.diff((valid > threshold).astype(np.int8)) > 0)),
        (
            float(np.percentile(above, 75) / max(np.percentile(above, 25), 0.01))
            if above.size > 0
            else 1.0
        ),
        float(np.sum(valid > 0.5)) / n,
        float(np.sum(valid > 1.0)) / n,
    ]
    return np.array(feats, dtype=np.float32)


def load_labels(label_dir: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(label_dir.glob("*_labels.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] in ("cdw", "no_cdw"):
                    rows.append(row)
    return rows


def build_features(rows: list[dict], chm_dir: Path, threshold: float = 0.15):
    """Return X (n_samples, n_features), y (n_samples,) arrays."""
    X, y = [], []
    open_rasters: dict[str, rasterio.DatasetReader] = {}
    missing: set[str] = set()

    for i, row in enumerate(rows):
        raster_name = row["raster"]
        if raster_name in missing:
            continue

        if raster_name not in open_rasters:
            # Try chm_dir/<raster_name>
            chm_path = chm_dir / raster_name
            if not chm_path.exists():
                print(f"  [skip] raster not found: {chm_path}", flush=True)
                missing.add(raster_name)
                continue
            open_rasters[raster_name] = rasterio.open(chm_path)

        src = open_rasters[raster_name]
        row_off = int(row["row_off"])
        col_off = int(row["col_off"])
        chunk_size = int(row["chunk_size"])

        tile = src.read(
            1,
            window=Window(col_off, row_off, chunk_size, chunk_size),
            boundless=True,
            fill_value=0,
        ).astype(np.float32)

        feat = _extract_features(tile, threshold)
        X.append(feat)
        y.append(1 if row["label"] == "cdw" else 0)

        if (i + 1) % 200 == 0:
            print(f"  Extracted features for {i+1}/{len(rows)} labels…", flush=True)

    for src in open_rasters.values():
        src.close()

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_ensemble(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    clfs = [
        (
            "lr",
            Pipeline(
                [
                    ("sc", StandardScaler()),
                    ("cl", LogisticRegression(max_iter=1000, C=2.0)),
                ]
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ]
    vc = VotingClassifier(clfs, voting="soft", n_jobs=1)

    # 5-fold cross-val for an honest accuracy estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(vc, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"  Cross-val AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    vc.fit(X, y)
    return vc


def save_model(model, output_path: Path, meta: dict) -> None:
    import joblib

    payload = {"model": model, "meta": meta}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path, compress=3)
    print(f"  Saved → {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Train CDW tile ensemble classifier")
    p.add_argument(
        "--labels", default="output/tile_labels", help="Directory containing *_labels.csv files"
    )
    p.add_argument("--chm-dir", default="chm_max_hag", help="Directory containing CHM GeoTIFFs")
    p.add_argument(
        "--output",
        default="output/tile_labels/ensemble_model.pkl",
        help="Output path for saved model (joblib)",
    )
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Height threshold for above-ground feature (default: 0.15 m)",
    )
    args = p.parse_args()

    label_dir = Path(args.labels)
    chm_dir = Path(args.chm_dir)
    out_path = Path(args.output)

    print(f"\nLoading labels from {label_dir} …")
    rows = load_labels(label_dir)
    if not rows:
        print("No labels found. Label some tiles first, then re-run.")
        sys.exit(1)

    n_cdw = sum(1 for r in rows if r["label"] == "cdw")
    n_no = sum(1 for r in rows if r["label"] == "no_cdw")
    print(f"  {len(rows)} labels  |  CDW: {n_cdw}  No CDW: {n_no}")

    if n_cdw < 8 or n_no < 8:
        print("Need ≥8 of each class to train. Keep labeling!")
        sys.exit(1)

    print("\nExtracting features …")
    X, y = build_features(rows, chm_dir, threshold=args.threshold)
    print(f"  Feature matrix: {X.shape}  (cdw={int(y.sum())}  no_cdw={int((y==0).sum())})")

    print("\nTraining ensemble (LR + RF + GB) …")
    model = train_ensemble(X, y, n_estimators=args.n_estimators)

    meta = {
        "n_samples": len(y),
        "n_cdw": int(y.sum()),
        "n_no_cdw": int((y == 0).sum()),
        "threshold": args.threshold,
        "version": 1,
    }
    save_model(model, out_path, meta)
    print(
        "\nDone.  The model will be loaded automatically next time you run label_all_rasters.py\n"
    )


if __name__ == "__main__":
    main()
