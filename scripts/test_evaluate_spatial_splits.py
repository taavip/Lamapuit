#!/usr/bin/env python3
"""
Test evaluation only for Option B spatial split retraining.

Loads pre-trained models from disk and evaluates on held-out test set.
Runs batched inference to avoid OOM on 56.5K test samples.

Usage:
  python scripts/test_evaluate_spatial_splits.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _instantiate_model_from_build_fn, _get_build_fn


CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("output/tile_labels_spatial_splits"),
    "LABELS_CSV": Path("data/chm_variants/labels_canonical_with_splits.csv"),
    "CHM_DIR": Path("data/lamapuit/chm_max_hag_13_drop"),
}


def normalize_chm(tile: np.ndarray) -> np.ndarray:
    """CHM normalization: clip to [0-20m] and scale to [0,1]."""
    return np.clip(tile, 0.0, 20.0) / 20.0


def load_chm_window(chm_dir: Path, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    """Load 128×128 CHM window from GeoTIFF."""
    chm_path = chm_dir / raster_name
    if not chm_path.exists():
        return None

    try:
        with rasterio.open(chm_path) as src:
            window = Window(col_off, row_off, 128, 128)
            data = src.read(1, window=window).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            data = np.nan_to_num(data, nan=0.0)
            return normalize_chm(data)
    except Exception:
        return None


def load_test_data(labels_csv: Path, chm_dir: Path):
    """Load test set from disk."""
    print("[load_test_data] Loading labels...")
    df = pd.read_csv(labels_csv)
    df_test = df[df['split'] == 'test'].copy()

    print(f"  Test: {len(df_test)} labels")
    print("[load_test_data] Loading test CHM windows...")

    X, y = [], []
    for idx, row in df_test.iterrows():
        chm = load_chm_window(chm_dir, row['raster'], int(row['row_off']), int(row['col_off']))
        if chm is None:
            continue
        label = 1 if row['label'] == 'cdw' else 0
        X.append(chm)
        y.append(label)

    X_test = np.array(X, dtype=np.float32)
    y_test = np.array(y, dtype=np.int64)

    print(f"  Loaded: X_test={X_test.shape}, y_test={y_test.shape}")
    return X_test, y_test


def load_models(output_dir: Path, device):
    """Load 4 pre-trained models from disk."""
    models = {}
    model_names = [
        ("cnn_seed42_spatial.pt", "CNN-seed42"),
        ("cnn_seed43_spatial.pt", "CNN-seed43"),
        ("cnn_seed44_spatial.pt", "CNN-seed44"),
        ("effnet_b2_spatial.pt", "EfficientNet-B2"),
    ]

    for fname, name in model_names:
        model_path = output_dir / fname
        if not model_path.exists():
            print(f"  ERROR: {model_path} not found")
            return None

        print(f"  Loading {name}...")
        checkpoint = torch.load(model_path, map_location=device)
        build_fn_name = checkpoint.get("build_fn_name")
        state_dict = checkpoint.get("state_dict")

        if not build_fn_name or not state_dict:
            print(f"  ERROR: Invalid checkpoint format for {name}")
            return None

        # Convert string name to actual build function
        build_fn = _get_build_fn(build_fn_name)
        model = _instantiate_model_from_build_fn(build_fn)
        model.load_state_dict(state_dict)
        model = model.to(device)
        models[name] = model

    return models


def evaluate_test_set_batched(models, X_test, y_test, device, batch_size=256):
    """Evaluate on test set with batched inference to avoid OOM."""
    print(f"\n[evaluate_test_set_batched] Testing {len(X_test)} samples...")
    print(f"  Batch size: {batch_size}, # batches: {(len(X_test) + batch_size - 1) // batch_size}")

    X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
    all_probs = None

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        model.eval()
        model_probs_list = []

        with torch.no_grad():
            for i in range(0, len(X_test_t), batch_size):
                batch_end = min(i + batch_size, len(X_test_t))
                batch = X_test_t[i:batch_end].to(device)
                logits = model(batch)
                batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                model_probs_list.append(batch_probs)

        model_probs = np.concatenate(model_probs_list, axis=0)
        if all_probs is None:
            all_probs = model_probs
        else:
            all_probs += model_probs
        del model_probs, model_probs_list

    ensemble_probs = all_probs / len(models)

    # Compute metrics
    auc = float(roc_auc_score(y_test, ensemble_probs))

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (ensemble_probs >= thr).astype(int)
        f1 = float(f1_score(y_test, preds, zero_division=0))
        if f1 >= best_f1:
            best_f1, best_thr = f1, float(thr)

    print(f"\nTest set evaluation:")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1: {best_f1:.4f} @ threshold={best_thr:.2f}")
    print(f"  n_test: {len(y_test)}")
    print(f"  CDW: {np.sum(y_test == 1)}, NO_CDW: {np.sum(y_test == 0)}")

    return {
        "ensemble_auc": auc,
        "ensemble_f1": best_f1,
        "ensemble_thresh": best_thr,
        "n_test": int(len(y_test)),
        "n_cdw": int(np.sum(y_test == 1)),
        "n_no_cdw": int(np.sum(y_test == 0)),
    }


def main():
    print("="*100)
    print("SPATIAL SPLIT ENSEMBLE — TEST EVALUATION (BATCHED)")
    print("="*100)

    output_dir = CONFIG["OUTPUT_DIR"]
    device = CONFIG["DEVICE"]

    # Verify model files exist
    print(f"\nChecking for pre-trained models in {output_dir}...")
    required_models = ["cnn_seed42_spatial.pt", "cnn_seed43_spatial.pt", "cnn_seed44_spatial.pt", "effnet_b2_spatial.pt"]
    missing = [m for m in required_models if not (output_dir / m).exists()]
    if missing:
        print(f"ERROR: Missing models: {missing}")
        print("Please ensure models are trained before running test evaluation.")
        return 1

    # Load data
    print(f"\n[step 1/2] Loading test data...")
    X_test, y_test = load_test_data(CONFIG["LABELS_CSV"], CONFIG["CHM_DIR"])

    # Load models
    print(f"\n[step 2/2] Loading pre-trained models...")
    models = load_models(output_dir, device)
    if models is None:
        print("ERROR: Failed to load models")
        return 1

    # Evaluate
    test_metrics = evaluate_test_set_batched(models, X_test, y_test, device, batch_size=256)

    # Save metrics
    print(f"\nSaving test metrics to metadata...")
    meta_path = output_dir / "training_metadata.json"

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata["test_metrics"] = test_metrics
    metadata["test_eval_timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {meta_path}")

    print("\n" + "="*100)
    print("TEST EVALUATION COMPLETE")
    print("="*100)
    print("\nNext steps:")
    print("  1. Run: python scripts/postprocess_spatial_split_retraining.py")
    print("  2. Review: OPTION_B_SPATIAL_SPLITS_COMPARISON.md")
    print("  3. Check: OPTION_B_SPATIAL_SPLITS_SUMMARY.md")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
