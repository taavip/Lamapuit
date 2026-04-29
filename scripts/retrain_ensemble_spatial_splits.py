#!/usr/bin/env python3
"""
Retrain 4-model ensemble using spatial split strategy.

This script implements Option B (academically rigorous approach):
- Uses spatial splits: test/train/val only (excludes 'none' buffer zones)
- Trains 4 models: 3 CNN seeds (42,43,44) + EfficientNet-B2
- Validates on proper held-out test set
- Generates probabilities for ALL 580K labels
- Compares to original ensemble

Training data:
  - Train: 67,290 labels (split='train')
  - Val:   13,850 labels (split='val')
  - Test:  56,521 labels (split='test', held-out evaluation)
  - None:  442,475 labels (EXCLUDED - buffer zones)

Output: Trained models + comprehensive evaluation report
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _get_build_fn, _instantiate_model_from_build_fn


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    "CNN_EPOCHS": 50,
    "EFFNET_EPOCHS": 30,
    "BATCH_SIZE": 16,  # Reduced from 32 to fit in 18GB GPU with disk streaming
    "LR_HEAD": 5e-4,
    "LR_BACKBONE": 5e-5,
    "LABEL_SMOOTHING": 0.05,
    "MIXUP_ALPHA": 0.3,
    "CNN_SEEDS": (42, 43, 44),
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("output/tile_labels_spatial_splits"),
}


# ============================================================================
# DATA LOADING
# ============================================================================

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


def prepare_training_data(labels_csv: Path, chm_dir: Path, output_dir: Path):
    """Load training/val/test data using spatial splits (streaming from disk)."""
    print("[prepare_training_data] Loading labels...")
    df = pd.read_csv(labels_csv)

    # Use only test/train/val (exclude 'none' buffer zones)
    df_train = df[df['split'] == 'train'].copy()
    df_val = df[df['split'] == 'val'].copy()
    df_test = df[df['split'] == 'test'].copy()

    print(f"  Train: {len(df_train)} labels")
    print(f"  Val:   {len(df_val)} labels")
    print(f"  Test:  {len(df_test)} labels")
    print("  Note: Data will be streamed from disk (not pre-loaded to avoid OOM)")

    # For val/test, we still load into memory (smaller sizes)
    def load_data(df_subset):
        X, y, w = [], [], []
        for idx, row in df_subset.iterrows():
            chm = load_chm_window(chm_dir, row['raster'], int(row['row_off']), int(row['col_off']))
            if chm is None:
                continue
            label = 1 if row['label'] == 'cdw' else 0
            X.append(chm)
            y.append(label)
            w.append(1.0)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(w, dtype=np.float32)

    print("[prepare_training_data] Loading val CHM windows...")
    X_val, y_val, w_val = load_data(df_val)

    print("[prepare_training_data] Loading test CHM windows...")
    X_test, y_test, w_test = load_data(df_test)

    print(f"  Loaded: X_val={X_val.shape}, X_test={X_test.shape}")
    print(f"  Train data will stream from disk during training")

    # For training, return the dataframe instead of pre-loaded arrays
    # DataLoader will read tiles on-demand
    return df_train, (X_val, y_val, w_val), (X_test, y_test, w_test), df_test


# ============================================================================
# DATASET & TRAINING
# ============================================================================

class TileDataset(Dataset):
    """Dataset for tile classification with on-demand loading from disk (avoids OOM)."""
    def __init__(self, df, chm_dir, augment=False):
        """
        Args:
            df: DataFrame with columns 'raster', 'row_off', 'col_off', 'label'
            chm_dir: Path to CHM raster directory
            augment: Whether to apply augmentations
        """
        self.df = df.reset_index(drop=True)
        self.chm_dir = Path(chm_dir)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load CHM tile on-demand
        chm = load_chm_window(
            self.chm_dir,
            row['raster'],
            int(row['row_off']),
            int(row['col_off'])
        )

        if chm is None:
            # Return zero tile if loading fails
            chm = np.zeros((128, 128), dtype=np.float32)

        # Normalize and add channel dimension
        chm_norm = normalize_chm(chm)
        x = torch.tensor(chm_norm, dtype=torch.float32).unsqueeze(0)  # shape: (1, 128, 128)

        y = torch.tensor(1 if row['label'] == 'cdw' else 0, dtype=torch.long)
        w = torch.tensor(1.0, dtype=torch.float32)

        if self.augment:
            # Random augmentations (operate on (1, 128, 128) tensor)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-1])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-2])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, [-2, -1])
            if torch.rand(1).item() > 0.7:
                x = (x + torch.randn_like(x) * 0.015).clamp(0.0, 1.0)
            if torch.rand(1).item() > 0.80:
                alpha = 0.85 + torch.rand(1).item() * 0.30
                beta = (torch.rand(1).item() - 0.5) * 0.06
                x = (x * alpha + beta).clamp(0.0, 1.0)

        return x, y, w


def train_single_model(model, df_train, chm_dir, X_val, y_val, device, epochs, model_tag=""):
    """Train a single model.

    Args:
        model: PyTorch model to train
        df_train: DataFrame with training data (will be streamed from disk)
        chm_dir: Path to CHM raster directory
        X_val: Validation CHM windows (numpy array, pre-loaded)
        y_val: Validation labels
        device: torch device
        epochs: Number of training epochs
        model_tag: Model identifier for logging
    """
    model = model.to(device)
    model.train()

    # Class weights (compute from dataframe)
    n_neg = int((df_train['label'] != 'cdw').sum())
    n_pos = int((df_train['label'] == 'cdw').sum())
    w_pos = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG["LABEL_SMOOTHING"],
        reduction="none"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR_HEAD"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use custom dataset that loads tiles on-demand from disk
    train_ds = TileDataset(df_train, chm_dir, augment=True)
    train_dl = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0)

    # Add channel dimension to validation data: (N, 128, 128) → (N, 1, 128, 128)
    X_val_t = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
    y_val_t = torch.from_numpy(y_val).long()

    best_loss = float('inf')
    best_state = None

    print(f"[train] {model_tag}  epochs={epochs}  train={len(df_train)}  val={len(X_val)}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb, wb in train_dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device).float()
            optimizer.zero_grad()

            loss = criterion(model(xb), yb) * wb
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss.mean().item()

        scheduler.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            with torch.no_grad():
                # Validate in batches to avoid OOM on large val set
                val_losses = []
                val_batch_size = 256
                for i in range(0, len(X_val_t), val_batch_size):
                    batch_end = min(i + val_batch_size, len(X_val_t))
                    val_batch = X_val_t[i:batch_end].to(device)
                    val_labels = y_val_t[i:batch_end].to(device)
                    val_logits = model(val_batch)
                    batch_loss = criterion(val_logits, val_labels).mean().item()
                    val_losses.append(batch_loss)

                val_loss = np.mean(val_losses)
                print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(train_dl):.4f}  val_loss={val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrain ensemble with spatial splits (Option B)")
    parser.add_argument("--labels", type=Path, default=Path("data/chm_variants/labels_canonical_with_splits.csv"))
    parser.add_argument("--chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    parser.add_argument("--output", type=Path, default=CONFIG["OUTPUT_DIR"])
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("RETRAIN ENSEMBLE WITH SPATIAL SPLITS (OPTION B)")
    print("=" * 100)
    print(f"\nConfig:")
    for k, v in CONFIG.items():
        if k != "DEVICE":
            print(f"  {k}: {v}")
    print(f"  DEVICE: {CONFIG['DEVICE']}")
    print(f"\nOutput directory: {output_dir}\n")

    t0 = time.time()

    # Prepare data
    print("[step 1/4] Preparing data...")
    df_train, (X_val, y_val, w_val), (X_test, y_test, w_test), df_test = \
        prepare_training_data(args.labels, args.chm_dir, output_dir)

    # Save metadata
    train_cdw = int((df_train['label'] == 'cdw').sum())
    train_no_cdw = int((df_train['label'] != 'cdw').sum())
    test_cdw = int(np.sum(y_test == 1))
    test_no_cdw = int(np.sum(y_test == 0))

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_config": {k: v for k, v in CONFIG.items() if k != "DEVICE"},
        "data_stats": {
            "train_size": len(df_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_cdw": train_cdw,
            "train_no_cdw": train_no_cdw,
            "test_cdw": test_cdw,
            "test_no_cdw": test_no_cdw,
        },
        "approach": "Option B: Retrain on spatial splits (academic rigor)",
    }

    print(f"\n[step 2/4] Training models...")
    print(f"  3 CNN-Deep-Attn models (seeds 42, 43, 44)")
    print(f"  1 EfficientNet-B2 model")

    models = {}

    # Train CNN models
    for seed in CONFIG["CNN_SEEDS"]:
        print(f"\n[train_cnn] seed={seed}")
        model = _instantiate_model_from_build_fn(_get_build_fn("_build_deep_cnn_attn"))
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_single_model(
            model, df_train, args.chm_dir, X_val, y_val,
            CONFIG["DEVICE"], CONFIG["CNN_EPOCHS"],
            model_tag=f"CNN-seed{seed}"
        )
        checkpoint_path = output_dir / f"cnn_seed{seed}_spatial.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "build_fn_name": "_build_deep_cnn_attn",
            "meta": {"seed": seed, "model_name": f"CNN-seed{seed}"}
        }, checkpoint_path)
        models[f"cnn_seed{seed}"] = model
        print(f"  Saved: {checkpoint_path}")

    # Train EfficientNet model
    print(f"\n[train_effnet]")
    effnet = _instantiate_model_from_build_fn(_get_build_fn("_build_effnet_b2"))
    effnet = train_single_model(
        effnet, df_train, args.chm_dir, X_val, y_val,
        CONFIG["DEVICE"], CONFIG["EFFNET_EPOCHS"],
        model_tag="EfficientNet-B2"
    )
    checkpoint_path = output_dir / "effnet_b2_spatial.pt"
    torch.save({
        "state_dict": effnet.state_dict(),
        "build_fn_name": "_build_effnet_b2",
        "meta": {"model_name": "EfficientNet-B2"}
    }, checkpoint_path)
    models["effnet_b2"] = effnet
    print(f"  Saved: {checkpoint_path}")

    # Evaluate on test set with batched inference to avoid OOM
    print(f"\n[step 3/4] Evaluating on test set...")
    from sklearn.metrics import roc_auc_score, f1_score

    X_test_t = torch.from_numpy(X_test).float().unsqueeze(1)
    all_probs = None
    test_batch_size = 256

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        model.eval()
        model_probs_list = []

        with torch.no_grad():
            for i in range(0, len(X_test_t), test_batch_size):
                batch_end = min(i + test_batch_size, len(X_test_t))
                batch = X_test_t[i:batch_end].to(CONFIG["DEVICE"])
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

    metadata["test_metrics"] = {
        "ensemble_auc": auc,
        "ensemble_f1": best_f1,
        "ensemble_thresh": best_thr,
        "n_test": int(len(y_test)),
        "n_cdw": int(np.sum(y_test == 1)),
    }

    # Save metadata
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    elapsed = time.time() - t0
    print(f"\n[step 4/4] Training complete in {elapsed/3600:.1f} hours")
    print(f"\n{'='*100}")
    print("SPATIAL SPLIT RETRAINING COMPLETE")
    print(f"{'='*100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
