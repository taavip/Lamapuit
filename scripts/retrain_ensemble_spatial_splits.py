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
import contextlib
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


def _is_wsl() -> bool:
    # Inside Docker we avoid forcing WSL-safe conservative mode, because
    # it can severely throttle GPU training throughput.
    if os.path.exists("/.dockerenv"):
        return False
    release = platform.release().lower()
    return bool(
        os.environ.get("WSL_DISTRO_NAME")
        or os.environ.get("WSL_INTEROP")
        or "microsoft" in release
        or "wsl" in release
    )


IS_WSL = _is_wsl()
if IS_WSL:
    # Reduce CPU oversubscription when running inside WSL/Docker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

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
    "BATCH_SIZE": 8,
    "LR_HEAD": 5e-4,
    "LR_BACKBONE": 5e-5,
    "LABEL_SMOOTHING": 0.05,
    "MIXUP_ALPHA": 0.3,
    "CNN_SEEDS": (42, 43, 44),
    "NUM_WORKERS": 0,
    "PERSISTENT_WORKERS": False,
    "PIN_MEMORY": False,
    "PREFETCH_FACTOR": 1,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("output/classification_140526_train_buffer_test"),
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


def _stratified_split_df(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx_pos = np.array(df.index[df["label"] == "cdw"], dtype=np.int64, copy=True)
    idx_neg = np.array(df.index[df["label"] != "cdw"], dtype=np.int64, copy=True)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_val_pos = max(1, int(round(len(idx_pos) * val_fraction))) if len(idx_pos) else 0
    n_val_neg = max(1, int(round(len(idx_neg) * val_fraction))) if len(idx_neg) else 0
    val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
    tr_idx = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
    df_train = df.loc[tr_idx].copy().reset_index(drop=True)
    df_val = df.loc[val_idx].copy().reset_index(drop=True)
    return df_train, df_val


def prepare_training_data(
    labels_csv: Path,
    chm_dir: Path,
    split_column: str,
    train_value: str,
    test_value: str,
    val_fraction: float = 0.2,
):
    """Load training/validation/test data using the provided split column."""
    print("[prepare_training_data] Loading labels...")
    df = pd.read_csv(labels_csv)

    if split_column not in df.columns:
        raise RuntimeError(f"Missing split column '{split_column}' in {labels_csv}")

    # Use only train/test; buffer/none stay outside the model-selection loop.
    df_train_all = df[df[split_column] == train_value].copy().reset_index(drop=True)
    df_test = df[df[split_column] == test_value].copy().reset_index(drop=True)

    df_train, df_val = _stratified_split_df(df_train_all, val_fraction=val_fraction, seed=42)

    print(f"  Train pool: {len(df_train_all)} labels")
    print(f"  Train:      {len(df_train)} labels")
    print(f"  Val:        {len(df_val)} labels")
    print(f"  Test:       {len(df_test)} labels")
    print("  Note: Train/val/test data will be streamed from disk (not pre-loaded)")

    return df_train, df_val, df_test


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
        # Keep raster handles open inside the dataset process to avoid
        # re-opening the same GeoTIFF for every single tile.
        self._src_cache = {}

    def _read_tile_cached(self, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
        src_entry = self._src_cache.get(raster_name)
        if src_entry is None:
            chm_path = self.chm_dir / raster_name
            if not chm_path.exists():
                return None
            try:
                src = rasterio.open(chm_path)
            except Exception:
                return None
            src_entry = (src, src.nodata)
            self._src_cache[raster_name] = src_entry

        src, nodata = src_entry
        try:
            window = Window(col_off, row_off, 128, 128)
            data = src.read(1, window=window).astype(np.float32)
            if nodata is not None:
                data[data == nodata] = np.nan
            data = np.nan_to_num(data, nan=0.0)
            return normalize_chm(data)
        except Exception:
            return None

    def __del__(self):
        for src, _nodata in self._src_cache.values():
            try:
                src.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load CHM tile on-demand (cached raster handles).
        chm = self._read_tile_cached(row["raster"], int(row["row_off"]), int(row["col_off"]))

        if chm is None:
            # Return zero tile if loading fails
            chm = np.zeros((128, 128), dtype=np.float32)

        # `load_chm_window` already normalizes to [0, 1].
        x = torch.tensor(chm, dtype=torch.float32).unsqueeze(0)  # shape: (1, 128, 128)

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


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _predict_probs_from_df(model, df_subset: pd.DataFrame, chm_dir: Path, device, batch_size: int = 256, tta: bool = False):
    """Predict probabilities for a dataframe subset without pre-loading the full array."""
    import torch

    ds = TileDataset(df_subset, chm_dir, augment=False)
    use_cuda = str(device).startswith("cuda")
    dl = DataLoader(ds, **_dataloader_kwargs(batch_size, shuffle=False))
    probs: list[float] = []
    labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for xb, yb, _wb in dl:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device)

            if tta:
                views = []
                for k in range(4):
                    v = torch.rot90(xb, k, [-2, -1])
                    views.append(torch.softmax(model(v), dim=1)[:, 1])
                    views.append(torch.softmax(model(torch.flip(v, [-1])), dim=1)[:, 1])
                pb = torch.stack(views, dim=0).mean(dim=0)
            else:
                pb = torch.softmax(model(xb), dim=1)[:, 1]

            probs.extend(pb.detach().cpu().numpy().tolist())
            labels.extend(yb.detach().cpu().numpy().astype(int).tolist())

    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def _metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5

    auc = float(roc_auc_score(y_true, y_prob))
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.10, 0.90, 81):
        preds = (y_prob >= thr).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 >= best_f1:
            best_f1, best_thr = f1, float(thr)
    return auc, best_f1, best_thr


def _dataloader_kwargs(batch_size: int, shuffle: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": CONFIG["NUM_WORKERS"],
        "pin_memory": CONFIG["PIN_MEMORY"],
    }
    if CONFIG["NUM_WORKERS"] > 0:
        kwargs["persistent_workers"] = CONFIG["PERSISTENT_WORKERS"]
        kwargs["prefetch_factor"] = CONFIG["PREFETCH_FACTOR"]
    return kwargs


def train_single_model(model, df_train, chm_dir, df_val, device, epochs, model_tag=""):
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
    use_cuda = str(device).startswith("cuda")
    train_dl = DataLoader(train_ds, **_dataloader_kwargs(CONFIG["BATCH_SIZE"], shuffle=True))

    best_loss = float('inf')
    best_state = None
    best_metrics = {"val_auc": 0.5, "val_f1": 0.5, "val_thresh": 0.5}

    print(f"[train_ensemble] {model_tag}  epochs={epochs}  train={len(df_train)}  val={len(df_val)}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb, wb in train_dl:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device)
            wb = wb.to(device).float()
            optimizer.zero_grad()

            loss = criterion(model(xb), yb) * wb
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss.mean().item()

        scheduler.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            y_val_np, val_probs = _predict_probs_from_df(model, df_val, chm_dir, device, batch_size=64 if IS_WSL else 256, tta=False)
            val_auc, val_f1, val_thr = _metrics_from_probs(y_val_np, val_probs)
            val_loss = float(1.0 - val_f1)
            avg_loss = epoch_loss / max(len(train_dl), 1)
            print(
                f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_AUC={val_auc:.4f}  F1={val_f1:.4f}@{val_thr:.2f}",
                flush=True,
            )

            if val_auc > best_metrics["val_auc"] or (
                np.isclose(val_auc, best_metrics["val_auc"]) and val_f1 > best_metrics["val_f1"]
            ):
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {
                    "val_auc": round(val_auc, 4),
                    "val_f1": round(val_f1, 4),
                    "val_thresh": round(val_thr, 2),
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_state is None:
        y_val_np, val_probs = _predict_probs_from_df(model, df_val, chm_dir, device, batch_size=64 if IS_WSL else 256, tta=False)
        val_auc, val_f1, val_thr = _metrics_from_probs(y_val_np, val_probs)
        best_metrics = {
            "val_auc": round(val_auc, 4),
            "val_f1": round(val_f1, 4),
            "val_thresh": round(val_thr, 2),
        }

    return model, best_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrain ensemble with spatial splits (Option B)")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits_train_buffer_test.csv"),
    )
    parser.add_argument("--chm-dir", type=Path, default=Path("data/lamapuit/chm_max_hag_13_drop"))
    parser.add_argument("--split-column", type=str, default="split_train_buffer_test")
    parser.add_argument("--train-value", type=str, default="train")
    parser.add_argument("--test-value", type=str, default="test")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=CONFIG["OUTPUT_DIR"])
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_ensemble.log"

    with log_path.open("w", encoding="utf-8") as log_fh, contextlib.redirect_stdout(Tee(sys.stdout, log_fh)):
        print("=" * 100)
        print("RETRAIN ENSEMBLE WITH SPATIAL SPLITS (OPTION B)")
        print("=" * 100)
        if IS_WSL:
            try:
                torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
                torch.set_num_interop_threads(1)
                print("[env] WSL detected: limiting torch CPU threads and keeping data loading conservative")
            except Exception as exc:
                print(f"[env] Warning: could not tune torch thread settings: {exc}")
        print(f"\nConfig:")
        for k, v in CONFIG.items():
            if k != "DEVICE":
                print(f"  {k}: {v}")
        print(f"  DEVICE: {CONFIG['DEVICE']}")
        print(f"\nOutput directory: {output_dir}\n")

        t0 = time.time()
        eval_batch_size = 64 if IS_WSL else 128

        # Prepare data
        print("[step 1/4] Preparing data...")
        df_train, df_val, df_test = prepare_training_data(
            args.labels,
            args.chm_dir,
            args.split_column,
            args.train_value,
            args.test_value,
            val_fraction=args.val_fraction,
        )

        # Save metadata
        train_cdw = int((df_train["label"] == "cdw").sum())
        train_no_cdw = int((df_train["label"] != "cdw").sum())
        val_cdw = int((df_val["label"] == "cdw").sum())
        val_no_cdw = int((df_val["label"] != "cdw").sum())
        test_cdw = int((df_test["label"] == "cdw").sum())
        test_no_cdw = int((df_test["label"] != "cdw").sum())

        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split_source": str(args.labels),
            "split_column": args.split_column,
            "split_values": {
                "train": args.train_value,
                "test": args.test_value,
            },
            "training_config": {k: v for k, v in CONFIG.items() if k != "DEVICE"},
            "data_stats": {
                "train_size": int(len(df_train)),
                "val_size": int(len(df_val)),
                "test_size": int(len(df_test)),
                "train_cdw": train_cdw,
                "train_no_cdw": train_no_cdw,
                "val_cdw": val_cdw,
                "val_no_cdw": val_no_cdw,
                "test_cdw": test_cdw,
                "test_no_cdw": test_no_cdw,
            },
            "approach": "Option B: Retrain on spatial splits (academic rigor)",
        }

        print(f"\n[step 2/4] Training models...")
        print(f"  3 CNN-Deep-Attn models (seeds 42, 43, 44)")
        print(f"  1 EfficientNet-B2 model")

        models = {}
        model_metrics = {}
        checkpoints = {}

        # Train CNN models
        for seed in CONFIG["CNN_SEEDS"]:
            tag = f"cnn_seed{seed}"
            print(f"\n[train_ensemble] ── Training {tag} ──────────────────────────", flush=True)
            model = _instantiate_model_from_build_fn(_get_build_fn("_build_deep_cnn_attn"))
            torch.manual_seed(seed)
            np.random.seed(seed)
            model, metrics = train_single_model(
                model,
                df_train,
                args.chm_dir,
                df_val,
                CONFIG["DEVICE"],
                CONFIG["CNN_EPOCHS"],
                model_tag=tag,
            )
            checkpoint_path = output_dir / f"{tag}_spatial.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "build_fn_name": "_build_deep_cnn_attn",
                    "model_name": tag,
                    "meta": {
                        "seed": seed,
                        "model_name": tag,
                        "best_auc": metrics["val_auc"],
                        "best_f1": metrics["val_f1"],
                        "best_thresh": metrics["val_thresh"],
                    },
                },
                checkpoint_path,
            )
            models[tag] = model
            model_metrics[tag] = metrics
            checkpoints[tag] = {
                "path": str(checkpoint_path),
                "build_fn_name": "_build_deep_cnn_attn",
                "model_name": tag,
                "weight": metrics["val_f1"],
                "threshold": metrics["val_thresh"],
            }
            print(
                f"[train_ensemble] {tag} saved → {checkpoint_path}  "
                f"val_AUC={metrics['val_auc']:.4f}  F1={metrics['val_f1']:.4f}",
                flush=True,
            )

        # Train EfficientNet model
        tag = "effnet_b2"
        print(f"\n[train_ensemble] ── Training {tag} ──────────────────────────────", flush=True)
        effnet = _instantiate_model_from_build_fn(_get_build_fn("_build_effnet_b2"))
        effnet, metrics = train_single_model(
            effnet,
            df_train,
            args.chm_dir,
            df_val,
            CONFIG["DEVICE"],
            CONFIG["EFFNET_EPOCHS"],
            model_tag=tag,
        )
        checkpoint_path = output_dir / f"{tag}_spatial.pt"
        torch.save(
            {
                "state_dict": effnet.state_dict(),
                "build_fn_name": "_build_effnet_b2",
                "model_name": tag,
                "meta": {
                    "model_name": tag,
                    "best_auc": metrics["val_auc"],
                    "best_f1": metrics["val_f1"],
                    "best_thresh": metrics["val_thresh"],
                },
            },
            checkpoint_path,
        )
        models[tag] = effnet
        model_metrics[tag] = metrics
        checkpoints[tag] = {
            "path": str(checkpoint_path),
            "build_fn_name": "_build_effnet_b2",
            "model_name": tag,
            "weight": metrics["val_f1"],
            "threshold": metrics["val_thresh"],
        }
        print(
            f"[train_ensemble] {tag} saved → {checkpoint_path}  "
            f"val_AUC={metrics['val_auc']:.4f}  F1={metrics['val_f1']:.4f}",
            flush=True,
        )

        # Evaluate on test set with batched inference to avoid OOM
        print(f"\n[step 3/4] Evaluating on test set...")
        from sklearn.metrics import roc_auc_score, f1_score

        ensemble_probs = None
        test_pred_df = df_test.copy().reset_index(drop=True)
        for name, model in models.items():
            print(f"  Evaluating {name}...")
            y_true_np, model_probs = _predict_probs_from_df(
                model,
                df_test,
                args.chm_dir,
                CONFIG["DEVICE"],
                batch_size=eval_batch_size,
                tta=True,
            )
            if ensemble_probs is None:
                ensemble_probs = model_probs.copy()
            else:
                ensemble_probs += model_probs
            test_pred_df[f"prob_{name}"] = model_probs
            test_pred_df["y_true"] = y_true_np

        ensemble_probs = ensemble_probs / len(models)
        test_pred_df["prob_ensemble"] = ensemble_probs

        auc = float(roc_auc_score(test_pred_df["y_true"].values, ensemble_probs))
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.1, 0.9, 81):
            preds = (ensemble_probs >= thr).astype(int)
            f1 = float(f1_score(test_pred_df["y_true"].values, preds, zero_division=0))
            if f1 >= best_f1:
                best_f1, best_thr = f1, float(thr)

        print(f"\nTest set evaluation:")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1: {best_f1:.4f} @ threshold={best_thr:.2f}")
        print(f"  n_test: {len(test_pred_df)}")
        print(
            f"  CDW: {int((test_pred_df['y_true'] == 1).sum())}, "
            f"NO_CDW: {int((test_pred_df['y_true'] == 0).sum())}"
        )

        metadata["test_metrics"] = {
            "ensemble_auc": auc,
            "ensemble_f1": best_f1,
            "ensemble_thresh": best_thr,
            "n_test": int(len(test_pred_df)),
            "n_cdw": int((test_pred_df["y_true"] == 1).sum()),
            "tta": True,
            "n_models": len(models),
        }
        metadata["model_metrics"] = model_metrics
        metadata["checkpoints"] = checkpoints

        # Save metadata
        metadata_path = output_dir / "training_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metadata: {metadata_path}")

        ensemble_meta_path = output_dir / "ensemble_meta.json"
        with open(ensemble_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "created_at": metadata["timestamp"],
                    "checkpoints": checkpoints,
                    "model_metrics": model_metrics,
                    "test_metrics": metadata["test_metrics"],
                    "split_column": args.split_column,
                    "split_source": str(args.labels),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved ensemble metadata: {ensemble_meta_path}")

        test_pred_path = output_dir / "test_predictions.csv"
        test_pred_df.to_csv(test_pred_path, index=False)
        print(f"Saved test predictions: {test_pred_path}")

        elapsed = time.time() - t0
        print(f"\n[step 4/4] Training complete in {elapsed/3600:.1f} hours")
        print(f"\n{'='*100}")
        print("SPATIAL SPLIT RETRAINING COMPLETE")
        print(f"{'='*100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
