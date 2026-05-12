#!/usr/bin/env python3
"""
Recalculate model probabilities using the exact ensemble TTA (Test Time Augmentation) logic.

This script reproduces the EXACT probabilities from train_ensemble.py by:
1. Loading all 4 models (cnn_seed42, cnn_seed43, cnn_seed44, effnet_b2)
2. For each label, performing 8x TTA (4 rotations × 2 flips per rotation)
3. Soft-voting probabilities across all 4 models
4. Normalizing with correct CHM formula: clip(raw, 0, 20) / 20

This matches the original ensemble probability generation exactly.

Usage:
  python scripts/recalculate_model_probs_tta_ensemble.py \\
    --labels data/chm_variants/labels_canonical_with_splits.csv \\
    --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \\
    --output data/chm_variants/labels_canonical_with_splits_tta_ensemble.csv

  python scripts/recalculate_model_probs_tta_ensemble.py \\
    --labels data/chm_variants/labels_canonical_with_splits.csv \\
    --baseline-chm-dir data/lamapuit/chm_max_hag_13_drop \\
    --model-dir output/tile_labels_spatial_splits \\
    --model-checkpoints cnn_seed42_spatial.pt cnn_seed43_spatial.pt cnn_seed44_spatial.pt effnet_b2_spatial.pt \\
    --model-name "Ensemble(4-spatial-TTA)" \\
    --output data/chm_variants/labels_canonical_with_splits_spatial_ensemble.csv
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window

# Import model builders from label_tiles
sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _get_build_fn, _instantiate_model_from_build_fn


def load_chm_window(chm_dir: Path, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    """Load 128×128 CHM window from GeoTIFF."""
    chm_path = chm_dir / raster_name
    if not chm_path.exists():
        return None

    try:
        with rasterio.open(chm_path) as src:
            return load_chm_window_from_src(src, row_off, col_off)
    except Exception:
        return None


def load_chm_window_from_src(src, row_off: int, col_off: int) -> np.ndarray:
    """Load a 128x128 CHM window from an already opened rasterio dataset."""
    window = Window(col_off, row_off, 128, 128)
    data = src.read(1, window=window).astype(np.float32)
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    return np.nan_to_num(data, nan=0.0)


def normalize_chm(tile: np.ndarray) -> np.ndarray:
    """Normalize CHM tile: clip to [0-20m] and scale to [0,1] (matches train_ensemble.py)."""
    return np.clip(tile, 0.0, 20.0) / 20.0


def load_model(checkpoint_path: Path, device) -> nn.Module | None:
    """Load a single model checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        build_fn_name = ckpt.get("build_fn_name", "_build_deep_cnn_attn")
        build_fn = _get_build_fn(build_fn_name)
        if build_fn is None:
            return None
        model = _instantiate_model_from_build_fn(build_fn).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR loading {checkpoint_path}: {e}")
        return None


def predict_proba_cdw_tta(model: nn.Module, device, chm_tile: np.ndarray) -> float | None:
    """
    Compute P(CDW) using 8x TTA (4 rotations × 2 flips per rotation).

    Exactly matches the TTA logic from train_ensemble.py lines 285-298:
    - For each of 4 rotations (0°, 90°, 180°, 270°)
    - For each rotation, compute both original and horizontally flipped
    - Average all 8 predictions
    """
    if chm_tile is None or np.all(~np.isfinite(chm_tile)):
        return None

    try:
        # Normalize CHM
        chm_norm = normalize_chm(chm_tile)
        x = torch.tensor(chm_norm[np.newaxis, np.newaxis], dtype=torch.float32).to(device)

        with torch.no_grad():
            views = []
            # 4 rotations
            for k in range(4):
                v = torch.rot90(x, k, [-2, -1])
                # Original rotation
                views.append(torch.softmax(model(v), dim=1)[0, 1].item())
                # Horizontal flip
                views.append(torch.softmax(model(torch.flip(v, [-1])), dim=1)[0, 1].item())

            # Average 8 views
            prob_cwd = float(np.mean(views))
            return prob_cwd
    except Exception:
        return None


def predict_proba_cdw_tta_batch(model: nn.Module, device, chm_tiles: list[np.ndarray]) -> np.ndarray:
    """
    Compute P(CDW) for a batch using 8x TTA.

    This uses the same views as predict_proba_cdw_tta, but runs them in batches
    so full-dataset inference can use the GPU efficiently.
    """
    batch = np.stack([normalize_chm(tile) for tile in chm_tiles]).astype(np.float32)
    x = torch.from_numpy(batch[:, np.newaxis]).to(device)

    with torch.no_grad():
        views = []
        for k in range(4):
            v = torch.rot90(x, k, [-2, -1])
            views.append(torch.softmax(model(v), dim=1)[:, 1])
            views.append(torch.softmax(model(torch.flip(v, [-1])), dim=1)[:, 1])

        probs = torch.stack(views, dim=0).mean(dim=0)
        return probs.detach().cpu().numpy()


def recalculate_probabilities_tta_ensemble(
    df,
    baseline_chm_dir,
    device,
    model_specs=None,
    model_name: str = "Ensemble(4-TTA)",
    batch_size: int = 256,
):
    """Recalculate model_prob using exact TTA + soft-voting ensemble."""
    df = df.copy()

    # Load all 4 models
    models_to_load = model_specs or [
        ("cnn_seed42", Path("output/tile_labels/cnn_seed42.pt")),
        ("cnn_seed43", Path("output/tile_labels/cnn_seed43.pt")),
        ("cnn_seed44", Path("output/tile_labels/cnn_seed44.pt")),
        ("effnet_b2", Path("output/tile_labels/effnet_b2.pt")),
    ]

    print("Loading 4-model ensemble with TTA...")
    models = {}
    for name, path in models_to_load:
        print(f"  Loading {name}...", end=" ", flush=True)
        model = load_model(path, device)
        if model is None:
            print("FAILED")
            return df
        models[name] = model
        print("OK")

    print(f"✓ All 4 models loaded successfully\n")

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    baseline_chm_dir = Path(baseline_chm_dir)

    # Track changes
    processed = 0
    failed = 0
    skipped = 0
    prob_changed = 0
    prob_changes = []

    print("Processing labels with 8x TTA + soft-voting (this will take a while)...", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    total = len(df)

    def process_batch(batch_indices, batch_tiles, batch_old_probs):
        nonlocal processed, failed, prob_changed
        if not batch_indices:
            return

        try:
            ensemble_probs = np.zeros(len(batch_tiles), dtype=np.float32)
            for model in models.values():
                ensemble_probs += predict_proba_cdw_tta_batch(model, device, batch_tiles)
            ensemble_probs /= len(models)
        except Exception as e:
            raise RuntimeError(f"Failed to process batch of {len(batch_indices)} rows") from e

        for row_idx, prob_cwd, old_prob in zip(batch_indices, ensemble_probs, batch_old_probs):
            prob_cwd = float(prob_cwd)
            if old_prob is not None:
                prob_diff = abs(prob_cwd - old_prob)
                prob_changes.append(prob_diff)
                if prob_diff > 0.01:
                    prob_changed += 1

            df.at[row_idx, "model_prob"] = prob_cwd
            df.at[row_idx, "model_name"] = model_name
            df.at[row_idx, "timestamp"] = timestamp
            processed += 1

    batch_indices = []
    batch_tiles = []
    batch_old_probs = []
    n_seen = 0
    progress_step = max(1, total // 20)

    for raster_name, raster_df in df.groupby("raster", sort=False):
        chm_path = baseline_chm_dir / raster_name
        if not chm_path.exists():
            raise FileNotFoundError(f"CHM raster not found: {chm_path}")

        with rasterio.open(chm_path) as src:
            for idx, row in raster_df.iterrows():
                n_seen += 1
                if n_seen % progress_step == 0:  # Progress every 5%
                    pct = 100.0 * n_seen / total
                    print(
                        f"  [{n_seen}/{total}] {pct:.1f}% | processed={processed}, "
                        f"changed={prob_changed}, failed={failed}",
                        flush=True,
                    )

                try:
                    chm_tile = load_chm_window_from_src(src, int(row["row_off"]), int(row["col_off"]))
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"    Error row {idx}: {e}", flush=True)
                    continue

                old_prob = float(row["model_prob"]) if pd.notna(row["model_prob"]) else None

                batch_indices.append(idx)
                batch_tiles.append(chm_tile)
                batch_old_probs.append(old_prob)

                if len(batch_indices) >= batch_size:
                    process_batch(batch_indices, batch_tiles, batch_old_probs)
                    batch_indices = []
                    batch_tiles = []
                    batch_old_probs = []

    process_batch(batch_indices, batch_tiles, batch_old_probs)

    print(f"\nProcessing complete:")
    print(f"  Total processed: {processed:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Probabilities changed (>1%): {prob_changed:,}")

    if prob_changes:
        prob_changes_arr = np.array(prob_changes)
        print(f"\nProbability change statistics:")
        print(f"  Mean change: {prob_changes_arr.mean():.4f}")
        print(f"  Std dev: {prob_changes_arr.std():.4f}")
        print(f"  Min: {prob_changes_arr.min():.4f}")
        print(f"  Max: {prob_changes_arr.max():.4f}")
        print(f"  Median: {np.median(prob_changes_arr):.4f}")

    return df


def print_stats(df):
    """Print statistics of recalculated probabilities."""
    print("\n" + "=" * 75)
    print("RECALCULATED MODEL PROBABILITY STATISTICS (TTA ENSEMBLE)")
    print("=" * 75)
    print()

    print("Overall statistics:")
    print(f"  Mean prob: {df['model_prob'].mean():.4f}")
    print(f"  Std dev:   {df['model_prob'].std():.4f}")
    print(f"  Min:       {df['model_prob'].min():.4f}")
    print(f"  Max:       {df['model_prob'].max():.4f}")
    print(f"  Median:    {df['model_prob'].median():.4f}")
    print()

    # By class
    print("By class label:")
    for label in ["cdw", "no_cdw"]:
        subset = df[df["label"] == label]
        if len(subset) > 0:
            print(f"\n  {label.upper()}:")
            print(f"    Count:     {len(subset):,}")
            print(f"    Mean prob: {subset['model_prob'].mean():.4f}")
            print(f"    Std dev:   {subset['model_prob'].std():.4f}")

    # By split
    print()
    print("By split:")
    for split in ["test", "val", "train", "none"]:
        subset = df[df["split"] == split]
        if len(subset) > 0:
            print(f"\n  {split.upper()}:")
            print(f"    Count:     {len(subset):,}")
            print(f"    Mean prob: {subset['model_prob'].mean():.4f}")
            print(f"    Std dev:   {subset['model_prob'].std():.4f}")

    print()
    print("=" * 75 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Recalculate model probabilities with exact TTA ensemble from train_ensemble.py"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits.csv"),
        help="Path to labels CSV",
    )
    parser.add_argument(
        "--baseline-chm-dir",
        type=Path,
        default=Path("data/lamapuit/chm_max_hag_13_drop"),
        help="Directory containing baseline CHM 20cm tif files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("output/tile_labels"),
        help="Directory containing ensemble checkpoint files",
    )
    parser.add_argument(
        "--model-checkpoints",
        nargs="+",
        default=["cnn_seed42.pt", "cnn_seed43.pt", "cnn_seed44.pt", "effnet_b2.pt"],
        help="Checkpoint file names or paths to load for the soft-vote ensemble",
    )
    parser.add_argument(
        "--model-name",
        default="Ensemble(4-TTA)",
        help="Value written to the output CSV model_name column",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of CHM tiles to evaluate per GPU batch",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    parser.add_argument(
        "--sample",
        type=int,
        help="Process only a sample of N labels (for testing)",
    )
    args = parser.parse_args()

    print(f"Loading labels from {args.labels}...")
    df = pd.read_csv(args.labels)
    print(f"Loaded {len(df)} labels")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42)
        print(f"Sampled {len(df)} labels for testing")

    print()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model_specs = []
    for checkpoint in args.model_checkpoints:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = args.model_dir / path
        model_specs.append((path.stem, path))

    print("Model checkpoints:")
    for name, path in model_specs:
        print(f"  {name}: {path}")
    print()

    # Recalculate probabilities with TTA
    df = recalculate_probabilities_tta_ensemble(
        df,
        args.baseline_chm_dir,
        device,
        model_specs=model_specs,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    # Print stats
    print_stats(df)

    if not args.dry_run:
        output_path = args.output or args.labels
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
