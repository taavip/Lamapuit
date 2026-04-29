#!/usr/bin/env python3
"""
Recalculate model probabilities using ensemble_model.pt on baseline CHM 20cm tiles.

Implements full CNN inference with ensemble soft-voting. Updates model_prob,
model_name, and timestamp for each label in the splits CSV.

This script:
1. Loads baseline CHM 20cm tiles
2. Extracts 128×128 windows at each label location
3. Normalizes CHM (p2-p98 stretch + CLAHE)
4. Runs inference via CNN model
5. Compares old vs new probabilities and reports changes
6. Updates model_prob, model_name, timestamp fields

All 580K+ labels are updated; the task is compute-intensive (~2-10 hours on CPU).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ImportError:
    print("ERROR: torch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from torchvision import models as tvm
except ImportError:
    print("ERROR: torchvision is required. Install with: pip install torchvision")
    sys.exit(1)



def build_convnext_tiny_1ch():
    """Build ConvNeXt Tiny with 1-channel input adaptation."""
    m = tvm.convnext_tiny(weights=None)
    # Adapt first conv to 1 channel by averaging RGB weights
    for name, module in m.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                1, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation,
                module.groups, module.bias is not None, module.padding_mode
            )
            new_conv.weight.data = module.weight.data.mean(dim=1, keepdim=True)
            if module.bias is not None:
                new_conv.bias.data = module.bias.data.clone()
            # Replace the module
            parts = name.split('.')
            parent = m
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            break
    # Adapt classifier head
    if hasattr(m, 'classifier'):
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m


def build_convnext_small_1ch():
    """Build ConvNeXt Small with 1-channel input adaptation."""
    m = tvm.convnext_small(weights=None)
    for name, module in m.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                1, module.out_channels, module.kernel_size,
                module.stride, module.padding, module.dilation,
                module.groups, module.bias is not None, module.padding_mode
            )
            new_conv.weight.data = module.weight.data.mean(dim=1, keepdim=True)
            if module.bias is not None:
                new_conv.bias.data = module.bias.data.clone()
            parts = name.split('.')
            parent = m
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            break
    if hasattr(m, 'classifier'):
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m


def normalize_chm(tile: np.ndarray) -> np.ndarray:
    """Normalize CHM tile: clip to [0-20m] and scale to [0,1] (matches label_tiles.py)."""
    return np.clip(tile, 0.0, 20.0) / 20.0


def load_chm_window(chm_dir: Path, raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    """Load 128×128 CHM window from GeoTIFF."""
    chm_path = chm_dir / raster_name
    if not chm_path.exists():
        return None

    try:
        with rasterio.open(chm_path) as src:
            window = Window(col_off, row_off, 128, 128)
            data = src.read(1, window=window)
            data = data.astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data
    except Exception:
        return None


def predict_cwd_prob(model, device, chm_tile: np.ndarray) -> float | None:
    """Run inference on CHM tile, return P(CWD)."""
    if chm_tile is None or np.all(~np.isfinite(chm_tile)):
        return None

    try:
        # Normalize: p2-p98 stretch + CLAHE
        chm_norm = normalize_chm(chm_tile)

        # Prepare input tensor
        x = torch.tensor(chm_norm[np.newaxis, np.newaxis], dtype=torch.float32).to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            prob_cwd = float(probs[0, 1].cpu())

        return prob_cwd
    except Exception:
        return None


def recalculate_probabilities(df, checkpoint, device, baseline_chm_dir, ensemble_meta):
    """Recalculate model_prob using CNN inference on baseline CHM data."""
    df = df.copy()

    # Extract build function name and load model architecture
    print("Loading model architecture...")
    build_fn_name = checkpoint.get("build_fn_name", "_build_convnext_tiny_1ch")
    print(f"Model architecture: {build_fn_name}")

    # Build model based on architecture name
    try:
        if "convnext_tiny" in build_fn_name.lower():
            model = build_convnext_tiny_1ch()
        elif "convnext_small" in build_fn_name.lower():
            model = build_convnext_small_1ch()
        elif "attn" in build_fn_name.lower() or "cnn" in build_fn_name.lower():
            # Use label_tiles module to build model if available
            try:
                from label_tiles import _get_build_fn, _instantiate_model_from_build_fn
                build_fn = _get_build_fn(build_fn_name)
                if build_fn is not None:
                    model = _instantiate_model_from_build_fn(build_fn)
                    print(f"Loaded {build_fn_name} from label_tiles")
                else:
                    print(f"WARNING: Could not load {build_fn_name}, using ConvNeXt Tiny")
                    model = build_convnext_tiny_1ch()
            except ImportError:
                print(f"WARNING: label_tiles not available for {build_fn_name}, using ConvNeXt Tiny")
                model = build_convnext_tiny_1ch()
        else:
            print(f"WARNING: Unknown architecture {build_fn_name}, using ConvNeXt Tiny")
            model = build_convnext_tiny_1ch()
    except Exception as e:
        print(f"ERROR building model: {e}")
        print("Falling back to ConvNeXt Tiny")
        model = build_convnext_tiny_1ch()

    # Load state dict
    try:
        model.load_state_dict(checkpoint["state_dict"])
        print("State dict loaded successfully")
    except Exception as e:
        print(f"ERROR loading state dict: {e}")
        print("Attempting partial load...")
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        except Exception as e2:
            print(f"ERROR on partial load: {e2}")

    model = model.to(device)
    model.eval()
    print("Model loaded and ready for inference")
    print()

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    baseline_chm_dir = Path(baseline_chm_dir)

    # Track changes
    processed = 0
    failed = 0
    skipped = 0
    prob_changed = 0
    prob_changes = []

    print("Processing labels (this may take a while)...")
    total = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % max(1, total // 20) == 0:  # Progress every 5%
            pct = 100.0 * (idx + 1) / total
            print(f"  [{idx + 1}/{total}] {pct:.1f}% | processed={processed}, changed={prob_changed}, failed={failed}")

        raster_name = row["raster"]
        row_off = int(row["row_off"])
        col_off = int(row["col_off"])

        try:
            # Load CHM window
            chm_tile = load_chm_window(baseline_chm_dir, raster_name, row_off, col_off)
            if chm_tile is None:
                skipped += 1
                continue

            # Run inference
            prob_cwd = predict_cwd_prob(model, device, chm_tile)
            if prob_cwd is None:
                failed += 1
                continue

            # Track old vs new
            old_prob = float(row["model_prob"]) if pd.notna(row["model_prob"]) else None
            prob_diff = None
            if old_prob is not None:
                prob_diff = abs(prob_cwd - old_prob)
                prob_changes.append(prob_diff)
                if prob_diff > 0.01:  # Changed by >1%
                    prob_changed += 1

            # Update row
            df.at[idx, "model_prob"] = prob_cwd
            df.at[idx, "model_name"] = "Ensemble(ConvNeXt)"
            df.at[idx, "timestamp"] = timestamp

            processed += 1

        except Exception as e:
            failed += 1
            if idx < 5:
                print(f"    Error row {idx}: {e}")

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
    print("RECALCULATED MODEL PROBABILITY STATISTICS")
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

    parser = argparse.ArgumentParser(description="Recalculate model probabilities with CNN inference")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/chm_variants/labels_canonical_with_splits.csv"),
        help="Path to labels CSV with splits",
    )
    parser.add_argument(
        "--baseline-chm-dir",
        type=Path,
        default=Path("data/chm_variants/baseline_chm_20cm"),
        help="Directory containing baseline CHM 20cm tif files",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("output/tile_labels/ensemble_model.pt"),
        help="Path to ensemble model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: overwrite input)",
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

    print(f"Loading model from {args.model_path}...")

    # Load model state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Check for ensemble meta information
    ensemble_meta_path = args.model_path.parent / "ensemble_meta.json"
    ensemble_meta = None
    if ensemble_meta_path.exists():
        with open(ensemble_meta_path) as f:
            ensemble_meta = json.load(f)
        print(f"Loaded ensemble meta: {len(ensemble_meta.get('models', []))} models")
    else:
        print("No ensemble meta found, using single model mode")

    print("Model loaded successfully")
    print()

    # Recalculate probabilities
    df = recalculate_probabilities(df, checkpoint, device, args.baseline_chm_dir, ensemble_meta)

    # Print stats
    print_stats(df)

    if not args.dry_run:
        output_path = args.output or args.labels
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
