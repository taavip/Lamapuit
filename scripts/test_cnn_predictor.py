#!/usr/bin/env python3
"""Test CNNPredictor directly to verify correct inference."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# Import CNNPredictor from label_tiles
sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import CNNPredictor


def main():
    # Load a few sample labels
    csv_path = Path("data/chm_variants/labels_canonical_with_splits.csv")
    df = pd.read_csv(csv_path)

    # Get 5 samples with real data
    samples = df[df['model_prob'].notna()].head(5)

    print("Testing CNNPredictor with correct normalization...\n")

    # Create predictor and load model
    pred = CNNPredictor()
    ok = pred.load_from_disk(Path("output/tile_labels/ensemble_model.pt"))
    if not ok:
        print("ERROR: Failed to load model")
        return 1

    chm_dir = Path("data/lamapuit/chm_max_hag_13_drop")

    for idx, (_, row) in enumerate(samples.iterrows()):
        raster_name = row['raster']
        row_off = int(row['row_off'])
        col_off = int(row['col_off'])
        original_prob = row['model_prob']

        # Load CHM window
        chm_path = chm_dir / raster_name
        if not chm_path.exists():
            print(f"[{idx}] SKIP {raster_name} (file not found)")
            continue

        try:
            with rasterio.open(chm_path) as src:
                window = Window(col_off, row_off, 128, 128)
                data = src.read(1, window=window).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan

            # Get prediction via CNNPredictor
            prob = pred.predict_proba_cdw(data)

            if prob is None:
                print(f"[{idx}] {raster_name} @ ({row_off},{col_off})")
                print(f"       Original: {original_prob:.6f}")
                print(f"       Predicted: None (FAILED)")
            else:
                diff = abs(prob - original_prob)
                print(f"[{idx}] {raster_name} @ ({row_off},{col_off})")
                print(f"       Original:  {original_prob:.6f}")
                print(f"       Predicted: {prob:.6f}")
                print(f"       Diff:      {diff:.6f}")

                if diff < 0.01:
                    print(f"       ✅ MATCH")
                else:
                    print(f"       ⚠️  DIFFERS")
            print()

        except Exception as e:
            print(f"[{idx}] ERROR: {e}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
