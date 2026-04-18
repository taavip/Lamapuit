import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import glob


def analyze_all_geotiffs_nodata(directory, pattern="*.tif"):
    files = sorted(Path(directory).glob(pattern))
    nodata_percents = []
    names = []
    for tif_path in files:
        try:
            with rasterio.open(tif_path) as src:
                arr = src.read(1)
                nodata_val = src.nodata
                if nodata_val is None:
                    nodata_val = np.nan
                nodata_mask = np.isnan(arr) if np.isnan(nodata_val) else (arr == nodata_val)
                nodata_pct = 100 * np.sum(nodata_mask) / arr.size
                nodata_percents.append(nodata_pct)
                names.append(tif_path.name)
                print(
                    f"{tif_path.name}: {nodata_pct:.2f}% nodata ({np.sum(nodata_mask)}/{arr.size})"
                )
        except Exception as e:
            print(f"[SKIP] {tif_path.name}: {e}")
    # Plot histogram of nodata percentages
    plt.figure(figsize=(7, 4))
    plt.hist(nodata_percents, bins=20, color="purple", alpha=0.7)
    plt.xlabel("% nodata pixels per file")
    plt.ylabel("Number of rasters")
    plt.title("Histogram of nodata % across rasters")
    plt.tight_layout()
    plt.show()
    # Optionally, print a table
    print("\nSummary table:")
    for n, pct in zip(names, nodata_percents):
        print(f"{n:40s}  {pct:6.2f}% nodata")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_nodata_hist_all.py <directory> [pattern]")
        sys.exit(1)
    directory = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*.tif"
    analyze_all_geotiffs_nodata(directory, pattern)
