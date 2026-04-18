import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# Usage: python calc_nodata_hist.py chm_max_hag/441643_2023_tava_chm_max_hag_20cm.tif [more.tif ...]
def analyze_geotiff_nodata(files):
    for tif_path in files:
        tif_path = Path(tif_path)
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            nodata_val = src.nodata
            if nodata_val is None:
                nodata_val = np.nan
            nodata_mask = np.isnan(arr) if np.isnan(nodata_val) else (arr == nodata_val)
            nodata_pct = 100 * np.sum(nodata_mask) / arr.size
            print(f"{tif_path.name}: {nodata_pct:.2f}% nodata ({np.sum(nodata_mask)}/{arr.size})")
            # Plot histogram of valid pixels
            valid = arr[~nodata_mask]
            plt.figure(figsize=(6, 3))
            plt.hist(valid.ravel(), bins=100, color="blue", alpha=0.7)
            plt.title(f"{tif_path.name} (nodata: {nodata_pct:.1f}%)")
            plt.xlabel("Value")
            plt.ylabel("Pixel count")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc_nodata_hist.py file1.tif [file2.tif ...]")
        sys.exit(1)
    analyze_geotiff_nodata(sys.argv[1:])
