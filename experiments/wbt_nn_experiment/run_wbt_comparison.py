import os
import subprocess
from pathlib import Path
import numpy as np
import rasterio
import laspy
import json
import traceback

def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)

def generate_nn_dtm(laz_file, class_str, out_dem, res=0.2):
    tmp_shp = out_dem.with_suffix('.shp')
    
    # Extract only the required class and voxel downsize it, then write as shapefile
    pipeline = [
        str(laz_file),
        {"type": "filters.range", "limits": f"Classification[{class_str}]"},
        {"type": "filters.voxeldownsize", "cell": res},
        {"type": "writers.ogr", "filename": str(tmp_shp)}
    ]
    pdal_conf = out_dem.with_suffix('.json')
    pdal_conf.write_text(json.dumps({"pipeline": pipeline}))
    run(["pdal", "pipeline", str(pdal_conf)])
    
    # Whitebox Natural Neighbour Interpolation
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    print(f"Running WBT Natural Neighbour on {tmp_shp}...")
    wbt.natural_neighbour_interpolation(i=str(tmp_shp), output=str(out_dem), use_z=True, cell_size=res)
    
    # Cleanup intermediate files
    for ext in ['.shp', '.dbf', '.shx', '.prj', '.json']:
        p = out_dem.with_suffix(ext)
        if p.exists(): 
            p.unlink()
    print(f"Generated DTM: {out_dem}")

def generate_chm_from_dtm_and_laz(laz_file, dtm_file, out_chm, hag_max=1.3):
    print(f"Generating CHM from {laz_file} and {dtm_file} ...")
    with rasterio.open(dtm_file) as src:
        dtm_arr = src.read(1, masked=True)
        # Convert infinite/nan to a masked array safely
        if type(dtm_arr) == np.ma.core.MaskedArray:
            dtm_arr = dtm_arr.filled(np.nan)
        
        width = src.width
        height = src.height
        ox = src.transform.c
        maxy = src.transform.f
        res = abs(src.transform.a)
        transform = src.transform
        crs = src.crs

    chm_arr = np.full((height, width), -9999.0, dtype=np.float32)
    
    with laspy.open(str(laz_file)) as fh:
        # We need the max Z over the grid, we chunk it to handle memory
        for pts_chunk in fh.chunk_iterator(2_000_000):
            x = np.asarray(pts_chunk.x, dtype=np.float64)
            y = np.asarray(pts_chunk.y, dtype=np.float64)
            z = np.asarray(pts_chunk.z, dtype=np.float32)
            c = np.asarray(pts_chunk.classification)
            
            # Optionally filter to unclassified / ground / veg if we want, but usually CHM takes max of all points (or veg points)
            # We'll take everything for the max DSM.
            if x.size == 0: continue
            
            c_idx = ((x - ox) / res).astype(np.int32)
            r_idx = ((maxy - y) / res).astype(np.int32)
            
            valid = (r_idx >= 0) & (r_idx < height) & (c_idx >= 0) & (c_idx < width)
            if not np.any(valid): continue
            
            x = x[valid]
            y = y[valid]
            z = z[valid]
            r = r_idx[valid]
            cn = c_idx[valid]
            
            flat = r * width + cn
            dtm_z = dtm_arr[r, cn]
            
            valid_dtm = np.isfinite(dtm_z)
            if not np.any(valid_dtm): continue
            
            # HAG = Z - DTM
            hag = z[valid_dtm] - dtm_z[valid_dtm]
            
            # Keeping the same bounds as previous experiments
            # Drop everything below -2.0m (invalid bounces)
            # Clip the ones between -2.0 and 0.0 to 0.0
            keep = (hag >= -2.0)
            if not np.any(keep): continue
            
            v_flat = flat[valid_dtm][keep]
            v_hag = np.clip(hag[keep], 0.0, None).astype(np.float32)
            
            # Compute max HAG per pixel
            np.maximum.at(chm_arr.ravel(), v_flat, v_hag)
            
    # Apply max drop for CHM (usually ~40m, but here we can just keep the actual max or clip to a specific tree drop max)
    # The previous code sometimes filtered hag <= hag_max. We'll stick to a reasonable max height or leave it.
    
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "transform": transform, "crs": crs,
        "nodata": -9999.0, "tiled": True, "compress": "lzw"
    }
    with rasterio.open(out_chm, 'w', **profile) as dst:
        dst.write(chm_arr, 1)
        
    print(f"Generated CHM: {out_chm}")

def compare_deltas(chm_orig, chm_randla, out_csv):
    print("Comparing deltas...")
    with rasterio.open(chm_orig) as src1:
        o = src1.read(1)
        nodata1 = src1.nodata
    with rasterio.open(chm_randla) as src2:
        r = src2.read(1)
        nodata2 = src2.nodata
    
    valid = np.isfinite(o) & np.isfinite(r) & (o != nodata1) & (r != nodata2)
    o_val = o[valid]
    r_val = r[valid]
    
    diff = r_val - o_val
    stats = {
        "pixels_compared": int(diff.size),
        "mean_delta_m": float(np.mean(diff)),
        "std_delta_m": float(np.std(diff)),
        "rmse_m": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff_m": float(np.max(np.abs(diff))),
        "p95_abs_diff_m": float(np.percentile(np.abs(diff), 95)),
        "p99_abs_diff_m": float(np.percentile(np.abs(diff), 99))
    }
    
    print(json.dumps(stats, indent=2))
    out_csv.write_text(json.dumps(stats, indent=2))
    print(f"Stats saved to {out_csv}")

if __name__ == "__main__":
    out_dir = Path("experiments/wbt_nn_experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    orig_laz = Path("data/lamapuit/laz/436646_2018_madal.laz")
    randla_laz = Path("output/laz_reclassified_randla/436646_2018_madal_reclassified_randla.laz")
    
    # 1. DTM generation using Ground points (Class 2) and Whitebox NN (0.2m pixel size)
    orig_dtm = out_dir / "2018_orig_nn_0.2m_dtm.tif"
    randla_dtm = out_dir / "2018_randla_nn_0.2m_dtm.tif"
    
    try:
        generate_nn_dtm(orig_laz, "2:2", orig_dtm)
        generate_nn_dtm(randla_laz, "2:2", randla_dtm)
        
        # 2. CHM generation using the standard max heights - DTM
        orig_chm = out_dir / "2018_orig_nn_0.2m_chm.tif"
        randla_chm = out_dir / "2018_randla_nn_0.2m_chm.tif"
        
        generate_chm_from_dtm_and_laz(orig_laz, orig_dtm, orig_chm)
        generate_chm_from_dtm_and_laz(randla_laz, randla_dtm, randla_chm)
        
        # 3. Compare deltas
        compare_deltas(orig_chm, randla_chm, out_dir / "chm_delta_stats.json")
    except Exception as e:
        traceback.print_exc()

