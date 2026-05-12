#!/usr/bin/env python3
"""Export full-tile probability GeoTIFF for a Phase 6 final model.

This utility creates a per-pixel probability map (float32 GeoTIFF), matching the
style of `phase5_predict_v10` outputs (for example `*_v10_prob.tif`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.raster_io import normalize_bands
from common.sliding_window import sliding_window_predict
from phase2_dataset_v3 import _get_binary_bands, _read_chm_path
from phase3_train_v10 import build_model


def _select_checkpoint(phase6_dir: Path, variant: str) -> Path:
    fold_dir = phase6_dir / variant / "fold0"
    metrics_path = fold_dir / "metrics.json"
    swa_path = fold_dir / "swa_model.pt"
    best_path = fold_dir / "best.pt"

    if not best_path.exists() and not swa_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if bool(metrics.get("use_swa_for_inference", False)) and swa_path.exists():
            return swa_path

    if best_path.exists():
        return best_path
    return swa_path


def _resolve_valid_area_mask(img: np.ndarray, variant: str) -> np.ndarray:
    # Default: all pixels are eligible unless nodata says otherwise.
    valid_area = np.ones(img.shape[1:], dtype=np.float32)

    # Composite carries validity mask in band 4 (0 or 255).
    if variant == "composite" and img.shape[0] >= 4:
        m = img[3]
        thr = 0.5 if float(np.nanmax(m)) <= 1.5 else 200.0
        valid_area = (m > thr).astype(np.float32)
    # Masked variant commonly carries validity in band 2.
    elif variant == "masked" and img.shape[0] >= 2:
        m = img[1]
        thr = 0.5 if float(np.nanmax(m)) <= 1.5 else 200.0
        valid_area = (m > thr).astype(np.float32)
    return valid_area


def infer_full_tile(
    ckpt_path: Path,
    arch: str,
    in_channels: int,
    variant: str,
    chm_tif: Path,
    band_stats: dict,
    binary_bands: list[int],
    device: torch.device,
    patch_size: int = 256,
    stride: int = 192,
    batch_size: int = 8,
    use_tta: bool = True,
) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(arch, in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with rasterio.open(chm_tif) as src:
        img = src.read(list(range(1, src.count + 1))).astype(np.float32)
        nodata = src.nodata

    nodata_mask = np.ones(img.shape[1:], dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        if nodata is not None:
            nodata_mask *= (band != nodata).astype(np.float32)
        nodata_mask *= (band != -9999.0).astype(np.float32)

    for i in range(img.shape[0]):
        img[i][nodata_mask < 0.5] = 0.0

    valid_area_mask = _resolve_valid_area_mask(img, variant=variant)
    img = normalize_bands(img, band_stats, binary_bands=binary_bands)
    img = np.nan_to_num(img, nan=0.0)

    with torch.no_grad():
        prob = sliding_window_predict(
            model=model,
            image=img,
            device=device,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            use_tta=use_tta,
        )

    combined_mask = valid_area_mask * nodata_mask
    return (prob * combined_mask).astype(np.float32)


def write_prob_tif(prob_map: np.ndarray, reference_tif: Path, output_tif: Path) -> None:
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=None,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        height=prob_map.shape[0],
        width=prob_map.shape[1],
    )
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(prob_map[np.newaxis, ...])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Phase 6 full-tile probability map.")
    p.add_argument("--phase6-dir", type=Path, required=True,
                   help="Path like .../phase6_<chain>__6_final_validation")
    p.add_argument("--dataset-dir", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v10_reconstructed")
    p.add_argument("--chm-tif", type=Path, default=None,
                   help="Optional override CHM path. If omitted, variant default is used.")
    p.add_argument("--output-tif", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    phase6_dir = args.phase6_dir.resolve()
    test_metrics_path = phase6_dir / "all_train_metrics_test.json"
    if not test_metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {test_metrics_path}")

    m = json.loads(test_metrics_path.read_text())
    variant = str(m["chm_variant"])
    arch = str(m["arch"])
    in_channels = int(m["in_channels"])

    ckpt_path = _select_checkpoint(phase6_dir, variant=variant)
    band_stats_path = args.dataset_dir / f"band_stats_{variant}.json"
    if not band_stats_path.exists():
        raise FileNotFoundError(f"Missing band stats: {band_stats_path}")
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(variant)
    chm_tif = args.chm_tif if args.chm_tif else _read_chm_path(variant, ROOT)

    if args.output_tif is not None:
        output_tif = args.output_tif
    else:
        output_tif = phase6_dir / f"406455_2021_tava_{m['run_id']}_prob.tif"

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Phase6 dir: {phase6_dir}")
    print(f"Run: {m['run_id']}")
    print(f"Variant: {variant}, arch: {arch}, in_channels: {in_channels}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"CHM: {chm_tif}")
    print(f"Band stats: {band_stats_path}")
    print(f"TTA: {'off' if args.no_tta else 'on'}")

    prob = infer_full_tile(
        ckpt_path=ckpt_path,
        arch=arch,
        in_channels=in_channels,
        variant=variant,
        chm_tif=chm_tif,
        band_stats=band_stats,
        binary_bands=binary_bands,
        device=device,
        batch_size=args.batch_size,
        use_tta=not args.no_tta,
    )
    write_prob_tif(prob, reference_tif=chm_tif, output_tif=output_tif)

    print(f"Wrote: {output_tif}")
    print(f"Stats: min={float(prob.min()):.6f} max={float(prob.max()):.6f} mean={float(prob.mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
