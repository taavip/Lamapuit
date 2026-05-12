#!/usr/bin/env python3
"""Threshold sweep analysis for a Phase 6 final model on the locked test stripe."""

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

from common.extended_metrics import boundary_iou, cldice_metric
from common.metrics import accumulate_pixel_metrics
from common.raster_io import normalize_bands
from common.sliding_window import sliding_window_predict
from phase2_dataset_v3 import STRIPE_WIDTH, _get_binary_bands
from phase3_train_v10 import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep thresholds for Phase 6 model.")
    p.add_argument("--phase6-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v10_reconstructed")
    p.add_argument("--chm-tif", type=Path,
                   default=ROOT / "seg_pipeline" / "input" / "baseline_chm.tif")
    p.add_argument("--mask-tif", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase1_masks" / "406455_2021_tava_truemask.tif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thr-start", type=float, default=0.05)
    p.add_argument("--thr-end", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.05)
    p.add_argument("--csv-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    phase6_dir = args.phase6_dir.resolve()
    metrics_path = phase6_dir / "all_train_metrics_test.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing: {metrics_path}")

    m = json.loads(metrics_path.read_text())
    variant = str(m["chm_variant"])
    arch = str(m["arch"])
    in_channels = int(m["in_channels"])

    ckpt_path = phase6_dir / variant / "fold0" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    band_stats_path = args.dataset_dir / f"band_stats_{variant}.json"
    if not band_stats_path.exists():
        raise FileNotFoundError(f"Missing: {band_stats_path}")
    band_stats = json.loads(band_stats_path.read_text())
    binary_bands = _get_binary_bands(variant)

    device = torch.device(args.device)
    model = build_model(arch, in_channels=in_channels, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with rasterio.open(args.chm_tif) as src:
        test_stripe = src.read(
            list(range(1, in_channels + 1)),
            window=((0, 5000), (0, STRIPE_WIDTH)),
        ).astype(np.float32)

    with rasterio.open(args.mask_tif) as src:
        raw = src.read([1, 2], window=((0, 5000), (0, STRIPE_WIDTH))).astype(np.float32)
        test_mask = raw[0].astype(np.uint8)
        test_valid = (raw[1] > 0.5).astype(np.uint8)

    test_stripe = normalize_bands(test_stripe, band_stats, binary_bands=binary_bands)
    with torch.no_grad():
        prob = sliding_window_predict(
            model=model,
            image=test_stripe,
            device=device,
            patch_size=256,
            stride=192,
        )

    thresholds = np.arange(args.thr_start, args.thr_end + 1e-9, args.thr_step)
    rows = []
    for thr in thresholds:
        thr = float(round(float(thr), 6))
        met = accumulate_pixel_metrics([prob], [test_mask], [test_valid], threshold=thr)
        pred = (prob >= thr).astype(np.uint8)
        rows.append({
            "threshold": thr,
            "precision": float(met["precision"]),
            "recall": float(met["recall"]),
            "f1": float(met["f1"]),
            "cldice": float(cldice_metric(pred, (test_mask > 0).astype(np.uint8))),
            "boundary_iou": float(boundary_iou(pred, (test_mask > 0).astype(np.uint8))),
        })

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_cldice = max(rows, key=lambda r: r["cldice"])

    print(f"run_id={m['run_id']}")
    print(f"variant={variant} arch={arch} in_channels={in_channels}")
    print("BEST_F1", json.dumps(best_f1))
    print("BEST_CLDICE", json.dumps(best_cldice))
    print("threshold,precision,recall,f1,cldice,boundary_iou")
    for r in rows:
        print(f"{r['threshold']:.2f},{r['precision']:.6f},{r['recall']:.6f},"
              f"{r['f1']:.6f},{r['cldice']:.6f},{r['boundary_iou']:.6f}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall,f1,cldice,boundary_iou\n")
            for r in rows:
                f.write(f"{r['threshold']:.6f},{r['precision']:.10f},{r['recall']:.10f},"
                        f"{r['f1']:.10f},{r['cldice']:.10f},{r['boundary_iou']:.10f}\n")
        print(f"Wrote CSV: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
