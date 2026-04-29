#!/usr/bin/env python3
"""Regenerate iter1 validation overlays with SLD CHM symbology.

This script does not retrain. It loads the saved iter1 checkpoint, runs
inference on manual validation tiles, and writes updated overlays/metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_onboarding_harmonized_seg_experiment import (  # type: ignore
    create_model,
    evaluate_manual_tiles,
    load_manual_validation_tiles,
    save_overlays,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate iter1 overlays with SLD + descriptions")
    p.add_argument(
        "--iter-dir",
        type=Path,
        default=Path("output/manual_mask_experiments/onboarding_harmonized_deeplab_v1/iter1_raw_fulltile"),
    )
    p.add_argument(
        "--chm-root",
        type=Path,
        default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss_stable"),
    )
    p.add_argument("--manual-mask-dir", type=Path, default=Path("output/manual_masks"))
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--max-count", type=int, default=0, help="0 means all usable validation tiles")
    p.add_argument("--output-subdir", type=str, default="overlays")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = args.iter_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    manual_tiles = load_manual_validation_tiles(args.manual_mask_dir, args.chm_root)
    if not manual_tiles:
        raise RuntimeError("No usable manual validation tiles loaded")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    encoder_name = str(cfg.get("encoder_name", "resnet18"))

    model = create_model(encoder_name=encoder_name, aspp_rates=(6, 12, 18))
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    incompatible = model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    metrics, probs = evaluate_manual_tiles(
        model=model,
        manual_tiles=manual_tiles,
        device=device,
        tile_size=args.tile_size,
        batch_size=max(1, args.batch_size),
        threshold=float(args.threshold),
    )

    out_dir = args.iter_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("overlay_*.png"):
        old.unlink()

    written = save_overlays(
        out_dir=out_dir,
        manual_tiles=manual_tiles,
        probs_by_id=probs,
        threshold=float(args.threshold),
        max_count=int(args.max_count),
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "threshold": float(args.threshold),
        "manual_tiles_loaded": int(len(manual_tiles)),
        "overlays_written": int(len(written)),
        "metrics": metrics,
        "state_dict_missing_keys": list(incompatible.missing_keys),
        "state_dict_unexpected_keys": list(incompatible.unexpected_keys),
    }
    (out_dir / "overlay_regeneration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Overlays written: {len(written)} -> {out_dir}")
    print(f"Metrics @ threshold {args.threshold:.2f}: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
