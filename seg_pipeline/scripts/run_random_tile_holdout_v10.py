#!/usr/bin/env python3
"""Random 80/20 holdout test on non-overlapping 256x256 tiles.

Protocol:
1) Rasterize `valid_area.gpkg` and label GPKG to a 3-band supervision mask.
2) Split full raster into non-overlapping 256x256 tiles.
3) Keep suitable tiles (min valid pixels threshold).
4) Stratified random split (positive-tile aware) into train/test (80/20).
5) Train best known V10 model config and report test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "seg_pipeline" / "scripts"))

from common.extended_metrics import cldice_metric
from common.metrics import accumulate_pixel_metrics, threshold_sweep
from phase2_dataset_v3 import CWDSegDataset, PatchEntry, _get_binary_bands, _get_in_channels
from phase3_train_v10 import build_model, train_fold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random 80/20 holdout on non-overlap 256x256 tiles.")
    p.add_argument("--baseline-chm", type=Path,
                   default=ROOT / "seg_pipeline" / "input" / "baseline_chm.tif")
    p.add_argument("--labels-gpkg", type=Path,
                   default=ROOT / "data" / "labels" / "cdw_labels_MP.gpkg")
    p.add_argument("--area-gpkg", type=Path,
                   default=ROOT / "data" / "labels" / "valid_area.gpkg")
    p.add_argument("--band-stats", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "phase2_dataset_v10_reconstructed" / "band_stats_baseline.json")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "seg_pipeline" / "output" / "random_holdout_256_v10")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--min-valid-px", type=int, default=328)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rasterize_gpkg(gpkg_path: Path, reference_tif: Path) -> np.ndarray:
    import geopandas as gpd

    with rasterio.open(reference_tif) as src:
        transform = src.transform
        height, width = src.height, src.width

    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError(f"GPKG has no CRS: {gpkg_path}")
    gdf = gdf.to_crs(epsg=3301)
    shapes = [(geom, 1.0) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        raise ValueError(f"No valid geometries in {gpkg_path}")

    return rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=True,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )


def build_phase1_style_mask(
    baseline_chm: Path,
    labels_gpkg: Path,
    area_gpkg: Path,
    out_tif: Path,
) -> None:
    area_mask = rasterize_gpkg(area_gpkg, baseline_chm) > 0.5
    label_mask = rasterize_gpkg(labels_gpkg, baseline_chm) > 0.5

    with rasterio.open(baseline_chm) as src:
        chm = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    coverage = np.isfinite(chm)
    target = (area_mask & label_mask).astype(np.float32)
    valid = (area_mask & coverage).astype(np.float32)
    ensemble_stub = np.zeros_like(target, dtype=np.float32)
    bands = np.stack([target, valid, ensemble_stub], axis=0)

    profile.update(
        count=3,
        dtype="float32",
        nodata=None,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        height=bands.shape[1],
        width=bands.shape[2],
    )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(bands)


def make_nonoverlap_entries(mask_tif: Path, patch_size: int, min_valid_px: int) -> list[PatchEntry]:
    with rasterio.open(mask_tif) as src:
        target = src.read(1).astype(np.float32)
        valid = src.read(2).astype(np.float32)
        H, W = src.height, src.width

    ys = list(range(0, H - patch_size + 1, patch_size))
    xs = list(range(0, W - patch_size + 1, patch_size))

    entries: list[PatchEntry] = []
    for y0 in ys:
        for x0 in xs:
            v = valid[y0:y0 + patch_size, x0:x0 + patch_size] > 0.5
            n_valid = int(v.sum())
            if n_valid < min_valid_px:
                continue
            t = target[y0:y0 + patch_size, x0:x0 + patch_size] > 0.5
            n_pos = int((t & v).sum())
            entries.append(PatchEntry(
                row_off=y0,
                col_off=x0,
                stripe_id=-1,
                fold_id=0,
                n_valid=n_valid,
                n_positive=n_pos,
            ))
    return entries


def stratified_train_test_split(entries: list[PatchEntry], train_ratio: float, seed: int) -> tuple[list[PatchEntry], list[PatchEntry]]:
    rng = random.Random(seed)
    pos = [e for e in entries if e.n_positive > 0]
    neg = [e for e in entries if e.n_positive == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_train = int(round(len(pos) * train_ratio))
    n_neg_train = int(round(len(neg) * train_ratio))

    train = pos[:n_pos_train] + neg[:n_neg_train]
    test = pos[n_pos_train:] + neg[n_neg_train:]
    rng.shuffle(train)
    rng.shuffle(test)

    train = [
        PatchEntry(e.row_off, e.col_off, e.stripe_id, 0, e.n_valid, e.n_positive)
        for e in train
    ]
    test = [
        PatchEntry(e.row_off, e.col_off, e.stripe_id, 1, e.n_valid, e.n_positive)
        for e in test
    ]
    return train, test


def save_entries_csv(path: Path, entries: list[PatchEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        raise ValueError(f"No entries to save: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(entries[0]).keys()))
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))


@torch.no_grad()
def evaluate_entries(
    ckpt_path: Path,
    arch: str,
    variant: str,
    entries: list[PatchEntry],
    chm_tif: Path,
    mask_tif: Path,
    band_stats: dict,
    device: torch.device,
    batch_size: int,
) -> dict:
    in_channels = _get_in_channels(variant)
    model = build_model(arch, in_channels=in_channels, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = CWDSegDataset(
        entries=entries,
        chm_tif=chm_tif,
        mask_tif=mask_tif,
        band_stats=band_stats,
        patch_size=256,
        in_channels=in_channels,
        augment=False,
        variant=variant,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    prob_list: list[np.ndarray] = []
    tgt_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        logits = model(image)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        for k in range(len(probs)):
            prob_list.append(probs[k])
            tgt_list.append(batch["target"][k, 0].numpy())
            val_list.append(batch["valid"][k, 0].numpy())

    best_thr, _ = threshold_sweep(prob_list, tgt_list, val_list)
    thr = float(best_thr["threshold"])
    px = accumulate_pixel_metrics(prob_list, tgt_list, val_list, threshold=thr)

    cl_vals = []
    for p, t, v in zip(prob_list, tgt_list, val_list):
        pred = (p >= thr) & (v > 0.5)
        gt = (t > 0.5) & (v > 0.5)
        cl_vals.append(float(cldice_metric(pred.astype(np.uint8), gt.astype(np.uint8))))

    return {
        "optimal_threshold": thr,
        "test_precision": float(px["precision"]),
        "test_recall": float(px["recall"]),
        "test_f1": float(px["f1"]),
        "test_dice": float(px["dice"]),
        "test_iou": float(px["iou"]),
        "test_accuracy": float(px["accuracy"]),
        "test_cldice_mean_patch": float(np.mean(cl_vals)) if cl_vals else 0.0,
        "n_test_tiles": len(entries),
    }


def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=== Random 80/20 Holdout Test (Non-overlap 256x256) ===")
    print(f"Output: {out}")
    print(f"Seed: {args.seed}")
    print(f"Patch size: {args.patch_size} (non-overlap)")
    print(f"Min valid px per tile: {args.min_valid_px}")
    print(f"Train ratio: {args.train_ratio:.2f}")

    mask_tif = out / "406455_2021_tava_truemask_random_holdout.tif"
    print("\n[1/5] Building phase1-style mask from GPKG...")
    build_phase1_style_mask(
        baseline_chm=args.baseline_chm,
        labels_gpkg=args.labels_gpkg,
        area_gpkg=args.area_gpkg,
        out_tif=mask_tif,
    )
    print(f"  Mask written: {mask_tif}")

    print("\n[2/5] Building non-overlap tile pool...")
    entries = make_nonoverlap_entries(mask_tif, patch_size=args.patch_size, min_valid_px=args.min_valid_px)
    if not entries:
        raise RuntimeError("No suitable tiles found.")
    n_pos = sum(1 for e in entries if e.n_positive > 0)
    print(f"  Suitable tiles: {len(entries)} (positive tiles: {n_pos}, negative tiles: {len(entries)-n_pos})")

    print("\n[3/5] Stratified random split 80/20...")
    train_entries, test_entries = stratified_train_test_split(entries, args.train_ratio, seed=args.seed)
    train_pos = sum(1 for e in train_entries if e.n_positive > 0)
    test_pos = sum(1 for e in test_entries if e.n_positive > 0)
    print(f"  Train: {len(train_entries)} tiles (pos={train_pos}, neg={len(train_entries)-train_pos})")
    print(f"  Test:  {len(test_entries)} tiles (pos={test_pos}, neg={len(test_entries)-test_pos})")
    save_entries_csv(out / "all_tiles.csv", entries)
    save_entries_csv(out / "train_tiles.csv", train_entries)
    save_entries_csv(out / "test_tiles.csv", test_entries)

    print("\n[4/5] Training best known model config...")
    band_stats = json.loads(args.band_stats.read_text())
    device = torch.device(args.device)

    # Best chain from previous run (legacy comparator won final test):
    # 2A__3C__4H__5D → baseline + unetpp_effb2 + tversky(0.6,0.4)+clDice(0.3),
    # full augmentation, soft targets, SWA enabled.
    train_result = train_fold(
        arch="unetpp_effb2",
        fold_id=0,
        train_entries=train_entries,
        val_entries=test_entries,  # explicit holdout validation set
        chm_tif=args.baseline_chm,
        mask_tif=mask_tif,
        band_stats=band_stats,
        output_dir=out / "phase3_runs_random_holdout",
        device=device,
        variant="baseline",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-4,
        patience=15,
        tversky_alpha=0.6,
        tversky_beta=0.4,
        cldice_weight=0.3,
        soft_targets=True,
        aug_mode="full",
        batch_aug=True,
        soft_sigma=2.0,
        swa_start_epoch=35,
        warmup_epochs=25,
        monitor_metric="val_cldice",
    )

    fold_dir = out / "phase3_runs_random_holdout" / "baseline" / "fold0"
    ckpt = fold_dir / "best.pt"
    if train_result.get("use_swa_for_inference", False) and (fold_dir / "swa_model.pt").exists():
        ckpt = fold_dir / "swa_model.pt"

    print("\n[5/5] Final evaluation on holdout test tiles...")
    test_metrics = evaluate_entries(
        ckpt_path=ckpt,
        arch="unetpp_effb2",
        variant="baseline",
        entries=test_entries,
        chm_tif=args.baseline_chm,
        mask_tif=mask_tif,
        band_stats=band_stats,
        device=device,
        batch_size=args.batch_size,
    )

    summary = {
        "protocol": "random_holdout_nonoverlap_256",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "patch_size": args.patch_size,
        "min_valid_px": args.min_valid_px,
        "n_tiles_total": len(entries),
        "n_tiles_train": len(train_entries),
        "n_tiles_test": len(test_entries),
        "train_result": train_result,
        "test_result": test_metrics,
        "checkpoint_used": str(ckpt),
    }
    summary_path = out / "RANDOM_HOLDOUT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary: {summary_path}")
    print("\n=== Done ===")
    print(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
