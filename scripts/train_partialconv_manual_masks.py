#!/usr/bin/env python3
"""Fine-tune PartialConv U-Net on manual RGBA masks from the labeler.

Mask semantics:
- R=255 and A>0 -> CWD positive
- R=0 and A>0 -> background negative
- R=128 or A=0 -> ignored (no supervision)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from cdw_detect.cwd_partialconv_pipeline import (
    NODATA_THRESHOLD,
    PartialConvUNet,
    SegmentationLoss,
    centered_pad,
    load_partialconv_checkpoint,
)


EPS = 1e-8


@dataclass
class ManualSample:
    tile_id: str
    sample_id: str
    variant: str
    row_off: int
    col_off: int
    raster_path: Path
    mask_path: Path


class ManualMaskDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], augment: bool = False) -> None:
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.items[idx]
        inp = item["input"]
        target = item["target"]
        confidence = item["confidence"]

        if self.augment:
            # Geometric augmentation keeps mask semantics and CHM alignment.
            if random.random() < 0.5:
                inp = np.flip(inp, axis=2).copy()
                target = np.flip(target, axis=2).copy()
                confidence = np.flip(confidence, axis=2).copy()
            if random.random() < 0.5:
                inp = np.flip(inp, axis=1).copy()
                target = np.flip(target, axis=1).copy()
                confidence = np.flip(confidence, axis=1).copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                inp = np.rot90(inp, k=k, axes=(1, 2)).copy()
                target = np.rot90(target, k=k, axes=(1, 2)).copy()
                confidence = np.rot90(confidence, k=k, axes=(1, 2)).copy()

        return {
            "input": torch.from_numpy(inp).float(),
            "target": torch.from_numpy(target).float(),
            "confidence": torch.from_numpy(confidence).float(),
            "has_label": torch.tensor([1.0 if item["has_label"] else 0.0], dtype=torch.float32),
            "tile_id": item["tile_id"],
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PartialConv with manual RGBA masks")
    parser.add_argument("--mask-dir", type=Path, default=Path("output/manual_masks"))
    parser.add_argument(
        "--chm-dir",
        type=Path,
        default=Path("output/chm_dataset_harmonized_0p8m_raw_gauss"),
        help="Directory containing *_harmonized_dem_last_{raw|gauss}_chm.tif",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/cwd_partialconv_gpu_multiepoch_20260417_sota_es/best_partialconv_unet.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/manual_mask_finetune_partialconv"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--use-cosine-scheduler", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--best-threshold-metric", type=str, default="f1", choices=["f1", "dice"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def freeze_encoder_blocks(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("enc") or name.startswith("bottleneck"):
            param.requires_grad = False


def build_raster_index(chm_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for tif in chm_dir.rglob("*.tif"):
        index[tif.stem] = tif
    return index


def parse_manual_samples(mask_dir: Path, raster_index: dict[str, Path]) -> list[ManualSample]:
    pat = re.compile(r"^(?P<sample>.+)_harmonized_dem_last_(?P<variant>raw|gauss)_chm__r(?P<row>\d+)_c(?P<col>\d+)$")
    out: list[ManualSample] = []

    for meta_path in sorted(mask_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tile_id = str(meta.get("tile_id", "")).strip()
        if not tile_id:
            continue

        match = pat.match(tile_id)
        if not match:
            continue

        raster_stem = f"{match.group('sample')}_harmonized_dem_last_{match.group('variant')}_chm"
        raster_path = raster_index.get(raster_stem)
        if raster_path is None:
            continue

        mask_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_mask.png"))
        if not mask_path.exists():
            continue

        out.append(
            ManualSample(
                tile_id=tile_id,
                sample_id=match.group("sample"),
                variant=match.group("variant"),
                row_off=int(match.group("row")),
                col_off=int(match.group("col")),
                raster_path=raster_path,
                mask_path=mask_path,
            )
        )

    return out


def read_chm_chip(raster_path: Path, row_off: int, col_off: int, size: int = 256) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        arr = src.read(1, window=Window(col_off, row_off, size, size), boundless=True, fill_value=np.nan)
    if arr.shape != (size, size):
        fixed = np.full((size, size), np.nan, dtype=np.float32)
        h = min(size, arr.shape[0])
        w = min(size, arr.shape[1])
        fixed[:h, :w] = arr[:h, :w]
        arr = fixed
    return arr.astype(np.float32)


def make_model_input(chm: np.ndarray) -> np.ndarray:
    invalid = ~np.isfinite(chm)
    invalid |= chm < NODATA_THRESHOLD
    h = chm.copy()
    h[invalid] = 0.0
    valid = (~invalid).astype(np.float32)
    h = centered_pad(h.astype(np.float32), target_size=256)
    valid = centered_pad(valid.astype(np.float32), target_size=256)
    return np.stack([h, valid], axis=0).astype(np.float32)


def load_mask_target(mask_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    rgba = np.asarray(Image.open(mask_path).convert("RGBA"), dtype=np.uint8)
    red = rgba[..., 0]
    alpha = rgba[..., 3]

    positive = (alpha > 0) & (red >= 200)
    negative = (alpha > 0) & (red <= 50)
    labeled = positive | negative

    target = positive.astype(np.float32)[None, ...]
    confidence = labeled.astype(np.float32)[None, ...]
    has_label = bool(np.any(labeled))
    return target, confidence, has_label


def build_items(samples: list[ManualSample]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sample in samples:
        chm = read_chm_chip(sample.raster_path, sample.row_off, sample.col_off, size=256)
        inp = make_model_input(chm)
        target, confidence, has_label = load_mask_target(sample.mask_path)
        items.append(
            {
                "tile_id": sample.tile_id,
                "input": inp,
                "target": target,
                "confidence": confidence,
                "has_label": has_label,
                "positive_tile": bool(target.max() > 0.5),
            }
        )
    return items


def stratified_split(items: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives = [i for i in items if i["positive_tile"]]
    negatives = [i for i in items if not i["positive_tile"]]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_pos_val = max(1, int(round(len(positives) * val_fraction))) if positives else 0
    n_neg_val = max(1, int(round(len(negatives) * val_fraction))) if negatives else 0

    val = positives[:n_pos_val] + negatives[:n_neg_val]
    train = positives[n_pos_val:] + negatives[n_neg_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _batch_confidence_mask(conf: torch.Tensor) -> torch.Tensor:
    return conf > 0.5


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    labeled_pixels = 0.0
    tiles = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            confidence = batch["confidence"].to(device)

            _, seg = model(x)
            pred = seg >= threshold
            gt = target > 0.5
            valid = _batch_confidence_mask(confidence)

            tp += torch.sum(pred & gt & valid).item()
            fp += torch.sum(pred & (~gt) & valid).item()
            fn += torch.sum((~pred) & gt & valid).item()
            tn += torch.sum((~pred) & (~gt) & valid).item()
            labeled_pixels += torch.sum(valid).item()
            tiles += x.shape[0]

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)

    return {
        "tiles": float(tiles),
        "labeled_pixels": float(labeled_pixels),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def evaluate_with_threshold_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric: str = "f1",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    thresholds = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for thr in thresholds:
        m = evaluate_model(model, loader, device=device, threshold=float(thr))
        row = {"threshold": float(thr), **m}
        rows.append(row)
        if best is None or row[metric] > best[metric]:
            best = row
    if best is None:
        raise RuntimeError("Threshold sweep produced no metrics")
    return best, rows


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    threshold: float,
    patience: int,
    optimizer_name: str,
    use_cosine_scheduler: bool,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_cosine_scheduler and epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    seg_loss = SegmentationLoss(alpha=0.25, gamma=2.0, smooth=1.0)

    best_state: dict[str, Any] | None = None
    best_dice = -1.0
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for batch in train_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            confidence = batch["confidence"].to(device)
            has_label = batch["has_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, seg = model(x)
            loss = seg_loss(seg, target, confidence, has_label)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().item())
            steps += 1

        train_loss = running / max(1, steps)
        val_metrics = evaluate_model(model, val_loader, device, threshold=threshold)
        row = {
            "epoch": float(epoch),
            "train_seg_loss": float(train_loss),
            "val_dice": float(val_metrics["dice"]),
            "val_iou": float(val_metrics["iou"]),
            "val_f1": float(val_metrics["f1"]),
        }
        history.append(row)

        if scheduler is not None:
            scheduler.step()

        current_dice = float(val_metrics["dice"])
        if current_dice > best_dice + 1e-6:
            best_dice = current_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"epoch={epoch:03d} train_seg_loss={train_loss:.5f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if patience > 0 and no_improve >= patience:
            print(f"early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raster_index = build_raster_index(args.chm_dir)
    manual_samples = parse_manual_samples(args.mask_dir, raster_index)
    if len(manual_samples) < 4:
        raise RuntimeError(f"Need at least 4 parsed manual samples, got {len(manual_samples)}")

    items = build_items(manual_samples)
    train_items, val_items = stratified_split(items, val_fraction=args.val_fraction, seed=args.seed)
    if not train_items or not val_items:
        raise RuntimeError("Train/val split is empty. Adjust --val-fraction or add more masks.")

    train_loader = DataLoader(
        ManualMaskDataset(train_items, augment=args.augment),
        batch_size=max(1, args.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        ManualMaskDataset(val_items, augment=False),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    all_loader = DataLoader(
        ManualMaskDataset(items, augment=False),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    baseline = load_partialconv_checkpoint(args.checkpoint, device=device)
    baseline_metrics_val = evaluate_model(baseline, val_loader, device=device, threshold=args.threshold)
    baseline_metrics_all = evaluate_model(baseline, all_loader, device=device, threshold=args.threshold)
    baseline_best_val, baseline_sweep_val = evaluate_with_threshold_sweep(
        baseline,
        val_loader,
        device=device,
        metric=args.best_threshold_metric,
    )
    baseline_best_all, baseline_sweep_all = evaluate_with_threshold_sweep(
        baseline,
        all_loader,
        device=device,
        metric=args.best_threshold_metric,
    )

    model = load_partialconv_checkpoint(args.checkpoint, device=device)
    if args.freeze_encoder:
        freeze_encoder_blocks(model)
    model.train()
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        patience=args.patience,
        optimizer_name=args.optimizer,
        use_cosine_scheduler=args.use_cosine_scheduler,
    )

    tuned_metrics_val = evaluate_model(model, val_loader, device=device, threshold=args.threshold)
    tuned_metrics_all = evaluate_model(model, all_loader, device=device, threshold=args.threshold)
    tuned_best_val, tuned_sweep_val = evaluate_with_threshold_sweep(
        model,
        val_loader,
        device=device,
        metric=args.best_threshold_metric,
    )
    tuned_best_all, tuned_sweep_all = evaluate_with_threshold_sweep(
        model,
        all_loader,
        device=device,
        metric=args.best_threshold_metric,
    )

    checkpoint_out = args.output_dir / "best_partialconv_manual_finetune.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint),
            "train_tiles": len(train_items),
            "val_tiles": len(val_items),
        },
        checkpoint_out,
    )

    report = {
        "config": {
            "mask_dir": str(args.mask_dir),
            "chm_dir": str(args.chm_dir),
            "checkpoint": str(args.checkpoint),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "optimizer": str(args.optimizer),
            "use_cosine_scheduler": bool(args.use_cosine_scheduler),
            "freeze_encoder": bool(args.freeze_encoder),
            "augment": bool(args.augment),
            "val_fraction": float(args.val_fraction),
            "threshold": float(args.threshold),
            "patience": int(args.patience),
            "best_threshold_metric": str(args.best_threshold_metric),
            "seed": int(args.seed),
            "device": str(device),
        },
        "dataset": {
            "total_tiles": len(items),
            "train_tiles": len(train_items),
            "val_tiles": len(val_items),
            "positive_tiles_total": int(sum(1 for it in items if it["positive_tile"])),
            "positive_tiles_train": int(sum(1 for it in train_items if it["positive_tile"])),
            "positive_tiles_val": int(sum(1 for it in val_items if it["positive_tile"])),
        },
        "baseline": {
            "val": baseline_metrics_val,
            "all": baseline_metrics_all,
            "best_val": baseline_best_val,
            "best_all": baseline_best_all,
        },
        "tuned": {
            "val": tuned_metrics_val,
            "all": tuned_metrics_all,
            "best_val": tuned_best_val,
            "best_all": tuned_best_all,
        },
        "delta": {
            "val_dice": float(tuned_metrics_val["dice"] - baseline_metrics_val["dice"]),
            "val_iou": float(tuned_metrics_val["iou"] - baseline_metrics_val["iou"]),
            "val_f1": float(tuned_metrics_val["f1"] - baseline_metrics_val["f1"]),
            "all_dice": float(tuned_metrics_all["dice"] - baseline_metrics_all["dice"]),
            "all_iou": float(tuned_metrics_all["iou"] - baseline_metrics_all["iou"]),
            "all_f1": float(tuned_metrics_all["f1"] - baseline_metrics_all["f1"]),
            "best_val_dice": float(tuned_best_val["dice"] - baseline_best_val["dice"]),
            "best_val_iou": float(tuned_best_val["iou"] - baseline_best_val["iou"]),
            "best_val_f1": float(tuned_best_val["f1"] - baseline_best_val["f1"]),
            "best_all_dice": float(tuned_best_all["dice"] - baseline_best_all["dice"]),
            "best_all_iou": float(tuned_best_all["iou"] - baseline_best_all["iou"]),
            "best_all_f1": float(tuned_best_all["f1"] - baseline_best_all["f1"]),
        },
        "threshold_sweep": {
            "baseline_val": baseline_sweep_val,
            "baseline_all": baseline_sweep_all,
            "tuned_val": tuned_sweep_val,
            "tuned_all": tuned_sweep_all,
        },
        "history": history,
        "outputs": {
            "checkpoint": str(checkpoint_out),
            "report": str(args.output_dir / "manual_mask_finetune_report.json"),
        },
    }

    report_path = args.output_dir / "manual_mask_finetune_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Manual-mask fine-tuning complete")
    print(f"train_tiles={len(train_items)} val_tiles={len(val_items)} total={len(items)}")
    print("Baseline VAL:", json.dumps(baseline_metrics_val, indent=2))
    print("Tuned VAL:", json.dumps(tuned_metrics_val, indent=2))
    print("Delta:", json.dumps(report["delta"], indent=2))
    print(f"Checkpoint: {checkpoint_out}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
