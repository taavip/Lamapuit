#!/usr/bin/env python3
"""
Train a 4-model soft-vote ensemble for CDW tile classification.

Models trained
--------------
  cnn_seed42.pt   CNN-Deep-Attn, seed 42
  cnn_seed43.pt   CNN-Deep-Attn, seed 43
  cnn_seed44.pt   CNN-Deep-Attn, seed 44
  effnet_b2.pt    EfficientNet-B2 (ImageNet pretrained, 1-channel adaptation)

All models are trained on the same training split (tiles NOT in
cnn_test_split.json).  The test-set evaluation uses TTA and soft-vote
averaging across all models.

Outputs (in --output directory)
--------------------------------
  cnn_seed42.pt / cnn_seed43.pt / cnn_seed44.pt / effnet_b2.pt  ← checkpoints
  ensemble_meta.json  ← per-model + ensemble test metrics + checkpoint paths

Usage
-----
  python scripts/train_ensemble.py \\
      --labels output/tile_labels \\
      --chm-dir chm_max_hag \\
      --test-split output/tile_labels/cnn_test_split.json \\
      --output output/tile_labels
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
ENSEMBLE_META = "ensemble_meta.json"
CNN_EPOCHS = 50
EFFNET_EPOCHS = 30
BATCH_SIZE = 32
LR_HEAD = 5e-4
LR_BACKBONE = 5e-5
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.3
CNN_SEEDS = (42, 43, 44)
MIN_PER_CLASS = 30


# ── Normalisation (must match fine_tune_cnn.py / label_tiles.py) ──────────────
def _norm_tile(raw: np.ndarray) -> np.ndarray:
    return np.clip(raw, 0.0, 20.0) / 20.0


# ── Reuse data helpers from fine_tune_cnn.py ──────────────────────────────────
def _import_helpers():
    """Import _load_labels and _build_arrays from fine_tune_cnn without side effects."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fine_tune_cnn",
        Path(__file__).parent / "fine_tune_cnn.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_labels, mod._build_arrays, mod._build_deep_cnn_attn_net, mod._compute_metrics


# ── EfficientNet-B2 (1-channel) ───────────────────────────────────────────────
def _build_effnet_b2(pretrained: bool = True):
    """EfficientNet-B2 adapted for single-channel 128×128 input.

    The first Conv2d is averaged over the 3 pretrained in-channels so the
    spatial feature extraction transfers from RGB imagery.
    The classifier head is replaced with a 2-way output.
    """
    import torch
    import torch.nn as nn

    try:
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b2(weights=weights)
    except Exception:
        from torchvision.models import efficientnet_b2

        model = efficientnet_b2(pretrained=pretrained)

    # Average RGB weights → single channel  (shape: out, 3, kH, kW → out, 1, kH, kW)
    first_conv = model.features[0][0]  # Conv2d(3, 32, ...)
    avg_w = first_conv.weight.data.mean(dim=1, keepdim=True)
    new_conv = nn.Conv2d(
        1,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight.data = avg_w
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()
    model.features[0][0] = new_conv

    # Replace classifier head: in_features → 2 logits
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 2)

    return model


def _param_groups_effnet(net, lr_head: float, lr_backbone: float):
    """Two param groups: backbone at low LR, head at higher LR."""
    head_params = list(net.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in net.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]


# ── Dataset ───────────────────────────────────────────────────────────────────
class _TileDataset:
    def __init__(self, X, y, w, augment: bool = False):
        import torch

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.w = torch.from_numpy(w.astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        import torch

        x = self.X[idx].clone()
        y = self.y[idx]
        w = self.w[idx]
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-1])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-2])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, [-2, -1])
            if torch.rand(1).item() > 0.7:
                x = (x + torch.randn_like(x) * 0.015).clamp(0.0, 1.0)
            if torch.rand(1).item() > 0.80:
                alpha = 0.85 + torch.rand(1).item() * 0.30
                beta = (torch.rand(1).item() - 0.5) * 0.06
                x = (x * alpha + beta).clamp(0.0, 1.0)
        return x, y, w


# ── Mixup ─────────────────────────────────────────────────────────────────────
def _mixup_batch(xb, yb, wb, alpha: float = 0.3):
    import torch

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xb.size(0), device=xb.device)
    return (
        lam * xb + (1.0 - lam) * xb[perm],
        yb,
        yb[perm],
        lam,
        lam * wb + (1.0 - lam) * wb[perm],
    )


# ── Generic training loop ─────────────────────────────────────────────────────
def _train_model(
    net,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device,
    epochs: int = CNN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    label_smooth: float = LABEL_SMOOTHING,
    mixup_alpha: float = MIXUP_ALPHA,
    param_groups=None,
    lr: float = LR_HEAD,
    model_tag: str = "",
) -> dict:
    """Train *net* and return best val metrics dict."""
    import torch
    from torch.utils.data import DataLoader

    _load_labels, _build_arrays, _build_cnn, _compute_metrics = _import_helpers()

    net = net.to(device)
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    w_pos = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smooth, reduction="none"
    )

    if param_groups is None:
        param_groups = [{"params": net.parameters(), "lr": lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = _TileDataset(X_train, y_train, w_train, augment=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    best_auc, best_f1, best_thresh = 0.0, 0.0, 0.5
    best_state = None
    log_every = max(1, epochs // 10)

    print(
        f"[train_ensemble] {model_tag}  epochs={epochs}  "
        f"train={len(X_train)}  val={len(X_val)}  device={device}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        for xb, yb, wb in train_dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device).float()
            optimizer.zero_grad()
            if mixup_alpha > 0.0 and torch.rand(1).item() > 0.5:
                xb_m, ya, yb_m, lam, wb_m = _mixup_batch(xb, yb, wb, alpha=mixup_alpha)
                loss = (
                    lam * criterion(net(xb_m), ya) + (1.0 - lam) * criterion(net(xb_m), yb_m)
                ) * wb_m
            else:
                loss = criterion(net(xb), yb) * wb
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss.mean().item()
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs:
            auc, f1_, thr_ = _compute_metrics(net, X_val_t, y_val_t, device, tta=False)
            avg_loss = epoch_loss / max(len(train_dl), 1)
            print(
                f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                f"val_AUC={auc:.4f}  F1={f1_:.4f}@{thr_:.2f}",
                flush=True,
            )
            if auc > best_auc or (auc == best_auc and f1_ > best_f1):
                best_auc, best_f1, best_thresh = auc, f1_, thr_
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    if best_state is None:
        best_auc, best_f1, best_thresh = _compute_metrics(net, X_val_t, y_val_t, device, tta=False)
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    return {
        "val_auc": round(best_auc, 4),
        "val_f1": round(best_f1, 4),
        "val_thresh": round(best_thresh, 2),
    }


# ── Test-set soft-vote evaluation ─────────────────────────────────────────────
def _ensemble_test_eval(nets, X_test: np.ndarray, y_test: np.ndarray, device) -> dict:
    """Soft-vote TTA across all nets → test metrics dict."""
    import torch
    from sklearn.metrics import roc_auc_score, f1_score

    X_t = torch.from_numpy(X_test).to(device)

    all_probs = None
    batch = 64

    def _tta_probs(net):
        net.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch):
                xb = X_t[i : i + batch]
                views = []
                for k in range(4):
                    v = torch.rot90(xb, k, [-2, -1])
                    views.append(torch.softmax(net(v), dim=1)[:, 1])
                    views.append(torch.softmax(net(torch.flip(v, [-1])), dim=1)[:, 1])
                pb = torch.stack(views, dim=0).mean(dim=0).cpu().numpy()
                probs_list.append(pb)
        return np.concatenate(probs_list)

    for net in nets:
        p = _tta_probs(net)
        all_probs = p if all_probs is None else all_probs + p

    ensemble_probs = all_probs / len(nets)
    labels = y_test.astype(int)

    auc = float(roc_auc_score(labels, ensemble_probs))
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.10, 0.90, 81):
        preds = (ensemble_probs >= thr).astype(int)
        f1_ = float(f1_score(labels, preds, zero_division=0))
        if f1_ >= best_f1:
            best_f1, best_thr = f1_, float(thr)

    return {
        "ensemble_auc": round(auc, 4),
        "ensemble_f1": round(best_f1, 4),
        "ensemble_thresh": round(best_thr, 2),
        "n_test": int(len(labels)),
        "n_cdw": int(np.sum(labels == 1)),
        "tta": True,
        "n_models": len(nets),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def train_ensemble(
    labels_dir: Path,
    chm_dir: Path,
    output_dir: Path,
    test_split: Path | None = None,
    dry_run: bool = False,
) -> int:
    import torch

    t0 = time.monotonic()

    _load_labels, _build_arrays, _build_cnn, _compute_metrics = _import_helpers()

    # ── Load labels ───────────────────────────────────────────────────────────
    print("[train_ensemble] Loading labels …", flush=True)
    records, _load_stats = _load_labels(labels_dir)
    n_cdw = sum(1 for r in records if r["label"] == 1)
    n_no = sum(1 for r in records if r["label"] == 0)
    print(f"[train_ensemble] Labels: {len(records)} total  CDW={n_cdw}  No-CDW={n_no}", flush=True)

    if n_cdw < MIN_PER_CLASS or n_no < MIN_PER_CLASS:
        print("[train_ensemble] Not enough data. Aborting.", flush=True)
        return 1

    # ── Load held-out test keys ───────────────────────────────────────────────
    test_keys: set[tuple] = set()
    if test_split is not None and test_split.exists():
        split_data = json.loads(test_split.read_text())
        test_keys = {(r, ro, co) for r, ro, co in split_data["keys"]}
        print(f"[train_ensemble] Test split: {len(test_keys)} tiles held out.", flush=True)

    # ── Build arrays (exclude test keys from training) ────────────────────────
    print(f"[train_ensemble] Reading tiles from {chm_dir} …", flush=True)
    train_records = [
        r for r in records if (r["raster"], r["row_off"], r["col_off"]) not in test_keys
    ]
    test_records = [r for r in records if (r["raster"], r["row_off"], r["col_off"]) in test_keys]

    X, y, w = _build_arrays(train_records, chm_dir)
    if X is None or len(X) < MIN_PER_CLASS * 2:
        print("[train_ensemble] Dataset too small after tile loading.", flush=True)
        return 1

    X_test_arr = y_test_arr = None
    if test_records:
        X_test_arr, y_test_arr, _ = _build_arrays(test_records, chm_dir)

    if dry_run:
        print("[train_ensemble] Dry-run: data OK, skipping training.", flush=True)
        return 0

    # ── Stratified train/val split ────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx_cdw = np.where(y == 1)[0]
    rng.shuffle(idx_cdw)
    idx_no = np.where(y == 0)[0]
    rng.shuffle(idx_no)
    n_val_cdw = max(1, len(idx_cdw) // 5)
    n_val_no = max(1, len(idx_no) // 5)
    val_idx = np.concatenate([idx_cdw[:n_val_cdw], idx_no[:n_val_no]])
    tr_idx = np.concatenate([idx_cdw[n_val_cdw:], idx_no[n_val_no:]])
    X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], w[tr_idx]
    X_vl, y_vl = X[val_idx], y[val_idx]
    print(f"[train_ensemble] Train={len(X_tr)}  Val={len(X_vl)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_ensemble] Device: {device}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    model_metrics: dict[str, dict] = {}
    trained_nets = []

    # ── Train 3× CNN-Deep-Attn with different seeds ───────────────────────────
    for seed in CNN_SEEDS:
        tag = f"cnn_seed{seed}"
        print(f"\n[train_ensemble] ── Training {tag} ──────────────────────────", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = _build_cnn()
        metrics = _train_model(
            net,
            X_tr,
            y_tr,
            w_tr,
            X_vl,
            y_vl,
            device,
            epochs=CNN_EPOCHS,
            label_smooth=LABEL_SMOOTHING,
            mixup_alpha=MIXUP_ALPHA,
            lr=LR_HEAD,
            model_tag=tag,
        )
        pt_path = output_dir / f"{tag}.pt"
        torch.save(
            {
                "state_dict": net.state_dict(),
                "tag": tag,
                "build_fn_name": "_build_deep_cnn_attn",
                "meta": metrics,
            },
            pt_path,
        )
        checkpoints[tag] = str(pt_path)
        model_metrics[tag] = metrics
        trained_nets.append(net)
        print(
            f"[train_ensemble] {tag} saved → {pt_path}  "
            f"val_AUC={metrics['val_auc']:.4f}  F1={metrics['val_f1']:.4f}",
            flush=True,
        )

    # ── Train EfficientNet-B2 ─────────────────────────────────────────────────
    tag = "effnet_b2"
    print(f"\n[train_ensemble] ── Training {tag} ──────────────────────────────", flush=True)
    torch.manual_seed(0)
    np.random.seed(0)
    effnet = _build_effnet_b2()
    pg = _param_groups_effnet(effnet.to(device), lr_head=LR_HEAD, lr_backbone=LR_BACKBONE)
    metrics = _train_model(
        effnet,
        X_tr,
        y_tr,
        w_tr,
        X_vl,
        y_vl,
        device,
        epochs=EFFNET_EPOCHS,
        label_smooth=LABEL_SMOOTHING,
        mixup_alpha=MIXUP_ALPHA,
        param_groups=pg,
        model_tag=tag,
    )
    pt_path = output_dir / f"{tag}.pt"
    torch.save(
        {
            "state_dict": effnet.state_dict(),
            "tag": tag,
            "build_fn_name": "_build_effnet_b2",
            "meta": metrics,
        },
        pt_path,
    )
    checkpoints[tag] = str(pt_path)
    model_metrics[tag] = metrics
    trained_nets.append(effnet)
    print(
        f"[train_ensemble] {tag} saved → {pt_path}  "
        f"val_AUC={metrics['val_auc']:.4f}  F1={metrics['val_f1']:.4f}",
        flush=True,
    )

    # ── Ensemble test-set evaluation ──────────────────────────────────────────
    test_metrics: dict | None = None
    if X_test_arr is not None and len(X_test_arr) >= 4 and len(np.unique(y_test_arr)) > 1:
        print(f"\n[train_ensemble] Ensemble TTA test eval  n={len(X_test_arr)} …", flush=True)
        test_metrics = _ensemble_test_eval(trained_nets, X_test_arr, y_test_arr, device)
        print(
            f"[train_ensemble] Ensemble test  "
            f"AUC={test_metrics['ensemble_auc']:.4f}  "
            f"F1={test_metrics['ensemble_f1']:.4f}@{test_metrics['ensemble_thresh']:.2f}",
            flush=True,
        )

    # ── Write ensemble_meta.json ──────────────────────────────────────────────
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": checkpoints,
        "model_metrics": model_metrics,
        "test_metrics": test_metrics,
        "cnn_epochs": CNN_EPOCHS,
        "effnet_epochs": EFFNET_EPOCHS,
        "label_smooth": LABEL_SMOOTHING,
        "mixup_alpha": MIXUP_ALPHA,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_vl)),
    }
    meta_path = output_dir / ENSEMBLE_META
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"\n[train_ensemble] ✓ Done in {time.monotonic() - t0:.1f}s  " f"→ {meta_path}", flush=True
    )
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 4-model soft-vote ensemble (3×CNN-Deep-Attn + EfficientNet-B2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        default="output/tile_labels",
        help="Directory containing *_labels.csv (default: output/tile_labels)",
    )
    parser.add_argument(
        "--chm-dir",
        default="chm_max_hag",
        help="CHM GeoTIFF directory (default: chm_max_hag)",
    )
    parser.add_argument(
        "--output",
        default="output/tile_labels",
        help="Output directory for checkpoints and ensemble_meta.json "
        "(default: output/tile_labels)",
    )
    parser.add_argument(
        "--test-split",
        default=None,
        help="Path to cnn_test_split.json for held-out evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check data pipeline only; skip training",
    )
    args = parser.parse_args()

    rc = train_ensemble(
        labels_dir=Path(args.labels),
        chm_dir=Path(args.chm_dir),
        output_dir=Path(args.output),
        test_split=Path(args.test_split) if args.test_split else None,
        dry_run=args.dry_run,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
