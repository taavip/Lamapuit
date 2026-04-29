#!/usr/bin/env python3
"""
Fine-tune CNN-Deep-Attn on human-labeled CHM tiles.

Loads all *_labels.csv from --labels-dir, reads the corresponding CHM tiles
from --chm-dir, builds a balanced dataset, and fine-tunes the existing
ensemble_model.pt checkpoint (or trains from scratch if not found).
Saves the updated checkpoint atomically and writes model_meta.json.

Artifact paths (relative to --output directory)
------------------------------------------------
  ensemble_model.pt      ← updated checkpoint (state_dict + build_fn_name + meta)
  model_meta.json        ← human-readable training metadata
  finetune.log           ← training log (written by parent process)

Usage
-----
# Fine-tune from existing checkpoint:
python scripts/fine_tune_cnn.py --labels-dir output/tile_labels --chm-dir chm_max_hag

# Dry-run (check data pipeline only, no training):
python scripts/fine_tune_cnn.py --labels-dir output/tile_labels --chm-dir chm_max_hag --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# ── File-naming constants ────────────────────────────────────────────────────
MODEL_PT = "ensemble_model.pt"  # main checkpoint
MODEL_PT_TMP = "ensemble_model.pt.tmp"  # atomic-save staging file
META_JSON = "model_meta.json"  # human-readable metadata
FINETUNE_DONE = "finetune_done.json"  # written on success (for hot-reload)

# ── Hyper-parameter defaults ─────────────────────────────────────────────────
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_MIN_PER_CLASS = 20
BACKBONE_LR_FACTOR = 0.1  # backbone gets lr * this (conservative fine-tune)
DEFAULT_LABEL_SMOOTH = 0.05  # CrossEntropyLoss label_smoothing


# ── Normalization (must match label_tiles.py CNNPredictor inference) ─────────
def _norm_tile(raw: np.ndarray) -> np.ndarray:
    """Clip CHM heights to 0–20 m and normalise to [0, 1]."""
    return np.clip(raw, 0.0, 20.0) / 20.0


# ── CNN architecture (must exactly match label_tiles.py _build_deep_cnn_attn_net) ──
def _build_deep_cnn_attn_net():
    """Build CNN-Deep-Attn network (exact copy from label_tiles / compare_classifiers)."""
    import torch.nn as nn

    class SE(nn.Module):
        def __init__(self, c, r=8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, max(c // r, 4)),
                nn.ReLU(),
                nn.Linear(max(c // r, 4), c),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)

    class AttnBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
            self.se = SE(out_c)
            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
            self.mp = nn.MaxPool2d(2)

        def forward(self, x):
            return self.mp(self.se(self.conv(x)) + self.skip(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(
                AttnBlock(1, 32),
                AttnBlock(32, 64),
                AttnBlock(64, 128),
                AttnBlock(128, 256),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2),
            )

        def forward(self, x):
            return self.head(self.blocks(x))

    return Net()


# ── Source quality weights ───────────────────────────────────────────────────
# Controls how much each label contributes to the training loss.
# Manually verified labels carry full weight; CNN auto-labels are down-weighted
# so the model trusts human annotations more than its own past predictions.
_SOURCE_WEIGHTS: dict[str, float] = {
    "manual": 1.00,
    "auto_reviewed": 0.85,
    "": 0.75,  # old CSVs without source column
    "auto": 0.60,
    "auto_skip": 0.30,
}


# ── Label loading ─────────────────────────────────────────────────────────────
def _load_labels(
    labels_dir: Path,
    review_set: set | None = None,
    review_action: str = "none",
    review_weight: float = 0.5,
) -> tuple[list[dict], dict]:
    """Return (records, stats).

    Supports optional review-set handling. *review_set* is a set of keys
    `(raster, row_off, col_off)` produced by the scanner. *review_action* can
    be:
      - "none"      : do nothing
      - "exclude"   : omit reviewed tiles from training
      - "downweight": include but set per-sample weight to *review_weight*

    Returns a tuple `(records, stats)` where stats contains counters useful
    for metadata (n_read, n_excluded, n_downweighted).
    """
    records: list[dict] = []
    stats = {"n_read": 0, "n_excluded": 0, "n_downweighted": 0}
    for csv_path in sorted(labels_dir.glob("*_labels.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                stats["n_read"] += 1
                lbl = row.get("label", "")
                if lbl not in ("cdw", "no_cdw"):
                    continue
                src = row.get("source", "")
                weight = _SOURCE_WEIGHTS.get(src, _SOURCE_WEIGHTS[""])

                key = (row["raster"], int(row["row_off"]), int(row["col_off"]))
                if review_set and key in review_set:
                    if review_action == "exclude":
                        stats["n_excluded"] += 1
                        continue
                    elif review_action == "downweight":
                        stats["n_downweighted"] += 1
                        weight = float(review_weight)

                records.append(
                    {
                        "raster": row["raster"],
                        "row_off": int(row["row_off"]),
                        "col_off": int(row["col_off"]),
                        "chunk_size": int(row.get("chunk_size", 128)),
                        "label": 1 if lbl == "cdw" else 0,
                        "source": src,
                        "weight": weight,
                    }
                )
    return records, stats


# ── Tile loading ──────────────────────────────────────────────────────────────
def _build_arrays(
    records: list[dict],
    chm_dir: Path,
    canonical_size: int = 128,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Read CHM tiles into (X, y, w) arrays.  X shape: (N, 1, H, W) float32.

    w  Per-sample quality weight derived from the `source` field.  Larger
       values mean the model trusts the label more during training.
    """
    by_raster: dict[str, list[dict]] = {}
    for rec in records:
        by_raster.setdefault(rec["raster"], []).append(rec)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []

    for raster_name, recs in by_raster.items():
        raster_path = chm_dir / raster_name
        if not raster_path.exists():
            matches = list(chm_dir.glob(f"{Path(raster_name).stem}*"))
            if not matches:
                print(
                    f"  [data] Raster not found: {raster_name} — skipping {len(recs)} tiles",
                    flush=True,
                )
                continue
            raster_path = matches[0]
        try:
            with rasterio.open(raster_path) as src:
                for rec in recs:
                    cs = rec["chunk_size"]
                    raw = src.read(
                        1,
                        window=Window(rec["col_off"], rec["row_off"], cs, cs),
                        boundless=True,
                        fill_value=0,
                    ).astype(np.float32)
                    if raw.shape != (canonical_size, canonical_size):
                        import cv2

                        raw = cv2.resize(raw, (canonical_size, canonical_size))
                    X_list.append(_norm_tile(raw))
                    y_list.append(rec["label"])
                    w_list.append(rec.get("weight", 1.0))
        except Exception as exc:
            print(f"  [data] Error reading {raster_path}: {exc} — skipping", flush=True)

    if not X_list:
        return None, None, None

    X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, H, W)
    y = np.array(y_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)
    return X, y, w


# ── Augmented dataset ─────────────────────────────────────────────────────────
class _TileDataset:
    """PyTorch dataset with augmentation and per-sample quality weights."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, augment: bool = True
    ):
        import torch

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.w = torch.from_numpy(w) if w is not None else torch.ones(len(y))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        import torch

        x = self.X[idx].clone()
        y = self.y[idx]
        w = self.w[idx]
        if self.augment:
            # Geometric augmentation (lossless reflections + 90° rotations)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-1])  # horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [-2])  # vertical flip
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, [-2, -1])  # 90° rotation
            # Intensity noise augmentations
            if torch.rand(1).item() > 0.7:  # Gaussian noise σ ≈ 1.5 %
                x = (x + torch.randn_like(x) * 0.015).clamp(0.0, 1.0)
            if torch.rand(1).item() > 0.75:  # Random pixel dropout
                x = x * (torch.rand_like(x) > 0.04)  # drop ~4 % of pixels
            if torch.rand(1).item() > 0.80:  # Brightness/contrast jitter
                alpha = 0.85 + torch.rand(1).item() * 0.30
                beta = (torch.rand(1).item() - 0.5) * 0.06
                x = (x * alpha + beta).clamp(0.0, 1.0)
            # Scale jitter: crop to 90–100 % then resize to canonical 128
            if torch.rand(1).item() > 0.70:
                h, w_s = x.shape[-2], x.shape[-1]
                scale = 0.90 + torch.rand(1).item() * 0.10  # [0.90, 1.00]
                ch = int(h * scale)
                cw = int(w_s * scale)
                r0 = torch.randint(0, max(h - ch, 1), (1,)).item()
                c0 = torch.randint(0, max(w_s - cw, 1), (1,)).item()
                patch = x[:, r0 : r0 + ch, c0 : c0 + cw]
                import torch.nn.functional as F

                x = F.interpolate(
                    patch.unsqueeze(0), size=(h, w_s), mode="bilinear", align_corners=False
                ).squeeze(0)
            # Random erasing: zero a rectangular patch (simulate occluded/missing data)
            if torch.rand(1).item() > 0.80:
                h, w_s = x.shape[-2], x.shape[-1]
                eh = int(h * (0.05 + torch.rand(1).item() * 0.10))  # 5–15 %
                ew = int(w_s * (0.05 + torch.rand(1).item() * 0.10))
                er = torch.randint(0, max(h - eh, 1), (1,)).item()
                ec = torch.randint(0, max(w_s - ew, 1), (1,)).item()
                x = x.clone()
                x[:, er : er + eh, ec : ec + ew] = 0.0
        return x, y, w


# ── Mixup ────────────────────────────────────────────────────────────────────
def _mixup_batch(xb, yb, wb, alpha: float = 0.3):
    """Mixup regularization: linearly interpolate pairs of training samples.

    Returns (mixed_x, y_a, y_b, lam, mixed_w) — loss computed as weighted
    sum of per-sample CrossEntropyLoss (requires criterion reduction='none').
    """
    import torch
    import numpy as np

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xb.size(0), device=xb.device)
    return (
        lam * xb + (1.0 - lam) * xb[perm],
        yb,
        yb[perm],
        lam,
        lam * wb + (1.0 - lam) * wb[perm],
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def _compute_metrics(
    net,
    X_t,
    y_t,
    device,
    batch_size: int = 64,
    tta: bool = False,
) -> tuple[float, float, float]:
    """Return (auc, best_f1, best_thresh) computed on a validation/test set.

    When *tta=True* averages predictions over 8 views (4 rotations × 2 h-flips)
    for more stable probability estimates.
    """
    import torch
    from sklearn.metrics import roc_auc_score, f1_score

    net.eval()
    all_probs: list[float] = []
    all_y: list[int] = []

    def _fwd(x_batch):
        return torch.softmax(net(x_batch), dim=1)[:, 1]

    with torch.no_grad():
        for i in range(0, len(y_t), batch_size):
            xb = X_t[i : i + batch_size].to(device)
            if tta:
                views = []
                for k in range(4):
                    v = torch.rot90(xb, k, [-2, -1])
                    views.append(_fwd(v))
                    views.append(_fwd(torch.flip(v, [-1])))
                pb = torch.stack(views, dim=0).mean(dim=0).cpu().numpy()
            else:
                pb = _fwd(xb).cpu().numpy()
            all_probs.extend(pb.tolist())
            all_y.extend(y_t[i : i + batch_size].numpy().tolist())

    probs = np.array(all_probs)
    labels = np.array(all_y)

    if len(np.unique(labels)) < 2:
        return 0.5, 0.5, 0.5

    auc = float(roc_auc_score(labels, probs))

    best_f1, best_thresh = 0.0, 0.5
    for thr in np.linspace(0.10, 0.90, 81):
        preds = (probs >= thr).astype(int)
        f1_ = float(f1_score(labels, preds, zero_division=0))
        if f1_ >= best_f1:
            best_f1, best_thresh = f1_, float(thr)

    return auc, best_f1, best_thresh


# ── Main training routine ─────────────────────────────────────────────────────
def fine_tune(
    labels_dir: Path,
    chm_dir: Path,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    lr: float = DEFAULT_LR,
    dry_run: bool = False,
    test_split: Path | None = None,
    label_smoothing: float = DEFAULT_LABEL_SMOOTH,
    mixup_alpha: float = 0.3,
    review_set: set | None = None,
    review_action: str = "none",
    review_weight: float = 0.5,
) -> int:
    """
    Fine-tune CNN-Deep-Attn on collected labels.

    Returns
    -------
    0  success
    1  not enough labeled data
    2  error
    """
    import torch
    from torch.utils.data import DataLoader

    t0 = time.monotonic()

    # ── Load labels ──────────────────────────────────────────────────────────
    print(f"[fine_tune] Loading labels from {labels_dir} …", flush=True)
    records, review_stats = _load_labels(
        labels_dir, review_set=review_set, review_action=review_action, review_weight=review_weight
    )
    n_cdw = sum(1 for r in records if r["label"] == 1)
    n_no = sum(1 for r in records if r["label"] == 0)
    n_total = len(records)
    print(f"[fine_tune] Labels: {n_total} total  CDW={n_cdw}  No-CDW={n_no}", flush=True)

    if n_cdw < min_per_class or n_no < min_per_class:
        print(
            f"[fine_tune] Not enough data (need ≥ {min_per_class} per class). Skipping.",
            flush=True,
        )
        return 1

    if dry_run:
        print("[fine_tune] Dry-run: data OK, skipping training.", flush=True)
        return 0

    # ── Build tile arrays ─────────────────────────────────────────────────────
    print(f"[fine_tune] Reading tiles from {chm_dir} …", flush=True)
    X, y, w = _build_arrays(records, chm_dir)
    if X is None or len(X) < min_per_class * 2:
        print("[fine_tune] Dataset too small after tile loading. Skipping.", flush=True)
        return 1

    y_cdw = int(np.sum(y == 1))
    y_no = int(np.sum(y == 0))
    print(f"[fine_tune] Dataset ready: {len(X)} tiles  CDW={y_cdw}  No-CDW={y_no}", flush=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fine_tune] Device: {device}", flush=True)

    # ── Load existing checkpoint (or start fresh) ─────────────────────────────
    pt_path = output_dir / MODEL_PT
    old_version = 0
    net = _build_deep_cnn_attn_net().to(device)

    if pt_path.exists():
        try:
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt["state_dict"])
            old_version = ckpt.get("meta", {}).get("version", 0)
            print(f"[fine_tune] Loaded checkpoint v{old_version} from {pt_path}", flush=True)
        except Exception as exc:
            print(
                f"[fine_tune] Warning: could not load {pt_path} ({exc}) — training from scratch",
                flush=True,
            )
    else:
        print(f"[fine_tune] No existing checkpoint — training from scratch", flush=True)

    new_version = old_version + 1

    # ── Stratified 80/20 train/val split ──────────────────────────────────────
    rng = np.random.default_rng(42)
    idx_cdw = np.where(y == 1)[0]
    rng.shuffle(idx_cdw)
    idx_no = np.where(y == 0)[0]
    rng.shuffle(idx_no)

    n_val_cdw = max(1, len(idx_cdw) // 5)
    n_val_no = max(1, len(idx_no) // 5)
    val_idx = np.concatenate([idx_cdw[:n_val_cdw], idx_no[:n_val_no]])
    train_idx = np.concatenate([idx_cdw[n_val_cdw:], idx_no[n_val_no:]])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"[fine_tune] Split: train={len(X_train)} / val={len(X_val)}", flush=True)

    # ── Class-weighted loss (handles imbalance) ───────────────────────────────
    n_neg_tr = int(np.sum(y_train == 0))
    n_pos_tr = int(np.sum(y_train == 1))
    w_pos = n_neg_tr / max(n_pos_tr, 1)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)
    # label_smoothing prevents overconfidence; reduction='none' needed for per-sample weights
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    # ── Optimizer: lower LR for backbone (conservative fine-tune) ─────────────
    optimizer = torch.optim.Adam(
        [
            {"params": list(net.blocks.parameters()), "lr": lr * BACKBONE_LR_FACTOR},
            {"params": list(net.head.parameters()), "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── Data loader ───────────────────────────────────────────────────────────
    train_ds = _TileDataset(X_train, y_train.astype(np.int64), w=w[train_idx], augment=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    # ── Training loop ─────────────────────────────────────────────────────────
    log_every = max(1, epochs // 5)
    best_f1 = 0.0
    best_thresh = 0.5
    best_auc = 0.5
    best_state: dict | None = None

    print(
        f"[fine_tune] Starting {epochs} epochs  "
        f"batch={batch_size}  lr={lr}  backbone_lr={lr * BACKBONE_LR_FACTOR}  "
        f"label_smooth={label_smoothing}  mixup_alpha={mixup_alpha}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        for xb, yb, wb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            wb = wb.to(device).float()
            optimizer.zero_grad()
            # Apply Mixup with probability 0.5 to avoid overfitting on small datasets
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
                f"[fine_tune] epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  val_auc={auc:.4f}  val_f1={f1_:.4f}@{thr_:.2f}",
                flush=True,
            )
            if f1_ >= best_f1:
                best_f1 = f1_
                best_thresh = thr_
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    # Fallback if best_state was never captured (should not happen)
    if best_state is None:
        best_auc, best_f1, best_thresh = _compute_metrics(net, X_val_t, y_val_t, device, tta=False)
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    # ── Atomic save ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_tmp = output_dir / MODEL_PT_TMP

    new_meta = {
        "model_name": "CNN-Deep-Attn",
        "version": new_version,
        "best_thresh": round(best_thresh, 4),
        "best_f1": round(best_f1, 4),
        "auc": round(best_auc, 4),
        "n_samples": int(len(X)),
        "n_cdw": int(y_cdw),
        "n_no_cdw": int(y_no),
        "n_labels_at_train": n_total,
        "n_review_excluded": int(review_stats.get("n_excluded", 0)),
        "n_review_downweighted": int(review_stats.get("n_downweighted", 0)),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
    }
    checkpoint = {
        "build_fn_name": "_build_deep_cnn_attn",
        "state_dict": best_state,
        "meta": new_meta,
    }
    torch.save(checkpoint, pt_tmp)
    pt_tmp.replace(pt_path)  # atomic rename on POSIX

    # Human-readable metadata sidecar
    (output_dir / META_JSON).write_text(json.dumps(new_meta, indent=2))

    # Done token (hot-reload detector in labeler)
    (output_dir / FINETUNE_DONE).write_text(
        json.dumps({"version": new_version, "done_at": datetime.now(timezone.utc).isoformat()})
    )

    # ── Optional test-set evaluation (TTA) ─────────────────────────────────
    if test_split is not None and test_split.exists():
        try:
            split_data = json.loads(test_split.read_text())
            test_keys = {(r, ro, co) for r, ro, co in split_data["keys"]}
            # Reload best weights before test eval
            net.load_state_dict(best_state)
            net.to(device)
            # Build test arrays from the held-out keys
            test_records = [
                rec
                for rec in records
                if (rec["raster"], rec["row_off"], rec["col_off"]) in test_keys
            ]
            if test_records:
                X_test, y_test, _ = _build_arrays(test_records, chm_dir)
                if X_test is not None and len(np.unique(y_test)) > 1:
                    X_test_t = torch.from_numpy(X_test).to(device)
                    y_test_t = torch.from_numpy(y_test.astype(np.int64))
                    test_auc, test_f1, test_thr = _compute_metrics(
                        net, X_test_t, y_test_t, device, tta=True
                    )
                    new_meta["test_metrics"] = {
                        "auc": round(test_auc, 4),
                        "f1": round(test_f1, 4),
                        "thresh": round(test_thr, 2),
                        "n_test": int(len(y_test)),
                        "n_cdw": int(np.sum(y_test == 1)),
                        "tta": True,
                    }
                    print(
                        f"[fine_tune] Test-set (TTA) "
                        f"AUC={test_auc:.4f}  F1={test_f1:.4f}@{test_thr:.2f}  "
                        f"n={len(y_test)}",
                        flush=True,
                    )
                    # Refresh meta sidecar with test metrics
                    (output_dir / META_JSON).write_text(json.dumps(new_meta, indent=2))
        except Exception as exc:
            print(f"[fine_tune] Test-split eval failed: {exc}", flush=True)

    elapsed = time.monotonic() - t0
    print(
        f"[fine_tune] ✓ Saved v{new_version} in {elapsed:.1f}s  "
        f"AUC={best_auc:.4f}  F1={best_f1:.4f}@{best_thresh:.2f}  → {pt_path}",
        flush=True,
    )
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune CNN-Deep-Attn on collected CDW tile labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Artifacts written to --output:\n"
            "  ensemble_model.pt  — updated PyTorch checkpoint\n"
            "  model_meta.json    — training statistics\n"
            "  finetune_done.json — version token for hot-reload detection"
        ),
    )
    parser.add_argument(
        "--labels-dir",
        default="output/tile_labels",
        help="Directory containing *_labels.csv files (default: output/tile_labels)",
    )
    parser.add_argument(
        "--chm-dir",
        default="chm_max_hag",
        help="Directory containing CHM GeoTIFF rasters (default: chm_max_hag)",
    )
    parser.add_argument(
        "--output",
        default="output/tile_labels",
        help="Output directory for checkpoint and metadata (default: output/tile_labels)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Fine-tuning epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=DEFAULT_MIN_PER_CLASS,
        help=f"Minimum labeled samples per class required (default: {DEFAULT_MIN_PER_CLASS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Head learning rate; backbone uses {BACKBONE_LR_FACTOR}× (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data pipeline only; do not train or save",
    )
    parser.add_argument(
        "--test-split",
        default=None,
        help="Path to cnn_test_split.json (from create_cnn_test_split.py). "
        "When provided, the best model is evaluated with TTA on the held-out test set "
        "and metrics are appended to model_meta.json.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTH,
        help=f"CrossEntropyLoss label_smoothing (default: {DEFAULT_LABEL_SMOOTH})",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.3,
        help="Mixup beta distribution alpha (0 disables Mixup, default: 0.3)",
    )
    parser.add_argument(
        "--review-queue",
        default=None,
        help="CSV file with raster,row_off,col_off for reviewed/problem tiles (from scanner).",
    )
    parser.add_argument(
        "--review-action",
        choices=("none", "exclude", "downweight"),
        default="none",
        help="What to do with tiles listed in --review-queue: exclude/downweight/none",
    )
    parser.add_argument(
        "--review-weight",
        type=float,
        default=0.5,
        help="Per-sample weight when --review-action=downweight (default 0.5)",
    )
    args = parser.parse_args()

    rc = fine_tune(
        labels_dir=Path(args.labels_dir),
        chm_dir=Path(args.chm_dir),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_per_class=args.min_per_class,
        lr=args.lr,
        dry_run=args.dry_run,
        test_split=Path(args.test_split) if args.test_split else None,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        review_set=None,  # filled below
        review_action=args.review_action,
        review_weight=args.review_weight,
    )
    # If a review-queue was provided, parse it and re-run with review_set
    if args.review_queue:
        try:
            import csv as _csv

            review_set = set()
            with open(args.review_queue, newline="") as _f:
                for _r in _csv.DictReader(_f):
                    review_set.add((_r["raster"], int(_r["row_off"]), int(_r["col_off"])))
            rc = fine_tune(
                labels_dir=Path(args.labels_dir),
                chm_dir=Path(args.chm_dir),
                output_dir=Path(args.output),
                epochs=args.epochs,
                batch_size=args.batch_size,
                min_per_class=args.min_per_class,
                lr=args.lr,
                dry_run=args.dry_run,
                test_split=Path(args.test_split) if args.test_split else None,
                label_smoothing=args.label_smoothing,
                mixup_alpha=args.mixup_alpha,
                review_set=review_set,
                review_action=args.review_action,
                review_weight=args.review_weight,
            )
        except Exception as exc:
            print(
                f"[fine_tune] Failed to parse review_queue {args.review_queue}: {exc}", flush=True
            )
    sys.exit(rc)


if __name__ == "__main__":
    main()
