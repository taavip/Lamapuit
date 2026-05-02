#!/usr/bin/env python3
"""
Ensemble 4th-model ablation: which architecture best complements CNN-Deep-Attn×3?

Research question
-----------------
The production ensemble (CNN-Deep-Attn×3 + EfficientNet-B2, F1=0.9819) chose
EfficientNet-B2 on principled reasoning (architectural diversity, ImageNet
pretraining) without a direct comparison to alternatives. This ablation
trains three architecturally distinct 4th-member candidates under fully
matched conditions and evaluates each as the ensemble's 4th member on the
same held-out test set.

Candidates (all trained from ImageNet weights on 15K subset, 30 epochs)
-----------------------------------------------------------------------
  A  CNN-Deep-Attn×3  +  EfficientNet-B2   (subset, ImageNet pretrained)
  B  CNN-Deep-Attn×3  +  ConvNeXt-tiny     (subset, ImageNet pretrained)
  C  CNN-Deep-Attn×3  +  ConvNeXt-small    (subset, ImageNet pretrained)

Reference (not part of ablation, listed for context)
----------------------------------------------------
  P  CNN-Deep-Attn×3  +  EfficientNet-B2   (production checkpoint, full 67K)
     F1 = 0.9819, AUC = 0.9885

Why this is a fair comparison
-----------------------------
All three candidates share:
  • identical training subset (15K tiles, stratified, production class ratio)
  • identical val subset (4K tiles)
  • identical optimizer, LR schedule, batch size, augmentation, MixUp, label smoothing
  • identical ImageNet-pretrained init (first conv averaged over 3 RGB channels
    to 1 input channel; classifier head replaced with 2-class output)
  • identical TTA at inference (4 rotations)
  • identical CNN×3 partners (loaded from production checkpoints)

The only varying factor is the 4th-model architecture itself. Any difference
in ensemble F1 reflects architectural fit, not training data, pretraining,
or hyperparameter advantages.

Protocol
--------
  • CNN×3: loaded from output/tile_labels_spatial_splits/ (NOT retrained).
  • A/B/C: trained on 15K stratified subset preserving production class ratio,
    30 epochs, ImageNet-pretrained init, identical hyperparameters.
  • Test set: full 56 521 held-out tiles (streamed, not pre-loaded).
  • TTA: 4 rotations averaged.

Reliability features
---------------------
  • Internal log file (no reliance on `tee`).
  • Val arrays cached to .npy (fast restart).
  • Checkpoint resumption: skip training if checkpoint exists.
  • Streaming test inference (no full RAM preload of 56K test windows).
  • Per-epoch validation; early stopping with patience=5.
  • Crash-safe: each completed step writes a partial-results JSON.

Outputs (output/ensemble_4th_model_ablation/)
---------------------------------------------
  convnext_tiny_ablation.pt
  convnext_small_ablation.pt
  ablation_results.json
  ablation_summary.txt
  run.log
  cache/X_val.npy, y_val.npy   (cached subsetted val arrays)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _get_build_fn, _instantiate_model_from_build_fn


# ── Local pretrained builders (ensures ImageNet init for fair comparison) ─────

def _adapt_first_conv_rgb_to_1ch(conv: nn.Conv2d) -> nn.Conv2d:
    """Average pretrained 3-channel weights to a 1-channel conv (preserves features)."""
    avg = conv.weight.data.mean(dim=1, keepdim=True)
    new = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    new.weight.data = avg
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    return new


def build_effnet_b2_pretrained() -> nn.Module:
    from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

    m = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def build_convnext_tiny_pretrained() -> nn.Module:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
    # ConvNeXt classifier: Sequential(LayerNorm2d, Flatten, Linear)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_f, 2)
    return m


def build_convnext_small_pretrained() -> nn.Module:
    from torchvision.models import ConvNeXt_Small_Weights, convnext_small

    m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_f, 2)
    return m

# ── Configuration ─────────────────────────────────────────────────────────────

LABELS_CSV = Path("data/chm_variants/labels_canonical_with_splits.csv")
CHM_DIR    = Path("data/chm_variants/baseline_chm_20cm")
CNN_CKPTS  = Path("output/tile_labels_spatial_splits")
OUTPUT_DIR = Path("output/ensemble_4th_model_ablation")
CACHE_DIR  = OUTPUT_DIR / "cache"

SUBSET_TRAIN_N = 15_000
SUBSET_VAL_N   =  4_000
EPOCHS         = 30
EARLY_STOP_PATIENCE = 5
BATCH_SIZE     = 16
LR_HEAD        = 5e-4
LR_BACKBONE    = 5e-5     # 10× lower than head; required for ImageNet-pretrained backbones
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA    = 0.3
CNN_SEEDS      = (42, 43, 44)
SUBSET_SEED    = 2026
TTA_VIEWS      = 4   # 4 rotations only (8-view added < 0.001 F1 but cost 2× inference)
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRODUCTION_F1  = 0.9819397825760232
PRODUCTION_AUC = 0.9884721096928946


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run.log"
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


log = setup_logging()


# ── CHM I/O ───────────────────────────────────────────────────────────────────

def _norm(tile: np.ndarray) -> np.ndarray:
    return np.clip(tile, 0.0, 20.0) / 20.0


def load_window(raster_name: str, row_off: int, col_off: int) -> np.ndarray | None:
    p = CHM_DIR / raster_name
    if not p.exists():
        return None
    try:
        with rasterio.open(p) as src:
            data = src.read(1, window=Window(col_off, row_off, 128, 128)).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return _norm(np.nan_to_num(data, nan=0.0))
    except Exception:
        return None


# ── Datasets ──────────────────────────────────────────────────────────────────

class TileDataset(Dataset):
    """For training (with optional augmentation) and val (no aug)."""

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        chm = load_window(row["raster"], int(row["row_off"]), int(row["col_off"]))
        if chm is None:
            chm = np.zeros((128, 128), dtype=np.float32)
        x = torch.tensor(chm, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(1 if row["label"] == "cdw" else 0, dtype=torch.long)
        if self.augment:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-1])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [-2])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, [-2, -1])
            if torch.rand(1) > 0.7:
                x = (x + torch.randn_like(x) * 0.015).clamp(0.0, 1.0)
            if torch.rand(1) > 0.8:
                a = 0.85 + torch.rand(1).item() * 0.30
                b = (torch.rand(1).item() - 0.5) * 0.06
                x = (x * a + b).clamp(0.0, 1.0)
        return x, y


class TestDataset(Dataset):
    """For streaming test inference — same as TileDataset but explicit-purpose."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        chm = load_window(row["raster"], int(row["row_off"]), int(row["col_off"]))
        if chm is None:
            chm = np.zeros((128, 128), dtype=np.float32)
        x = torch.tensor(chm, dtype=torch.float32).unsqueeze(0)
        y = 1 if row["label"] == "cdw" else 0
        return x, y


# ── Subset sampling (preserves class ratio) ───────────────────────────────────

def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample n rows preserving the original class ratio (production behavior)."""
    n = min(n, len(df))
    cdw_frac = (df["label"] == "cdw").mean()
    n_cdw = int(round(n * cdw_frac))
    n_no  = n - n_cdw
    cdw   = df[df["label"] == "cdw"].sample(n=min(n_cdw, (df["label"] == "cdw").sum()),
                                            random_state=seed)
    no_cdw = df[df["label"] != "cdw"].sample(n=min(n_no, (df["label"] != "cdw").sum()),
                                             random_state=seed)
    return pd.concat([cdw, no_cdw]).sample(frac=1, random_state=seed).reset_index(drop=True)


def cached_arrays(df: pd.DataFrame, cache_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load val arrays from cache, building if missing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x_path = CACHE_DIR / f"X_{cache_prefix}.npy"
    y_path = CACHE_DIR / f"y_{cache_prefix}.npy"
    if x_path.exists() and y_path.exists():
        log.info(f"  cache hit: {x_path.name}")
        return np.load(x_path), np.load(y_path)
    log.info(f"  building cache for {cache_prefix} ({len(df)} tiles)...")
    X, y = [], []
    for _, row in df.iterrows():
        chm = load_window(row["raster"], int(row["row_off"]), int(row["col_off"]))
        if chm is None:
            continue
        X.append(chm)
        y.append(1 if row["label"] == "cdw" else 0)
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    np.save(x_path, X_arr); np.save(y_path, y_arr)
    log.info(f"  cached → {x_path.name} ({X_arr.nbytes/1e9:.2f} GB)")
    return X_arr, y_arr


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    df_train: pd.DataFrame,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    tag: str,
) -> dict:
    """Train with per-epoch validation, MixUp, early stopping. Returns best metrics.

    Uses dual learning rates for ImageNet-pretrained models:
    - backbone (pretrained features): LR_BACKBONE = 5e-5
    - classifier head (randomly initialized): LR_HEAD = 5e-4
    Both EfficientNet-B2 and ConvNeXt expose .classifier — same code path works.
    """
    model = model.to(DEVICE)

    n_neg = int((df_train["label"] != "cdw").sum())
    n_pos = int((df_train["label"] == "cdw").sum())
    w_pos = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()

    # Dual-LR: pretrained backbone gets 10× lower LR than the new classifier head.
    # This matches V2 ConvNeXt training (lr_head=5e-4, lr_backbone=5e-5) and
    # the production EfficientNet-B2 setup. Without it, ConvNeXt collapses to
    # trivial-prediction loss (val_auc ≈ 0.5) on the first epoch.
    if hasattr(model, "classifier"):
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        param_groups = [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ]
        log.info(f"  dual-LR: backbone={LR_BACKBONE:g}  head={LR_HEAD:g}  "
                 f"({len(backbone_params)} backbone tensors, {len(head_params)} head tensors)")
    else:
        param_groups = [{"params": list(model.parameters()), "lr": LR_HEAD}]
        log.info(f"  uniform LR={LR_HEAD:g}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(SUBSET_SEED)

    train_dl = DataLoader(TileDataset(df_train, augment=True), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)

    X_val_t = torch.from_numpy(X_val).float().unsqueeze(1)
    y_val_t = torch.from_numpy(y_val).long().to(DEVICE)

    best_val_loss = float("inf")
    best_state: dict | None = None
    best_metrics = {"val_f1": 0.0, "val_auc": 0.0, "val_thresh": 0.5, "best_val_loss": float("inf"),
                    "best_epoch": 0}
    no_improve = 0
    t0 = time.monotonic()

    log.info(f"[{tag}] train={len(df_train)} (CDW={n_pos}, NO_CDW={n_neg})  "
             f"val={len(X_val)}  epochs={epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            lam = float(rng.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            perm = torch.randperm(xb.size(0), device=DEVICE)
            xb_m = lam * xb + (1 - lam) * xb[perm]
            optimizer.zero_grad()
            logits = model(xb_m)
            loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[perm])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()

        # Per-epoch validation (cheap with cached arrays)
        model.eval()
        logits_parts, probs_parts = [], []
        with torch.no_grad():
            for i in range(0, len(X_val_t), 512):
                xb = X_val_t[i:i+512].to(DEVICE)
                lg = model(xb)
                logits_parts.append(lg.cpu())
                probs_parts.append(torch.softmax(lg, dim=1)[:, 1].cpu().numpy())
        logits_all = torch.cat(logits_parts).to(DEVICE)
        val_loss = float(eval_criterion(logits_all, y_val_t).item())
        vp = np.concatenate(probs_parts)
        val_auc = float(roc_auc_score(y_val, vp))
        best_f1_e, best_thr_e = 0.0, 0.5
        for thr in np.linspace(0.1, 0.9, 81):
            f1 = float(f1_score(y_val, (vp >= thr).astype(int), zero_division=0))
            if f1 >= best_f1_e:
                best_f1_e, best_thr_e = f1, float(thr)

        improved = val_loss < best_val_loss
        log.info(f"  ep {epoch:3d}/{epochs}  loss={ep_loss/len(train_dl):.4f}  "
                 f"val_loss={val_loss:.4f}  val_f1={best_f1_e:.4f}  val_auc={val_auc:.4f}  "
                 f"{'★' if improved else ' '}  [{time.monotonic()-t0:.0f}s]")

        if improved:
            best_val_loss = val_loss
            best_metrics = {"val_f1": best_f1_e, "val_auc": val_auc, "val_thresh": best_thr_e,
                            "best_val_loss": val_loss, "best_epoch": epoch}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                log.info(f"  early stop @ epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics


# ── Streaming TTA inference ───────────────────────────────────────────────────

def streaming_tta_probs(model: nn.Module, df_test: pd.DataFrame, num_views: int = 4) -> np.ndarray:
    """Stream test set through 4-view (rotation-only) TTA. No full preload."""
    model.eval()
    dl = DataLoader(TestDataset(df_test), batch_size=256, shuffle=False, num_workers=2)
    probs = np.zeros(len(df_test), dtype=np.float64)
    cursor = 0
    with torch.no_grad():
        for xb, _yb in dl:
            xb = xb.to(DEVICE)
            view_acc = torch.zeros(xb.size(0), device=DEVICE, dtype=torch.float32)
            for k in range(num_views):
                v = xb if k == 0 else torch.rot90(xb, k, [-2, -1])
                view_acc += torch.softmax(model(v), dim=1)[:, 1]
            view_acc /= num_views
            probs[cursor:cursor + xb.size(0)] = view_acc.cpu().numpy()
            cursor += xb.size(0)
    return probs.astype(np.float32)


# ── Ensemble eval (probabilities-only API; no logits needed) ──────────────────

def eval_ensemble(probs_list: list[np.ndarray], y: np.ndarray) -> dict:
    ens = np.mean(probs_list, axis=0)
    auc = float(roc_auc_score(y, ens))
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        f1 = float(f1_score(y, (ens >= thr).astype(int), zero_division=0))
        if f1 >= best_f1:
            best_f1, best_thr = f1, float(thr)
    preds = (ens >= best_thr).astype(int)
    return {
        "ensemble_f1": best_f1,
        "ensemble_auc": auc,
        "ensemble_thresh": best_thr,
        "ensemble_precision": float(precision_score(y, preds, zero_division=0)),
        "ensemble_recall": float(recall_score(y, preds, zero_division=0)),
    }


def write_partial_results(results: dict) -> None:
    (OUTPUT_DIR / "ablation_results.json").write_text(json.dumps(results, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.monotonic()
    log.info(f"\n{'='*72}")
    log.info("Ensemble 4th-model ablation")
    log.info(f"Device: {DEVICE}   subset_train={SUBSET_TRAIN_N}   epochs={EPOCHS}   tta={TTA_VIEWS}")
    log.info(f"{'='*72}")

    # ── 1. Load CSV, build subsets ────────────────────────────────────────────
    log.info("[1/5] Loading labels CSV...")
    df = pd.read_csv(LABELS_CSV)
    df_train_full = df[df["split"] == "train"].copy()
    df_test_full  = df[df["split"] == "test"].copy()
    df_val_full   = df[df["split"] == "val"].copy()
    log.info(f"  full splits — train={len(df_train_full)}  val={len(df_val_full)}  "
             f"test={len(df_test_full)}")

    df_train_sub = stratified_subset(df_train_full, SUBSET_TRAIN_N, SUBSET_SEED)
    df_val_sub   = stratified_subset(df_val_full,   SUBSET_VAL_N,   SUBSET_SEED)
    cdw_tr = int((df_train_sub["label"] == "cdw").sum())
    log.info(f"  subsets — train={len(df_train_sub)} (CDW={cdw_tr}, "
             f"NO_CDW={len(df_train_sub)-cdw_tr})  val={len(df_val_sub)}")

    log.info("[1/5] Loading val arrays (cached)...")
    X_val, y_val = cached_arrays(df_val_sub, "val_subset")
    log.info(f"  X_val={X_val.shape}")

    y_test = df_test_full["label"].apply(lambda v: 1 if v == "cdw" else 0).to_numpy(dtype=np.int64)
    log.info(f"[1/5] Test set: {len(df_test_full)} tiles  CDW={int(np.sum(y_test==1))}  "
             f"NO_CDW={int(np.sum(y_test==0))} (streamed, not preloaded)")

    # ── 2. CNN×3 streaming test probabilities ─────────────────────────────────
    log.info("[2/5] Loading pre-trained CNN×3 and computing test probs...")
    cnn_test_probs: list[np.ndarray] = []
    for seed in CNN_SEEDS:
        ckpt_path = CNN_CKPTS / f"cnn_seed{seed}_spatial.pt"
        log.info(f"  {ckpt_path.name} ...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        net = _instantiate_model_from_build_fn(_get_build_fn("_build_deep_cnn_attn")).to(DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        cache_p = CACHE_DIR / f"probs_cnn_seed{seed}.npy"
        if cache_p.exists():
            p = np.load(cache_p)
            log.info(f"    cache hit  mean={p.mean():.3f}")
        else:
            p = streaming_tta_probs(net, df_test_full, TTA_VIEWS)
            np.save(cache_p, p)
            log.info(f"    computed   mean={p.mean():.3f}")
        cnn_test_probs.append(p)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 3. EfficientNet-B2 (candidate A) — train on subset with ImageNet init ─
    log.info("[3/5] EfficientNet-B2 (candidate A) — pretrained, 15K subset...")
    ckpt_a = OUTPUT_DIR / "effnet_b2_ablation.pt"
    if ckpt_a.exists():
        log.info(f"  resume: loading {ckpt_a.name}")
        ca = torch.load(ckpt_a, map_location=DEVICE)
        effnet = build_effnet_b2_pretrained().to(DEVICE)
        effnet.load_state_dict(ca["state_dict"])
        val_a = ca.get("meta", {})
    else:
        torch.manual_seed(0); np.random.seed(0)
        effnet = build_effnet_b2_pretrained()
        val_a = train_model(effnet, df_train_sub, X_val, y_val, EPOCHS, "EfficientNet-B2")
        torch.save({"state_dict": effnet.state_dict(),
                    "build_fn_name": "build_effnet_b2_pretrained", "meta": val_a}, ckpt_a)
        log.info(f"  saved: {ckpt_a.name}")
    cache_a = CACHE_DIR / "probs_effnet_b2_ablation.npy"
    if cache_a.exists():
        eff_probs = np.load(cache_a)
    else:
        eff_probs = streaming_tta_probs(effnet, df_test_full, TTA_VIEWS)
        np.save(cache_a, eff_probs)
    del effnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ens_a = eval_ensemble(cnn_test_probs + [eff_probs], y_test)
    log.info(f"  Ensemble A (CNN×3 + EfficientNet-B2): F1={ens_a['ensemble_f1']:.4f}  "
             f"AUC={ens_a['ensemble_auc']:.4f}")

    partial = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fourth_model_val_metrics": {"A_efficientnet_b2": val_a},
        "ensemble_test_results": {"A_cnn3_efficientnet_b2": {**ens_a,
                                  "label": "CNN×3 + EfficientNet-B2 (15K subset, pretrained)"}},
    }
    write_partial_results(partial)

    # ── 4. ConvNeXt-tiny (candidate B) — pretrained, same 15K subset ─────────
    log.info("[4/5] ConvNeXt-tiny (candidate B) — pretrained, 15K subset...")
    ckpt_b = OUTPUT_DIR / "convnext_tiny_ablation.pt"
    if ckpt_b.exists():
        log.info(f"  resume: loading {ckpt_b.name}")
        cb = torch.load(ckpt_b, map_location=DEVICE)
        cxt = build_convnext_tiny_pretrained().to(DEVICE)
        cxt.load_state_dict(cb["state_dict"])
        val_b = cb.get("meta", {})
    else:
        torch.manual_seed(0); np.random.seed(0)
        cxt = build_convnext_tiny_pretrained()
        val_b = train_model(cxt, df_train_sub, X_val, y_val, EPOCHS, "ConvNeXt-tiny")
        torch.save({"state_dict": cxt.state_dict(),
                    "build_fn_name": "build_convnext_tiny_pretrained", "meta": val_b}, ckpt_b)
        log.info(f"  saved: {ckpt_b.name}")
    cache_b = CACHE_DIR / "probs_convnext_tiny.npy"
    if cache_b.exists():
        cxt_probs = np.load(cache_b)
    else:
        cxt_probs = streaming_tta_probs(cxt, df_test_full, TTA_VIEWS)
        np.save(cache_b, cxt_probs)
    del cxt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ens_b = eval_ensemble(cnn_test_probs + [cxt_probs], y_test)
    log.info(f"  Ensemble B: F1={ens_b['ensemble_f1']:.4f}  AUC={ens_b['ensemble_auc']:.4f}")

    partial["ensemble_test_results"]["B_cnn3_convnext_tiny"] = {**ens_b,
        "label": "CNN×3 + ConvNeXt-tiny (15K subset)"}
    partial["fourth_model_val_metrics"] = {"B_convnext_tiny": val_b}
    write_partial_results(partial)

    # ── 5. ConvNeXt-small (candidate C) — pretrained, same 15K subset ────────
    log.info("[5/5] ConvNeXt-small (candidate C) — pretrained, 15K subset...")
    ckpt_c = OUTPUT_DIR / "convnext_small_ablation.pt"
    if ckpt_c.exists():
        log.info(f"  resume: loading {ckpt_c.name}")
        cc = torch.load(ckpt_c, map_location=DEVICE)
        cxs = build_convnext_small_pretrained().to(DEVICE)
        cxs.load_state_dict(cc["state_dict"])
        val_c = cc.get("meta", {})
    else:
        torch.manual_seed(0); np.random.seed(0)
        cxs = build_convnext_small_pretrained()
        val_c = train_model(cxs, df_train_sub, X_val, y_val, EPOCHS, "ConvNeXt-small")
        torch.save({"state_dict": cxs.state_dict(),
                    "build_fn_name": "build_convnext_small_pretrained", "meta": val_c}, ckpt_c)
        log.info(f"  saved: {ckpt_c.name}")
    cache_c = CACHE_DIR / "probs_convnext_small.npy"
    if cache_c.exists():
        cxs_probs = np.load(cache_c)
    else:
        cxs_probs = streaming_tta_probs(cxs, df_test_full, TTA_VIEWS)
        np.save(cache_c, cxs_probs)
    del cxs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ens_c = eval_ensemble(cnn_test_probs + [cxs_probs], y_test)
    log.info(f"  Ensemble C: F1={ens_c['ensemble_f1']:.4f}  AUC={ens_c['ensemble_auc']:.4f}")

    partial["ensemble_test_results"]["C_cnn3_convnext_small"] = {**ens_c,
        "label": "CNN×3 + ConvNeXt-small (15K subset)"}
    partial["fourth_model_val_metrics"]["C_convnext_small"] = val_c

    # ── Final results ─────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    winner = max(
        {"A": ens_a["ensemble_f1"], "B": ens_b["ensemble_f1"], "C": ens_c["ensemble_f1"]}.items(),
        key=lambda kv: kv[1],
    )[0]

    final = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed),
        "protocol": {
            "labels_csv": str(LABELS_CSV),
            "chm_dir": str(CHM_DIR),
            "subset_train_n": SUBSET_TRAIN_N,
            "subset_val_n": SUBSET_VAL_N,
            "test_n": int(len(df_test_full)),
            "test_cdw": int(np.sum(y_test == 1)),
            "test_no_cdw": int(np.sum(y_test == 0)),
            "epochs": EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD,
            "lr_backbone": LR_BACKBONE,
            "label_smoothing": LABEL_SMOOTHING,
            "mixup_alpha": MIXUP_ALPHA,
            "tta_views": TTA_VIEWS,
            "subset_seed": SUBSET_SEED,
            "scope_note": (
                "Fair comparison: all three 4th-model candidates trained on the SAME "
                "15K stratified subset, with ImageNet pretraining (first conv averaged "
                "RGB→1ch), identical hyperparameters, optimizer, schedule, augmentation, "
                "MixUp, and label smoothing. Only the architecture varies."
            ),
        },
        "production_reference": {
            "ensemble_f1": PRODUCTION_F1, "ensemble_auc": PRODUCTION_AUC,
            "note": "Production CNN×3 + EfficientNet-B2 on full 67K (not in ablation, listed for context)"
        },
        "fourth_model_val_metrics": {"A_efficientnet_b2": val_a,
                                     "B_convnext_tiny":   val_b,
                                     "C_convnext_small":  val_c},
        "ensemble_test_results": partial["ensemble_test_results"],
        "winner": winner,
        "delta_B_vs_A_f1": round(ens_b["ensemble_f1"] - ens_a["ensemble_f1"], 6),
        "delta_C_vs_A_f1": round(ens_c["ensemble_f1"] - ens_a["ensemble_f1"], 6),
    }
    (OUTPUT_DIR / "ablation_results.json").write_text(json.dumps(final, indent=2))

    summary = "\n".join([
        "Ensemble 4th-model ablation — results",
        f"Run: {final['created_at']}   elapsed: {elapsed/3600:.1f}h",
        f"Test set: {int(len(df_test_full))} tiles  "
        f"(CDW={int(np.sum(y_test==1))}, NO_CDW={int(np.sum(y_test==0))})",
        f"Production reference (full 67K training):  F1={PRODUCTION_F1:.4f}  "
        f"AUC={PRODUCTION_AUC:.4f}",
        "",
        f"{'Ensemble (4th model)':<42}  {'F1':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'Thr':>5}",
        "-" * 84,
        f"{'A: EfficientNet-B2 (15K, ImageNet)':<42}  "
        f"{ens_a['ensemble_f1']:>6.4f}  {ens_a['ensemble_auc']:>6.4f}  "
        f"{ens_a['ensemble_precision']:>6.4f}  {ens_a['ensemble_recall']:>6.4f}  "
        f"{ens_a['ensemble_thresh']:>5.2f}",
        f"{'B: ConvNeXt-tiny  (15K, ImageNet)':<42}  "
        f"{ens_b['ensemble_f1']:>6.4f}  {ens_b['ensemble_auc']:>6.4f}  "
        f"{ens_b['ensemble_precision']:>6.4f}  {ens_b['ensemble_recall']:>6.4f}  "
        f"{ens_b['ensemble_thresh']:>5.2f}",
        f"{'C: ConvNeXt-small (15K, ImageNet)':<42}  "
        f"{ens_c['ensemble_f1']:>6.4f}  {ens_c['ensemble_auc']:>6.4f}  "
        f"{ens_c['ensemble_precision']:>6.4f}  {ens_c['ensemble_recall']:>6.4f}  "
        f"{ens_c['ensemble_thresh']:>5.2f}",
        "-" * 84,
        f"Delta B vs A (F1): {ens_b['ensemble_f1']-ens_a['ensemble_f1']:+.4f}",
        f"Delta C vs A (F1): {ens_c['ensemble_f1']-ens_a['ensemble_f1']:+.4f}",
        f"Winner: {winner}",
    ])
    (OUTPUT_DIR / "ablation_summary.txt").write_text(summary)
    log.info("\n" + summary)
    log.info(f"\nDone in {elapsed/3600:.1f}h → {OUTPUT_DIR / 'ablation_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
