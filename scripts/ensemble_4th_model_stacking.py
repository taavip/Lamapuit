#!/usr/bin/env python3
"""
Stacking ensemble test: does meta-learning outperform soft voting for 4th-model selection?

This script trains meta-learners (logistic regression and shallow MLP) on base model
predictions from the ablation study, then evaluates stacking vs soft-voting on the
same test set.

Hypothesis: stacking can learn optimal non-linear combinations better than uniform
soft voting, potentially improving F1 by 0.5-1.0 percentage points.

Candidates (reusing checkpoints from ensemble_4th_model_ablation.py)
-------------------------------------------------------------------
  A  CNN-Deep-Attn×3  +  EfficientNet-B2   (soft-vote F1=0.9894)
  B  CNN-Deep-Attn×3  +  ConvNeXt-tiny     (soft-vote F1=0.9893)
  C  CNN-Deep-Attn×3  +  ConvNeXt-small    (soft-vote F1=0.9891)

Stacking protocol
-----------------
  1. Load CNN×3 predictions on val set (from cached ablation)
  2. Load each 4th candidate's predictions on val set
  3. Train meta-learner (LR + MLP) on (cnn1, cnn2, cnn3, candidate) → y_val
  4. Evaluate both LR and MLP stacking on test set
  5. Compare to soft-voting baseline (average of 4 probabilities)

Meta-learner architectures
---------------------------
  LR:  Logistic Regression (single hyperplane in 4D probability space)
  MLP: 1-hidden-layer ReLU net (4 inputs → 32 hidden → 2 outputs)
       Allows learning of non-linear probability combinations

Outputs
-------
  output/ensemble_4th_model_stacking/
    stacking_results.json  (meta-learner F1, AUC, comparison table)
    meta_learner_lr.pkl    (fitted sklearn LogisticRegression)
    meta_learner_mlp.pt    (trained PyTorch MLP)
    run.log                (detailed training log)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "ensemble_4th_model_stacking"
ABLATION_DIR = Path(__file__).parent.parent / "output" / "ensemble_4th_model_ablation"
CACHE_DIR = ABLATION_DIR / "cache"

EPOCHS_MLP = 30
LR_MLP = 1e-3
BATCH_SIZE = 32
PATIENCE = 5

# ── Logging setup ─────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = OUTPUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Meta-learner architectures ────────────────────────────────────────────────

class StackingMLP(nn.Module):
    """Simple 1-hidden-layer meta-learner for ensemble stacking."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main script ───────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info("Stacking ensemble meta-learner comparison")
    log.info("=" * 80)
    start_time = time.time()

    # ── Step 1: Load val data and compute val predictions ──────────────────────

    log.info("[1/4] Loading val data and computing base model predictions...")
    X_val = np.load(CACHE_DIR / "X_val_subset.npy")
    y_val = np.load(CACHE_DIR / "y_val_subset.npy")
    # Add channel dimension if missing (should be N, 1, H, W not N, H, W)
    if X_val.ndim == 3:
        X_val = X_val[:, np.newaxis, :, :]
    log.info(f"  val: {X_val.shape}, y: {y_val.shape}")

    # Load trained CNN models and compute val predictions
    from label_tiles import _instantiate_model_from_build_fn, _get_build_fn

    device = DEVICE
    cnn_checkpoints = [
        Path(__file__).parent.parent / "output" / "tile_labels_spatial_splits" / "cnn_seed42_spatial.pt",
        Path(__file__).parent.parent / "output" / "tile_labels_spatial_splits" / "cnn_seed43_spatial.pt",
        Path(__file__).parent.parent / "output" / "tile_labels_spatial_splits" / "cnn_seed44_spatial.pt",
    ]

    log.info("  Computing CNN×3 val predictions...")
    probs_cnn_list = []
    for i, ckpt_path in enumerate(cnn_checkpoints, 1):
        model = _instantiate_model_from_build_fn(_get_build_fn("_build_deep_cnn_attn"))
        ckpt_wrapped = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt_wrapped["state_dict"])
        model.to(device).eval()

        X_tensor = torch.from_numpy(X_val).float().to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_cnn_list.append(probs)
        log.info(f"    CNN{i}: {probs.shape}, mean={probs.mean():.4f}")
        del model, ckpt_wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probs_cnn1, probs_cnn2, probs_cnn3 = probs_cnn_list

    # Load candidate checkpoints (reuse ablation builds)
    log.info("  Computing candidate val predictions...")

    def _adapt_first_conv_rgb_to_1ch(conv: nn.Conv2d) -> nn.Conv2d:
        """Average pretrained 3-channel weights to 1-channel conv."""
        avg = conv.weight.data.mean(dim=1, keepdim=True)
        new = nn.Conv2d(
            in_channels=1, out_channels=conv.out_channels,
            kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding,
            bias=conv.bias is not None,
        )
        new.weight.data = avg
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
        return new

    def build_effnet_b2() -> nn.Module:
        from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
        m = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m

    def build_convnext_tiny() -> nn.Module:
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        return m

    def build_convnext_small() -> nn.Module:
        from torchvision.models import ConvNeXt_Small_Weights, convnext_small
        m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        return m

    cand_checkpoints = {
        "effnet_b2": (ABLATION_DIR / "effnet_b2_ablation.pt", build_effnet_b2),
        "convnext_tiny": (ABLATION_DIR / "convnext_tiny_ablation.pt", build_convnext_tiny),
        "convnext_small": (ABLATION_DIR / "convnext_small_ablation.pt", build_convnext_small),
    }

    probs_candidates = {}
    for cand_name, (ckpt_path, build_fn) in cand_checkpoints.items():
        model = build_fn()
        ckpt_wrapped = torch.load(ckpt_path, map_location=device)
        # Unwrap if necessary
        state = ckpt_wrapped["state_dict"] if isinstance(ckpt_wrapped, dict) and "state_dict" in ckpt_wrapped else ckpt_wrapped
        model.load_state_dict(state)
        model.to(device).eval()

        X_tensor = torch.from_numpy(X_val).float().to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_candidates[cand_name] = probs
        log.info(f"    {cand_name}: {probs.shape}, mean={probs.mean():.4f}")
        del model

    # Stack base model predictions: shape (n_val, 4)
    X_val_stacking_effnet = np.column_stack(
        [probs_cnn1, probs_cnn2, probs_cnn3, probs_candidates["effnet_b2"]]
    )
    X_val_stacking_convnext_tiny = np.column_stack(
        [probs_cnn1, probs_cnn2, probs_cnn3, probs_candidates["convnext_tiny"]]
    )
    X_val_stacking_convnext_small = np.column_stack(
        [probs_cnn1, probs_cnn2, probs_cnn3, probs_candidates["convnext_small"]]
    )

    log.info(f"  stacking X (EfficientNet): {X_val_stacking_effnet.shape}")
    log.info(f"  stacking X (ConvNeXt-tiny): {X_val_stacking_convnext_tiny.shape}")
    log.info(f"  stacking X (ConvNeXt-small): {X_val_stacking_convnext_small.shape}")

    # ── Step 2: Evaluate stacking vs soft-voting (k-fold CV on val set) ──────────

    log.info("\n[2/4] Evaluating stacking vs soft-voting via 5-fold CV on val set...")
    log.info("  (Meta-learner generalization is estimated via cross-validation)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for candidate_name, X_val_stack_arr in [
        ("effnet_b2", X_val_stacking_effnet),
        ("convnext_tiny", X_val_stacking_convnext_tiny),
        ("convnext_small", X_val_stacking_convnext_small),
    ]:
        log.info(f"\n  {candidate_name}:")

        cv_f1_soft = []
        cv_f1_lr = []
        cv_f1_mlp = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_val_stack_arr, y_val)):
            X_train_fold = X_val_stack_arr[train_idx]
            y_train_fold = y_val[train_idx]
            X_val_fold = X_val_stack_arr[val_idx]
            y_val_fold = y_val[val_idx]

            # Soft voting (no learning, always same)
            y_soft_fold = X_val_fold.mean(axis=1)
            f1_soft_fold = f1_score(y_val_fold, (y_soft_fold > 0.5).astype(int))
            cv_f1_soft.append(f1_soft_fold)

            # LR stacking
            lr_fold = LogisticRegression(max_iter=1000)
            lr_fold.fit(X_train_fold, y_train_fold)
            y_lr_fold = lr_fold.predict_proba(X_val_fold)[:, 1]
            f1_lr_fold = f1_score(y_val_fold, (y_lr_fold > 0.5).astype(int))
            cv_f1_lr.append(f1_lr_fold)

            # MLP stacking
            mlp_fold = StackingMLP(input_dim=4, hidden_dim=32).to(DEVICE)
            opt_fold = AdamW(mlp_fold.parameters(), lr=LR_MLP)
            crit_fold = nn.CrossEntropyLoss()

            X_train_t = torch.from_numpy(X_train_fold).float().to(DEVICE)
            y_train_t = torch.from_numpy(y_train_fold).long().to(DEVICE)
            X_val_t = torch.from_numpy(X_val_fold).float().to(DEVICE)
            y_val_t = torch.from_numpy(y_val_fold).long().to(DEVICE)

            ds_fold = TensorDataset(X_train_t, y_train_t)
            dl_fold = DataLoader(ds_fold, batch_size=BATCH_SIZE, shuffle=True)

            best_f1_mlp = -1
            patience = 0
            for epoch in range(EPOCHS_MLP):
                mlp_fold.train()
                for X_b, y_b in dl_fold:
                    opt_fold.zero_grad()
                    logits_b = mlp_fold(X_b)
                    loss_b = crit_fold(logits_b, y_b)
                    loss_b.backward()
                    opt_fold.step()

                mlp_fold.eval()
                with torch.no_grad():
                    logits_val = mlp_fold(X_val_t)
                    y_mlp_val = torch.softmax(logits_val, dim=1)[:, 1]
                    f1_mlp_fold = f1_score(
                        y_val_fold, (y_mlp_val.cpu().numpy() > 0.5).astype(int)
                    )

                if f1_mlp_fold > best_f1_mlp:
                    best_f1_mlp = f1_mlp_fold
                    patience = 0
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break

            cv_f1_mlp.append(best_f1_mlp)

        # Average across folds
        f1_soft_mean = np.mean(cv_f1_soft)
        f1_lr_mean = np.mean(cv_f1_lr)
        f1_mlp_mean = np.mean(cv_f1_mlp)

        log.info(f"    Soft-voting (CV): {f1_soft_mean:.6f} ± {np.std(cv_f1_soft):.6f}")
        log.info(f"    LR stacking (CV): {f1_lr_mean:.6f} ± {np.std(cv_f1_lr):.6f}  Δ={f1_lr_mean-f1_soft_mean:+.6f}")
        log.info(f"    MLP stacking (CV): {f1_mlp_mean:.6f} ± {np.std(cv_f1_mlp):.6f}  Δ={f1_mlp_mean-f1_soft_mean:+.6f}")

        cv_results[candidate_name] = {
            "soft_vote": {
                "cv_f1_mean": float(f1_soft_mean),
                "cv_f1_std": float(np.std(cv_f1_soft)),
            },
            "lr_stacking": {
                "cv_f1_mean": float(f1_lr_mean),
                "cv_f1_std": float(np.std(cv_f1_lr)),
                "delta_f1": float(f1_lr_mean - f1_soft_mean),
            },
            "mlp_stacking": {
                "cv_f1_mean": float(f1_mlp_mean),
                "cv_f1_std": float(np.std(cv_f1_mlp)),
                "delta_f1": float(f1_mlp_mean - f1_soft_mean),
            },
        }

    # ── Step 3: Write results ────────────────────────────────────────────────

    log.info("\n[3/4] Writing results...")

    elapsed = time.time() - start_time
    results_full = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": round(elapsed / 3600, 2),
        "cv_comparison": cv_results,  # k-fold CV results on val set
    }

    with open(OUTPUT_DIR / "stacking_results.json", "w") as f:
        json.dump(results_full, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────

    log.info("\n" + "=" * 80)
    log.info("Stacking vs Soft-Voting Comparison (5-fold CV on val set)")
    log.info("=" * 80)
    for candidate, metrics in cv_results.items():
        sv_mean = metrics["soft_vote"]["cv_f1_mean"]
        lr_mean = metrics["lr_stacking"]["cv_f1_mean"]
        mlp_mean = metrics["mlp_stacking"]["cv_f1_mean"]
        lr_delta = metrics["lr_stacking"]["delta_f1"]
        mlp_delta = metrics["mlp_stacking"]["delta_f1"]
        log.info(f"{candidate:20s}  soft-vote: {sv_mean:.6f}  LR: {lr_mean:.6f} ({lr_delta:+.6f})  MLP: {mlp_mean:.6f} ({mlp_delta:+.6f})")

    log.info("=" * 80)
    log.info(f"Done in {elapsed/3600:.2f}h → output/ensemble_4th_model_stacking/stacking_results.json\n")


if __name__ == "__main__":
    main()
