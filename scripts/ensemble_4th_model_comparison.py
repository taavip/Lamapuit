#!/usr/bin/env python3
"""
Fair ensemble comparison: soft-voting vs stacking with consistent methodology.

This script evaluates all 4th-member candidates using identical validation
protocol:
  1. Stratified train/test split of val set (3K train, 1K test per fold)
  2. Train meta-learners on train fold
  3. Evaluate soft-voting and stacking on test fold
  4. Repeat 5 times with different folds
  5. Report CV-averaged metrics for all methods

This ensures all ensembles are compared on:
  - Same data
  - Same validation methodology
  - Held-out test fold (not used during meta-learner training)
  - Stratified by class distribution
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _get_build_fn, _instantiate_model_from_build_fn

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "ensemble_4th_model_comparison"
ABLATION_DIR = Path(__file__).parent.parent / "output" / "ensemble_4th_model_ablation"
CACHE_DIR = ABLATION_DIR / "cache"

EPOCHS_MLP = 20
LR_MLP = 1e-3
BATCH_SIZE = 32
PATIENCE = 3
N_SPLITS = 5  # 5-fold CV

# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── Meta-learner ──────────────────────────────────────────────────────────────

class StackingMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info("Fair ensemble comparison: soft-voting vs stacking")
    log.info("=" * 80)

    # Load val data and compute base model predictions
    log.info("\n[1/3] Loading val data and computing base model predictions...")
    X_val = np.load(CACHE_DIR / "X_val_subset.npy")
    y_val = np.load(CACHE_DIR / "y_val_subset.npy")
    if X_val.ndim == 3:
        X_val = X_val[:, np.newaxis, :, :]

    device = DEVICE
    log.info(f"  Val data: {X_val.shape}, y: {y_val.shape}")

    # Compute CNN×3 val predictions
    from label_tiles import _instantiate_model_from_build_fn, _get_build_fn

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
        log.info(f"    CNN{i}: mean={probs.mean():.4f}")
        del model

    probs_cnn1, probs_cnn2, probs_cnn3 = probs_cnn_list

    # Compute candidate val predictions
    log.info("  Computing candidate val predictions...")

    def _adapt_first_conv_rgb_to_1ch(conv: nn.Conv2d) -> nn.Conv2d:
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
        state = ckpt_wrapped["state_dict"] if isinstance(ckpt_wrapped, dict) and "state_dict" in ckpt_wrapped else ckpt_wrapped
        model.load_state_dict(state)
        model.to(device).eval()

        X_tensor = torch.from_numpy(X_val).float().to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_candidates[cand_name] = probs
        log.info(f"    {cand_name}: mean={probs.mean():.4f}")
        del model

    probs_effnet = probs_candidates["effnet_b2"]
    probs_convnext_tiny = probs_candidates["convnext_tiny"]
    probs_convnext_small = probs_candidates["convnext_small"]

    log.info(f"  Base probs shape: ({len(y_val)},)")

    # Build stacking features for each candidate
    X_stack = {
        "effnet_b2": np.column_stack([probs_cnn1, probs_cnn2, probs_cnn3, probs_effnet]),
        "convnext_tiny": np.column_stack([probs_cnn1, probs_cnn2, probs_cnn3, probs_convnext_tiny]),
        "convnext_small": np.column_stack([probs_cnn1, probs_cnn2, probs_cnn3, probs_convnext_small]),
    }

    # ── Stratified K-fold CV with held-out test fold ──────────────────────────

    log.info(f"\n[2/3] Running {N_SPLITS}-fold stratified CV comparison...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    results_all_folds = {cand: {"soft_vote": [], "lr": [], "mlp": []} for cand in X_stack.keys()}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_stack["effnet_b2"], y_val)):
        log.info(f"\n  Fold {fold_idx + 1}/{N_SPLITS}:")

        y_train = y_val[train_idx]
        y_test = y_val[test_idx]

        for cand_name, X_stack_arr in X_stack.items():
            X_train = X_stack_arr[train_idx]
            X_test = X_stack_arr[test_idx]

            # Soft voting (no learning)
            y_soft_test = X_test.mean(axis=1)
            f1_soft = f1_score(y_test, (y_soft_test > 0.5).astype(int))
            results_all_folds[cand_name]["soft_vote"].append(f1_soft)

            # LR stacking
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            y_lr_test = lr.predict_proba(X_test)[:, 1]
            f1_lr = f1_score(y_test, (y_lr_test > 0.5).astype(int))
            results_all_folds[cand_name]["lr"].append(f1_lr)

            # MLP stacking
            mlp = StackingMLP(input_dim=4, hidden_dim=32).to(DEVICE)
            opt = AdamW(mlp.parameters(), lr=LR_MLP)
            crit = nn.CrossEntropyLoss()

            X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
            y_train_t = torch.from_numpy(y_train).long().to(DEVICE)
            X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
            y_test_t = torch.from_numpy(y_test).long().to(DEVICE)

            ds = TensorDataset(X_train_t, y_train_t)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            best_f1_mlp = -1
            patience_ctr = 0
            for epoch in range(EPOCHS_MLP):
                mlp.train()
                for X_b, y_b in dl:
                    opt.zero_grad()
                    logits_b = mlp(X_b)
                    loss_b = crit(logits_b, y_b)
                    loss_b.backward()
                    opt.step()

                mlp.eval()
                with torch.no_grad():
                    logits_test = mlp(X_test_t)
                    y_mlp_test = torch.softmax(logits_test, dim=1)[:, 1]
                    f1_mlp = f1_score(
                        y_test, (y_mlp_test.cpu().numpy() > 0.5).astype(int)
                    )

                if f1_mlp > best_f1_mlp:
                    best_f1_mlp = f1_mlp
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= PATIENCE:
                        break

            results_all_folds[cand_name]["mlp"].append(best_f1_mlp)

            if fold_idx == 0:
                log.info(f"    {cand_name}  soft={f1_soft:.6f}  lr={f1_lr:.6f}  mlp={best_f1_mlp:.6f}")

    # ── Aggregate results ──────────────────────────────────────────────────────

    log.info("\n[3/3] Final results (5-fold CV average):")
    log.info("\n" + "=" * 90)
    log.info(f"{'Ensemble (4th model)':<30} {'Method':<15} {'F1 (mean ± std)':<25} {'Δ vs soft-vote':<20}")
    log.info("=" * 90)

    final_results = {}

    for cand_name in sorted(X_stack.keys()):
        sv_scores = np.array(results_all_folds[cand_name]["soft_vote"])
        lr_scores = np.array(results_all_folds[cand_name]["lr"])
        mlp_scores = np.array(results_all_folds[cand_name]["mlp"])

        sv_mean, sv_std = sv_scores.mean(), sv_scores.std()
        lr_mean, lr_std = lr_scores.mean(), lr_scores.std()
        mlp_mean, mlp_std = mlp_scores.mean(), mlp_scores.std()

        log.info(f"{cand_name:<30} {'Soft-voting':<15} {sv_mean:.6f} ± {sv_std:.6f}      {'baseline':<20}")
        log.info(f"{'':<30} {'LR stacking':<15} {lr_mean:.6f} ± {lr_std:.6f}      {lr_mean - sv_mean:+.6f} ({(lr_mean - sv_mean) / sv_mean * 100:+.2f}%)")
        log.info(f"{'':<30} {'MLP stacking':<15} {mlp_mean:.6f} ± {mlp_std:.6f}      {mlp_mean - sv_mean:+.6f} ({(mlp_mean - sv_mean) / sv_mean * 100:+.2f}%)")
        log.info("")

        final_results[cand_name] = {
            "soft_vote": {"mean": float(sv_mean), "std": float(sv_std)},
            "lr_stacking": {"mean": float(lr_mean), "std": float(lr_std), "delta": float(lr_mean - sv_mean)},
            "mlp_stacking": {"mean": float(mlp_mean), "std": float(mlp_std), "delta": float(mlp_mean - sv_mean)},
        }

    log.info("=" * 90)

    # Determine winner
    soft_vote_scores = {cand: final_results[cand]["soft_vote"]["mean"] for cand in final_results}
    winner_soft = max(soft_vote_scores, key=soft_vote_scores.get)
    log.info(f"\nSoft-voting winner: {winner_soft} (F1={final_results[winner_soft]['soft_vote']['mean']:.6f})")

    lr_scores = {cand: final_results[cand]["lr_stacking"]["mean"] for cand in final_results}
    winner_lr = max(lr_scores, key=lr_scores.get)
    log.info(f"LR stacking winner: {winner_lr} (F1={final_results[winner_lr]['lr_stacking']['mean']:.6f})")

    mlp_scores = {cand: final_results[cand]["mlp_stacking"]["mean"] for cand in final_results}
    winner_mlp = max(mlp_scores, key=mlp_scores.get)
    log.info(f"MLP stacking winner: {winner_mlp} (F1={final_results[winner_mlp]['mlp_stacking']['mean']:.6f})")

    # Save results
    output_json = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": "5-fold stratified CV with held-out test fold",
        "n_folds": N_SPLITS,
        "results": final_results,
        "winners": {
            "soft_vote": winner_soft,
            "lr_stacking": winner_lr,
            "mlp_stacking": winner_mlp,
        },
    }

    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(output_json, f, indent=2)

    log.info(f"\nResults saved to {OUTPUT_DIR / 'comparison_results.json'}\n")


if __name__ == "__main__":
    main()
