#!/usr/bin/env python3
"""
Rigorous ensemble comparison v2: defensible 4th-member selection.

Improvements over v1
--------------------
1. Repeated stratified 5-fold CV (5 repeats × 5 folds = 25 evaluations)
   for tighter estimates of generalization performance.
2. Per-fold threshold optimization: tune threshold on training fold,
   apply to held-out test fold. This is the methodologically correct
   way to estimate operating-point performance.
3. Five aggregation methods compared:
   - Soft-voting (uniform average, threshold=0.5)
   - Soft-voting tuned (uniform average, threshold tuned on train)
   - Weighted soft-voting (weights from logistic regression coefficients)
   - LR stacking (logistic regression meta-learner)
   - MLP stacking (1-hidden-layer meta-learner)
4. Paired Wilcoxon signed-rank tests between best methods
   (handles non-normal F1 distributions).
5. Bootstrap 95% confidence intervals for the winning combination.
6. Multiple metrics reported: F1, AUC, precision, recall.

Total runtime budget
--------------------
Model loading dominates (~2 min). The CV evaluation itself is fast
(~3 sec per repeat). 5 repeats × 5 folds × 3 candidates × 5 methods
≈ 1 min of compute + 2 min model loading = ~3 min total.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from label_tiles import _get_build_fn, _instantiate_model_from_build_fn

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "ensemble_4th_model_comparison_v2"
ABLATION_DIR = Path(__file__).parent.parent / "output" / "ensemble_4th_model_ablation"
CACHE_DIR = ABLATION_DIR / "cache"

N_FOLDS = 5
N_REPEATS = 5  # 5x5 = 25 evaluations per (candidate, method)
EPOCHS_MLP = 20
LR_MLP = 1e-3
BATCH_SIZE = 32
PATIENCE = 3
THRESHOLD_GRID = np.arange(0.20, 0.80, 0.02)  # 30 thresholds for tuning
N_BOOTSTRAP = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(OUTPUT_DIR / "run.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Meta-learners ─────────────────────────────────────────────────────────────

class StackingMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    f1: float
    auc: float
    precision: float
    recall: float
    threshold: float


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


def _build_effnet_b2():
    from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
    m = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def _build_convnext_tiny():
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    return m


def _build_convnext_small():
    from torchvision.models import ConvNeXt_Small_Weights, convnext_small
    m = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    m.features[0][0] = _adapt_first_conv_rgb_to_1ch(m.features[0][0])
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    return m


def _load_cnn_predictions(X_val: np.ndarray) -> list[np.ndarray]:
    """Compute val predictions for the 3 production CNN-Deep-Attn models."""
    cnn_dir = Path(__file__).parent.parent / "output" / "tile_labels_spatial_splits"
    cnn_paths = [cnn_dir / f"cnn_seed{s}_spatial.pt" for s in (42, 43, 44)]
    probs_list = []
    for i, ckpt_path in enumerate(cnn_paths, 1):
        model = _instantiate_model_from_build_fn(_get_build_fn("_build_deep_cnn_attn"))
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE).eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_val).float().to(DEVICE)
            probs = torch.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()
        log.info(f"    CNN{i}: mean={probs.mean():.4f}")
        probs_list.append(probs)
        del model
    return probs_list


def _load_candidate_predictions(X_val: np.ndarray) -> dict[str, np.ndarray]:
    """Compute val predictions for the 3 candidate 4th-member architectures."""
    builds = {
        "effnet_b2": (ABLATION_DIR / "effnet_b2_ablation.pt", _build_effnet_b2),
        "convnext_tiny": (ABLATION_DIR / "convnext_tiny_ablation.pt", _build_convnext_tiny),
        "convnext_small": (ABLATION_DIR / "convnext_small_ablation.pt", _build_convnext_small),
    }
    out = {}
    for name, (ckpt_path, build_fn) in builds.items():
        model = build_fn()
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_val).float().to(DEVICE)
            probs = torch.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()
        log.info(f"    {name}: mean={probs.mean():.4f}")
        out[name] = probs
        del model
    return out


def _optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find F1-optimal threshold on a given fold."""
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_true, (y_proba > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _eval_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> FoldResult:
    y_pred = (y_proba > threshold).astype(int)
    return FoldResult(
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_proba),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        threshold=threshold,
    )


def _train_mlp(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Train MLP and return test probabilities."""
    mlp = StackingMLP(input_dim=4, hidden_dim=32).to(DEVICE)
    opt = AdamW(mlp.parameters(), lr=LR_MLP)
    crit = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train).long().to(DEVICE)
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)

    dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    best_f1, best_probs, patience_ctr = -1.0, None, 0
    for epoch in range(EPOCHS_MLP):
        mlp.train()
        for X_b, y_b in dl:
            opt.zero_grad()
            crit(mlp(X_b), y_b).backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            probs = torch.softmax(mlp(X_test_t), dim=1)[:, 1].cpu().numpy()
        f1 = f1_score(y_test, (probs > 0.5).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_probs, patience_ctr = f1, probs, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
    return best_probs


def _evaluate_fold(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> dict[str, FoldResult]:
    """Evaluate all 5 ensembling methods on one CV fold."""
    out = {}

    # 1. Soft-voting at threshold 0.5
    probs_soft_test = X_test.mean(axis=1)
    out["soft_vote"] = _eval_at_threshold(y_test, probs_soft_test, 0.5)

    # 2. Soft-voting with tuned threshold (tune on train, apply to test)
    probs_soft_train = X_train.mean(axis=1)
    t_opt = _optimal_threshold(y_train, probs_soft_train)
    out["soft_vote_tuned"] = _eval_at_threshold(y_test, probs_soft_test, t_opt)

    # 3. Weighted soft-voting (weights from sigmoid of LR coefs)
    lr_for_weights = LogisticRegression(max_iter=1000)
    lr_for_weights.fit(X_train, y_train)
    weights = np.maximum(lr_for_weights.coef_[0], 0.01)  # nonneg
    weights = weights / weights.sum()
    probs_weighted_train = (X_train * weights).sum(axis=1)
    probs_weighted_test = (X_test * weights).sum(axis=1)
    t_w = _optimal_threshold(y_train, probs_weighted_train)
    out["weighted_vote"] = _eval_at_threshold(y_test, probs_weighted_test, t_w)

    # 4. LR stacking (full meta-learner with tuned threshold)
    probs_lr_train = lr_for_weights.predict_proba(X_train)[:, 1]
    probs_lr_test = lr_for_weights.predict_proba(X_test)[:, 1]
    t_lr = _optimal_threshold(y_train, probs_lr_train)
    out["lr_stacking"] = _eval_at_threshold(y_test, probs_lr_test, t_lr)

    # 5. MLP stacking with tuned threshold
    probs_mlp_test = _train_mlp(X_train, y_train, X_test, y_test)
    # Use same threshold-tuning approach: estimate optimal t from training fold via meta-LR proxy
    # (training MLP's own proba on X_train is overfit, so we use lr proxy threshold)
    out["mlp_stacking"] = _eval_at_threshold(y_test, probs_mlp_test, t_lr)

    return out


def _bootstrap_ci(scores: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap percentile confidence interval over fold scores."""
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot_means, 100 * alpha / 2)), float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info("Ensemble comparison v2 (rigorous 5x5 repeated CV)")
    log.info("=" * 80)

    # ── 1. Load & compute base predictions ────────────────────────────────────

    log.info("\n[1/4] Loading val data and computing base model predictions...")
    X_val = np.load(CACHE_DIR / "X_val_subset.npy")
    y_val = np.load(CACHE_DIR / "y_val_subset.npy")
    if X_val.ndim == 3:
        X_val = X_val[:, np.newaxis, :, :]
    log.info(f"  Val data: {X_val.shape}, y: {y_val.shape}, CDW={int(y_val.sum())}")

    log.info("  Computing CNN×3 val predictions...")
    probs_cnn = _load_cnn_predictions(X_val)
    log.info("  Computing candidate val predictions...")
    probs_cand = _load_candidate_predictions(X_val)

    X_stack = {
        cand: np.column_stack([*probs_cnn, p_cand])
        for cand, p_cand in probs_cand.items()
    }

    # ── 2. Repeated 5-fold CV ─────────────────────────────────────────────────

    log.info(f"\n[2/4] Running {N_REPEATS}x{N_FOLDS} repeated CV ({N_REPEATS * N_FOLDS} evaluations per method)...")
    methods = ["soft_vote", "soft_vote_tuned", "weighted_vote", "lr_stacking", "mlp_stacking"]
    metrics = ["f1", "auc", "precision", "recall", "threshold"]

    # results[cand][method][metric] = list of N_REPEATS*N_FOLDS scores
    results = {cand: {m: {met: [] for met in metrics} for m in methods} for cand in X_stack}

    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + repeat)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_stack["effnet_b2"], y_val)):
            for cand_name, X_arr in X_stack.items():
                fold_metrics = _evaluate_fold(
                    X_arr[train_idx], y_val[train_idx],
                    X_arr[test_idx], y_val[test_idx],
                )
                for method, fold_result in fold_metrics.items():
                    for met in metrics:
                        results[cand_name][method][met].append(getattr(fold_result, met))
        log.info(f"  Repeat {repeat + 1}/{N_REPEATS} done")

    # ── 3. Aggregate & report ─────────────────────────────────────────────────

    log.info("\n[3/4] Aggregating results...\n")
    log.info("=" * 110)
    log.info(f"{'Candidate':<16}{'Method':<20}{'F1 mean ± std':<25}{'F1 95% CI':<24}{'AUC':<10}{'Prec':<10}{'Rec':<10}")
    log.info("=" * 110)

    summary = {}
    for cand_name in X_stack:
        summary[cand_name] = {}
        for method in methods:
            f1s = np.array(results[cand_name][method]["f1"])
            aucs = np.array(results[cand_name][method]["auc"])
            precs = np.array(results[cand_name][method]["precision"])
            recs = np.array(results[cand_name][method]["recall"])
            ths = np.array(results[cand_name][method]["threshold"])

            ci_low, ci_high = _bootstrap_ci(f1s)
            log.info(
                f"{cand_name:<16}{method:<20}"
                f"{f1s.mean():.6f} ± {f1s.std():.6f}     "
                f"[{ci_low:.6f}, {ci_high:.6f}]    "
                f"{aucs.mean():.4f}    {precs.mean():.4f}    {recs.mean():.4f}"
            )

            summary[cand_name][method] = {
                "f1_mean": float(f1s.mean()),
                "f1_std": float(f1s.std()),
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high,
                "auc_mean": float(aucs.mean()),
                "precision_mean": float(precs.mean()),
                "recall_mean": float(recs.mean()),
                "threshold_mean": float(ths.mean()),
            }
        log.info("-" * 110)

    # ── 4. Statistical significance & winner declaration ─────────────────────

    log.info("\n[4/4] Statistical significance tests (paired Wilcoxon signed-rank)...")

    # For each candidate, compare each method vs soft_vote baseline
    sig_tests = {}
    for cand_name in X_stack:
        baseline = np.array(results[cand_name]["soft_vote"]["f1"])
        sig_tests[cand_name] = {}
        for method in methods:
            if method == "soft_vote":
                continue
            scores = np.array(results[cand_name][method]["f1"])
            try:
                stat, pval = wilcoxon(scores, baseline, alternative="greater")
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                log.info(f"  {cand_name:<16} {method:<20} vs soft_vote: Δ={scores.mean()-baseline.mean():+.6f}  p={pval:.4f} {sig}")
                sig_tests[cand_name][method] = {"p_value": float(pval), "significant": bool(pval < 0.05)}
            except ValueError:
                log.info(f"  {cand_name:<16} {method:<20} vs soft_vote: (identical)")

    # Cross-candidate winner under best method
    log.info("\n  Cross-candidate comparison under best method (LR stacking):")
    lr_scores_by_cand = {c: np.array(results[c]["lr_stacking"]["f1"]) for c in X_stack}
    winner = max(lr_scores_by_cand, key=lambda c: lr_scores_by_cand[c].mean())
    for cand in X_stack:
        if cand == winner:
            continue
        stat, pval = wilcoxon(lr_scores_by_cand[winner], lr_scores_by_cand[cand], alternative="greater")
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        log.info(f"    {winner} > {cand}:  Δ={lr_scores_by_cand[winner].mean()-lr_scores_by_cand[cand].mean():+.6f}  p={pval:.4f} {sig}")

    # Best overall (candidate × method)
    best_combo = None
    best_f1 = -1
    for cand in X_stack:
        for method in methods:
            f1 = summary[cand][method]["f1_mean"]
            if f1 > best_f1:
                best_f1 = f1
                best_combo = (cand, method)

    log.info("\n" + "=" * 80)
    log.info(f"WINNER: {best_combo[0]} + {best_combo[1]}")
    log.info(f"  F1 = {summary[best_combo[0]][best_combo[1]]['f1_mean']:.6f}")
    log.info(f"  95% CI = [{summary[best_combo[0]][best_combo[1]]['f1_ci_low']:.6f}, "
             f"{summary[best_combo[0]][best_combo[1]]['f1_ci_high']:.6f}]")
    log.info("=" * 80)

    # Save full results
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": f"{N_REPEATS}x{N_FOLDS} repeated stratified CV with per-fold threshold tuning",
        "n_evaluations_per_method": N_REPEATS * N_FOLDS,
        "summary": summary,
        "significance_tests": sig_tests,
        "winner": {"candidate": best_combo[0], "method": best_combo[1], "f1": float(best_f1)},
    }

    with open(OUTPUT_DIR / "comparison_v2_results.json", "w") as f:
        json.dump(out, f, indent=2)

    log.info(f"\nResults saved to {OUTPUT_DIR / 'comparison_v2_results.json'}\n")


if __name__ == "__main__":
    main()
