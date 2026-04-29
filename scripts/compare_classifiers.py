#!/usr/bin/env python3
"""
CDW tile classifier comparison -- sklearn models + PyTorch CNNs (GPU).

Models
------
Sklearn  (22 hand-crafted height features):
  LogisticRegression, RandomForest, GradientBoosting, ExtraTrees,
  AdaBoost, SVM-RBF, KNN, MLP, GaussianNB, DecisionTree,
  Ensemble-3 (LR+RF+GB), Ensemble-5 (LR+RF+GB+ET+SVM)

PyTorch  (128x128 single-channel CHM tile, GPU):
  CNN-Simple, CNN-Deep, CNN-Mobile, ResNet50-FT,
  EfficientNet-B0, ConvNeXt-Tiny,
  CNN-Deep-Attn (SE channel attention), CNN-Deep-Pro (longer+Mixup+aug)

All PyTorch models use light augmentation (random flip + 90-deg rotate).
CNN-Deep-Pro additionally uses Mixup.
Threshold tuning for best-F1 reported per model.

Usage
-----
python scripts/compare_classifiers.py \
    --labels  output/tile_labels \
    --chm-dir chm_max_hag \
    --output  output/tile_labels/ensemble_model.pkl
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -- feature extraction -------------------------------------------------------

_HEIGHT_BANDS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.30]
_CHUNK_SIZE = 128


def _extract_features(tile: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    valid = tile[np.isfinite(tile)]
    n = max(valid.size, 1)
    above = valid[valid > threshold]
    feats: list[float] = [
        *(float(np.sum(valid > b)) / n for b in _HEIGHT_BANDS),
        *(
            float(np.percentile(above, q)) if above.size > 0 else 0.0
            for q in [10, 25, 50, 75, 90, 99]
        ),
        float(above.mean()) if above.size > 0 else 0.0,
        float(above.std()) if above.size > 0 else 0.0,
        float(above.max()) if above.size > 0 else 0.0,
        float(above.size) / n,
        float(np.sum(np.diff((valid > threshold).astype(np.int8)) > 0)),
        (
            float(np.percentile(above, 75) / max(np.percentile(above, 25), 0.01))
            if above.size > 0
            else 1.0
        ),
        float(np.sum(valid > 0.5)) / n,
        float(np.sum(valid > 1.0)) / n,
    ]
    return np.array(feats, dtype=np.float32)


# -- I/O ----------------------------------------------------------------------


def load_labels(label_dir: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(label_dir.glob("*_labels.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] in ("cdw", "no_cdw"):
                    rows.append(row)
    return rows


def build_features(rows, chm_dir, threshold=0.15, also_return_tiles=False):
    X_feat, X_tiles, y = [], [], []
    open_rasters: dict[str, rasterio.DatasetReader] = {}
    missing: set[str] = set()

    for i, row in enumerate(rows):
        rname = row["raster"]
        if rname in missing:
            continue
        if rname not in open_rasters:
            p = chm_dir / rname
            if not p.exists():
                print(f"  [skip] raster not found: {p}", flush=True)
                missing.add(rname)
                continue
            open_rasters[rname] = rasterio.open(p)

        src = open_rasters[rname]
        cs = int(row["chunk_size"])
        tile = src.read(
            1,
            window=Window(int(row["col_off"]), int(row["row_off"]), cs, cs),
            boundless=True,
            fill_value=0,
        ).astype(np.float32)

        X_feat.append(_extract_features(tile, threshold))
        if also_return_tiles:
            if tile.shape != (_CHUNK_SIZE, _CHUNK_SIZE):
                import cv2

                tile = cv2.resize(tile, (_CHUNK_SIZE, _CHUNK_SIZE), interpolation=cv2.INTER_LINEAR)
            X_tiles.append(np.clip(tile, 0.0, 20.0) / 20.0)
        y.append(1 if row["label"] == "cdw" else 0)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)} features extracted ...", flush=True)

    for src in open_rasters.values():
        src.close()

    Xf = np.array(X_feat, dtype=np.float32)
    Xt = np.array(X_tiles, dtype=np.float32) if also_return_tiles else None
    yy = np.array(y, dtype=np.int64)
    return Xf, Xt, yy


# -- GPU helper ---------------------------------------------------------------


def _get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- augmentation + mixup -----------------------------------------------------


def _augment_batch(xb):
    import torch

    if torch.rand(1).item() > 0.5:
        xb = torch.flip(xb, dims=[3])
    if torch.rand(1).item() > 0.5:
        xb = torch.flip(xb, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        xb = torch.rot90(xb, k=k, dims=[2, 3])
    return xb


def _mixup_batch(xb, yb, alpha=0.4):
    import torch

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(xb.size(0), device=xb.device)
    xb_mix = lam * xb + (1 - lam) * xb[idx]
    ya = torch.nn.functional.one_hot(yb, 2).float()
    yb2 = torch.nn.functional.one_hot(yb[idx], 2).float()
    return xb_mix, lam * ya + (1 - lam) * yb2


# -- architectures ------------------------------------------------------------


def _build_simple_cnn():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(8),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 2),
    )


def _build_deep_cnn():
    import torch.nn as nn

    class Block(nn.Module):
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
            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
            self.mp = nn.MaxPool2d(2)

        def forward(self, x):
            return self.mp(self.conv(x) + self.skip(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(
                Block(1, 32),
                Block(32, 64),
                Block(64, 128),
                Block(128, 128),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            )

        def forward(self, x):
            return self.head(self.blocks(x))

    return Net()


def _build_deep_cnn_attn():
    """CNN-Deep + Squeeze-and-Excitation channel attention."""
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


def _build_mobile_cnn():
    import torch.nn as nn

    def dw(in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        dw(16, 32, stride=2),
        dw(32, 64, stride=2),
        dw(64, 96, stride=2),
        dw(96, 128, stride=2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    )


def _build_resnet50():
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    w = model.conv1.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = w
    for name, param in model.named_parameters():
        if not any(name.startswith(s) for s in ("layer3", "layer4", "fc", "conv1")):
            param.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    return model


def _build_efficientnet_b0():
    """EfficientNet-B0 finetuned from ImageNet, 1-channel input."""
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    old = model.features[0][0]
    new_conv = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    new_conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    model.features[0][0] = new_conv
    for name, param in model.named_parameters():
        if not any(
            name.startswith(s)
            for s in ("features.5", "features.6", "features.7", "features.8", "classifier")
        ):
            param.requires_grad = False
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_feat, 2))
    return model


def _build_convnext_tiny():
    """ConvNeXt-Tiny (torchvision >=0.13), 1-channel input."""
    import torch.nn as nn
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    old = model.features[0][0]
    new_conv = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    new_conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)
    if old.bias is not None:
        new_conv.bias = old.bias
    model.features[0][0] = new_conv
    for name, param in model.named_parameters():
        if not any(
            name.startswith(s)
            for s in ("features.4", "features.5", "features.6", "features.7", "classifier")
        ):
            param.requires_grad = False
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feat, 2)
    return model


# -- TorchWrapper (sklearn-compatible) ----------------------------------------


class TorchWrapper:
    def __init__(
        self,
        build_fn,
        epochs=40,
        batch_size=32,
        lr=1e-3,
        patience=10,
        augment=True,
        mixup_alpha=0.0,
    ):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.model_ = None

    def get_params(self, deep=True):
        return dict(
            build_fn=self.build_fn,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            augment=self.augment,
            mixup_alpha=self.mixup_alpha,
        )

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        dev = _get_device()
        self.model_ = self.build_fn().to(dev)
        Xt = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        weight = torch.tensor([n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(dev)
        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(dev.type == "cuda"),
            num_workers=0,
        )
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model_.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.epochs,
            eta_min=self.lr * 0.02,
        )
        use_mixup = self.mixup_alpha > 0
        best_loss = float("inf")
        no_imp = 0
        best_state = copy.deepcopy(self.model_.state_dict())

        for ep in range(self.epochs):
            self.model_.train()
            ep_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                if self.augment:
                    xb = _augment_batch(xb)
                opt.zero_grad()
                if use_mixup:
                    xb_m, yb_soft = _mixup_batch(xb, yb, alpha=self.mixup_alpha)
                    logits = self.model_(xb_m)
                    log_p = torch.nn.functional.log_softmax(logits, dim=1)
                    loss = -(yb_soft * log_p).sum(dim=1).mean()
                else:
                    loss = nn.CrossEntropyLoss(weight=weight)(self.model_(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(xb)
            sched.step()
            ep_loss /= max(len(Xt), 1)
            if ep_loss < best_loss - 1e-4:
                best_loss, no_imp = ep_loss, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        dev = _get_device()
        self.model_.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = torch.tensor(X[i : i + 256, np.newaxis], dtype=torch.float32).to(dev)
                all_probs.append(torch.softmax(self.model_(xb), dim=1).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# -- catalogues ---------------------------------------------------------------


def build_sklearn_catalogue(n_est=100):
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    lr = Pipeline(
        [
            ("sc", StandardScaler()),
            ("cl", LogisticRegression(max_iter=2000, C=2.0, random_state=42)),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=42
    )
    et = ExtraTreesClassifier(
        n_estimators=n_est, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    ada = AdaBoostClassifier(n_estimators=n_est, learning_rate=0.5, random_state=42)
    svm = Pipeline(
        [
            ("sc", StandardScaler()),
            ("cl", SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=42)),
        ]
    )
    knn = Pipeline(
        [
            ("sc", StandardScaler()),
            ("cl", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
        ]
    )
    mlp = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "cl",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42
                ),
            ),
        ]
    )
    gnb = Pipeline([("sc", StandardScaler()), ("cl", GaussianNB())])
    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, random_state=42)
    vc3 = VotingClassifier([("lr", lr), ("rf", rf), ("gb", gb)], voting="soft", n_jobs=1)
    vc5 = VotingClassifier(
        [("lr", lr), ("rf", rf), ("gb", gb), ("et", et), ("svm", svm)],
        voting="soft",
        n_jobs=1,
    )
    return [
        ("LogisticRegression", lr, "feat"),
        ("RandomForest", rf, "feat"),
        ("GradientBoosting", gb, "feat"),
        ("ExtraTrees", et, "feat"),
        ("AdaBoost", ada, "feat"),
        ("SVM-RBF", svm, "feat"),
        ("KNN", knn, "feat"),
        ("MLP", mlp, "feat"),
        ("GaussianNB", gnb, "feat"),
        ("DecisionTree", dt, "feat"),
        ("Ensemble-3 (LR+RF+GB)", vc3, "feat"),
        ("Ensemble-5 (LR+RF+GB+ET+SVM)", vc5, "feat"),
    ]


def build_torch_catalogue():
    W = TorchWrapper
    return [
        ("CNN-Simple", W(_build_simple_cnn, epochs=40, lr=1e-3, augment=True), "tile"),
        ("CNN-Deep", W(_build_deep_cnn, epochs=40, lr=5e-4, augment=True), "tile"),
        ("CNN-Mobile", W(_build_mobile_cnn, epochs=40, lr=1e-3, augment=True), "tile"),
        (
            "ResNet50-FT",
            W(_build_resnet50, epochs=25, lr=3e-4, batch_size=16, augment=True),
            "tile",
        ),
        (
            "EfficientNet-B0",
            W(_build_efficientnet_b0, epochs=30, lr=2e-4, batch_size=32, augment=True),
            "tile",
        ),
        (
            "ConvNeXt-Tiny",
            W(_build_convnext_tiny, epochs=30, lr=2e-4, batch_size=32, augment=True),
            "tile",
        ),
        ("CNN-Deep-Attn", W(_build_deep_cnn_attn, epochs=50, lr=5e-4, augment=True), "tile"),
        (
            "CNN-Deep-Pro",
            W(_build_deep_cnn, epochs=80, lr=5e-4, augment=True, mixup_alpha=0.4, patience=15),
            "tile",
        ),
    ]


# -- threshold tuning ---------------------------------------------------------


def _best_threshold(y_true, y_prob):
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.02):
        yp = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# -- evaluation ---------------------------------------------------------------


def evaluate_all(catalogue, X_feat, X_tiles, y, n_splits=5):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    results = []
    total = len(catalogue)

    for model_idx, (name, clf, inp) in enumerate(catalogue, 1):
        X = X_feat if inp == "feat" else X_tiles
        is_gpu = inp == "tile"
        aucs, accs, f1s, precs, recs, bf1s, bts = [], [], [], [], [], [], []
        t0 = time.perf_counter()
        tag = " [GPU]" if is_gpu else "      "
        sym = "star" if "Ensemble" in name else ("GPU" if is_gpu else "cpu")
        full = name + tag

        print(f"\n[{model_idx}/{total}] {full}", flush=True)

        for fold, (tr, te) in enumerate(cv.split(X, y), 1):
            m = copy.deepcopy(clf)
            m.fit(X[tr], y[tr])
            yp = m.predict(X[te])
            try:
                yprob = m.predict_proba(X[te])[:, 1]
                auc = roc_auc_score(y[te], yprob)
                bt, bf1 = _best_threshold(y[te], yprob)
            except Exception:
                yprob = None
                auc = float("nan")
                bt, bf1 = 0.5, float(f1_score(y[te], yp, zero_division=0))
            aucs.append(auc)
            accs.append(accuracy_score(y[te], yp))
            cur_f1 = f1_score(y[te], yp, zero_division=0)
            f1s.append(cur_f1)
            precs.append(precision_score(y[te], yp, zero_division=0))
            recs.append(recall_score(y[te], yp, zero_division=0))
            bf1s.append(bf1)
            bts.append(bt)
            elapsed_fold = time.perf_counter() - t0
            print(
                f"   fold {fold}/{n_splits}  AUC {auc:.3f}  "
                f"F1@0.50={cur_f1:.3f}  best-F1={bf1:.3f}@{bt:.2f}  "
                f"({elapsed_fold:.1f}s)",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        res = {
            "name": full,
            "bare_name": name,
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
            "acc_mean": float(np.mean(accs)),
            "f1_mean": float(np.mean(f1s)),
            "prec_mean": float(np.mean(precs)),
            "rec_mean": float(np.mean(recs)),
            "best_f1": float(np.mean(bf1s)),
            "best_thresh": float(np.median(bts)),
            "time_s": elapsed,
            "clf": clf,
            "inp": inp,
        }
        results.append(res)
        print(
            f"   ==> AUC {res['auc_mean']:.3f}+-{res['auc_std']:.3f}  "
            f"Acc {res['acc_mean']:.3f}  F1 {res['f1_mean']:.3f}  "
            f"best-F1 {res['best_f1']:.3f}@{res['best_thresh']:.2f}  "
            f"total {elapsed:.1f}s",
            flush=True,
        )

    return sorted(results, key=lambda r: r["auc_mean"], reverse=True)


# -- report -------------------------------------------------------------------


def print_table(results):
    SEP = "=" * 122
    hdr = (
        f"{'Rk':<4} {'Model':<44} {'AUC':>8} {'+-':>6}"
        f" {'Acc':>7} {'F1@.5':>7} {'bestF1':>8} {'thr':>6}"
        f" {'Prec':>7} {'Rec':>7} {'Time':>7}"
    )
    print(f"\n{SEP}\n{hdr}\n{SEP}")
    for i, r in enumerate(results, 1):
        tag = "  <-- BEST" if i == 1 else ""
        print(
            f"  {i:<3} {r['name']:<44}"
            f" {r['auc_mean']:>7.3f} {r['auc_std']:>6.3f}"
            f" {r['acc_mean']:>7.3f} {r['f1_mean']:>7.3f}"
            f" {r['best_f1']:>8.3f} {r['best_thresh']:>6.2f}"
            f" {r['prec_mean']:>7.3f} {r['rec_mean']:>7.3f}"
            f" {r['time_s']:>6.0f}s{tag}"
        )
    print(f"{SEP}\n")


# -- save ---------------------------------------------------------------------


def save_best(res, X_feat, X_tiles, y, out_path):
    import torch

    clf = copy.deepcopy(res["clf"])
    X = X_feat if res["inp"] == "feat" else X_tiles
    print(f"  Refitting {res['bare_name']} on full dataset ...", flush=True)
    clf.fit(X, y)
    meta = {
        "model_name": res["bare_name"],
        "auc": res["auc_mean"],
        "best_f1": res["best_f1"],
        "best_thresh": res["best_thresh"],
        "n_samples": len(y),
        "n_cdw": int(y.sum()),
        "n_no_cdw": int((y == 0).sum()),
        "version": 2,
        "inp_type": res["inp"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if res["inp"] == "tile":
        # PyTorch models: save state_dict + wrapper params (avoids pickle of local classes)
        pt_path = out_path.with_suffix(".pt")
        torch.save(
            {
                "state_dict": clf.model_.state_dict(),
                "build_fn_name": clf.build_fn.__name__,
                "meta": meta,
            },
            pt_path,
        )
        # Also write a small joblib stub pointing to the .pt file
        import joblib

        stub = {"type": "torch", "pt_path": str(pt_path), "meta": meta}
        joblib.dump(stub, out_path, compress=3)
        size_kb = pt_path.stat().st_size / 1024
        print(f"  Saved --> {pt_path}  ({size_kb:.0f} KB)")
    else:
        import joblib

        joblib.dump({"model": clf, "meta": meta}, out_path, compress=3)
        size_kb = out_path.stat().st_size / 1024
        print(f"  Saved --> {out_path}  ({size_kb:.0f} KB)")
    print(f"  Best threshold for F1: {res['best_thresh']:.2f}  " f"(best-F1={res['best_f1']:.3f})")


# -- main ---------------------------------------------------------------------


def main():
    import torch

    p = argparse.ArgumentParser(description="Compare CDW tile classifiers (sklearn + PyTorch GPU)")
    p.add_argument("--labels", default="output/tile_labels")
    p.add_argument("--chm-dir", default="chm_max_hag")
    p.add_argument("--output", default="output/tile_labels/ensemble_model.pkl")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--sklearn-only", action="store_true")
    p.add_argument("--torch-only", action="store_true")
    p.add_argument(
        "--filter",
        default="",
        help="Comma-separated name substrings to run, e.g. 'CNN-Deep,ResNet'",
    )
    args = p.parse_args()

    dev = _get_device()
    print(f"\nDevice : {dev}")
    if dev.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    label_dir = Path(args.labels)
    chm_dir = Path(args.chm_dir)
    out_path = Path(args.output)

    print(f"\nLoading labels from {label_dir} ...")
    rows = load_labels(label_dir)
    if not rows:
        print("No labels found.")
        sys.exit(1)
    n_cdw = sum(1 for r in rows if r["label"] == "cdw")
    n_no = sum(1 for r in rows if r["label"] == "no_cdw")
    print(f"Labels : {len(rows)}  (CDW: {n_cdw}  No CDW: {n_no})")
    if n_cdw < 8 or n_no < 8:
        print("Need >=8 of each class.")
        sys.exit(1)

    need_tiles = not args.sklearn_only
    print("\nExtracting features ...")
    X_feat, X_tiles, y = build_features(rows, chm_dir, args.threshold, need_tiles)
    print(f"  feat : {X_feat.shape}")
    if X_tiles is not None:
        print(f"  tiles: {X_tiles.shape}")

    catalogue: list = []
    if not args.torch_only:
        catalogue += build_sklearn_catalogue(args.n_estimators)
    if not args.sklearn_only:
        torch_cat = []
        for entry in build_torch_catalogue():
            name2, clf2, inp2 = entry
            if name2 == "ConvNeXt-Tiny":
                try:
                    _build_convnext_tiny()
                    torch_cat.append(entry)
                except Exception as e:
                    print(f"  [skip] ConvNeXt-Tiny: {e}")
            else:
                torch_cat.append(entry)
        catalogue += torch_cat

    if args.filter:
        patterns = [s.strip().lower() for s in args.filter.split(",") if s.strip()]
        catalogue = [e for e in catalogue if any(pt in e[0].lower() for pt in patterns)]
        if not catalogue:
            print(f"No models matched --filter '{args.filter}'.")
            sys.exit(1)

    print(f"\nWill evaluate {len(catalogue)} model(s):")
    for name2, _, inp2 in catalogue:
        print(f"   {'[GPU]' if inp2 == 'tile' else '[cpu]'}  {name2}")

    results = evaluate_all(catalogue, X_feat, X_tiles, y)
    print_table(results)

    best = results[0]
    print(f"Best model : {best['name']}")
    print(f"  AUC      : {best['auc_mean']:.3f} +- {best['auc_std']:.3f}")
    print(f"  Accuracy : {best['acc_mean']:.3f}")
    print(f"  F1@0.50  : {best['f1_mean']:.3f}")
    print(f"  best-F1  : {best['best_f1']:.3f} @ threshold {best['best_thresh']:.2f}")

    if not args.no_save:
        print("\nSaving best model ...")
        save_best(best, X_feat, X_tiles, y, out_path)
        print("Production model updated.\n")
    else:
        print("\n--no-save: production model not updated.\n")


if __name__ == "__main__":
    main()
