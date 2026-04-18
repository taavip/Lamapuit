#!/usr/bin/env python3
"""
Interactive CDW tile labeling tool.

Splits a CHM GeoTIFF into 128×128 px chunks and displays each one for
binary classification using arrow keys:
  →  (right)   CDW present
  ←  (left)    No CDW
  ↑  (up)      Unknown / skip
  Esc or q     Save progress and quit

Auto-skips ground-only chunks (max pixel value < threshold).

Usage
-----
# Label one raster from scratch:
python scripts/label_tiles.py --chm chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif

# Resume from where you stopped:
python scripts/label_tiles.py --chm chm_max_hag/406455_2021_tava_chm_max_hag_20cm.tif --resume

# Custom chunk size / overlap:
python scripts/label_tiles.py --chm ... --chunk-size 128 --overlap 0.5 --output output/tile_labels
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import Window

try:
    from cdw_detect.wms_utils import (
        build_wms_layer_name,
        fetch_wms_for_bbox,
        parse_chm_filename,
    )
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[1]
    _src_dir = _repo_root / "src"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from cdw_detect.wms_utils import (
        build_wms_layer_name,
        fetch_wms_for_bbox,
        parse_chm_filename,
    )

# ── SLD terrain colormap (0→1.3m height) ────────────────────────────────────
_SLD_BREAKPOINTS = [
    (0.000, "#580a0c"),
    (0.065, "#f2854e"),
    (0.130, "#f9ab66"),
    (0.195, "#fcbf75"),
    (0.260, "#fec57b"),
    (0.325, "#fed68f"),
    (0.390, "#fee29e"),
    (0.455, "#fdedaa"),
    (0.520, "#f7f4b3"),
    (0.585, "#e4f2b4"),
    (0.650, "#d6eeb1"),
    (0.715, "#c9e9ae"),
    (0.780, "#bce4a9"),
    (0.845, "#addca8"),
    (0.910, "#9dd3a7"),
    (0.975, "#8bc6aa"),
    (1.040, "#78b9ad"),
    (1.105, "#65acb0"),
    (1.170, "#529eb4"),
    (1.235, "#3e91b7"),
    (1.300, "#2b83ba"),
]
_MAX_HAG = 1.3  # meters


def _make_sld_cmap():
    vals = [v / _MAX_HAG for v, _ in _SLD_BREAKPOINTS]
    colors = [c for _, c in _SLD_BREAKPOINTS]
    return mcolors.LinearSegmentedColormap.from_list("sld_terrain", list(zip(vals, colors)))


SLD_CMAP = _make_sld_cmap()


# ── Normalization (same as training pipeline) ─────────────────────────────────


def _normalize_for_model(tile: np.ndarray) -> np.ndarray:
    """Return uint8 grayscale tile via p2-p98 stretch + CLAHE (what model sees)."""
    nodata = ~np.isfinite(tile)
    t = tile.copy().astype(np.float32)
    t[nodata] = np.nan
    valid = t[~nodata]
    if valid.size > 10:
        p2, p98 = np.nanpercentile(t, 2), np.nanpercentile(t, 98)
        rng = p98 - p2
        norm = np.clip((t - p2) / (rng if rng > 1e-6 else 1.0), 0.0, 1.0)
        norm[nodata] = 0.0
        uint8 = (norm * 255).astype(np.uint8)
    else:
        uint8 = np.zeros(tile.shape, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    uint8 = clahe.apply(uint8)
    uint8[nodata] = 0
    return uint8


_DARK_THRESHOLD = 0.15 / _MAX_HAG  # values below 0.15 m fade to black


def _apply_sld(tile: np.ndarray) -> np.ndarray:
    """Return RGB uint8 image using SLD terrain colormap (0→1.3m range).

    Nodata and zero-height pixels are rendered as black.
    Heights 0–0.15 m are progressively darkened so they blend into nodata,
    helping the eye focus on structures above ground level.
    """
    nodata = ~np.isfinite(tile)
    is_zero = tile <= 0
    black_mask = nodata | is_zero
    t = tile.copy().astype(np.float32)
    t[black_mask] = 0.0
    t = np.clip(t, 0.0, _MAX_HAG) / _MAX_HAG  # normalise to [0,1]
    rgb = (SLD_CMAP(t)[:, :, :3] * 255).astype(np.uint8)
    # Darken low-height pixels: 0 m → fully black, 0.15 m → full colour
    dark_factor = np.where(
        t < _DARK_THRESHOLD,
        (t / _DARK_THRESHOLD) ** 0.7,  # concave ramp: dark near 0, quicker rise
        1.0,
    ).astype(np.float32)
    rgb = (rgb.astype(np.float32) * dark_factor[:, :, np.newaxis]).astype(np.uint8)
    rgb[black_mask] = 0  # force black for nodata / zero
    return rgb


# Feature extraction for sklearn removed — CNNPredictor uses raw pixel tiles directly.


# ── CNN-Deep-Attn architecture (mirrored from compare_classifiers.py) ─────────


def _build_deep_cnn_attn_net():
    """Rebuild CNN-Deep-Attn network for inference (must match training arch)."""
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


_CNN_BUILD_FNS = {
    "_build_deep_cnn_attn": _build_deep_cnn_attn_net,
}

_MODEL_NAME_TO_BUILD_FN = {
    "convnext_tiny": "_build_convnext_tiny_1ch",
    "convnext_small": "_build_convnext_small_1ch",
    "efficientnet_b2": "_build_effnet_b2",
}


def _adapt_first_conv_to_1ch(model) -> None:
    import torch.nn as nn

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            new_conv.weight.data = module.weight.data.mean(dim=1, keepdim=True)
            if module.bias is not None:
                new_conv.bias.data = module.bias.data.clone()

            parent = model
            parts = module_name.split(".")
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            leaf = parts[-1]
            if leaf.isdigit():
                parent[int(leaf)] = new_conv
            else:
                setattr(parent, leaf, new_conv)
            return


def _replace_classifier_head(model, num_classes: int = 2) -> None:
    import torch.nn as nn

    if hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes=num_classes)
            return
        except TypeError:
            model.reset_classifier(num_classes)
            return
        except Exception:
            pass

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Linear):
            model.classifier = nn.Linear(cls.in_features, num_classes)
            return
        if isinstance(cls, nn.Sequential):
            for i in range(len(cls) - 1, -1, -1):
                if isinstance(cls[i], nn.Linear):
                    cls[i] = nn.Linear(cls[i].in_features, num_classes)
                    return

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return

    raise RuntimeError("Unsupported model head replacement path")


def _build_convnext_tiny_1ch(pretrained: bool = False):
    from torchvision import models as tvm

    m = tvm.convnext_tiny(weights=None)
    _adapt_first_conv_to_1ch(m)
    _replace_classifier_head(m, 2)
    return m


def _build_convnext_small_1ch(pretrained: bool = False):
    from torchvision import models as tvm

    m = tvm.convnext_small(weights=None)
    _adapt_first_conv_to_1ch(m)
    _replace_classifier_head(m, 2)
    return m


_CNN_BUILD_FNS.update(
    {
        "_build_convnext_tiny_1ch": _build_convnext_tiny_1ch,
        "_build_convnext_small_1ch": _build_convnext_small_1ch,
    }
)


def _get_build_fn_for_model_name(model_name: str):
    canon = str(model_name or "").strip().lower()
    fn_name = _MODEL_NAME_TO_BUILD_FN.get(canon)
    if not fn_name:
        return None
    return _get_build_fn(fn_name)


def _get_build_fn(name: str):
    """Return a network build function by name.

    For functions not in _CNN_BUILD_FNS (e.g. EfficientNet from
    train_ensemble.py), attempt a lazy import from that module.
    """
    if name in _CNN_BUILD_FNS:
        return _CNN_BUILD_FNS[name]
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_ensemble",
            Path(__file__).parent / "train_ensemble.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, name, None)
        if fn is not None:
            _CNN_BUILD_FNS[name] = fn  # cache for subsequent calls
        return fn
    except Exception as exc:
        print(f"  [CNN] Cannot import build_fn '{name}': {exc}")
        return None


def _instantiate_model_from_build_fn(build_fn):
    """Instantiate a model from build_fn without forcing weight downloads.

    For builders that support it (e.g. _build_effnet_b2 in train_ensemble.py),
    pass pretrained=False during inference because checkpoint state_dict will be
    loaded immediately afterwards.
    """
    try:
        sig = inspect.signature(build_fn)
        if "pretrained" in sig.parameters:
            return build_fn(pretrained=False)
    except Exception:
        pass
    return build_fn()


# Minimum number of labeled tiles before a fine-tune subprocess is launched
_MIN_LABELS_FOR_FINETUNE = 50


class CNNPredictor:
    """CNN predictor supporting single-model and soft-vote ensemble inference.

    * Single-model path: loads ``ensemble_model.pt`` directly.
    * Ensemble path: reads ``ensemble_meta.json`` (written by train_ensemble.py)
      and soft-votes P(CDW) across all checkpoint models.

    Supports atomic hot-reload when the checkpoint is updated by the background
    fine-tune subprocess.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._net = None  # primary net (single-model fallback)
        self._nets: list = []  # all nets (ensemble mode, includes _net)
        self._weights: list[float] = []
        self._device = None
        self._thresh = 0.5  # best-F1 threshold from saved meta
        self._trained = False
        self._version = 0
        self._model_name = "CNN-Deep-Attn"

    def load_from_disk(self, pt_path: Path) -> bool:
        """Load CNN-Deep-Attn checkpoint directly from a .pt file.

        If an ``ensemble_meta.json`` exists alongside the checkpoint directory,
        the full ensemble is loaded automatically.

        Returns True on success.  Thread-safe.
        """
        # Force single-model CNN-Deep-Attn load for inference.
        # Deliberately ignore any ensemble_meta.json so the labeler uses the
        # single best checkpoint (CNN-Deep-Attn) and does not soft-vote across
        # multiple models.
        return self._load_cnn_pt(pt_path)

    def load_ensemble_meta(self, meta_path: Path) -> bool:
        """Load all checkpoints listed in *ensemble_meta.json*.

        Returns True if at least one model loaded successfully.
        """
        try:
            import torch

            meta = json.loads(meta_path.read_text())
            ckpts = meta.get("checkpoints", {})
            if not ckpts:
                return False

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            nets = []
            weights: list[float] = []
            thresh = 0.5
            version = 0
            for tag, entry in ckpts.items():
                if isinstance(entry, dict):
                    pt_path = Path(entry.get("path", ""))
                    model_name_hint = str(entry.get("model_name", "")).strip().lower()
                    build_fn_name_hint = str(entry.get("build_fn_name", "")).strip()
                    weight_hint = entry.get("weight", None)
                    threshold_hint = entry.get("threshold", None)
                else:
                    pt_path = Path(str(entry))
                    model_name_hint = ""
                    build_fn_name_hint = ""
                    weight_hint = None
                    threshold_hint = None

                if not pt_path.exists():
                    print(f"  [CNN] Ensemble checkpoint not found: {pt_path}")
                    continue
                try:
                    ckpt = torch.load(pt_path, map_location=dev, weights_only=False)

                    fn_name = build_fn_name_hint or ckpt.get("build_fn_name", "")
                    if model_name_hint:
                        model_name = model_name_hint
                    else:
                        model_name = str(ckpt.get("model_name", "")).strip().lower()

                    build_fn = _get_build_fn(fn_name) if fn_name else None
                    if build_fn is None and model_name:
                        build_fn = _get_build_fn_for_model_name(model_name)
                    if build_fn is None:
                        build_fn = _get_build_fn("_build_deep_cnn_attn")

                    if build_fn is None:
                        print(
                            f"  [CNN] Unknown build_fn for {tag} "
                            f"(hint={fn_name}, model={model_name}) — skipping"
                        )
                        continue

                    net = _instantiate_model_from_build_fn(build_fn).to(dev)
                    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
                    load_res = net.load_state_dict(state_dict, strict=False)
                    if getattr(load_res, "missing_keys", None):
                        if load_res.missing_keys:
                            print(f"  [CNN] {tag} missing keys: {load_res.missing_keys}")
                        if load_res.unexpected_keys:
                            print(f"  [CNN] {tag} unexpected keys: {load_res.unexpected_keys}")
                    net.eval()
                    nets.append(net)

                    m_thr = threshold_hint
                    if m_thr is None:
                        m_thr = ckpt.get("meta", {}).get("best_thresh", 0.5)
                    if m_thr:
                        thresh = float(m_thr)

                    weight_value = weight_hint
                    if weight_value is None:
                        weight_value = ckpt.get("meta", {}).get("best_f1", None)
                    if weight_value is None:
                        weight_value = 1.0
                    weights.append(float(weight_value))

                    print(f"  [CNN] Ensemble loaded {tag} from {pt_path.name}")
                except Exception as exc:
                    print(f"  [CNN] Failed to load ensemble checkpoint {tag}: {exc}")

            if not nets:
                return False

            # Use ensemble test threshold if available
            test_m = meta.get("test_metrics") or {}
            if test_m.get("ensemble_thresh"):
                thresh = float(test_m["ensemble_thresh"])

            if len(weights) != len(nets) or not weights:
                weights = [1.0] * len(nets)
            wsum = float(sum(max(0.0, w) for w in weights))
            if wsum <= 0.0:
                weights = [1.0 / len(nets)] * len(nets)
            else:
                weights = [max(0.0, w) / wsum for w in weights]

            with self._lock:
                self._nets = nets
                self._net = nets[0]
                self._weights = weights
                self._device = dev
                self._thresh = thresh
                self._trained = True
                self._model_name = f"Ensemble({len(nets)} models)"

            print(
                f"  [CNN] Ensemble ready: {len(nets)} model(s)  "
                f"thresh={thresh:.2f}  device={dev}"
            )
            return True
        except Exception as exc:
            print(f"  [CNN] Failed to load ensemble_meta: {exc}")
            return False

    def _load_cnn_pt(self, pt_path: Path) -> bool:
        """Internal: load/reload CNN from .pt checkpoint file."""
        try:
            import torch

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(pt_path, map_location=dev, weights_only=False)
            build_fn_name = ckpt.get("build_fn_name", "_build_deep_cnn_attn")
            build_fn = _get_build_fn(build_fn_name)
            if build_fn is None:
                print(f"  [CNN] Unknown build_fn '{build_fn_name}' — cannot load")
                return False
            net = _instantiate_model_from_build_fn(build_fn).to(dev)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            m = ckpt.get("meta", {})
            auc = m.get("auc", "?")
            best_f1 = m.get("best_f1", "?")
            best_thr = float(m.get("best_thresh", 0.5))
            n = m.get("n_samples", "?")
            ver = m.get("version", 1)
            with self._lock:
                self._net = net
                self._nets = [net]
                self._weights = [1.0]
                self._device = dev
                self._thresh = best_thr
                self._trained = True
                self._version = ver
                self._model_name = m.get("model_name", "CNN-Deep-Attn")
            print(
                f"  [CNN] Loaded {self._model_name} on {dev}  "
                f"AUC={auc}  F1={best_f1}@{best_thr:.2f}  "
                f"n={n}  v{ver}"
            )
            return True
        except Exception as exc:
            print(f"  [CNN] Failed to load {pt_path}: {exc}")
            return False

    def hot_reload(self, pt_path: Path, known_mtime: float) -> float:
        """Reload checkpoint if the .pt file has been replaced.

        Compares file mtime against *known_mtime*.  Returns the new mtime
        (unchanged if no reload occurred).  Safe to call frequently.
        """
        try:
            mtime = pt_path.stat().st_mtime
            if mtime > known_mtime + 1.0:  # 1 s grace to avoid partial writes
                print(f"  [CNN] Checkpoint updated — hot-reloading …", flush=True)
                ok = self._load_cnn_pt(pt_path)
                return mtime if ok else known_mtime
            return known_mtime
        except Exception:
            return known_mtime

    def predict_proba_cdw(self, tile: np.ndarray) -> float | None:
        """Return P(CDW) in [0, 1] using CNN inference, or None if not loaded.

        *tile* must be a 128x128 float32 array of raw CHM heights (metres).
        When multiple models are loaded (ensemble), their probabilities are
        soft-voted (averaged) for a more stable estimate.
        """
        with self._lock:
            nets = list(self._nets)
            weights = list(self._weights)
            net = self._net
            dev = self._device
            ready = self._trained
        if not ready or net is None:
            return None
        try:
            import torch

            # Normalise raw CHM heights [0-20 m] -> [0, 1]  (same as fine_tune_cnn.py)
            tile_norm = np.clip(tile, 0.0, 20.0) / 20.0
            x = torch.tensor(tile_norm[np.newaxis, np.newaxis], dtype=torch.float32).to(dev)
            with torch.no_grad():
                if not nets:
                    nets = [net]
                if len(weights) != len(nets):
                    weights = [1.0 / len(nets)] * len(nets)

                probs = []
                for model in nets:
                    model.eval()
                    probs.append(float(torch.softmax(model(x), dim=1)[0, 1].cpu()))
                prob = float(sum(w * p for w, p in zip(weights, probs)))
            return prob
        except Exception as exc:
            print(f"  [CNN] inference error: {exc}")
            return None

    def shutdown(self) -> None:
        pass  # no background threads to clean up


# ── Heatmap helpers ───────────────────────────────────────────────────────────

_HEATMAP_CYCLE = ("off", "IntGrad", "HiResCAM", "GradCAM+", "RISE")


def _tile_tensor(raw_tile: np.ndarray, device: "torch.device") -> "torch.Tensor":
    """Normalise a raw CHM tile to a (1,1,H,W) float32 tensor on *device*.

    Uses the same normalisation as fine_tune_cnn.py / backfill_model_prob.py:
    clip heights to [0, 20 m] and divide by 20 — identical to training time.
    """
    import torch

    normed = np.clip(raw_tile, 0.0, 20.0) / 20.0
    return torch.from_numpy(normed.copy()).float().unsqueeze(0).unsqueeze(0).to(device)


def _to_saliency_map(arr: np.ndarray) -> np.ndarray:
    """Normalise a 2-D float array to uint8 in [0, 255]."""
    arr = np.abs(arr).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)


def _last_3x3_conv(net: "torch.nn.Module") -> "torch.nn.Module | None":
    """Return the last Conv2d with kernel_size > 1 in the network.

    The AttnBlock architecture contains 3×3 feature convs AND 1×1 skip Conv2d.
    Iterating net.modules() returns the 1×1 skip as the very last Conv2d, which
    has no spatial receptive field and produces a flat/zero CAM.  We skip it.
    """
    import torch

    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size[0] > 1:
            last = m
    return last


def _hirescam(net: "torch.nn.Module", device: "torch.device", raw_tile: np.ndarray) -> np.ndarray:
    """HiResCAM: element-wise grad × feature-map, full spatial resolution."""
    import torch

    net.eval()
    t = _tile_tensor(raw_tile, device)
    last_conv = _last_3x3_conv(net)
    if last_conv is None:
        return np.zeros(raw_tile.shape, dtype=np.uint8)

    feature_map: list = []

    def _fwd_hook(_, __, out):
        # store the activation tensor (not detached) so autograd can compute grads
        feature_map.append(out)

    fh = last_conv.register_forward_hook(_fwd_hook)
    try:
        t.requires_grad_(True)
        logits = net(t)
        score = torch.softmax(logits, dim=1)[0, 1]
        # Compute gradients of score w.r.t. the saved feature map via autograd
        grads = torch.autograd.grad(score, feature_map[0], retain_graph=False)[0]

        F = feature_map[0][0].detach()  # (C, H, W)
        G = grads[0].detach()  # (C, H, W)
        cam = (G * F).sum(dim=0)  # (H, W)
        cam = torch.clamp(cam, min=0).cpu().numpy()
    finally:
        fh.remove()

    cam_resized = cv2.resize(
        cam, (raw_tile.shape[1], raw_tile.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return _to_saliency_map(cam_resized)


def _gradcam_pp(net: "torch.nn.Module", device: "torch.device", raw_tile: np.ndarray) -> np.ndarray:
    """GradCAM++: 2nd-order gradient weights for multi-instance detection."""
    import torch

    net.eval()
    last_conv = _last_3x3_conv(net)
    if last_conv is None:
        return np.zeros(raw_tile.shape, dtype=np.uint8)

    t = _tile_tensor(raw_tile, device)
    feature_map: list = []

    def _fh(_, __, out):
        feature_map.append(out)

    fh = last_conv.register_forward_hook(_fh)
    try:
        t.requires_grad_(True)
        logits = net(t)
        score = torch.softmax(logits, dim=1)[0, 1]
        # First-order gradients w.r.t. activation
        gp = torch.autograd.grad(score, feature_map[0], retain_graph=False)[0]

        A = feature_map[0][0].detach()  # (C, H, W)
        gp = gp[0].detach()  # (C, H, W)
        gp2 = gp**2
        gp3 = gp**3
        denom = 2.0 * gp2 + (A * gp3).sum(dim=(1, 2), keepdim=True)
        denom = torch.clamp(denom, min=1e-7)
        alpha = gp2 / denom  # (C, H, W)
        weights = (alpha * torch.clamp(gp, min=0)).sum(dim=(1, 2))  # (C,)
        cam = torch.clamp((weights[:, None, None] * A).sum(dim=0), min=0)
        cam = cam.cpu().numpy()
    finally:
        fh.remove()

    cam_resized = cv2.resize(
        cam, (raw_tile.shape[1], raw_tile.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return _to_saliency_map(cam_resized)


def _integrated_gradients(
    net: "torch.nn.Module", device: "torch.device", raw_tile: np.ndarray, steps: int = 50
) -> np.ndarray:
    """Integrated Gradients (Sundararajan 2017) — axiomatically faithful."""
    import torch

    net.eval()
    t = _tile_tensor(raw_tile, device)
    base = torch.zeros_like(t)
    grads = []
    for k in range(steps):
        alpha = (k + 1) / steps
        inp = (base + alpha * (t - base)).requires_grad_(True)
        logits = net(inp)
        score = torch.softmax(logits, dim=1)[0, 1]
        net.zero_grad()
        score.backward()
        grads.append(inp.grad.detach().cpu().squeeze().numpy())  # (H, W)
    ig = np.mean(grads, axis=0) * (t - base).squeeze().detach().cpu().numpy()
    return _to_saliency_map(ig)


def _rise(
    predictor: "CNNPredictor",
    raw_tile: np.ndarray,
    n_masks: int = 300,
    mask_res: int = 8,
    p_keep: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """RISE (Petsiuk BMVC2018) — model-agnostic causal attribution."""
    import torch

    rng = np.random.default_rng(seed)
    H, W = raw_tile.shape
    sal = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)

    for _ in range(n_masks):
        # Low-res binary mask, upsampled with bilinear smoothing
        small = rng.random((mask_res, mask_res)) < p_keep
        upmask = cv2.resize(small.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        upmask = (upmask > 0.5).astype(np.float32)
        masked = raw_tile * upmask
        prob = predictor.predict_proba_cdw(masked)
        if prob is None:
            continue
        sal += prob * upmask
        cnt += upmask

    with np.errstate(invalid="ignore", divide="ignore"):
        sal = np.where(cnt > 0, sal / cnt, 0.0)
    return _to_saliency_map(sal)


def _compute_heatmap(
    method: str,
    predictor: "CNNPredictor",
    raw_tile: np.ndarray,
    row_off: int,
    col_off: int,
    cache: dict,
) -> np.ndarray:
    """Compute (or return cached) saliency map for *method*.

    Returns a uint8 (H, W) array.
    """
    key = (method, row_off, col_off)
    if key in cache:
        return cache[key]

    if not predictor._trained or not predictor._nets:
        result = np.zeros(raw_tile.shape[:2], dtype=np.uint8)
        cache[key] = result
        return result

    import torch

    net = predictor._nets[0]
    device = predictor._device

    if method == "HiResCAM":
        result = _hirescam(net, device, raw_tile)
        # Fallback: if CAM returns all zeros, use Integrated Gradients instead
        if result.max() == 0:
            result = _integrated_gradients(net, device, raw_tile)
    elif method == "GradCAM+":
        result = _gradcam_pp(net, device, raw_tile)
        if result.max() == 0:
            result = _integrated_gradients(net, device, raw_tile)
    elif method == "IntGrad":
        result = _integrated_gradients(net, device, raw_tile)
    elif method == "RISE":
        result = _rise(predictor, raw_tile)
    else:
        result = np.zeros(raw_tile.shape[:2], dtype=np.uint8)

    cache[key] = result
    return result


# ── Chunk generator ───────────────────────────────────────────────────────────


def _iter_chunks(
    height: int,
    width: int,
    chunk_size: int,
    overlap: float,
) -> list[tuple[int, int]]:
    """Return list of (row_off, col_off) for all chunk origins.

    Always starts at (0, 0).  A border-clamped chunk is appended only if the
    uncovered gap at the far edge is more than chunk_size // 4 pixels, which
    avoids near-duplicate tiles when the raster size is only a few pixels
    larger than a multiple of the stride.
    """
    stride = max(1, int(chunk_size * (1.0 - overlap)))
    min_gap = chunk_size // 4  # minimum uncovered pixels before we add an edge chunk

    def make_offsets(size: int) -> list[int]:
        if size <= chunk_size:
            return [0]
        offsets = list(range(0, size - chunk_size + 1, stride))
        last_border = size - chunk_size
        # Gap between end of last chunk and raster edge
        gap = size - (offsets[-1] + chunk_size)
        if gap > min_gap and last_border not in offsets:
            offsets.append(last_border)
        return offsets

    rows = make_offsets(height)
    cols = make_offsets(width)
    # Deduplicate while preserving row-major order
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            if (r, c) not in seen:
                seen.add((r, c))
                result.append((r, c))
    return result


_YEAR_VIEW_FILENAME_RE = re.compile(
    r"^(?P<grid>\d+)_(?P<year>\d{4})_(?P<token>[a-z]+)(?P<tail>_chm_max_hag.*\.tif)$",
    re.IGNORECASE,
)


def _discover_temporal_rasters(chm_path: Path, chm_dir: Path | None = None) -> list[dict]:
    """Return same-place rasters across years for W/S archive browsing.

    Matching is intentionally strict to keep year switching deterministic and
    fast: same grid id + same filename tail (e.g. `_chm_max_hag_20cm.tif`).
    We pick one raster per year, preferring the current token when available.
    """
    match = _YEAR_VIEW_FILENAME_RE.match(chm_path.name)
    if match is None:
        return [{"year": None, "path": chm_path, "token": ""}]

    grid = match.group("grid")
    current_year = int(match.group("year"))
    current_token = match.group("token").lower()
    tail = match.group("tail").lower()
    root = chm_dir if chm_dir is not None else chm_path.parent

    by_year: dict[int, list[tuple[Path, str]]] = {}
    for candidate in sorted(root.glob(f"{grid}_*_*.tif")):
        cand_match = _YEAR_VIEW_FILENAME_RE.match(candidate.name)
        if cand_match is None:
            continue
        if cand_match.group("grid") != grid:
            continue
        if cand_match.group("tail").lower() != tail:
            continue
        year = int(cand_match.group("year"))
        token = cand_match.group("token").lower()
        by_year.setdefault(year, []).append((candidate, token))

    if current_year not in by_year:
        by_year[current_year] = [(chm_path, current_token)]

    entries: list[dict] = []
    for year in sorted(by_year):
        options = sorted(by_year[year], key=lambda t: t[0].name)
        if year == current_year:
            chosen_path = chm_path
            chosen_token = current_token
        else:
            preferred = [opt for opt in options if opt[1] == current_token]
            chosen_path, chosen_token = preferred[0] if preferred else options[0]
        entries.append(
            {
                "year": year,
                "path": chosen_path,
                "token": chosen_token,
            }
        )
    return entries


# ── Session save/load ─────────────────────────────────────────────────────────

# Canonical CSV header — includes provenance columns (backwards-compatible: old
# CSVs without these columns are read fine; missing fields default to "").
_CSV_HEADER = [
    "raster",
    "row_off",
    "col_off",
    "chunk_size",
    "label",
    "source",
    "annotator",
    "model_name",
    "model_prob",
    "timestamp",
]
# source values written to CSVs
_SRC_MANUAL = "manual"  # human pressed a key
_SRC_AUTO = "auto"  # CNN auto-advance (high confidence)
_SRC_AUTO_REVIEWED = "auto_reviewed"  # auto-labeled then shown to human for spot-check
_SRC_AUTO_SKIP = "auto_skip"  # auto-skipped because height < threshold


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """Return standard binary-classification metrics with zero-safe division."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def _parse_prob(value: object) -> float | None:
    """Parse probability-like value from CSV/user metadata, else None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _proposal_from_queue_meta(meta: dict) -> str | None:
    """Infer previous auto proposal from queue metadata.

    Prefers explicit `last_label` when available; otherwise derives label from
    `last_model_prob` (or `model_prob`) using 0.5 as the binary threshold.
    """
    last_label = str(meta.get("last_label", "")).strip().lower()
    if last_label in {"cdw", "no_cdw"}:
        return last_label
    prob = _parse_prob(meta.get("last_model_prob", ""))
    if prob is None:
        prob = _parse_prob(meta.get("model_prob", ""))
    if prob is None:
        return None
    return "cdw" if prob >= 0.5 else "no_cdw"


def _load_existing(
    csv_path: Path,
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, int], dict]]:
    """Return ({(row,col): label}, {(row,col): prov_dict}) from an existing CSV.

    Backwards-compatible: if source/annotator/model_prob columns are absent
    (old CSV format) those provenance fields default to empty strings.
    """
    done: dict[tuple[int, int], str] = {}
    prov: dict[tuple[int, int], dict] = {}
    if not csv_path.exists():
        return done, prov
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["row_off"]), int(row["col_off"]))
            done[key] = row["label"]
            prov[key] = {
                "source": row.get("source", ""),
                "annotator": row.get("annotator", ""),
                "model_name": row.get("model_name", ""),
                "model_prob": row.get("model_prob", ""),
                "timestamp": row.get("timestamp", ""),
            }
    return done, prov


def _append_row(
    csv_path: Path,
    raster: str,
    row_off: int,
    col_off: int,
    label: str,
    chunk_size: int,
    *,
    source: str = _SRC_MANUAL,
    annotator: str = "",
    model_name: str = "",
    model_prob: float | None = None,
) -> dict:
    """Append one label record to the CSV (creates header if new).

    Returns the provenance dict that was written so callers can store it.
    """
    write_header = not csv_path.exists()
    ts = datetime.now(tz=None).isoformat(timespec="seconds")
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(_CSV_HEADER)
        prob_str = "" if model_prob is None else f"{model_prob:.4f}"
        w.writerow(
            [
                raster,
                row_off,
                col_off,
                chunk_size,
                label,
                source,
                annotator,
                model_name,
                prob_str,
                ts,
            ]
        )
        f.flush()
        os.fsync(f.fileno())
    return {
        "source": source,
        "annotator": annotator,
        "model_name": model_name,
        "model_prob": prob_str,
        "timestamp": ts,
    }


def _rewrite_csv(
    csv_path: Path,
    done: dict[tuple[int, int], str],
    raster: str,
    chunk_size: int,
    prov: dict[tuple[int, int], dict] | None = None,
) -> None:
    """Rewrite the full CSV from the done dict (used after undo).

    Preserves provenance from *prov* for every tile that has it; tiles
    without an entry get source='manual' as a safe default.
    """
    prov = prov or {}
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        ts = datetime.now(tz=None).isoformat(timespec="seconds")
        for (r, c), label in done.items():
            p = prov.get((r, c), {})
            w.writerow(
                [
                    raster,
                    r,
                    c,
                    chunk_size,
                    label,
                    p.get("source", _SRC_MANUAL),
                    p.get("annotator", ""),
                    p.get("model_name", ""),
                    p.get("model_prob", ""),
                    p.get("timestamp", ts),
                ]
            )
        f.flush()
        os.fsync(f.fileno())


# ── Active-learning helpers ──────────────────────────────────────────────────


def _entropy_score(p: float) -> float:
    """Binary entropy H(p) ∈ [0, 1].  Maximum at p = 0.5 (most uncertain).

    For binary classification, entropy-based and margin-based active-learning
    rankings are equivalent (both monotone in |p − 0.5|), so entropy is used.
    A tile with entropy near 1.0 is the most informative for the model to learn
    from; sort descending to show those tiles first in the GUI.
    """
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _cdw_prob_color(cdw_pct: float) -> str:
    """Return a 5-band color for CDW probability percent.

    Lower CDW confidence is red, higher confidence is green.
    """
    if cdw_pct < 20.0:
        return "#e53935"  # red
    if cdw_pct < 40.0:
        return "#fb8c00"  # orange
    if cdw_pct < 60.0:
        return "#fdd835"  # yellow
    if cdw_pct < 80.0:
        return "#9ccc65"  # light green
    return "#43a047"  # green


# ── Main labeling session ─────────────────────────────────────────────────────


class QuitAllException(Exception):
    """Raised when user wants to quit the entire batch session."""

    pass


# ── Prefetch helpers ───────────────────────────────────────────────────

_PREFETCH_AHEAD = 8  # how many upcoming chunks to pre-render


def run_labeling_session(
    chm_path: Path,
    output_dir: Path,
    chm_dir: Path | None = None,  # fine-tune subprocess + temporal year discovery root
    chunk_size: int = 128,
    overlap: float = 0.5,
    auto_skip_threshold: float = 0.15,
    resume: bool = False,
    display_scale: int = 3,
    auto_advance_thresh: float = 0.0,  # 0=off; e.g. 0.92 → auto-label if CNN ≥92% or ≤8%
    review_pct: float = 0.0,  # fraction of auto-labeled tiles forced to GUI review (0–1)
    no_finetune: bool = False,  # skip background fine-tune launch
    annotator: str = "",  # human annotator id / name (written to CSV + session JSON)
    single_model: bool = False,  # force single checkpoint (ensemble_model.pt) instead of ensemble
    model_path: Path | None = None,  # optional explicit checkpoint path
    tile_list: "dict[str, list[tuple[int,int]]] | None" = None,  # audit queue filter
    tile_meta: "dict[str, dict[tuple[int,int], dict]] | None" = None,
) -> bool:
    """Run interactive labeling session for one CHM raster.

    Returns True if user quit (Esc/q), False if all chunks were labeled.
    Raises QuitAllException to signal "quit all rasters".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{chm_path.stem}_labels.csv"
    raster_name = chm_path.name
    raster_tile_meta: dict[tuple[int, int], dict] = (
        tile_meta.get(raster_name, {}) if tile_meta is not None else {}
    )

    # Load previous labels + provenance if resuming
    if resume:
        done, existing_prov = _load_existing(csv_path)
    else:
        done, existing_prov = {}, {}
    # Tracks provenance for tiles labeled *this* session (captured by closures).
    session_prov: dict[tuple[int, int], dict] = {}

    # ── Session provenance sidecar ───────────────────────────────────────────────────
    _session_tag = datetime.now(tz=None).strftime("%Y%m%d_%H%M%S")
    _session_json = output_dir / f"session_{_session_tag}.json"
    _session_meta: dict = {
        "raster": chm_path.name,
        "started_at": datetime.now(tz=None).isoformat(timespec="seconds"),
        "annotator": annotator,
        "auto_advance_thresh": auto_advance_thresh,
        "review_pct": review_pct,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "auto_skip_threshold": auto_skip_threshold,
        "tiles_at_start": len(done),
        "finished_at": None,
        "tiles_labeled_this_session": 0,
        "tile_list_mode": tile_list is not None,
        # Provenance counters to help QA and training decisions
        "manual_corrections": 0,
        "auto_corrections": 0,
        "fp_count": 0,
        "fn_count": 0,
        "model_path": str(model_path) if model_path is not None else "",
    }
    _session_json.write_text(json.dumps(_session_meta, indent=2))

    with rasterio.open(chm_path) as src:
        height, width = src.height, src.width
        res = abs(src.transform.a)  # metres/pixel, typically 0.2

    temporal_views = _discover_temporal_rasters(
        chm_path,
        chm_dir if chm_dir is not None else chm_path.parent,
    )
    label_year_idx = 0
    for i_view, entry in enumerate(temporal_views):
        entry_path = Path(entry["path"])
        try:
            same_path = entry_path.resolve() == chm_path.resolve()
        except Exception:
            same_path = entry_path == chm_path
        if same_path:
            label_year_idx = i_view
            break

    chunks = _iter_chunks(height, width, chunk_size, overlap)
    # Build work queue: chunks not yet labeled
    remaining = [c for c in chunks if c not in done]

    # When a tile-list is provided (e.g. from audit_review_queue.csv), restrict to
    # flagged tiles only.  Non-listed tiles are left as-is in the CSV.
    relabel: list[tuple[int, int]] = []
    if tile_list is not None:
        allowed_ordered = list(dict.fromkeys(tile_list.get(raster_name, [])))
        allowed_set = set(allowed_ordered)
        if not allowed_ordered:
            print(f"  [tile-list] No tiles for {raster_name} in queue — skipping raster.")
            return False
        # Keep queue-file order for deterministic resume and audit progression.
        remaining = [c for c in allowed_ordered if c in set(remaining)]
        # In tile-list/audit mode, only re-open already-labeled tiles when their
        # latest source is non-manual (auto/auto-skip). Tiles already reviewed by
        # a human should be skipped on resume.
        skipped_manual = 0
        for _rc in allowed_ordered:
            if _rc not in done:
                continue
            _src = str(existing_prov.get(_rc, {}).get("source", "")).strip().lower()
            if _src in {_SRC_MANUAL, _SRC_AUTO_REVIEWED}:
                skipped_manual += 1
                continue
            relabel.append(_rc)
        if relabel:
            print(f"  [tile-list] Including {len(relabel)} non-manual tiles for re-review")
            for _rc in relabel:
                done.pop(_rc, None)  # remove from done → GUI will show them
        if skipped_manual:
            print(f"  [tile-list] Skipping {skipped_manual} already-manual tiles")
        # Keep order stable and avoid duplicates when mixing relabel + unseen queue tiles.
        merged: list[tuple[int, int]] = []
        seen_rc: set[tuple[int, int]] = set()
        for _rc in (relabel + remaining):
            if _rc in seen_rc or _rc not in allowed_set:
                continue
            merged.append(_rc)
            seen_rc.add(_rc)
        remaining = merged
        print(f"  [tile-list] Restricted to {len(remaining)} audited tiles")
    _audit_mode = bool(tile_list is not None and (relabel or remaining))

    n_total = len(chunks)
    n_done_start = len(done)
    auto_count = 0
    # Tracks tiles diverted for 5% spot-check and their original auto proposal.
    spotcheck_candidates: dict[tuple[int, int], dict] = {}
    auto_advance_count = 0  # chunks auto-labeled by CNN this session

    # ── CNN predictor ─────────────────────────────────────────────────────────
    predictor = CNNPredictor()
    _model_path = model_path if model_path is not None else (output_dir / "ensemble_model.pt")
    _ensemble_meta = _model_path.parent / "ensemble_meta.json"
    _pt_mtime = 0.0
    _hot_reload_tick = 0  # counter; check mtime every 50 display updates

    # Default behaviour: prefer ensemble meta when available (more reliable
    # probabilities). Allow forcing single-model for ultra-low-latency via
    # the `single_model` flag.
    if (not single_model) and _ensemble_meta.exists():
        if predictor.load_ensemble_meta(_ensemble_meta):
            # If an explicit single .pt exists, keep its mtime for hot-reload checks
            _pt_mtime = _model_path.stat().st_mtime if _model_path.exists() else 0.0
        else:
            # Fall back to single checkpoint if ensemble load fails
            if _model_path.exists() and predictor.load_from_disk(_model_path):
                _pt_mtime = _model_path.stat().st_mtime
            else:
                print(
                    "  [CNN] No model found — auto-advance disabled.\n"
                    "  [CNN] Run:  python scripts/fine_tune_cnn.py  to train the model first."
                )
    else:
        # Single-model path (explicitly requested or no ensemble meta present)
        if _model_path.exists():
            if predictor.load_from_disk(_model_path):
                _pt_mtime = _model_path.stat().st_mtime
        else:
            print(
                "  [CNN] No model found — auto-advance disabled.\n"
                "  [CNN] Run:  python scripts/fine_tune_cnn.py  to train the model first."
            )

    # ── Launch fine-tune in background at session start (always) ──────────────
    _ft_proc: subprocess.Popen | None = None
    if not no_finetune:
        _ft_script = Path(__file__).parent / "fine_tune_cnn.py"
        # Count existing labels to determine if training is viable
        _n_labels = (
            sum(
                sum(1 for line in open(f) if ",cdw," in line or ",no_cdw," in line)
                for f in output_dir.glob("*_labels.csv")
            )
            if output_dir.exists()
            else 0
        )
        if _ft_script.exists() and _n_labels >= _MIN_LABELS_FOR_FINETUNE:
            _ft_log = output_dir / "finetune.log"
            _ft_chm = str(chm_dir if chm_dir is not None else chm_path.parent)
            _ft_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(_ft_script),
                    "--labels-dir",
                    str(output_dir),
                    "--chm-dir",
                    _ft_chm,
                    "--output",
                    str(output_dir),
                ],
                stdout=open(_ft_log, "w"),
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent,
            )
            print(
                f"  [CNN] Fine-tune started in background  " f"PID={_ft_proc.pid}  log → {_ft_log}",
                flush=True,
            )
        elif not _ft_script.exists():
            print(f"  [CNN] fine_tune_cnn.py not found — skipping background fine-tune")
        else:
            print(
                f"  [CNN] Only {_n_labels} labeled tiles — need≥{_MIN_LABELS_FOR_FINETUNE} "
                f"to launch fine-tune"
            )

    # Prefetch: background thread renders context images ahead of time
    # Key: (row_off, col_off) → (raw_ctx, color_rgb, raw_tile_128)
    _pf_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    _pf_lock = threading.Lock()
    _pf_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")
    # Open a second file handle exclusively for the prefetch thread
    _pf_handle = rasterio.open(chm_path)

    # Context window: 5×2 chunks — label 1 chunk from left edge
    #   left  : 1 chunk context
    #   right : 3 chunks (yellow preview box) + 0.5 buffer  → 3 right
    #   top   : 0.5 chunk buffer
    #   bottom: 0.5 chunk buffer
    CTX_PAD_LEFT = 1 * chunk_size
    CTX_PAD_RIGHT = 3 * chunk_size
    CTX_PAD_TOP = chunk_size // 2
    CTX_PAD_BOTTOM = chunk_size // 2
    context_cols = CTX_PAD_LEFT + chunk_size + CTX_PAD_RIGHT  # 5 × chunk
    context_rows = CTX_PAD_TOP + chunk_size + CTX_PAD_BOTTOM  # 2 × chunk

    print(f"\nRaster : {raster_name}")
    print(f"Size   : {width}×{height} px  ({width*res:.0f}×{height*res:.0f} m)")
    print(
        f"Chunks : {n_total} total  |  {n_done_start} already labeled  |  {len(remaining)} remaining"
    )
    if len(temporal_views) > 1:
        years = ", ".join(str(v.get("year", "?")) for v in temporal_views)
        print(f"Years  : {years}  (W newer / S older, loops both ways)")
    if len(remaining) == 0:
        print("All chunks already labeled. Nothing to do.")
        return False
    _audit_mode_str = "  [AUDIT — Esc advances; q quits all]" if tile_list is not None else ""
    print(
        f"\nControls:  → CDW   ← No CDW   Space No CDW   ↑ Unknown   ↓ Skip 5   "
        f"z Undo   b Focus   h Heatmap   o Orthophoto   w Newer year   s Older year   "
        f"Esc Next   q QuitAll{_audit_mode_str}\n"
    )

    # ── Headless auto-advance pre-pass (no GUI needed for high-confidence tiles) ──
    if auto_advance_thresh > 0.0 and predictor._trained:
        import random as _random

        _rng = _random.Random(42)
        print(
            f"  [auto-adv] Headless pre-pass at thresh={auto_advance_thresh:.2f}  "
            f"review_pct={review_pct*100:.0f}% ...",
            flush=True,
        )
        _uncertain_remaining: list[tuple[int, int]] = []
        _pre_count = 0
        for _row_off, _col_off in remaining:
            # Auto-skip: height below noise threshold
            _raw_t = _pf_handle.read(
                1,
                window=Window(_col_off, _row_off, chunk_size, chunk_size),
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
            _valid_t = _raw_t[np.isfinite(_raw_t)]
            if _valid_t.size > 0 and float(_valid_t.max()) < auto_skip_threshold:
                done[(_row_off, _col_off)] = "no_cdw"
                prov_entry = _append_row(
                    csv_path,
                    raster_name,
                    _row_off,
                    _col_off,
                    "no_cdw",
                    chunk_size,
                    source=_SRC_AUTO_SKIP,
                    model_name=getattr(predictor, "_model_name", ""),
                )
                session_prov[(_row_off, _col_off)] = prov_entry
                auto_count += 1
                continue
            # CNN auto-advance
            _prob_t = predictor.predict_proba_cdw(_raw_t)
            if _prob_t is not None:
                if _prob_t >= auto_advance_thresh:
                    _aa_lbl = "cdw"
                elif _prob_t <= 1.0 - auto_advance_thresh:
                    _aa_lbl = "no_cdw"
                else:
                    _aa_lbl = None
                if _aa_lbl is not None:
                    # Randomly divert a fraction to GUI review regardless of confidence
                    if review_pct > 0.0 and _rng.random() < review_pct:
                        spotcheck_candidates[(_row_off, _col_off)] = {
                            "proposed_label": _aa_lbl,
                            "model_prob": float(_prob_t),
                        }
                        _uncertain_remaining.append((_row_off, _col_off))
                        continue
                    done[(_row_off, _col_off)] = _aa_lbl
                    prov_entry = _append_row(
                        csv_path,
                        raster_name,
                        _row_off,
                        _col_off,
                        _aa_lbl,
                        chunk_size,
                        source=_SRC_AUTO,
                        model_name=getattr(predictor, "_model_name", ""),
                        model_prob=_prob_t,
                    )
                    session_prov[(_row_off, _col_off)] = prov_entry
                    _pre_count += 1
                    if _pre_count % 500 == 0:
                        print(f"  [auto-adv] {_pre_count} / {len(remaining)} ...", flush=True)
                    continue
            # Uncertain → needs manual review via GUI
            _uncertain_remaining.append((_row_off, _col_off))

        remaining = _uncertain_remaining
        print(
            f"  [auto-adv] Pre-pass done: {_pre_count} auto-labeled, "
            f"{len(remaining)} need review, {auto_count} auto-skipped (low height)",
            flush=True,
        )

        if not remaining:
            # Every chunk handled headlessly — skip GUI entirely
            n_cdw_h = sum(1 for v in done.values() if v == "cdw")
            n_no_h = sum(1 for v in done.values() if v == "no_cdw")
            print(f"\nSaved → {csv_path}")
            print(f"  CDW: {n_cdw_h}  No CDW: {n_no_h}  Auto-skip: {auto_count}")
            print(f"  Total labeled: {len(done)} / {n_total}")
            _pf_handle.close()
            _pf_executor.shutdown(wait=False)
            predictor.shutdown()
            return False

    # ── Active-learning sort: show most uncertain tiles first ─────────────────
    # Tiles with CNN probability near 0.5 (max entropy) are hardest for the
    # model and most informative to label manually.  Sort them first so the
    # human's effort is concentrated where it reduces model uncertainty most.
    # For binary classification, entropy-based and margin sampling give the same
    # ranking — entropy used here because it is directly interpretable.
    if predictor._trained and remaining:
        print(f"  [active-learn] Scoring {len(remaining)} tiles by entropy …", flush=True)
        _scored: list[tuple[float, tuple[int, int]]] = []
        for _rc in remaining:
            _r2, _c2 = _rc
            _rt = _pf_handle.read(
                1,
                window=Window(_c2, _r2, chunk_size, chunk_size),
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
            _p2 = predictor.predict_proba_cdw(_rt)
            _ent2 = _entropy_score(_p2) if _p2 is not None else 1.0
            _scored.append((_ent2, _rc))
        _scored.sort(key=lambda t: t[0], reverse=True)  # high entropy first
        remaining = [rc for _, rc in _scored]
        if _scored:
            print(
                f"  [active-learn] Sorted. Top tile H={_scored[0][0]:.3f} "
                f"(most uncertain first)",
                flush=True,
            )

    # ── matplotlib setup ────────────────────────────────────────────────────
    # Select interactive backend: prefer MPLBACKEND env, then Qt if available, else Tk.
    import os as _os
    _preferred_backend = _os.environ.get("MPLBACKEND")
    if _preferred_backend:
        matplotlib.use(_preferred_backend, force=True)
    else:
        # Try a range of Qt bindings (PyQt5/6, PySide2/6). Prefer Qt5Agg for Qt5
        # and the generic QtAgg for Qt6 bindings.
        try:
            import PyQt5  # type: ignore
            matplotlib.use("Qt5Agg", force=True)
        except Exception:
            try:
                import PyQt6  # type: ignore
                matplotlib.use("QtAgg", force=True)
            except Exception:
                try:
                    import PySide2  # type: ignore
                    matplotlib.use("Qt5Agg", force=True)
                except Exception:
                    try:
                        import PySide6  # type: ignore
                        matplotlib.use("QtAgg", force=True)
                    except Exception:
                        matplotlib.use("TkAgg", force=True)

    display_cols = context_cols * display_scale  # 5×128×scale px
    display_rows = context_rows * display_scale  # 2×128×scale px
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")
    ax.set_title("HEIGHT  (SLD terrain colour — 0–1.3 m)", color="white", fontsize=12)
    # Try to maximise the window so chunks appear as large as possible
    try:
        fig.canvas.manager.window.attributes("-zoomed", True)
    except Exception:
        pass

    instruction = fig.text(
        0.5,
        0.01,
        "→/c CDW  ← No CDW  Space No CDW  ↑ Unk  ↓ Skip5  z Undo  b Focus  h Heatmap(IntGrad/HiResCAM/GradCAM+/RISE)  o Orthophoto  w/s Year  Esc Next  q QuitAll",
        ha="center",
        va="bottom",
        color="#cccccc",
        fontsize=11,
        bbox=dict(facecolor="#333333", edgecolor="none", pad=4),
    )

    img_color = ax.imshow(np.zeros((display_rows, display_cols, 3), dtype=np.uint8))

    chunk_px = chunk_size * display_scale
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # Corner-only focus marker: thinner and less intrusive than a full square.
    corner_len_px = max(10.0, min(chunk_px * 0.2, chunk_px * 0.35))
    focus_corner_lines: list[mlines.Line2D] = []
    for _ in range(8):
        ln = mlines.Line2D(
            [0, 0],
            [0, 0],
            color="#ff4444",
            linewidth=1.15,
            solid_capstyle="round",
            zorder=5,
        )
        ax.add_line(ln)
        focus_corner_lines.append(ln)

    archive_border = mpatches.Rectangle(
        (-0.5, -0.5),
        display_cols,
        display_rows,
        linewidth=6.0,
        edgecolor="#ff2f2f",
        facecolor="none",
        zorder=6,
        visible=False,
    )
    ax.add_patch(archive_border)

    def _set_focus_corner_lines(x_px: float, y_px: float) -> None:
        x0 = x_px - 0.5
        y0 = y_px - 0.5
        x1 = x0 + chunk_px
        y1 = y0 + chunk_px
        l = corner_len_px
        segments = [
            ((x0, x0 + l), (y0, y0)),
            ((x0, x0), (y0, y0 + l)),
            ((x1 - l, x1), (y0, y0)),
            ((x1, x1), (y0, y0 + l)),
            ((x0, x0 + l), (y1, y1)),
            ((x0, x0), (y1 - l, y1)),
            ((x1 - l, x1), (y1, y1)),
            ((x1, x1), (y1 - l, y1)),
        ]
        for ln, (xs, ys) in zip(focus_corner_lines, segments):
            ln.set_data(xs, ys)

    progress_text = fig.text(
        0.5,
        0.96,
        "",
        ha="center",
        va="top",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    year_mode_text = fig.text(
        0.01,
        0.985,
        "",
        ha="left",
        va="top",
        color="#dddddd",
        fontsize=9,
        bbox=dict(facecolor="#222222", edgecolor="none", alpha=0.65, pad=2),
    )
    focus_info_text = fig.text(
        0.99,
        0.985,
        "",
        ha="right",
        va="top",
        color="#dddddd",
        fontsize=9,
        bbox=dict(facecolor="#222222", edgecolor="none", alpha=0.65, pad=2),
    )
    eta_text = fig.text(
        0.5,
        0.92,
        "",
        ha="center",
        va="top",
        color="#aaaaaa",
        fontsize=10,
    )

    def _wms_footer_for_path(view_path: Path) -> str:
        parsed = parse_chm_filename(view_path.name)
        layer_name = build_wms_layer_name(view_path.name)
        if parsed is not None and layer_name is not None:
            return f"WMS: {parsed[0]} | {layer_name}"
        return "WMS: n/a"

    wms_text = fig.text(
        0.99,
        0.045,
        _wms_footer_for_path(chm_path),
        ha="right",
        va="bottom",
        color="#dddddd",
        fontsize=9,
        bbox=dict(facecolor="#222222", edgecolor="none", alpha=0.65, pad=2),
    )
    # CDW prediction readout (heuristic % of above-ground pixels)
    pred_text = fig.text(
        0.5,
        0.88,
        "",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    label_flash = fig.text(
        0.5,
        0.84,
        "",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="white",
    )
    focus_copy_path = output_dir / "current_focus_lest97.txt"

    plt.tight_layout(rect=[0, 0.06, 1, 0.87])

    # Shared state
    state = {
        "idx": 0,
        "done": dict(done),  # (row,col) → label
        "quit": False,
        "quit_all": False,  # True → stop the entire batch
        "auto": auto_count,
        "auto_adv": 0,  # CNN auto-advance count
        "flushed": n_done_start,
        "t_start": time.monotonic(),  # for ETA calculation
        "manual": 0,  # manual labels this session (for speed calc)
        "history": [],  # undo stack: [(row_off, col_off, label)]
        "last_prob": None,  # CNN P(CDW) of the *currently displayed* tile
        "last_coord_print": "",  # terminal print debounce for copy-friendly coords
        "year_views": temporal_views,
        "year_idx": label_year_idx,
        "label_year_idx": label_year_idx,
        # QA counters updated during the session
        "manual_corrections": 0,
        "auto_corrections": 0,
        "fp_count": 0,
        "fn_count": 0,
        # 5% spot-check QA counters (auto proposal vs human final label)
        "spotcheck_total": 0,
        "spotcheck_tp": 0,
        "spotcheck_tn": 0,
        "spotcheck_fp": 0,
        "spotcheck_fn": 0,
        "spotcheck_unknown": 0,
        "spotcheck_matches": 0,
        # Queue-aware QA counters (stratified for statistically meaningful reporting).
        "queue_qa": {
            "all_auto": {
                "reviewed": 0,
                "evaluated": 0,
                "unknown": 0,
                "unknown_proposal": 0,
                "matches": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            },
            "spotcheck": {
                "reviewed": 0,
                "evaluated": 0,
                "unknown": 0,
                "unknown_proposal": 0,
                "matches": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            },
            "low_confidence": {
                "reviewed": 0,
                "evaluated": 0,
                "unknown": 0,
                "unknown_proposal": 0,
                "matches": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            },
        },
        # Global display modifiers: must behave consistently regardless of
        # whether CHM or orthophoto is currently active.
        "blank_surround": False,
        "heatmap_mode": "off",
        "_heatmap_cache": {},  # (method, row_off, col_off) → uint8 ndarray
        "show_orthophoto": False,  # O — toggle orthophoto WMS as base layer
        "_wms_cache": {},  # cache_key -> RGB image
        "_wms_cache_order": [],  # insertion order for bounded cache
        "_notice_text": "",
        "_notice_color": "#ffaa44",
        "_notice_ttl": 0,
    }

    _WMS_CACHE_MAX = 16
    _ARCHIVE_CTX_CACHE_MAX = 48
    _archive_handles: dict[str, object] = {}
    _archive_ctx_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    _archive_ctx_order: list[tuple[str, int, int]] = []

    def _set_notice(text: str, color: str = "#ffaa44", ttl: int = 30) -> None:
        state["_notice_text"] = text
        state["_notice_color"] = color
        state["_notice_ttl"] = max(1, int(ttl))

    def _cache_put(key: tuple, img: np.ndarray) -> None:
        cache: dict = state["_wms_cache"]
        order: list = state["_wms_cache_order"]
        if key in cache:
            try:
                order.remove(key)
            except ValueError:
                pass
        cache[key] = img
        order.append(key)
        while len(order) > _WMS_CACHE_MAX:
            old = order.pop(0)
            cache.pop(old, None)

    def _active_layer() -> str:
        return "orthophoto" if state.get("show_orthophoto", False) else "chm"

    def _is_archive_view() -> bool:
        return int(state.get("year_idx", 0)) != int(state.get("label_year_idx", 0))

    def _current_year_view() -> dict:
        views = state.get("year_views") or [{"year": None, "path": chm_path, "token": ""}]
        idx = int(state.get("year_idx", 0)) % len(views)
        return views[idx]

    def _current_view_path() -> Path:
        return Path(_current_year_view().get("path", chm_path))

    def _current_view_year_label() -> str:
        year = _current_year_view().get("year")
        return str(year) if year is not None else "n/a"

    def _cycle_year(step: int) -> None:
        views = state.get("year_views", [])
        if len(views) <= 1:
            _set_notice("No archive years found for this raster", color="#ffaa44", ttl=25)
            if state["idx"] < len(remaining):
                _update_display(*remaining[state["idx"]])
            fig.canvas.draw_idle()
            return

        state["year_idx"] = (int(state.get("year_idx", 0)) + step) % len(views)
        year_label = _current_view_year_label()
        if _is_archive_view():
            _set_notice(
                f"Archive year {year_label} (labeling locked)",
                color="#ff5555",
                ttl=35,
            )
        else:
            _set_notice(
                f"Label year {year_label} (labeling enabled)",
                color="#66cc66",
                ttl=20,
            )

        if state["idx"] < len(remaining):
            _update_display(*remaining[state["idx"]])
        fig.canvas.draw_idle()

    def _update_queue_qa(bucket: str, proposed: str | None, label: str) -> None:
        q = state["queue_qa"][bucket]
        q["reviewed"] += 1
        if label == "unknown":
            q["unknown"] += 1
            return
        if proposed is None:
            q["unknown_proposal"] += 1
            return
        q["evaluated"] += 1
        if proposed == label:
            q["matches"] += 1
        if proposed == "cdw" and label == "cdw":
            q["tp"] += 1
        elif proposed == "no_cdw" and label == "no_cdw":
            q["tn"] += 1
        elif proposed == "cdw" and label == "no_cdw":
            q["fp"] += 1
        elif proposed == "no_cdw" and label == "cdw":
            q["fn"] += 1

    # Pre-open raster for fast reads
    src_handle = rasterio.open(chm_path)

    def _read_context_from_offsets(ds, r: int, c: int):
        """Read one context window + focus tile using raster pixel offsets."""
        ctx_r = max(0, r - CTX_PAD_TOP)
        ctx_c = max(0, c - CTX_PAD_LEFT)
        raw = ds.read(
            1,
            window=Window(ctx_c, ctx_r, context_cols, context_rows),
            boundless=True,
            fill_value=0,
        ).astype(np.float32)
        color = _apply_sld(raw)
        raw_tile = ds.read(
            1,
            window=Window(c, r, chunk_size, chunk_size),
            boundless=True,
            fill_value=0,
        ).astype(np.float32)
        return raw, color, raw_tile

    def _read_context_from_bounds(ds, r: int, c: int):
        """Read context from another year by reprojecting same map bbox to that raster."""
        ctx_r = max(0, r - CTX_PAD_TOP)
        ctx_c = max(0, c - CTX_PAD_LEFT)
        ctx_bbox = window_bounds(
            Window(ctx_c, ctx_r, context_cols, context_rows),
            src_handle.transform,
        )
        tile_bbox = window_bounds(
            Window(c, r, chunk_size, chunk_size),
            src_handle.transform,
        )
        raw = ds.read(
            1,
            window=window_from_bounds(*ctx_bbox, transform=ds.transform),
            out_shape=(context_rows, context_cols),
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        color = _apply_sld(raw)
        raw_tile = ds.read(
            1,
            window=window_from_bounds(*tile_bbox, transform=ds.transform),
            out_shape=(chunk_size, chunk_size),
            boundless=True,
            fill_value=0,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        return raw, color, raw_tile

    def _get_archive_handle(view_path: Path):
        key = str(view_path)
        handle = _archive_handles.get(key)
        if handle is None:
            handle = rasterio.open(view_path)
            _archive_handles[key] = handle
        return handle

    def _archive_cache_put(
        key: tuple[str, int, int],
        val: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        if key in _archive_ctx_cache:
            try:
                _archive_ctx_order.remove(key)
            except ValueError:
                pass
        _archive_ctx_cache[key] = val
        _archive_ctx_order.append(key)
        while len(_archive_ctx_order) > _ARCHIVE_CTX_CACHE_MAX:
            old = _archive_ctx_order.pop(0)
            _archive_ctx_cache.pop(old, None)

    def _get_archive_context(r: int, c: int, view_path: Path):
        key = (str(view_path), int(r), int(c))
        cached = _archive_ctx_cache.get(key)
        if cached is not None:
            return cached
        try:
            ds = _get_archive_handle(view_path)
            val = _read_context_from_bounds(ds, r, c)
        except Exception:
            return None
        _archive_cache_put(key, val)
        return val

    def _prefetch_one(r: int, c: int) -> None:
        """Read + colorise one context window in the background."""
        key = (r, c)
        with _pf_lock:
            if key in _pf_cache:
                return
        try:
            raw, color, raw_tile = _read_context_from_offsets(_pf_handle, r, c)
        except Exception:
            return
        with _pf_lock:
            _pf_cache[key] = (raw, color, raw_tile)

    def _schedule_prefetch(from_idx: int) -> None:
        """Queue prefetch jobs for the next _PREFETCH_AHEAD chunks."""
        for ahead in range(_PREFETCH_AHEAD):
            ni = from_idx + ahead
            if ni < len(remaining):
                nr, nc = remaining[ni]
                with _pf_lock:
                    if (nr, nc) not in _pf_cache:
                        _pf_executor.submit(_prefetch_one, nr, nc)

    def _get_context(r: int, c: int):
        """Return (raw_ctx, color_rgb, raw_tile) from cache or sync read."""
        key = (r, c)
        with _pf_lock:
            cached = _pf_cache.get(key)
        if cached is not None:
            return cached
        # Cache miss — read synchronously (should rarely happen)
        return _read_context_from_offsets(src_handle, r, c)

    def _get_view_context(r: int, c: int):
        view_path = _current_view_path()
        try:
            is_base = view_path.resolve() == chm_path.resolve()
        except Exception:
            is_base = view_path == chm_path
        if is_base:
            return _get_context(r, c)
        archive_ctx = _get_archive_context(r, c, view_path)
        if archive_ctx is not None:
            return archive_ctx
        _set_notice(f"Archive load failed for {view_path.name}; showing label year", color="#ff5555", ttl=35)
        return _get_context(r, c)

    def _update_display(row_off: int, col_off: int) -> None:
        nonlocal _pt_mtime, _hot_reload_tick
        view_path = _current_view_path()
        view_year_label = _current_view_year_label()
        archive_mode = _is_archive_view()
        raw_ctx, color_rgb, raw_tile = _get_view_context(row_off, col_off)
        ctx_row = max(0, row_off - CTX_PAD_TOP)
        ctx_col = max(0, col_off - CTX_PAD_LEFT)

        # Periodic hot-reload check (every 50 display refreshes)
        _hot_reload_tick += 1
        if _hot_reload_tick % 50 == 0 and _model_path.exists():
            _pt_mtime = predictor.hot_reload(_model_path, _pt_mtime)

        # Upscale for visibility (cv2 expects (width, height))
        color_big = cv2.resize(
            color_rgb, (display_cols, display_rows), interpolation=cv2.INTER_NEAREST
        )

        if state.get("show_orthophoto", False):
            layer = build_wms_layer_name(view_path.name)
            if layer is None:
                state["show_orthophoto"] = False
                _set_notice(
                    f"Orthophoto unavailable for year {view_year_label}; returned to CHM",
                    color="#ff5555",
                    ttl=30,
                )
            else:
                bbox = window_bounds(
                    Window(ctx_col, ctx_row, context_cols, context_rows),
                    src_handle.transform,
                )
                cache_key = (
                    layer,
                    round(float(bbox[0]), 3),
                    round(float(bbox[1]), 3),
                    round(float(bbox[2]), 3),
                    round(float(bbox[3]), 3),
                    display_cols,
                    display_rows,
                )
                ortho_img = state["_wms_cache"].get(cache_key)
                if ortho_img is None:
                    ortho_img = fetch_wms_for_bbox(
                        layer=layer,
                        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        width=display_cols,
                        height=display_rows,
                    )
                    if ortho_img is not None:
                        _cache_put(cache_key, ortho_img)
                if ortho_img is None:
                    state["show_orthophoto"] = False
                    _set_notice("Orthophoto request failed; returned to CHM")
                else:
                    if ortho_img.shape[:2] != (display_rows, display_cols):
                        ortho_img = cv2.resize(
                            ortho_img,
                            (display_cols, display_rows),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    # Never mutate cached orthophoto in-place via overlays/focus mask.
                    color_big = ortho_img.copy()

        # Work on a private frame so B/H operations are always reversible.
        color_big = color_big.copy()

        # Reposition focus marker (must be computed before overlay)
        actual_pad_col = (col_off - ctx_col) * display_scale
        actual_pad_row = (row_off - ctx_row) * display_scale
        _set_focus_corner_lines(actual_pad_col, actual_pad_row)

        # Focus-tile top-left coordinate in map CRS (L-EST97 in this project).
        x0, y0 = src_handle.transform * (col_off, row_off)
        coord_str = f"L-EST97 {y0:.2f}, {x0:.2f}"
        focus_line = f"{view_path.name} | {coord_str}"
        focus_info_text.set_text(focus_line)
        # Emit once per focus tile in terminal to make copying easy.
        if state.get("last_coord_print") != focus_line:
            print(f"  [focus-coord] {focus_line}", flush=True)
            state["last_coord_print"] = focus_line
            # Keep a live copy-friendly text file in output dir.
            try:
                focus_copy_path.write_text(focus_line + "\n")
            except Exception:
                pass

        # ── H: heatmap overlay (focused tile only) ──────────────────────────
        hm_active = state.get("heatmap_mode", "off")
        if hm_active != "off":
            sal = _compute_heatmap(
                hm_active,
                predictor,
                raw_tile,
                row_off,
                col_off,
                state["_heatmap_cache"],
            )

            # The heatmap is defined for the current tile only, so project it
            # into the focused tile ROI rather than stretching across context.
            r0 = max(0, int(actual_pad_row))
            c0 = max(0, int(actual_pad_col))
            r1 = min(display_rows, r0 + chunk_px)
            c1 = min(display_cols, c0 + chunk_px)
            if r1 > r0 and c1 > c0:
                roi_h = r1 - r0
                roi_w = c1 - c0
                sal_roi = cv2.resize(sal, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                sal_color = cv2.applyColorMap(sal_roi, cv2.COLORMAP_INFERNO)[:, :, ::-1]

                base_roi = color_big[r0:r1, c0:c1].astype(np.float32)
                blended = np.clip(
                    0.35 * base_roi + 0.65 * sal_color.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                color_big[r0:r1, c0:c1] = blended

        # ── B: blank surroundings — zero out everything outside current tile ─
        if state.get("blank_surround", False):
            mask = np.zeros((display_rows, display_cols), dtype=bool)
            r0 = int(actual_pad_row)
            c0 = int(actual_pad_col)
            r1 = min(display_rows, r0 + chunk_px)
            c1 = min(display_cols, c0 + chunk_px)
            mask[r0:r1, c0:c1] = True
            color_big[~mask] = 0

        img_color.set_data(color_big)

        if archive_mode:
            archive_border.set_visible(True)
            year_mode_text.set_text(
                f"Year {view_year_label} — ARCHIVE VIEW (locked; W newer / S older)"
            )
            year_mode_text.set_color("#ff5555")
        else:
            archive_border.set_visible(False)
            year_mode_text.set_text(f"Year {view_year_label} — LABEL VIEW")
            year_mode_text.set_color("#dddddd")

        # Stats
        n_labeled = len(state["done"])
        n_cdw = sum(1 for v in state["done"].values() if v == "cdw")
        n_no = sum(1 for v in state["done"].values() if v == "no_cdw")
        n_unk = sum(1 for v in state["done"].values() if v == "unknown")
        n_auto = state["auto"]
        n_adv = state["auto_adv"]
        pos = state["idx"] + n_done_start
        pct = 100.0 * n_labeled / n_total if n_total else 0

        progress_text.set_text(
            f"[{pos+1} / {n_total}]  {pct:.1f}%  "
            f"CDW: {n_cdw}  No: {n_no}  Unk: {n_unk}  "
            f"Auto-skip: {n_auto}  CNN-auto: {n_adv}"
        )

        # ETA based on manual labeling speed
        elapsed = time.monotonic() - state["t_start"]
        manual = state["manual"]
        n_remaining = len(remaining) - state["idx"]
        if manual > 3 and elapsed > 0:
            speed = manual / elapsed  # chunks / sec
            eta_s = n_remaining / speed
            if eta_s < 60:
                eta_str = f"{eta_s:.0f}s"
            elif eta_s < 3600:
                eta_str = f"{eta_s/60:.1f}min"
            else:
                eta_str = f"{eta_s/3600:.1f}h"
            eta_text.set_text(f"{n_remaining} remaining  |  {speed:.1f} chunks/s  |  ETA {eta_str}")
        else:
            eta_text.set_text(f"{n_remaining} remaining")

        hm_active = state.get("heatmap_mode", "off")
        hm_suffix = f"  [{hm_active}]" if hm_active != "off" else ""
        if archive_mode:
            state["last_prob"] = None
            pred_label = f"Archive year {view_year_label} — labeling disabled" + hm_suffix
            pred_color = "#ff5555"
        else:
            ml_prob = predictor.predict_proba_cdw(raw_tile)
            state["last_prob"] = ml_prob  # saved so _record can write it to CSV
            if ml_prob is not None:
                cdw_pct = ml_prob * 100.0
                pred_label = (
                    f"CNN CDW: {cdw_pct:.0f}%  "
                    f"(v{predictor._version}  thresh={predictor._thresh:.2f})" + hm_suffix
                )
            else:
                pred_label = "CNN not loaded — run fine_tune_cnn.py"
                cdw_pct = 0.0
            pred_color = _cdw_prob_color(cdw_pct)

        pred_text.set_text(pred_label)
        pred_text.set_color(pred_color)
        # Color the focus corner markers the same as the prediction
        for ln in focus_corner_lines:
            ln.set_color(pred_color)

        # Keep WMS validation info visible in the lower-right corner.
        wms_text.set_text(_wms_footer_for_path(view_path))

        if state.get("_notice_ttl", 0) > 0 and state.get("_notice_text"):
            label_flash.set_text(state["_notice_text"])
            label_flash.set_color(state.get("_notice_color", "#ffaa44"))
            state["_notice_ttl"] -= 1
            if state["_notice_ttl"] <= 0:
                state["_notice_text"] = ""
        else:
            label_flash.set_text("")
        fig.canvas.draw_idle()

    def _next_chunk() -> None:
        """Advance to next chunk, auto-skipping ground-only tiles."""
        _schedule_prefetch(state["idx"])
        while state["idx"] < len(remaining):
            row_off, col_off = remaining[state["idx"]]
            if (row_off, col_off) in state["done"]:
                state["idx"] += 1
                continue
            # Auto-skip low tiles (height below threshold)
            raw = src_handle.read(
                1, window=Window(col_off, row_off, chunk_size, chunk_size)
            ).astype(np.float32)
            valid = raw[np.isfinite(raw)]
            if valid.size > 0 and float(valid.max()) < auto_skip_threshold:
                state["done"][(row_off, col_off)] = "no_cdw"
                prov_entry = _append_row(
                    csv_path,
                    raster_name,
                    row_off,
                    col_off,
                    "no_cdw",
                    chunk_size,
                    source=_SRC_AUTO_SKIP,
                    model_name=getattr(predictor, "_model_name", ""),
                )
                session_prov[(row_off, col_off)] = prov_entry
                state["auto"] += 1
                state["idx"] += 1
                continue
            # Auto-advance: CNN confident → label silently without showing UI
            if auto_advance_thresh > 0.0 and predictor._trained:
                _, _, aa_tile = _get_context(row_off, col_off)
                aa_prob = predictor.predict_proba_cdw(aa_tile)
                if aa_prob is not None:
                    if aa_prob >= auto_advance_thresh:
                        aa_label = "cdw"
                    elif aa_prob <= 1.0 - auto_advance_thresh:
                        aa_label = "no_cdw"
                    else:
                        aa_label = None
                    if aa_label is not None:
                        state["done"][(row_off, col_off)] = aa_label
                        prov_entry = _append_row(
                            csv_path,
                            raster_name,
                            row_off,
                            col_off,
                            aa_label,
                            chunk_size,
                            source=_SRC_AUTO,
                            model_name=getattr(predictor, "_model_name", ""),
                            model_prob=aa_prob,
                        )
                        session_prov[(row_off, col_off)] = prov_entry
                        state["auto_adv"] += 1
                        state["idx"] += 1
                        if state["auto_adv"] % 50 == 0:
                            print(
                                f"  [auto-adv] {state['auto_adv']} chunks "
                                f"auto-labeled at thresh={auto_advance_thresh:.2f}",
                                flush=True,
                            )
                        continue
            # Show this chunk
            _update_display(row_off, col_off)
            return
        # All done
        progress_text.set_text("✓ All chunks labeled!")
        fig.canvas.draw_idle()
        plt.pause(1.5)
        _save_and_quit(quit_all=False)

    def _record(label: str) -> None:
        if _is_archive_view():
            return
        if state["idx"] >= len(remaining):
            return
        row_off, col_off = remaining[state["idx"]]
        prob = state.get("last_prob")
        state["done"][(row_off, col_off)] = label
        prov_entry = _append_row(
            csv_path,
            raster_name,
            row_off,
            col_off,
            label,
            chunk_size,
            source=_SRC_MANUAL,
            annotator=annotator,
            model_name=getattr(predictor, "_model_name", ""),
            model_prob=prob,
        )
        session_prov[(row_off, col_off)] = prov_entry
        state["history"].append((row_off, col_off, label))
        # Increment manual label counter
        state["manual"] += 1

        # QA counters: detect whether this manual label corrects a previous
        # high-confidence model or auto label. We inspect any existing
        # provenance for this tile (from prior sessions / pre-fill).
        prev = existing_prov.get((row_off, col_off))
        try:
            prev_prob = float(prev.get("model_prob", "")) if prev else None
        except Exception:
            prev_prob = None
        if prev_prob is not None:
            prev_pred = "cdw" if prev_prob >= 0.5 else "no_cdw"
            if prev_pred != label:
                state["manual_corrections"] += 1
                # Count FP/FN depending on direction of mismatch
                if prev_pred == "cdw" and label == "no_cdw":
                    state["fp_count"] += 1
                elif prev_pred == "no_cdw" and label == "cdw":
                    state["fn_count"] += 1
                # If the previous source was an auto label, mark as auto_correction
                if prev.get("source", "") in (_SRC_AUTO, _SRC_AUTO_REVIEWED):
                    state["auto_corrections"] += 1

        # Spot-check accounting: compare model auto-proposal against human label.
        sc = spotcheck_candidates.get((row_off, col_off))
        if sc is not None:
            state["spotcheck_total"] += 1
            proposed = sc.get("proposed_label")
            if label == "unknown":
                state["spotcheck_unknown"] += 1
            else:
                if proposed == label:
                    state["spotcheck_matches"] += 1
                if proposed == "cdw" and label == "cdw":
                    state["spotcheck_tp"] += 1
                elif proposed == "no_cdw" and label == "no_cdw":
                    state["spotcheck_tn"] += 1
                elif proposed == "cdw" and label == "no_cdw":
                    state["spotcheck_fp"] += 1
                elif proposed == "no_cdw" and label == "cdw":
                    state["spotcheck_fn"] += 1

        # Queue-based QA accounting (tile-list workflows): keep random
        # spot-check and low-confidence strata separate for valid interpretation.
        qmeta = raster_tile_meta.get((row_off, col_off))
        if qmeta is not None:
            q_last_source = str(qmeta.get("last_source", qmeta.get("source", ""))).strip().lower()
            if q_last_source == _SRC_AUTO:
                q_reason = str(qmeta.get("reason", "")).strip().lower()
                q_proposed = _proposal_from_queue_meta(qmeta)
                _update_queue_qa("all_auto", q_proposed, label)
                if q_reason == "spotcheck":
                    _update_queue_qa("spotcheck", q_proposed, label)
                elif q_reason == "low_confidence":
                    _update_queue_qa("low_confidence", q_proposed, label)

        flash_colors = {"cdw": "#e05c5c", "no_cdw": "#5cae5c", "unknown": "#d4a017"}
        label_flash.set_text(
            {"cdw": "✓  CDW", "no_cdw": "✗  No CDW", "unknown": "?  Unknown"}[label]
        )
        label_flash.set_color(flash_colors[label])
        fig.canvas.draw_idle()

        state["idx"] += 1
        plt.pause(0.1)
        _next_chunk()

    def _save_and_quit(quit_all: bool = True) -> None:
        src_handle.close()
        _pf_handle.close()
        _pf_executor.shutdown(wait=False)
        for _h in list(_archive_handles.values()):
            try:
                _h.close()
            except Exception:
                pass
        _archive_handles.clear()
        predictor.shutdown()
        state["quit"] = True
        state["quit_all"] = quit_all
        # In audit/re-label mode the CSV has old + new rows for corrected tiles.
        # Rewrite it once at exit so each tile has exactly one (latest) row.
        if _audit_mode:
            _rewrite_csv(
                csv_path,
                state["done"],
                raster_name,
                chunk_size,
                prov={**existing_prov, **session_prov},
            )
        n_cdw = sum(1 for v in state["done"].values() if v == "cdw")
        n_no = sum(1 for v in state["done"].values() if v == "no_cdw")
        n_unk = sum(1 for v in state["done"].values() if v == "unknown")
        print(f"\nSaved → {csv_path}")
        print(f"  CDW: {n_cdw}  No CDW: {n_no}  Unknown: {n_unk}  " f"Auto-skip: {state['auto']}")
        print(f"  Total labeled: {len(state['done'])} / {n_total}")
        if _ft_proc is not None:
            if _ft_proc.poll() is None:
                print(
                    f"  [CNN] Fine-tune still running (PID {_ft_proc.pid}) — "
                    f"will complete in background  log → {output_dir / 'finetune.log'}",
                    flush=True,
                )
            else:
                print(f"  [CNN] Fine-tune completed (exit {_ft_proc.returncode})", flush=True)
        # Finalise session provenance JSON
        _session_meta["finished_at"] = datetime.now(tz=None).isoformat(timespec="seconds")
        _session_meta["tiles_labeled_this_session"] = state["manual"]
        _session_meta["auto_skip_count"] = state["auto"]
        _session_meta["auto_advance_count"] = state["auto_adv"]
        _session_meta["manual_corrections"] = state.get("manual_corrections", 0)
        _session_meta["auto_corrections"] = state.get("auto_corrections", 0)
        _session_meta["fp_count"] = state.get("fp_count", 0)
        _session_meta["fn_count"] = state.get("fn_count", 0)
        _session_meta["model_version"] = predictor._version

        # Spot-check quality metrics for 5% reviewed auto-labeled tiles.
        sc_tp = int(state.get("spotcheck_tp", 0))
        sc_tn = int(state.get("spotcheck_tn", 0))
        sc_fp = int(state.get("spotcheck_fp", 0))
        sc_fn = int(state.get("spotcheck_fn", 0))
        sc_total = int(state.get("spotcheck_total", 0))
        sc_unknown = int(state.get("spotcheck_unknown", 0))
        sc_matches = int(state.get("spotcheck_matches", 0))
        sc_eval = sc_tp + sc_tn + sc_fp + sc_fn
        _session_meta["spotcheck"] = {
            "candidate_count": int(len(spotcheck_candidates)),
            "reviewed_count": sc_total,
            "evaluated_count": sc_eval,
            "unknown_count": sc_unknown,
            "agreement_count": sc_matches,
            "agreement_rate": (sc_matches / sc_eval) if sc_eval else 0.0,
            "tp": sc_tp,
            "tn": sc_tn,
            "fp": sc_fp,
            "fn": sc_fn,
            **_binary_metrics(sc_tp, sc_tn, sc_fp, sc_fn),
        }

        def _queue_payload(bucket: str) -> dict:
            q = state["queue_qa"][bucket]
            tp = int(q["tp"])
            tn = int(q["tn"])
            fp = int(q["fp"])
            fn = int(q["fn"])
            evaluated = int(q["evaluated"])
            matches = int(q["matches"])
            return {
                "reviewed_count": int(q["reviewed"]),
                "evaluated_count": evaluated,
                "unknown_count": int(q["unknown"]),
                "unknown_proposal_count": int(q["unknown_proposal"]),
                "agreement_count": matches,
                "agreement_rate": (matches / evaluated) if evaluated else 0.0,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                **_binary_metrics(tp, tn, fp, fn),
            }

        _session_meta["queue_spotcheck"] = _queue_payload("spotcheck")
        _session_meta["queue_low_confidence"] = _queue_payload("low_confidence")
        _session_meta["queue_all_auto"] = _queue_payload("all_auto")
        _session_json.write_text(json.dumps(_session_meta, indent=2))

        # Emit concise QA summary in terminal when spot-check data exists.
        if sc_total > 0:
            sc = _session_meta["spotcheck"]
            print(
                "  [spot-check] "
                f"reviewed={sc['reviewed_count']} eval={sc['evaluated_count']} "
                f"agree={sc['agreement_rate']*100:.1f}% "
                f"F1={sc['f1']:.3f} Acc={sc['accuracy']:.3f}",
                flush=True,
            )
        q_sc = _session_meta["queue_spotcheck"]
        q_lc = _session_meta["queue_low_confidence"]
        q_all = _session_meta["queue_all_auto"]
        if q_all["reviewed_count"] > 0:
            print(
                "  [queue-qa:auto-all] "
                f"reviewed={q_all['reviewed_count']} eval={q_all['evaluated_count']} "
                f"agree={q_all['agreement_rate']*100:.1f}% "
                f"F1={q_all['f1']:.3f} Acc={q_all['accuracy']:.3f}",
                flush=True,
            )
        if q_sc["reviewed_count"] > 0:
            print(
                "  [queue-qa:spotcheck] "
                f"reviewed={q_sc['reviewed_count']} eval={q_sc['evaluated_count']} "
                f"agree={q_sc['agreement_rate']*100:.1f}% "
                f"F1={q_sc['f1']:.3f} Acc={q_sc['accuracy']:.3f}",
                flush=True,
            )
        if q_lc["reviewed_count"] > 0:
            print(
                "  [queue-qa:low-confidence] "
                f"reviewed={q_lc['reviewed_count']} eval={q_lc['evaluated_count']} "
                f"agree={q_lc['agreement_rate']*100:.1f}% "
                f"F1={q_lc['f1']:.3f} Acc={q_lc['accuracy']:.3f}",
                flush=True,
            )
        try:
            plt.close(fig)  # close THIS figure; don't kill others
        except Exception:
            pass

    def _skip_batch(count: int = 5) -> None:
        """Label next `count` chunks as no_cdw without showing them."""
        if _is_archive_view():
            return
        skipped = 0
        while skipped < count and state["idx"] < len(remaining):
            row_off, col_off = remaining[state["idx"]]
            if (row_off, col_off) not in state["done"]:
                state["done"][(row_off, col_off)] = "no_cdw"
                prov_entry = _append_row(
                    csv_path,
                    raster_name,
                    row_off,
                    col_off,
                    "no_cdw",
                    chunk_size,
                    source=_SRC_MANUAL,
                    annotator=annotator,
                    model_name=getattr(predictor, "_model_name", ""),
                )
                session_prov[(row_off, col_off)] = prov_entry
                state["history"].append((row_off, col_off, "no_cdw"))
                state["manual"] += 1
                skipped += 1
            state["idx"] += 1
        label_flash.set_text(f"⇣ Skipped {skipped} as No CDW")
        label_flash.set_color("#5cae5c")
        fig.canvas.draw_idle()
        plt.pause(0.05)
        _next_chunk()

    def _undo() -> None:
        """Undo the last manual label and re-show that chunk."""
        if _is_archive_view():
            label_flash.set_text("Archive view locked — return to label year")
            label_flash.set_color("#ff5555")
            fig.canvas.draw_idle()
            return
        if not state["history"]:
            label_flash.set_text("Nothing to undo")
            label_flash.set_color("#888888")
            fig.canvas.draw_idle()
            return
        row_off, col_off, _old_label = state["history"].pop()
        # Remove from done dict
        state["done"].pop((row_off, col_off), None)
        # We can’t remove from the CSV easily, so rewrite it
        _rewrite_csv(
            csv_path, state["done"], raster_name, chunk_size, prov={**existing_prov, **session_prov}
        )
        # Find the index in remaining and rewind
        try:
            idx = remaining.index((row_off, col_off))
            state["idx"] = idx
        except ValueError:
            pass
        state["manual"] = max(0, state["manual"] - 1)
        label_flash.set_text(f"↩ Undo ({_old_label})")
        label_flash.set_color("#d4a017")
        _update_display(row_off, col_off)

    def _on_key(event) -> None:
        if state["quit"]:
            return
        key = event.key
        key_norm = (key or "").strip()
        key_norm_lower = key_norm.lower()

        # Single-key quit-all: q should always stop the full batch.
        if key_norm_lower == "q":
            _save_and_quit(quit_all=True)
            return

        if key_norm_lower == "w":
            _cycle_year(+1)
            return
        if key_norm_lower == "s":
            _cycle_year(-1)
            return

        if _is_archive_view() and (
            key in {"right", "left", "up", "down", "z", " ", "space"}
            or key_norm_lower in {"c", "z"}
        ):
            _set_notice(
                "Archive view is locked — press W/S to return to label year",
                color="#ff5555",
                ttl=30,
            )
            label_flash.set_text("ARCHIVE VIEW — labeling disabled")
            label_flash.set_color("#ff5555")
            if state["idx"] < len(remaining):
                _update_display(*remaining[state["idx"]])
            fig.canvas.draw_idle()
            return

        if key == "right" or key == "c":
            _record("cdw")
        elif key == "left":
            _record("no_cdw")
        elif key == "up":
            _record("unknown")
        elif key == "down":
            _skip_batch(5)
            return
        elif key in (" ", "space"):
            _record("no_cdw")
        elif key == "z":
            _undo()
            return
        elif key == "escape":
            # In tile-list / audit mode Escape means "done with this raster,
            # advance to the next one" — quit_all would silently stop everything.
            _save_and_quit(quit_all=not _audit_mode)
        elif key == "b":
            state["blank_surround"] = not state.get("blank_surround", False)
            if state["idx"] < len(remaining):
                _update_display(*remaining[state["idx"]])
            mode_str = "ON" if state["blank_surround"] else "OFF"
            label_flash.set_text(f"Focus mode: {mode_str}")
            label_flash.set_color("#88aaff")
            fig.canvas.draw_idle()
            return
        elif key == "h":
            cur = _HEATMAP_CYCLE.index(state.get("heatmap_mode", "off"))
            nxt = _HEATMAP_CYCLE[(cur + 1) % len(_HEATMAP_CYCLE)]
            state["heatmap_mode"] = nxt
            if state["idx"] < len(remaining):
                rc = remaining[state["idx"]]
                # Evict cache for this method+tile so it recomputes fresh
                state["_heatmap_cache"].pop((nxt, rc[0], rc[1]), None)
                _update_display(*rc)
            if nxt != "off":
                label_flash.set_text(f"Heatmap: {nxt}")
            else:
                label_flash.set_text("Heatmap: off")
            label_flash.set_color("#88aaff")
            fig.canvas.draw_idle()
            return
        elif key_norm_lower == "o":
            state["show_orthophoto"] = not state.get("show_orthophoto", False)
            if state["idx"] < len(remaining):
                _update_display(*remaining[state["idx"]])
            if state["show_orthophoto"]:
                _set_notice("View: ORTHOPHOTO", color="#88aaff", ttl=20)
            else:
                _set_notice("View: CHM", color="#88aaff", ttl=20)
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # Show first chunk
    _next_chunk()

    # Block until quit
    plt.show(block=True)

    # Explicit figure teardown — ensures the Tk window is fully destroyed before
    # the next raster's plt.subplots() runs (prevents window accumulation).
    try:
        plt.close(fig)
    except Exception:
        pass

    if not state["quit"]:
        _save_and_quit(quit_all=False)

    if state.get("quit_all"):
        raise QuitAllException("User quit all rasters")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive CDW tile labeling tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls:\n"
            "  → (right)   CDW present\n"
            "  ← (left)    No CDW\n"
            "  ↑ (up)      Unknown / skip\n"
            "  o           Toggle orthophoto (WMS)\n"
            "  w / s       Newer / older year view (loops; archive is locked)\n"
            "  Esc         Save progress and continue to next raster\n"
            "  q           Save progress and quit ALL rasters"
        ),
    )
    p.add_argument("--chm", required=True, help="Path to CHM GeoTIFF")
    p.add_argument(
        "--output",
        default="output/tile_labels",
        help="Output directory for label CSVs (default: output/tile_labels)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Chunk size in pixels (default: 128 = 25.6m at 20cm/px)",
    )
    p.add_argument(
        "--overlap", type=float, default=0.5, help="Chunk overlap fraction (default: 0.5)"
    )
    p.add_argument(
        "--auto-skip-threshold",
        type=float,
        default=0.15,
        help="Auto-label as no_cdw if max height < this value in metres " "(default: 0.15)",
    )
    p.add_argument("--resume", action="store_true", help="Resume from previously saved labels")
    p.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Display upscale factor (default: 3 → 3072×1152 px image in maximised window)",
    )
    p.add_argument(
        "--auto-advance",
        type=float,
        default=0.0,
        help="CNN confidence threshold for auto-labeling (0=off; e.g. 0.97)",
    )
    p.add_argument(
        "--review-pct",
        type=float,
        default=0.05,
        help="Fraction of auto-labeled tiles sent to GUI for spot-check (default: 0.05)",
    )
    p.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip the background CNN fine-tune at session start",
    )
    p.add_argument(
        "--start-finetune",
        action="store_true",
        help="Explicitly start background CNN fine-tune at session start (opt-in)."
        " By default finetune is disabled unless this flag is used.",
    )
    p.add_argument(
        "--single-model",
        action="store_true",
        help="Force using the single checkpoint (ensemble_model.pt) for ultra-low-latency sessions; "
        "by default the tool will load ensemble_meta.json when available.",
    )
    p.add_argument(
        "--model-path",
        default="",
        help="Optional explicit checkpoint path (.pt). If set, it overrides output/ensemble_model.pt.",
    )
    p.add_argument(
        "--tile-list",
        default=None,
        help="CSV file with raster/row_off/col_off columns (e.g. audit_review_queue.csv). "
        "When given, only the listed tiles are shown for labeling.",
    )
    args = p.parse_args()

    # Load optional tile-list (audit queue)
    tile_list: "dict[str, list[tuple[int,int]]] | None" = None
    tile_meta: "dict[str, dict[tuple[int,int], dict]] | None" = None
    if args.tile_list:
        import csv as _csv
        from collections import defaultdict as _dd

        tile_list = _dd(list)
        tile_meta = _dd(dict)
        with open(args.tile_list, newline="") as _f:
            for _row in _csv.DictReader(_f):
                _raster = _row["raster"]
                _key = (int(_row["row_off"]), int(_row["col_off"]))
                tile_list[_raster].append(_key)
                tile_meta[_raster][_key] = {
                    "reason": _row.get("reason", ""),
                    "last_source": _row.get("last_source", _row.get("source", "")),
                    "last_model_prob": _row.get("last_model_prob", _row.get("model_prob", "")),
                    "last_label": _row.get("last_label", ""),
                }
        tile_list = dict(tile_list)
        tile_meta = dict(tile_meta)
        print(
            f"[tile-list] Loaded {sum(len(v) for v in tile_list.values()):,} tiles "
            f"across {len(tile_list)} raster(s) from {args.tile_list}"
        )

    # finetune will only run when explicitly requested (--start-finetune),
    # unless --no-finetune is provided which always disables it.
    run_labeling_session(
        chm_path=Path(args.chm),
        output_dir=Path(args.output),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        auto_skip_threshold=args.auto_skip_threshold,
        resume=args.resume,
        display_scale=args.scale,
        auto_advance_thresh=args.auto_advance,
        review_pct=args.review_pct,
        no_finetune=(args.no_finetune or not args.start_finetune),
        single_model=args.single_model,
        model_path=Path(args.model_path) if args.model_path else None,
        tile_list=tile_list,
        tile_meta=tile_meta,
    )


if __name__ == "__main__":
    main()
