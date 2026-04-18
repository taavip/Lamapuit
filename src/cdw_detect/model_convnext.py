"""
ConvNeXt V2 backbone + FPN + Mask R-CNN for CDW instance segmentation.

Replaces YOLO11-seg with a two-stage detector:
  - Backbone : ConvNeXt V2 (timm, pretrained on ImageNet-22k+1k)
  - Neck     : Feature Pyramid Network (torchvision FPN)
  - Head     : Mask R-CNN (torchvision, standard RPN + RoI heads)

Strides produced by ConvNeXt V2 Tiny:
  stage-0 → 4×    [96  ch]  (not used – too high res for FPN)
  stage-1 → 8×    [192 ch]
  stage-2 → 16×   [384 ch]
  stage-3 → 32×   [768 ch]
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import timm
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)

# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────


class ConvNeXtFPN(nn.Module):
    """ConvNeXt V2 body + FPN neck.

    Exposes ``out_channels`` attribute required by torchvision detectors.
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny",
        pretrained: bool = True,
        out_channels: int = 256,
        in_channels_override: Optional[list[int]] = None,
        use_stages: tuple[int, ...] = (1, 2, 3),  # skip stage-0 (stride 4)
    ) -> None:
        super().__init__()
        self.use_stages = use_stages
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        all_channels: list[int] = self.body.feature_info.channels()
        in_ch = (
            in_channels_override if in_channels_override else [all_channels[i] for i in use_stages]
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_ch,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        all_feats = self.body(x)
        feat_dict: OrderedDict = OrderedDict()
        for key_idx, stage_idx in enumerate(self.use_stages):
            feat_dict[str(key_idx)] = all_feats[stage_idx]
        return self.fpn(feat_dict)


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────


def build_convnext_maskrcnn(
    model_name: str = "convnextv2_tiny",
    num_classes: int = 2,  # 1 CDW class + background
    pretrained_backbone: bool = True,
    fpn_out_channels: int = 256,
    # Anchor settings tuned for small CDW objects (20-200 px at 20 cm/px).
    # FPN with LastLevelMaxPool on 3 backbone stages → 4 output levels.
    anchor_sizes: tuple = ((32,), (64,), (128,), (256,)),
    aspect_ratios: tuple = ((0.5, 1.0, 2.0),) * 4,
    # RPN / detector thresholds
    box_score_thresh: float = 0.25,
    box_nms_thresh: float = 0.40,
    box_detections_per_img: int = 200,
) -> MaskRCNN:
    """Return a Mask R-CNN with ConvNeXt V2 backbone.

    All detection/NMS thresholds match the existing YOLO inference defaults.
    """
    backbone = ConvNeXtFPN(
        model_name=model_name,
        pretrained=pretrained_backbone,
        out_channels=fpn_out_channels,
    )
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios,
    )
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_convnext_maskrcnn(pretrained_backbone=False)
    model.eval()
    dummy = [torch.zeros(3, 640, 640)]
    with torch.no_grad():
        out = model(dummy)
    print("Output keys:", out[0].keys())
    print("Masks shape:", out[0]["masks"].shape)
    print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
