"""Hybrid Dice + Focal loss with valid-pixel masking.

Ported from scripts/train_deeplabv3plus_manual_masks.py lines 173-219.
Math is unchanged; added PositiveWeightedDiceFocalLoss for class-imbalance tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class DiceFocalLoss(nn.Module):
    """Hybrid Dice + Focal loss with valid-pixel masking.

    Args:
        dice_weight: Weight for the Dice term.
        focal_weight: Weight for the Focal term.
        focal_alpha: Class-balance factor for focal loss (alpha for positives).
        focal_gamma: Focusing parameter; higher → harder examples weighted more.
        smooth: Laplace smoothing added to Dice numerator and denominator.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = EPS,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.smooth = float(smooth)

    def _masked_focal(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - p_t).pow(self.focal_gamma)) * bce
        denom = valid.sum().clamp_min(1.0)
        return (focal * valid).sum() / denom

    def _masked_dice(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits) * valid
        truth = targets * valid
        inter = (probs * truth).sum(dim=(1, 2, 3))
        den = probs.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (den + self.smooth)
        return 1.0 - dice.mean()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        focal = self._masked_focal(logits, targets, valid)
        dice = self._masked_dice(logits, targets, valid)
        return self.focal_weight * focal + self.dice_weight * dice


class TverskyFocalLoss(nn.Module):
    """Tversky index loss combined with Focal loss, with valid-pixel masking.

    The Tversky index (Salehi et al., 2017) generalises the Dice coefficient by
    independently weighting false positives (α) and false negatives (β).  Setting
    α > β penalises FP more heavily than FN, shifting the precision–recall balance
    toward higher precision — addressing V2's observed pattern of recall=0.55 but
    precision=0.076.

    Recommended starting point for CWD segmentation:
        alpha=0.6, beta=0.4  → mild FP penalty, keeps recall workable
        alpha=0.7, beta=0.3  → stronger FP penalty, use if FP rate is very high

    References:
        Salehi, S.S.M., Erdogmus, D., Gholipour, A. (2017). Tversky loss function
        for image segmentation using 3D fully convolutional deep networks. MLMI.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = EPS,
        pos_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.focal_weight = float(focal_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.smooth = float(smooth)
        self.pos_weight = float(pos_weight)

    def _masked_tversky(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits) * valid
        truth = targets * valid
        tp = (probs * truth).sum(dim=(1, 2, 3))
        fp = (probs * (1.0 - truth)).sum(dim=(1, 2, 3))
        fn = ((1.0 - probs) * truth).sum(dim=(1, 2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()

    def _masked_focal(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - p_t).pow(self.focal_gamma)) * bce
        class_weight = 1.0 + (self.pos_weight - 1.0) * targets
        focal = focal * class_weight
        denom = valid.sum().clamp_min(1.0)
        return (focal * valid).sum() / denom

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        tversky = self._masked_tversky(logits, targets, valid)
        focal = self._masked_focal(logits, targets, valid)
        return tversky + self.focal_weight * focal


class SoftCLDiceLoss(nn.Module):
    """Centerline Dice loss for thin/tubular structure segmentation.

    Computes Dice on differentiable soft skeletons of the prediction and ground
    truth rather than on their full masks.  Thin structures like fallen logs
    (1–5 px wide at 0.2 m/px) are penalised for connectivity breaks and blob
    fragmentation that area-based Dice/Tversky ignores.

    Soft skeleton approximation (Shit et al. 2021):
        skel(x) = x - open(x)   where open = dilate(erode(x))
    Implemented with 2-D min/max pooling (differentiable, GPU-compatible).

    References:
        Shit, S. et al. (2021). clDice — a Novel Topology-Preserving Loss
        Function for Tubular Structure Segmentation. CVPR.
    """

    def __init__(self, iter_: int = 3, smooth: float = EPS) -> None:
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth

    def _soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

    def _soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

    def _soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(img))

    def _soft_skel(self, img: torch.Tensor) -> torch.Tensor:
        skel = F.relu(img - self._soft_open(img))
        for _ in range(self.iter_ - 1):
            img = self._soft_erode(img)
            skel = skel + F.relu(img - self._soft_open(img))
        return skel

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(logits) * valid
        gt = targets * valid
        skel_pred = self._soft_skel(pred)
        skel_gt = self._soft_skel(gt)
        # Topology precision: skeleton of pred overlaps skeleton of GT
        tprec = (skel_pred * gt).sum() + self.smooth
        tprec = tprec / (skel_pred.sum() + self.smooth)
        # Topology recall: skeleton of GT overlaps pred
        tsens = (skel_gt * pred).sum() + self.smooth
        tsens = tsens / (skel_gt.sum() + self.smooth)
        cl_dice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
        return cl_dice


class PositiveWeightedDiceFocalLoss(DiceFocalLoss):
    """DiceFocalLoss with an extra positive-class multiplier on the focal term.

    Use when the positive (CWD) class is severely underrepresented; the
    pos_weight parameter scales the focal loss contribution for positive pixels.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = EPS,
        pos_weight: float = 3.0,
    ) -> None:
        super().__init__(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            smooth=smooth,
        )
        self.pos_weight = float(pos_weight)

    def _masked_focal(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - p_t).pow(self.focal_gamma)) * bce
        # Upweight positive pixels by pos_weight
        class_weight = 1.0 + (self.pos_weight - 1.0) * targets
        focal = focal * class_weight
        denom = valid.sum().clamp_min(1.0)
        return (focal * valid).sum() / denom
