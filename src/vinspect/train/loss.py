"""The segmentation objective: focal on every pixel, Tversky on the shapes.

Two losses answering different questions, because neither alone covers this
data.

**Focal** decides *which pixels deserve gradient*. It is binary cross-entropy
scaled by ``(1 - p_t) ** gamma``, so a pixel the model already has right
contributes almost nothing. That matters because only 0.86% of pixels in the
training split are defect: measured on one carpet image mid-training, plain BCE
spends 54% of its total on background pixels already at 99% confidence. Focal
reclaims essentially all of it.

**Tversky** decides *what the finished region should look like, and which way to
err*. It compares two sets rather than scoring pixels independently, and true
negatives never enter the formula -- so it is immune to class imbalance by
construction rather than by tuning. ``beta > alpha`` makes a missed defect cost
more than a false alarm, which is the only place in this training setup that
encodes the brief's asymmetric cost argument.

The pairing matters because they are blind in opposite directions. Focal scores
each pixel independently and would accept a correct-but-speckled prediction that
forms no coherent region. Tversky sees the region but produces a gradient that
is identical for every pixel of the same label, so it can never say *which*
pixel is wrong. Focal teaches; Tversky constrains.

Setting ``tversky_weight=0`` gives focal alone, and ``alpha=beta=0.5`` makes the
region term plain Dice. Both are wanted as controls: DRAEM reaches near-SOTA on
this dataset with focal alone, so the region term has to earn its place by
measurement rather than argument.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: Focal's exponent. 2.0 is the value from the RetinaNet paper and the one
#: DRAEM uses for its segmentation head on this dataset.
DEFAULT_GAMMA = 2.0

#: Tversky's false-positive and false-negative weights. 0.3 / 0.7 are the values
#: Salehi et al. found best for imbalanced segmentation. Constrained to sum to
#: 1.0 so the trade-off is a single dial.
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.7


class FocalTverskyLoss(nn.Module):
    """Focal over all pixels plus Tversky over the images that have a defect.

    Returns ``(total, components)``. The components are detached and exist so
    the training loop can log both terms: the weighting between them is the one
    thing here that cannot be reasoned out in advance, and watching one drown
    the other is much easier than predicting it.
    """

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        tversky_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be non-negative, got {alpha}, {beta}")
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta must equal 1.0 so the trade-off is one dial, "
                f"got {alpha} + {beta} = {alpha + beta}"
            )
        if tversky_weight < 0:
            raise ValueError(f"tversky_weight must be non-negative, got {tversky_weight}")
        if smooth <= 0:
            raise ValueError(f"smooth must be positive, got {smooth}")

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tversky_weight = float(tversky_weight)
        self.smooth = float(smooth)

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, alpha={self.alpha}, beta={self.beta}, "
            f"tversky_weight={self.tversky_weight}, smooth={self.smooth}"
        )

    def forward(
        self, logits: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """``logits`` are raw scores, not probabilities. ``targets`` are 0 or 1.

        Both are ``(B, 1, H, W)``.
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must match, got {tuple(logits.shape)} "
                f"and {tuple(targets.shape)}"
            )
        if logits.dim() != 4:
            raise ValueError(f"expected (B, 1, H, W), got {tuple(logits.shape)}")
        targets = targets.to(logits.dtype)

        # Forced to float32 because the Tversky term sums 262,144 values per
        # image. Under bfloat16 autocast sigmoid returns bfloat16, which carries
        # roughly three significant decimal digits, and a sum that long
        # accumulates enough error to make TP, FP and FN unreliable.
        logits = logits.float()
        # binary_cross_entropy_with_logits is exactly -log(p_t), computed in a
        # numerically stable way. Never sigmoid-then-log by hand: at |logit| of
        # 50 the naive version underflows to log(0).
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probability = torch.sigmoid(logits)

        # Probability the model assigned to the *correct* answer, written as
        # arithmetic rather than a branch so it runs over the whole batch at
        # once: the target being 0 or 1 switches one term off.
        p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
        focal = cross_entropy * (1.0 - p_t).pow(self.gamma)

        # Normalised by the positive-pixel count, following RetinaNet, and not
        # by the pixel count. Averaged over pixels this term sits near 0.003 on
        # MVTec, two orders of magnitude under Tversky's [0, 1] range, and the
        # auxiliary term would dominate the one carrying the gradient.
        n_positive = targets.sum().clamp(min=1.0)
        focal_term = focal.sum() / n_positive

        # Soft counts: a defect pixel predicted at 0.6 contributes 0.6 to the
        # hit and 0.4 to the miss. Thresholding here would zero the gradient.
        per_image = (1, 2, 3)
        true_positive = (probability * targets).sum(per_image)
        false_positive = (probability * (1.0 - targets)).sum(per_image)
        false_negative = ((1.0 - probability) * targets).sum(per_image)

        tversky_index = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        # On a clean image the true positives and false negatives are both zero,
        # so beta -- the entire reason for choosing Tversky -- multiplies zero
        # and only a smoothing-scaled residue is left. 81% of the training split
        # is clean, so those images are excluded rather than averaged in.
        has_defect = (targets.sum(per_image) > 0).to(logits.dtype)
        tversky_term = ((1.0 - tversky_index) * has_defect).sum() / has_defect.sum().clamp(
            min=1.0
        )

        total = focal_term + self.tversky_weight * tversky_term
        return total, {
            "focal": focal_term.detach(),
            "tversky": tversky_term.detach(),
            "n_defective_images": has_defect.sum().detach(),
        }
