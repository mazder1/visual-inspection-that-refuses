"""Segmentation metrics, never averaged into a single number.

The brief is explicit that per-pixel IoU and Dice are reported per defect
category rather than averaged, and the measured data says why: defect area
ranges from 1.67% of pixels on carpet to 8.60% on bottle. A single mean is
dominated by that spread rather than by model quality.

One subtlety drives the whole design. On a clean image the target is empty, and
any overlap metric on two empty sets is degenerate -- a model predicting nothing
scores a perfect 1.0. So **defect metrics are computed over defective images
only**, and clean images are reported separately as a false-alarm rate. Mixing
them produces a number that rewards doing nothing, which is exactly the failure
this project exists to avoid.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor

EPS = 1e-8


def dice_and_iou(
    prediction: Tensor, target: Tensor, eps: float = EPS
) -> Tuple[Tensor, Tensor]:
    """Per-image Dice and IoU from binary masks, shape ``(B, 1, H, W)``.

    Returns one value per image. Both are degenerate when the target is empty --
    an empty prediction scores 1.0 -- so callers must filter to defective
    images before aggregating.
    """
    dims = (1, 2, 3)
    intersection = (prediction * target).sum(dims)
    predicted_area = prediction.sum(dims)
    target_area = target.sum(dims)

    dice = (2 * intersection + eps) / (predicted_area + target_area + eps)
    union = predicted_area + target_area - intersection
    iou = (intersection + eps) / (union + eps)
    return dice, iou


class SegmentationMetrics:
    """Accumulate per-image results, then report them grouped.

    Holds one record per image rather than a running mean, so the report can be
    grouped by category and by defect type after the fact, and so a bootstrap
    interval can be taken over the records. With 44 defective test images across
    three categories, a point estimate on its own would overstate what was
    measured.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.records: List[Dict[str, object]] = []

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        categories: Sequence[str],
        defect_types: Sequence[str],
    ) -> None:
        prediction = (torch.sigmoid(logits.float()) > self.threshold).float()
        target = (targets > 0.5).float()
        dice, iou = dice_and_iou(prediction, target)

        dims = (1, 2, 3)
        predicted_fraction = prediction.mean(dims)
        target_area = target.sum(dims)

        for i, (category, defect_type) in enumerate(zip(categories, defect_types)):
            self.records.append(
                {
                    "category": category,
                    "defect_type": defect_type,
                    "defective": bool(target_area[i] > 0),
                    "dice": float(dice[i]),
                    "iou": float(iou[i]),
                    "predicted_fraction": float(predicted_fraction[i]),
                }
            )

    # --- aggregates -------------------------------------------------------

    @property
    def defective(self) -> List[Dict[str, object]]:
        return [r for r in self.records if r["defective"]]

    @property
    def clean(self) -> List[Dict[str, object]]:
        return [r for r in self.records if not r["defective"]]

    def mean_dice(self) -> float:
        """Mean Dice over defective images. The model-selection metric.

        Deliberately excludes clean images: including them would let a model
        that predicts nothing win.
        """
        rows = self.defective
        return sum(float(r["dice"]) for r in rows) / len(rows) if rows else 0.0

    def false_alarm_area(self) -> float:
        """Mean fraction of pixels wrongly called defect on clean images.

        Tracked alongside Dice because selecting on Dice alone is blind to
        whether the recall was bought with false alarms.
        """
        rows = self.clean
        return (
            sum(float(r["predicted_fraction"]) for r in rows) / len(rows)
            if rows
            else 0.0
        )

    def clean_images_touched(self) -> float:
        """Fraction of clean images with any predicted defect pixel at all."""
        rows = self.clean
        if not rows:
            return 0.0
        return sum(1 for r in rows if float(r["predicted_fraction"]) > 0) / len(rows)

    def by_defect_type(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for record in self.defective:
            grouped[(str(record["category"]), str(record["defect_type"]))].append(record)
        return {
            key: {
                "n": len(rows),
                "dice": sum(float(r["dice"]) for r in rows) / len(rows),
                "iou": sum(float(r["iou"]) for r in rows) / len(rows),
            }
            for key, rows in sorted(grouped.items())
        }

    def by_category(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for record in self.defective:
            grouped[str(record["category"])].append(record)
        return {
            category: {
                "n": len(rows),
                "dice": sum(float(r["dice"]) for r in rows) / len(rows),
                "iou": sum(float(r["iou"]) for r in rows) / len(rows),
            }
            for category, rows in sorted(grouped.items())
        }

    def summary(self) -> Dict[str, object]:
        return {
            "n_images": len(self.records),
            "n_defective": len(self.defective),
            "n_clean": len(self.clean),
            "mean_dice": self.mean_dice(),
            "false_alarm_area": self.false_alarm_area(),
            "clean_images_touched": self.clean_images_touched(),
            "by_category": self.by_category(),
            "by_defect_type": {
                f"{c}/{d}": v for (c, d), v in self.by_defect_type().items()
            },
        }

    def format_report(self) -> str:
        header = f"{'category / defect':<34} {'n':>4} {'IoU':>8} {'Dice':>8}"
        lines = [header, "-" * len(header)]
        for (category, defect), row in self.by_defect_type().items():
            lines.append(
                f"{category + ' / ' + defect:<34} {int(row['n']):>4} "
                f"{row['iou']:>8.3f} {row['dice']:>8.3f}"
            )
        lines.append("-" * len(header))
        for category, row in self.by_category().items():
            lines.append(
                f"{category + ' (all defects)':<34} {int(row['n']):>4} "
                f"{row['iou']:>8.3f} {row['dice']:>8.3f}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'clean images':<34} {len(self.clean):>4}   "
            f"false-alarm area {self.false_alarm_area():.4%}, "
            f"{self.clean_images_touched():.1%} touched"
        )
        return "\n".join(lines)
