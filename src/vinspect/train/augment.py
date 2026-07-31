"""Augmentation, justified per transform against the physics of the defect.

The brief asks for each augmentation to be argued from what the process can
actually produce, rather than copied from a default list. So each entry in
:data:`POLICIES` carries the argument, and one transform is switched off for one
category on those grounds.

**Geometric transforms are free here.** Bottles are photographed top-down and
are circular; hazelnuts sit at arbitrary orientations on the rig; carpet is
continuous texture. None of the three has a canonical up, so flips and
quarter-turns produce images the process could genuinely have produced. Small
rotation, translation and scale model a part being set down slightly differently
between acquisitions.

**Photometric transforms need more care.** Mild brightness and contrast model
lighting drift over a shift, which is real. Saturation is different: `color` is
one of carpet's five defect classes, so shifting saturation on carpet would
augment away the signal the model is being asked to find. It is switched off
there and left mild elsewhere.

**Elastic deformation is deliberately absent** for every category. It would bend
a crack into a shape the fracture could not take, which teaches the model that
impossible defects are ordinary.

Geometric transforms are applied identically to image and mask, and the mask is
always resampled with nearest-neighbour so it stays binary. Interpolating a mask
invents partial defects along every boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class Policy:
    """What is allowed for one category, and by how much."""

    flips: bool = True
    rotate90: bool = True
    degrees: float = 10.0
    translate: float = 0.05
    scale: Tuple[float, float] = (0.95, 1.05)
    brightness: float = 0.15
    contrast: float = 0.15
    #: Zero where colour itself is a defect class.
    saturation: float = 0.10


POLICIES: Dict[str, Policy] = {
    # `color` is one of carpet's five defect classes, so saturation jitter would
    # remove the evidence for it.
    "carpet": Policy(saturation=0.0),
    # Top-down and circular, so orientation carries no information at all.
    "bottle": Policy(),
    # Photographed at arbitrary rotations already; the dataset itself contains
    # the same nut at several orientations.
    "hazelnut": Policy(),
}

DEFAULT_POLICY = Policy()


def policy_for(category: str) -> Policy:
    """Policy for a category, falling back to the conservative default."""
    return POLICIES.get(category, DEFAULT_POLICY)


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high))


class DefectAugmentation:
    """Apply one category's policy to an (image, mask) pair.

    Randomness is drawn from torch's global generator, which ``DataLoader``
    seeds separately per worker, so multiple workers do not replay the same
    sequence.
    """

    def __init__(self, category: str, enabled: bool = True) -> None:
        self.category = category
        self.policy = policy_for(category)
        self.enabled = enabled

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.enabled:
            return image, mask
        policy = self.policy

        if policy.flips:
            if torch.rand(1).item() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if torch.rand(1).item() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

        if policy.rotate90:
            turns = int(torch.randint(0, 4, (1,)).item())
            if turns:
                image = torch.rot90(image, turns, dims=(-2, -1))
                mask = torch.rot90(mask, turns, dims=(-2, -1))

        if policy.degrees or policy.translate or policy.scale != (1.0, 1.0):
            angle = _uniform(-policy.degrees, policy.degrees)
            max_shift = int(policy.translate * image.shape[-1])
            shift = [
                int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                for _ in range(2)
            ]
            scale = _uniform(*policy.scale)
            # Identical geometry for both; nearest-neighbour on the mask so it
            # stays binary.
            image = TF.affine(
                image, angle=angle, translate=shift, scale=scale, shear=[0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask, angle=angle, translate=shift, scale=scale, shear=[0.0],
                interpolation=InterpolationMode.NEAREST,
            )

        if policy.brightness:
            image = TF.adjust_brightness(
                image, _uniform(1 - policy.brightness, 1 + policy.brightness)
            )
        if policy.contrast:
            image = TF.adjust_contrast(
                image, _uniform(1 - policy.contrast, 1 + policy.contrast)
            )
        if policy.saturation:
            image = TF.adjust_saturation(
                image, _uniform(1 - policy.saturation, 1 + policy.saturation)
            )

        return image.clamp(0.0, 1.0), (mask > 0.5).to(mask.dtype)


class AugmentedDataset(Dataset):
    """Wrap a dataset so augmentation is a separate concern from loading.

    A wrapper rather than a parameter on ``MVTecDataset``, so the evaluation
    path cannot accidentally become non-deterministic: if augmentation is not
    wrapped around it, it is not applied.
    """

    def __init__(self, base: Dataset, augmentation: DefectAugmentation) -> None:
        self.base = base
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = dict(self.base[index])
        sample["image"], sample["mask"] = self.augmentation(
            sample["image"], sample["mask"]
        )
        return sample
