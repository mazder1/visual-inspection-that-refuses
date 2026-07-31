"""Tests for the augmentation policy.

Two things are worth pinning. Geometric transforms must be applied *identically*
to image and mask, or every augmented sample teaches the model a mask that is
in the wrong place. And carpet must not get saturation jitter, because `color`
is one of its five defect classes and the augmentation would remove the evidence
for it.
"""

from __future__ import annotations

import pytest
import torch

from vinspect.train.augment import (
    POLICIES,
    AugmentedDataset,
    DefectAugmentation,
    Policy,
    policy_for,
)


def _pair(size=64):
    """An image whose bright square coincides exactly with the mask."""
    image = torch.zeros(3, size, size)
    mask = torch.zeros(1, size, size)
    image[:, 8:24, 40:56] = 1.0
    mask[:, 8:24, 40:56] = 1.0
    return image, mask


@pytest.mark.parametrize("seed", range(8))
def test_image_and_mask_stay_aligned(seed):
    """The mask must move exactly with the image, every time.

    Constructed so the bright region of the image *is* the mask; if the two
    transforms ever diverge, they stop overlapping.
    """
    torch.manual_seed(seed)
    image, mask = _pair()
    augmented_image, augmented_mask = DefectAugmentation("bottle")(image, mask)

    bright = (augmented_image.mean(0) > 0.5).float()
    overlap = (bright * augmented_mask[0]).sum()
    union = ((bright + augmented_mask[0]) > 0).float().sum()
    assert float(overlap / union) > 0.8, "image and mask drifted apart"


@pytest.mark.parametrize("seed", range(8))
def test_mask_stays_binary(seed):
    torch.manual_seed(seed)
    image, mask = _pair()
    _, augmented_mask = DefectAugmentation("hazelnut")(image, mask)
    assert torch.equal(augmented_mask, augmented_mask.round()), (
        "mask picked up interpolated values; it must resample nearest-neighbour"
    )


@pytest.mark.parametrize("seed", range(6))
def test_image_stays_in_range(seed):
    torch.manual_seed(seed)
    image, mask = _pair()
    augmented, _ = DefectAugmentation("bottle")(image, mask)
    assert float(augmented.min()) >= 0.0 and float(augmented.max()) <= 1.0


def test_disabled_augmentation_is_the_identity():
    image, mask = _pair()
    result_image, result_mask = DefectAugmentation("bottle", enabled=False)(image, mask)
    assert torch.equal(result_image, image)
    assert torch.equal(result_mask, mask)


def test_augmentation_actually_changes_something():
    torch.manual_seed(0)
    image, mask = _pair()
    changed = any(
        not torch.equal(DefectAugmentation("bottle")(image, mask)[0], image)
        for _ in range(5)
    )
    assert changed


def test_carpet_has_no_saturation_jitter():
    """`color` is one of carpet's defect classes; jittering saturation would
    augment away the signal the model is asked to find."""
    assert POLICIES["carpet"].saturation == 0.0
    assert policy_for("bottle").saturation > 0.0
    assert policy_for("hazelnut").saturation > 0.0


def test_unknown_category_falls_back_to_the_conservative_default():
    assert policy_for("screw") == Policy()


def test_no_elastic_deformation_anywhere():
    # Absent by design: it would bend a crack into a shape the fracture cannot
    # take. This asserts the policy has no knob for it.
    assert not any("elastic" in field for field in Policy.__dataclass_fields__)


def test_wrapper_preserves_the_rest_of_the_sample():
    class _Stub:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            image, mask = _pair()
            return {
                "image": image,
                "mask": mask,
                "label": torch.tensor(1),
                "category": "bottle",
                "defect_type": "crack",
                "key": "bottle/test/crack/000",
            }

    torch.manual_seed(0)
    sample = AugmentedDataset(_Stub(), DefectAugmentation("bottle"))[0]
    assert sample["category"] == "bottle"
    assert sample["key"] == "bottle/test/crack/000"
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
