"""Tests for the segmentation objective.

The properties worth pinning are the ones a later change could silently break:
that clean images really are excluded from the region term, that focal is
normalised by positive count rather than pixel count, and that the asymmetry
actually points the way it is supposed to. Each of those is a decision that was
argued for, so each gets a test rather than a comment.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vinspect.train.loss import FocalTverskyLoss

CONFIDENT = 10.0  # a logit of +-10 is sigmoid ~= 0.99995 / 0.00005


def _blank(batch=1, size=32):
    return torch.zeros(batch, 1, size, size)


def _with_box(tensor, index=0, rows=slice(0, 10), cols=slice(0, 10), value=1.0):
    tensor = tensor.clone()
    tensor[index, 0, rows, cols] = value
    return tensor


@pytest.fixture
def criterion():
    return FocalTverskyLoss()


def test_perfect_prediction_is_almost_zero(criterion):
    targets = _with_box(_blank())
    logits = _with_box(_blank(), value=CONFIDENT) - CONFIDENT * (1 - targets)
    total, parts = criterion(logits, targets)

    assert float(total) < 0.05
    assert float(parts["focal"]) < 0.02
    assert float(parts["tversky"]) < 0.05


def test_confidently_wrong_is_expensive(criterion):
    targets = _with_box(_blank())
    inverted = CONFIDENT * (1 - targets) - CONFIDENT * targets
    wrong, _ = criterion(inverted, targets)
    right, _ = criterion(-inverted, targets)
    assert float(wrong) > 10 * float(right)


def test_gradient_reaches_the_logits(criterion):
    targets = _with_box(_blank())
    logits = torch.zeros_like(targets, requires_grad=True)
    total, _ = criterion(logits, targets)
    total.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


# --- the masking decision -------------------------------------------------


def test_clean_images_are_excluded_from_the_region_term(criterion):
    """Adding a clean image to the batch must not move the Tversky term.

    On a clean image both TP and FN are zero, so beta multiplies nothing and
    only a smoothing residue is left. Averaging that in would dilute the term
    with a number that carries no information -- and 81% of the training split
    is clean.
    """
    targets_one = _with_box(_blank(batch=1))
    logits_one = torch.full_like(targets_one, -2.0)

    targets_two = torch.cat([targets_one, _blank(batch=1)], dim=0)
    logits_two = torch.cat([logits_one, torch.full_like(logits_one, -2.0)], dim=0)

    _, one = criterion(logits_one, targets_one)
    _, two = criterion(logits_two, targets_two)

    assert float(one["tversky"]) == pytest.approx(float(two["tversky"]), abs=1e-6)
    assert float(two["n_defective_images"]) == 1.0


def test_all_clean_batch_has_no_region_term(criterion):
    targets = _blank(batch=3)
    logits = torch.full_like(targets, -5.0)
    total, parts = criterion(logits, targets)

    assert float(parts["tversky"]) == 0.0
    assert float(parts["n_defective_images"]) == 0.0
    assert torch.isfinite(total)


def test_all_clean_batch_still_penalises_false_alarms(criterion):
    """With no region term left, focal alone must keep clean images honest."""
    targets = _blank(batch=2)
    quiet, _ = criterion(torch.full_like(targets, -CONFIDENT), targets)
    noisy, _ = criterion(torch.full_like(targets, CONFIDENT), targets)
    assert float(noisy) > float(quiet)


# --- the focal normalisation decision -------------------------------------


def test_focal_is_normalised_by_positives_not_by_pixels(criterion):
    """Averaging focal over pixels puts it ~100x below Tversky on this data.

    If that regression ever lands, the auxiliary region term silently becomes
    the dominant one.
    """
    targets = _with_box(_blank(size=100))
    logits = torch.full_like(targets, -2.0)
    _, parts = criterion(logits, targets)

    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    focal = cross_entropy * (1 - p_t) ** criterion.gamma

    by_positives = focal.sum() / targets.sum()
    by_pixels = focal.mean()

    assert float(parts["focal"]) == pytest.approx(float(by_positives), rel=1e-5)
    assert float(parts["focal"]) > 50 * float(by_pixels)


def test_gamma_zero_reduces_to_normalised_bce():
    targets = _with_box(_blank())
    logits = torch.full_like(targets, 0.3)
    _, parts = FocalTverskyLoss(gamma=0.0)(logits, targets)

    expected = (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
        / targets.sum()
    )
    assert float(parts["focal"]) == pytest.approx(float(expected), rel=1e-5)


def test_higher_gamma_suppresses_easy_pixels():
    # A batch that is almost entirely confident, correct background: exactly
    # the mass focal exists to discount.
    targets = _with_box(_blank(size=100))
    logits = torch.full_like(targets, -6.0)
    logits = logits + 12.0 * targets

    charges = [
        float(FocalTverskyLoss(gamma=g)(logits, targets)[1]["focal"])
        for g in (0.0, 1.0, 2.0, 5.0)
    ]
    assert charges == sorted(charges, reverse=True), charges


# --- the asymmetry decision -----------------------------------------------


def _segmentation_case(kind):
    """A 32x32 image with a 100-pixel defect, under- or over-segmented by 50."""
    targets = _with_box(_blank())
    logits = torch.full_like(targets, -CONFIDENT)
    if kind == "under":
        logits[0, 0, 0:5, 0:10] = CONFIDENT  # finds 50 of the 100
    else:
        logits[0, 0, 0:10, 0:10] = CONFIDENT  # finds all 100
        logits[0, 0, 10:15, 0:10] = CONFIDENT  # plus 50 that are not there
    return logits, targets


def test_tversky_punishes_misses_harder_than_dice_does():
    logits, targets = _segmentation_case("under")
    _, dice = FocalTverskyLoss(alpha=0.5, beta=0.5)(logits, targets)
    _, tversky = FocalTverskyLoss(alpha=0.3, beta=0.7)(logits, targets)
    assert float(tversky["tversky"]) > float(dice["tversky"])


def test_tversky_forgives_false_alarms_more_than_dice_does():
    logits, targets = _segmentation_case("over")
    _, dice = FocalTverskyLoss(alpha=0.5, beta=0.5)(logits, targets)
    _, tversky = FocalTverskyLoss(alpha=0.3, beta=0.7)(logits, targets)
    assert float(tversky["tversky"]) < float(dice["tversky"])


def test_reversing_the_dial_reverses_the_preference():
    under, targets_u = _segmentation_case("under")
    over, targets_o = _segmentation_case("over")
    recall_first = FocalTverskyLoss(alpha=0.3, beta=0.7)
    precision_first = FocalTverskyLoss(alpha=0.7, beta=0.3)

    assert float(recall_first(under, targets_u)[1]["tversky"]) > float(
        recall_first(over, targets_o)[1]["tversky"]
    )
    assert float(precision_first(under, targets_u)[1]["tversky"]) < float(
        precision_first(over, targets_o)[1]["tversky"]
    )


def test_alpha_beta_half_is_dice():
    logits, targets = _segmentation_case("under")
    criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, smooth=1e-6)
    _, parts = criterion(logits, targets)

    probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum()
    dice = 2 * intersection / (probability.sum() + targets.sum())
    assert float(parts["tversky"]) == pytest.approx(float(1 - dice), abs=1e-4)


# --- configuration --------------------------------------------------------


def test_zero_weight_gives_focal_alone():
    logits, targets = _segmentation_case("under")
    total, parts = FocalTverskyLoss(tversky_weight=0.0)(logits, targets)
    assert float(total) == pytest.approx(float(parts["focal"]), rel=1e-6)
    # The term is still reported, so the control arm can be compared without
    # it influencing training.
    assert float(parts["tversky"]) > 0


def test_weight_scales_the_region_term():
    logits, targets = _segmentation_case("under")
    light, parts = FocalTverskyLoss(tversky_weight=1.0)(logits, targets)
    heavy, _ = FocalTverskyLoss(tversky_weight=3.0)(logits, targets)
    assert float(heavy) == pytest.approx(
        float(light) + 2 * float(parts["tversky"]), rel=1e-5
    )


@pytest.mark.parametrize("logit", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_extreme_logits_stay_finite(criterion, logit):
    targets = _with_box(_blank())
    logits = torch.full_like(targets, logit).requires_grad_(True)
    total, parts = criterion(logits, targets)
    total.backward()

    assert torch.isfinite(total), logit
    assert all(torch.isfinite(v).all() for v in parts.values())
    assert torch.isfinite(logits.grad).all()


def test_components_are_detached(criterion):
    targets = _with_box(_blank())
    logits = torch.zeros_like(targets, requires_grad=True)
    _, parts = criterion(logits, targets)
    assert all(not v.requires_grad for v in parts.values())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 0.3, "beta": 0.3}, "must equal 1.0"),
        ({"gamma": -1.0}, "gamma must be non-negative"),
        ({"tversky_weight": -1.0}, "must be non-negative"),
        ({"smooth": 0.0}, "smooth must be positive"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FocalTverskyLoss(**kwargs)


def test_shape_mismatch_is_rejected(criterion):
    with pytest.raises(ValueError, match="must match"):
        criterion(_blank(size=32), _blank(size=16))
    with pytest.raises(ValueError, match=r"expected \(B, 1, H, W\)"):
        criterion(torch.zeros(4, 32), torch.zeros(4, 32))
