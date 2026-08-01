"""Tests for the metric accumulator.

The property that matters most: clean images must not enter the Dice average.
On an empty target every overlap metric is degenerate, and a model predicting
nothing scores a perfect 1.0 -- so a selection metric that includes them would
crown exactly the model this project exists to avoid shipping.
"""

from __future__ import annotations

import pytest
import torch

from vinspect.eval.metrics import SegmentationMetrics, dice_and_iou

BIG = 20.0  # a logit large enough to be a confident 1 after sigmoid


def _mask(size=16, rows=slice(0, 8)):
    mask = torch.zeros(1, 1, size, size)
    mask[0, 0, rows, :] = 1.0
    return mask


def test_perfect_overlap_scores_one():
    target = _mask()
    dice, iou = dice_and_iou(target.clone(), target)
    assert float(dice) == pytest.approx(1.0, abs=1e-5)
    assert float(iou) == pytest.approx(1.0, abs=1e-5)


def test_half_overlap_matches_hand_computation():
    target = _mask(rows=slice(0, 8))
    prediction = _mask(rows=slice(4, 12))  # half the pixels overlap
    dice, iou = dice_and_iou(prediction, target)
    # |A|=|B|=128, intersection 64: Dice = 2*64/256 = 0.5, IoU = 64/192 = 1/3
    assert float(dice) == pytest.approx(0.5, abs=1e-4)
    assert float(iou) == pytest.approx(1 / 3, abs=1e-4)


def test_no_overlap_scores_zero():
    dice, iou = dice_and_iou(_mask(rows=slice(0, 4)), _mask(rows=slice(8, 12)))
    assert float(dice) == pytest.approx(0.0, abs=1e-5)
    assert float(iou) == pytest.approx(0.0, abs=1e-5)


def test_clean_images_are_excluded_from_mean_dice():
    """A model that predicts nothing must not score well."""
    metrics = SegmentationMetrics()
    empty = torch.zeros(1, 1, 16, 16)

    # One clean image, predicted perfectly (empty), then one defective image
    # the model completely misses.
    metrics.update(torch.full_like(empty, -BIG), empty, ["bottle"], ["good"])
    metrics.update(torch.full_like(empty, -BIG), _mask(), ["bottle"], ["broken"])

    assert metrics.mean_dice() == pytest.approx(0.0, abs=1e-5), (
        "the clean image's degenerate perfect score leaked into the average"
    )
    assert len(metrics.defective) == 1
    assert len(metrics.clean) == 1


def test_false_alarm_area_counts_only_clean_images():
    metrics = SegmentationMetrics()
    empty = torch.zeros(1, 1, 16, 16)
    # Clean image, model calls the top half defect: 50% of pixels.
    metrics.update(torch.where(_mask() > 0, BIG, -BIG), empty, ["carpet"], ["good"])
    assert metrics.false_alarm_area() == pytest.approx(0.5, abs=1e-3)
    assert metrics.clean_images_touched() == pytest.approx(1.0)


def test_untouched_clean_images_report_zero():
    metrics = SegmentationMetrics()
    empty = torch.zeros(2, 1, 16, 16)
    metrics.update(
        torch.full_like(empty, -BIG), empty, ["carpet", "carpet"], ["good", "good"]
    )
    assert metrics.false_alarm_area() == 0.0
    assert metrics.clean_images_touched() == 0.0


def test_results_group_by_category_and_defect_type():
    metrics = SegmentationMetrics()
    target = _mask()
    perfect = torch.where(target > 0, BIG, -BIG)
    for category, defect in (("bottle", "crack"), ("bottle", "hole"), ("carpet", "cut")):
        metrics.update(perfect, target, [category], [defect])

    by_type = metrics.by_defect_type()
    assert set(by_type) == {("bottle", "crack"), ("bottle", "hole"), ("carpet", "cut")}
    assert set(metrics.by_category()) == {"bottle", "carpet"}
    assert metrics.by_category()["bottle"]["n"] == 2


def test_threshold_is_respected():
    target = _mask()
    logits = torch.where(target > 0, 0.4, -5.0)  # sigmoid(0.4) = 0.599
    strict = SegmentationMetrics(threshold=0.7)
    loose = SegmentationMetrics(threshold=0.5)
    strict.update(logits, target, ["bottle"], ["crack"])
    loose.update(logits, target, ["bottle"], ["crack"])

    assert strict.mean_dice() < loose.mean_dice()


def _predict(rows):
    """Confident prediction covering ``rows``, so overlap varies per image."""
    return torch.where(_mask(rows=rows) > 0, BIG, -BIG)


def test_bootstrap_brackets_the_point_estimate():
    metrics = SegmentationMetrics()
    target = _mask(rows=slice(0, 8))
    # Genuinely different overlaps, so the scores have a spread to resample.
    for rows in (slice(0, 8), slice(2, 10), slice(4, 12), slice(0, 6), slice(6, 14)):
        metrics.update(_predict(rows), target, ["bottle"], ["crack"])

    interval = metrics.bootstrap("iou", "bottle", resamples=2000)
    assert interval["n"] == 5
    assert interval["low"] <= interval["point"] <= interval["high"]
    assert interval["high"] > interval["low"]


def test_bootstrap_is_deterministic_and_ignores_clean_images():
    metrics = SegmentationMetrics()
    target = _mask()
    metrics.update(torch.where(target > 0, BIG, -BIG), target, ["bottle"], ["crack"])
    metrics.update(
        torch.full_like(target, -BIG), torch.zeros_like(target), ["bottle"], ["good"]
    )

    first = metrics.bootstrap("iou", "bottle", resamples=500, seed=3)
    again = metrics.bootstrap("iou", "bottle", resamples=500, seed=3)
    assert first == again
    assert first["n"] == 1, "the clean image leaked into the bootstrap sample"


def test_bootstrap_narrows_as_images_are_added():
    """More images, tighter interval. This is why the delta table needs one:
    at 12 to 18 defective test images the interval is wide."""
    target = _mask(rows=slice(0, 8))
    widths = []
    for count in (4, 40):
        metrics = SegmentationMetrics()
        for i in range(count):
            rows = slice(0, 8) if i % 2 else slice(4, 12)
            metrics.update(_predict(rows), target, ["bottle"], ["crack"])
        interval = metrics.bootstrap("iou", "bottle", resamples=3000)
        widths.append(interval["high"] - interval["low"])
    assert widths[1] < widths[0]


def test_bootstrap_on_an_empty_category_is_safe():
    assert SegmentationMetrics().bootstrap("iou", "bottle")["n"] == 0


def test_summary_and_report_render():
    metrics = SegmentationMetrics()
    target = _mask()
    metrics.update(torch.where(target > 0, BIG, -BIG), target, ["bottle"], ["crack"])
    metrics.update(
        torch.full_like(target, -BIG), torch.zeros_like(target), ["bottle"], ["good"]
    )

    summary = metrics.summary()
    assert summary["n_defective"] == 1 and summary["n_clean"] == 1
    assert "bottle" in metrics.format_report()
