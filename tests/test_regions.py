"""Tests for region-level scoring.

The first test is the one that matters: the same number of hot pixels, once
scattered and once in a blob, must not score the same. Every per-pixel
statistic fails it, which is why none of them belongs here.
"""

from __future__ import annotations

import pytest
import torch

from vinspect.uncertainty.mc_dropout import MCPrediction
from vinspect.uncertainty.regions import (
    calibrate_min_area,
    extract_regions,
    ground_truth_region_areas,
    score_image,
)

SIZE = 64


def _prediction(hot, passes=8, hot_value=0.95, agreement=1.0):
    """Build an MCPrediction whose mean is ``hot_value`` on ``hot``.

    ``agreement`` is the share of passes in which the hot pixels stay high, so
    a value below 1 simulates a region the model only sometimes believes in.
    """
    mean = torch.full((SIZE, SIZE), 0.01)
    mean[hot] = hot_value

    stack = torch.full((passes, SIZE, SIZE), 0.01)
    believing = max(1, int(round(agreement * passes)))
    stack[:believing, hot[0], hot[1]] = hot_value
    return MCPrediction(mean=mean, std=stack.std(dim=0), passes=stack)


def _blob(rows=slice(10, 20), cols=slice(10, 20)):
    grid = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    grid[rows, cols] = True
    return grid.nonzero(as_tuple=True)


def _scattered(count=100, seed=0):
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randperm(SIZE * SIZE, generator=generator)[:count]
    return (flat // SIZE, flat % SIZE)


# --- the point of the whole module ----------------------------------------


def test_scattered_pixels_score_far_below_a_blob_without_any_gate():
    """With no minimum-area gate at all, the score alone must separate them.

    This is what forces the score to take the strongest single region rather
    than summing over all of them: a sum over regions is a sum over pixels in
    disguise, and would score these two identically.
    """
    blob = score_image(_prediction(_blob()))
    scattered = score_image(_prediction(_scattered(100)))
    assert blob.defect_score > 20 * scattered.defect_score


def test_a_blob_and_the_same_pixels_scattered_score_differently():
    """100 hot pixels in a 10x10 square against 100 spread at random.

    Identical under max, top-k mean and area-above-threshold. A defect is
    produced by a physical process and is contiguous, so these must differ.
    """
    blob = score_image(_prediction(_blob()), min_area=50)
    scattered = score_image(_prediction(_scattered(100)), min_area=50)

    assert blob.defect_score > 0
    assert scattered.defect_score == 0.0
    assert scattered.n_regions == 0


def test_per_pixel_statistics_would_not_have_separated_them():
    # Documents why the earlier design was wrong: the maximum and the mean of
    # the top 100 pixels are identical for both layouts.
    blob = _prediction(_blob())
    scattered = _prediction(_scattered(100))

    assert float(blob.mean.max()) == pytest.approx(float(scattered.mean.max()))
    top_blob = blob.mean.flatten().topk(100).values.mean()
    top_scattered = scattered.mean.flatten().topk(100).values.mean()
    assert float(top_blob) == pytest.approx(float(top_scattered), abs=1e-6)


# --- filtering ------------------------------------------------------------


def test_min_area_removes_small_regions():
    small = _blob(slice(10, 14), slice(10, 14))  # 16 px
    assert score_image(_prediction(small), min_area=10).n_regions == 1
    assert score_image(_prediction(small), min_area=50).n_regions == 0


def test_multi_part_defects_are_scored_by_their_largest_part():
    """The documented cost of taking the maximum rather than the sum.

    Summing over regions would raise the score here, but a sum over regions is
    a sum over pixels in disguise and cannot tell a blob from scattered noise.
    The second part is still recorded, it just does not inflate the verdict --
    which is fine, since the question is whether the part is defective, not how
    many defects it carries.
    """
    rows, cols = _blob(slice(10, 20), slice(10, 20))
    rows2, cols2 = _blob(slice(40, 48), slice(40, 48))
    both = (torch.cat([rows, rows2]), torch.cat([cols, cols2]))

    one = score_image(_prediction(_blob()), min_area=20)
    two = score_image(_prediction(both), min_area=20)

    assert two.n_regions == 2, "both parts must still be found and reported"
    assert len(two.regions) == 2
    assert two.defect_score == pytest.approx(one.defect_score), (
        "the score comes from the strongest region, not the sum"
    )


def test_regions_come_back_ordered_by_mass():
    rows, cols = _blob(slice(10, 20), slice(10, 20))
    rows2, cols2 = _blob(slice(40, 45), slice(40, 45))
    both = (torch.cat([rows, rows2]), torch.cat([cols, cols2]))

    regions = extract_regions(_prediction(both), min_area=10)
    assert [r.mass for r in regions] == sorted((r.mass for r in regions), reverse=True)
    assert regions[0].area > regions[1].area


def test_an_empty_prediction_scores_zero():
    empty = MCPrediction(
        mean=torch.zeros(SIZE, SIZE),
        std=torch.zeros(SIZE, SIZE),
        passes=torch.zeros(4, SIZE, SIZE),
    )
    score = score_image(empty)
    assert score.defect_score == 0.0
    assert not score.predicted_defective


# --- region-level uncertainty ---------------------------------------------


def test_persistence_separates_a_believed_region_from_a_flickering_one():
    """Existence doubt, which a per-pixel average cannot express.

    A blob present in every pass and one present in half carry very different
    claims, even where their mean maps look similar.
    """
    solid = score_image(_prediction(_blob(), agreement=1.0), min_area=20)
    flickering = score_image(_prediction(_blob(), agreement=0.5), min_area=20)

    assert solid.persistence == pytest.approx(1.0)
    assert flickering.persistence < 0.75
    assert flickering.persistence > 0.0


def test_a_small_certain_region_outweighs_a_large_uncertain_one():
    """The property plain mass gets backwards.

    Evidence adds in log-odds, not in probability: logit(0.99) is 11x
    logit(0.6), while 0.99 is only 1.65x 0.6. So a small, very confident region
    can outweigh a large, hesitant one -- which is what lets a genuine small
    defect survive without needing a minimum-area gate to protect it.
    """
    small_certain = score_image(  # 20x20 = 400 px, very sure
        _prediction(_blob(slice(10, 30), slice(10, 30)), hot_value=0.99)
    )
    large_unsure = score_image(  # 30x40 = 1200 px, hesitant
        _prediction(_blob(slice(10, 40), slice(10, 50)), hot_value=0.60)
    )

    assert large_unsure.largest_area > small_certain.largest_area
    assert large_unsure.mass_score > small_certain.mass_score, "mass prefers size"
    assert small_certain.defect_score > large_unsure.defect_score, (
        "log-odds should let confidence compensate for size"
    )


def test_no_minimum_area_is_needed_by_default():
    # The gate is off: a real but small region must still produce a score
    # rather than being deleted and the part called clean.
    small = score_image(_prediction(_blob(slice(10, 18), slice(10, 18))))
    assert small.n_regions == 1
    assert small.defect_score > 0


def test_region_features_are_populated():
    score = score_image(_prediction(_blob()), min_area=20)
    region = score.regions[0]
    assert region.area == 100
    assert region.mean_probability == pytest.approx(0.95, abs=0.01)
    assert region.mass == pytest.approx(95.0, abs=1.0)
    assert region.logodds == pytest.approx(100 * torch.logit(torch.tensor(0.95)), rel=0.02)
    assert 0.0 <= region.persistence <= 1.0
    assert region.area_cv >= 0.0
    assert region.interior_probability > 0.0


def test_interior_ignores_the_boundary_band():
    # A region whose edge is uncertain but whose middle is solid should read as
    # confident: only its outline is in doubt, not its existence.
    hot = _blob(slice(10, 30), slice(10, 30))
    prediction = _prediction(hot)
    mean = prediction.mean.clone()
    mean[10, 10:30] = 0.55  # a wobbly top edge
    mean[29, 10:30] = 0.55
    wobbly = MCPrediction(mean=mean, std=prediction.std, passes=prediction.passes)

    region = extract_regions(wobbly, min_area=20)[0]
    assert region.interior_probability > region.mean_probability


def test_thin_regions_still_report_an_interior():
    # Eroding a 2px-wide region would leave nothing; it must fall back rather
    # than crash or report zero.
    thin = _blob(slice(10, 40), slice(10, 12))
    region = extract_regions(_prediction(thin), min_area=10)[0]
    assert region.interior_probability > 0.0


# --- calibrating the threshold --------------------------------------------


def _mask_with(main_area, speck=True):
    mask = torch.zeros(1, SIZE, SIZE)
    side = int(main_area ** 0.5)
    mask[0, 0:side, 0:side] = 1.0
    if speck:
        mask[0, SIZE - 1, SIZE - 1] = 1.0  # a one-pixel annotation artefact
    return mask


def test_calibration_ignores_annotation_specks():
    """MVTec's masks are hand-drawn and carry stray single pixels.

    Taking every region, hazelnut's 5th percentile comes out at 1 px because
    49% of its ground-truth regions are specks. Calibrating on the largest
    region per image is immune to that.
    """
    masks = [_mask_with(area) for area in (400, 625, 900, 1225, 1600)]

    every = ground_truth_region_areas(masks)
    largest = ground_truth_region_areas(masks, largest_only=True)
    assert every.min() == 1.0, "the fixture should contain specks"
    assert largest.min() > 100, "largest-only must not see them"

    assert calibrate_min_area(masks, percentile=5.0) > 100


def test_calibration_is_a_percentile_of_real_defect_sizes():
    masks = [_mask_with(area, speck=False) for area in (400, 900, 1600, 2500)]
    low = calibrate_min_area(masks, percentile=0.0)
    high = calibrate_min_area(masks, percentile=50.0)
    assert low == 400
    assert low < high < 2500


def test_calibration_needs_defective_masks():
    with pytest.raises(ValueError, match="no defective masks"):
        calibrate_min_area([torch.zeros(1, SIZE, SIZE)])


# --- hysteresis -----------------------------------------------------------


def _chain_prediction(strong_value=0.7, bridge_value=0.4, passes=6):
    """Two strong fragments joined by a weak bridge -- a faint defect's shape."""
    mean = torch.full((SIZE, SIZE), 0.01)
    mean[20, 5:15] = strong_value    # fragment A
    mean[20, 15:25] = bridge_value   # the valley between them
    mean[20, 25:35] = strong_value   # fragment B
    stack = mean.unsqueeze(0).repeat(passes, 1, 1)
    return MCPrediction(mean=mean, std=stack.std(dim=0), passes=stack)


def test_hysteresis_reconnects_fragments_across_a_weak_bridge():
    shattered = extract_regions(_chain_prediction(), threshold=0.5)
    joined = extract_regions(_chain_prediction(), threshold=0.5, weak_threshold=0.33)

    assert len(shattered) == 2, "one hard threshold shatters the chain"
    assert len(joined) == 1, "the weak bridge must reconnect it"
    # The joined region's evidence is the SUM of both fragments' cores.
    assert joined[0].logodds == pytest.approx(
        shattered[0].logodds + shattered[1].logodds, rel=1e-4
    )


def test_weak_pixels_connect_but_do_not_add_evidence():
    """At p < 0.5 the log-odds are negative; letting bridges into the sum would
    make a defect's evidence shrink as more of it comes into faint view."""
    joined = extract_regions(_chain_prediction(), threshold=0.5, weak_threshold=0.33)[0]
    assert joined.extent > joined.area, "bridge is in the footprint"
    assert joined.area == 20, "core is only the strong pixels"
    assert joined.logodds > 0


def test_weak_whisper_without_a_seed_never_activates():
    mean = torch.full((SIZE, SIZE), 0.01)
    mean[10:20, 10:20] = 0.4  # a big whisper, no pixel above 0.5
    stack = mean.unsqueeze(0).repeat(4, 1, 1)
    prediction = MCPrediction(mean=mean, std=stack.std(dim=0), passes=stack)

    assert extract_regions(prediction, threshold=0.5, weak_threshold=0.33) == []
    score = score_image(prediction, weak_threshold=0.33)
    assert score.defect_score == 0.0


def test_hysteresis_changes_nothing_when_evidence_is_compact():
    prediction = _prediction(_blob())
    plain = score_image(prediction)
    hysteresis = score_image(prediction, weak_threshold=0.33)
    assert hysteresis.defect_score == pytest.approx(plain.defect_score, rel=1e-4)
    assert hysteresis.n_regions == plain.n_regions


def test_weak_threshold_must_sit_below_strong():
    with pytest.raises(ValueError, match="below threshold"):
        extract_regions(_chain_prediction(), threshold=0.5, weak_threshold=0.6)
