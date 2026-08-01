"""Tests for the calibrator.

The properties argued for in the design each get a test: monotone always, never
certain from small counts, ties share a step, and the whole thing actually
reduces Brier score on overconfident inputs -- which is its one job.
"""

from __future__ import annotations

import numpy as np
import pytest

from vinspect.eval.calibration import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


def _separable(n=60, seed=0):
    """Scores where higher genuinely means more defective, with overlap."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(100, 60, n).clip(0)
    defective = rng.normal(700, 300, n).clip(0)
    scores = np.concatenate([clean, defective])
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return scores, labels


def test_predictions_are_monotone_in_the_score():
    scores, labels = _separable()
    calibrator = IsotonicCalibrator().fit(scores, labels)
    grid = np.linspace(scores.min() - 50, scores.max() + 50, 500)
    predicted = calibrator.predict(grid)
    assert (np.diff(predicted) >= -1e-12).all(), "a higher score mapped lower"


def test_probabilities_track_reality():
    scores, labels = _separable(n=300)
    calibrator = IsotonicCalibrator().fit(scores, labels)
    predicted = calibrator.predict(scores)
    # Low-scoring images should be told low numbers, high-scoring high.
    assert predicted[labels == 0].mean() < 0.35
    assert predicted[labels == 1].mean() > 0.65


def test_never_certain_from_small_counts():
    """The raw staircase says 0% and 100% at its ends. Shrinkage must not."""
    scores, labels = _separable(n=25)
    calibrator = IsotonicCalibrator(pseudocount=1.0).fit(scores, labels)
    predicted = calibrator.predict(np.array([-1e6, 1e6]))
    assert predicted[0] > 0.0, "claimed impossible from a handful of images"
    assert predicted[1] < 1.0, "claimed certain from a handful of images"


def test_zero_pseudocount_reproduces_the_raw_staircase():
    scores, labels = _separable(n=40)
    calibrator = IsotonicCalibrator(pseudocount=0.0).fit(scores, labels)
    predicted = calibrator.predict(np.array([-1e6, 1e6]))
    assert predicted[0] == 0.0 and predicted[1] == 1.0


def test_steps_resting_on_less_data_move_further():
    """The Laplace property: shrinkage is proportional to ignorance."""
    # Bottom step: 30 clean images. Top step: 2 defective images.
    scores = np.concatenate([np.arange(30), [1000.0, 1001.0]])
    labels = np.array([0] * 30 + [1, 1])
    calibrator = IsotonicCalibrator(pseudocount=1.0).fit(scores, labels)

    bottom, top = calibrator.steps[0], calibrator.steps[-1]
    assert abs(bottom.probability - bottom.raw_rate) < abs(
        top.probability - top.raw_rate
    )


def test_tied_scores_share_a_probability():
    scores = np.array([5.0, 5.0, 5.0, 5.0, 50.0, 50.0])
    labels = np.array([0, 1, 0, 0, 1, 1])
    calibrator = IsotonicCalibrator().fit(scores, labels)
    low, high = calibrator.predict(np.array([5.0, 50.0]))
    assert 0.0 < low < high < 1.0


def test_monotone_survives_adversarial_shrinkage():
    """Shrinkage toward a high base rate can swap steps when the lower one
    rests on fewer images; the final pass must re-impose order."""
    # Mostly defective overall (high base rate), a thin low block.
    scores = np.concatenate([[10.0, 11.0], np.arange(100, 200)])
    labels = np.concatenate([[0, 1], np.ones(100)]).astype(int)
    labels[50] = 0  # keep both outcomes above
    calibrator = IsotonicCalibrator(pseudocount=5.0).fit(scores, labels)
    values = [step.probability for step in calibrator.steps]
    assert values == sorted(values)


def test_calibration_reduces_brier_of_an_overconfident_model():
    """The one job: an overconfident score squashed through sigmoid should get
    honest -- and measurably so -- after calibration on held-out outcomes."""
    scores, labels = _separable(n=200, seed=1)
    holdout_scores, holdout_labels = _separable(n=200, seed=2)

    overconfident = 1.0 / (1.0 + np.exp(-(holdout_scores - 400.0) / 20.0))
    calibrator = IsotonicCalibrator().fit(scores, labels)
    calibrated = calibrator.predict(holdout_scores)

    assert brier_score(calibrated, holdout_labels) < brier_score(
        overconfident, holdout_labels
    )


def test_fit_and_predict_are_deterministic():
    scores, labels = _separable()
    first = IsotonicCalibrator().fit(scores, labels).predict(scores)
    second = IsotonicCalibrator().fit(scores, labels).predict(scores)
    assert np.array_equal(first, second)


def test_summary_reports_the_evidence():
    scores, labels = _separable(n=30)
    summary = IsotonicCalibrator().fit(scores, labels).summary()
    assert summary["n_steps"] >= 2
    assert sum(step["count"] for step in summary["steps"]) == 60
    assert 0.0 < summary["base_rate"] < 1.0


@pytest.mark.parametrize(
    "scores, labels, match",
    [
        ([1.0], [1], "fewer than 2"),
        ([1.0, 2.0], [0, 2], "labels must be 0 or 1"),
        ([1.0, 2.0], [1, 1], "both outcomes"),
        ([1.0, 2.0, 3.0], [0, 1], "equal-length"),
    ],
)
def test_bad_inputs_are_rejected(scores, labels, match):
    with pytest.raises(ValueError, match=match):
        IsotonicCalibrator().fit(np.array(scores), np.array(labels))


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        IsotonicCalibrator().predict(np.array([1.0]))


def test_negative_pseudocount_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        IsotonicCalibrator(pseudocount=-1.0)


# --- reliability ----------------------------------------------------------


def test_reliability_bins_are_equal_count():
    probabilities = np.concatenate([np.full(70, 0.05), np.linspace(0.5, 1.0, 10)])
    labels = (probabilities > 0.5).astype(int)
    rows = reliability_bins(probabilities, labels, n_bins=8)
    counts = [row["count"] for row in rows]
    assert max(counts) - min(counts) <= 1, "bins should hold equal evidence"


def test_reliability_of_perfect_predictions_sits_on_the_diagonal():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, 4000)
    labels = (rng.uniform(0, 1, 4000) < probabilities).astype(int)
    for row in reliability_bins(probabilities, labels, n_bins=5):
        assert abs(row["claimed"] - row["observed"]) < 0.08


def test_ece_zero_for_perfect_and_large_for_inverted():
    probabilities = np.array([0.1] * 50 + [0.9] * 50)
    labels = np.array([0] * 50 + [1] * 50)
    good = expected_calibration_error(probabilities, labels)
    bad = expected_calibration_error(1.0 - probabilities, labels)
    assert good < 0.11
    assert bad > 0.7


def test_empty_inputs_are_safe():
    assert reliability_bins([], []) == []
    assert expected_calibration_error([], []) == 0.0
