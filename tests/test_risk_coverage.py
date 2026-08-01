"""Tests for the risk-coverage curve.

The property that matters: routing the least-decidable parts first must never
lower recall, and a perfectly-ranked model reaches full recall by routing
exactly its mistakes.
"""

from __future__ import annotations

import numpy as np

from vinspect.eval.risk_coverage import points_at, risk_coverage_curve


def test_full_coverage_matches_the_raw_model():
    probabilities = np.array([0.9, 0.8, 0.1, 0.05, 0.6, 0.4])
    labels = np.array([1, 1, 0, 0, 1, 1])
    # Machine at 0.5: catches the 0.9, 0.8, 0.6; misses the 0.4.
    row = risk_coverage_curve(probabilities, labels)[0]
    assert row["coverage"] == 1.0
    assert row["recall"] == 0.75
    assert row["machine_missed"] == 1


def test_recall_never_falls_as_coverage_drops():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, 200)
    labels = (rng.uniform(0, 1, 200) < probabilities).astype(int)
    curve = risk_coverage_curve(probabilities, labels)
    recalls = [row["recall"] for row in curve]
    assert all(b >= a - 1e-12 for a, b in zip(recalls, recalls[1:]))


def test_the_fence_sitters_are_routed_first():
    probabilities = np.array([0.51, 0.02, 0.98, 0.49])
    labels = np.array([1, 0, 1, 1])
    curve = risk_coverage_curve(probabilities, labels)
    # After routing two, the 0.51 and 0.49 should be with the human.
    row = curve[2]
    assert row["routed"] == 2
    assert row["recall"] == 1.0, "routing the two fence-sitters catches the 0.49 miss"


def test_zero_coverage_catches_everything():
    probabilities = np.array([0.9, 0.1, 0.3])
    labels = np.array([1, 1, 1])
    assert risk_coverage_curve(probabilities, labels)[-1]["recall"] == 1.0


def test_false_alarms_are_counted_on_the_machine_share_only():
    probabilities = np.array([0.9, 0.6])
    labels = np.array([0, 0])
    curve = risk_coverage_curve(probabilities, labels)
    assert curve[0]["machine_false_alarms"] == 2
    # 0.6 is nearer the fence, so it is routed first.
    assert curve[1]["machine_false_alarms"] == 1
    assert curve[-1]["machine_false_alarms"] == 0


def test_points_at_picks_nearest_levels():
    probabilities = np.linspace(0.01, 0.99, 10)
    labels = (probabilities > 0.5).astype(int)
    curve = risk_coverage_curve(probabilities, labels)
    picked = points_at(curve, (1.0, 0.5))
    assert picked[0]["coverage"] == 1.0
    assert abs(picked[1]["coverage"] - 0.5) <= 0.1


def test_unsupported_scores_are_routed_before_fence_sitters():
    """The lesson of the three misses: a probability the calibrator
    extrapolated is not a verdict. An unsupported confident-clean must go to
    the human before a supported fence-sitter."""
    probabilities = np.array([0.005, 0.48, 0.97, 0.01])
    labels = np.array([1, 0, 1, 0])  # the unsupported 0.005 is a real defect
    supported = np.array([False, True, True, True])

    curve = risk_coverage_curve(probabilities, labels, supported=supported)
    # Routing exactly one part must pick the unsupported one and catch the miss.
    assert curve[0]["recall"] == 0.5
    assert curve[1]["recall"] == 1.0


def test_supported_default_changes_nothing():
    probabilities = np.array([0.9, 0.1, 0.55])
    labels = np.array([1, 0, 1])
    plain = risk_coverage_curve(probabilities, labels)
    explicit = risk_coverage_curve(
        probabilities, labels, supported=np.ones(3, dtype=bool)
    )
    assert plain == explicit


def test_empty_input_is_safe():
    assert risk_coverage_curve([], []) == []
