"""Tests for derived component grouping.

The interesting fixture is ``planted_root``, where component identity is known
by construction: the same synthetic part appears at several rotations. A
grouping method that works must recover exactly those components. A method that
merges everything, or nothing, fails visibly here rather than silently on real
data.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from vinspect.data.grouping import (
    DEFAULT_INLIER_THRESHOLD,
    calibrate_thresholds,
    geometric_inliers,
    load_scores,
    save_scores,
    group_by_hash,
    group_by_keypoints,
    group_records,
    inlier_distribution,
    keypoint_signature,
    signature_records,
)
from vinspect.data.mvtec import index_mvtec


def _clusters(grouping):
    """group id -> set of record keys."""
    out = defaultdict(set)
    for key, group_id in grouping.groups.items():
        out[group_id].add(key)
    return {frozenset(v) for v in out.values()}


def _truth_clusters(planted_groups):
    out = defaultdict(set)
    for key, component in planted_groups.items():
        out[component].add(key)
    return {frozenset(v) for v in out.values()}


def test_keypoint_grouping_recovers_planted_components(planted_root, planted_groups):
    records = index_mvtec(planted_root)
    grouping = group_by_keypoints(records, threshold=DEFAULT_INLIER_THRESHOLD)

    planted = _truth_clusters(planted_groups)
    recovered = {c for c in _clusters(grouping) if len(c) > 1}
    expected = {c for c in planted if len(c) > 1}
    assert recovered == expected


def test_rotated_copies_score_far_above_unrelated_parts(planted_root, planted_groups):
    # The threshold is only safe if there is a gap. This asserts the gap exists
    # on data where the answer is known.
    records = index_mvtec(planted_root)
    signatures = signature_records(records)
    scored = inlier_distribution(records, signatures=signatures)

    same, different = [], []
    for pairs in scored.values():
        for n, a, b in pairs:
            if a in planted_groups and b in planted_groups:
                bucket = same if planted_groups[a] == planted_groups[b] else different
                bucket.append(n)

    assert same and different
    assert min(same) > max(different), (
        f"no separating gap: same-part min {min(same)}, "
        f"different-part max {max(different)}"
    )


def test_grouping_never_crosses_categories(planted_root):
    records = index_mvtec(planted_root)
    grouping = group_by_keypoints(records, threshold=DEFAULT_INLIER_THRESHOLD)
    by_key = {r.key: r for r in records}

    members = defaultdict(set)
    for key, group_id in grouping.groups.items():
        members[group_id].add(by_key[key].category)
    assert all(len(c) == 1 for c in members.values())


def test_every_record_lands_in_exactly_one_group(planted_root):
    records = index_mvtec(planted_root)
    grouping = group_by_keypoints(records)
    assert set(grouping.groups) == {r.key for r in records}
    assert sum(grouping.sizes.values()) == len(records)


def test_grouping_is_deterministic(planted_root):
    records = index_mvtec(planted_root)
    first = group_by_keypoints(records, threshold=DEFAULT_INLIER_THRESHOLD)
    second = group_by_keypoints(records, threshold=DEFAULT_INLIER_THRESHOLD)
    assert first.groups == second.groups


def test_raising_the_threshold_never_merges_more(planted_root):
    records = index_mvtec(planted_root)
    scores = inlier_distribution(records)
    counts = [
        group_by_keypoints(records, threshold=t, scores=scores).n_groups
        for t in (5, 10, 20, 40, 200)
    ]
    assert counts == sorted(counts), f"group count not monotonic in threshold: {counts}"


def test_self_match_is_strong(planted_root):
    records = index_mvtec(planted_root)
    signature = keypoint_signature(records[0].image_path)
    assert geometric_inliers(signature, signature) > DEFAULT_INLIER_THRESHOLD


def test_featureless_images_do_not_match(fake_mvtec_root):
    # The flat-colour fixture has no keypoints at all. The method must return
    # zero rather than raising or matching everything to everything.
    records = index_mvtec(fake_mvtec_root)
    signatures = signature_records(records)
    grouping = group_by_keypoints(records, signatures=signatures)
    assert grouping.n_grouped_images == 0


def test_hash_grouping_still_runs_as_the_documented_control(planted_root):
    # Kept working on purpose: the README reports it as the method that fails.
    records = index_mvtec(planted_root)
    grouping = group_by_hash(records, threshold=6)
    assert set(grouping.groups) == {r.key for r in records}
    assert "dhash" in grouping.method


def test_dispatch_and_bad_method(planted_root):
    records = index_mvtec(planted_root)
    assert "ORB" in group_records(records, method="keypoints").method
    with pytest.raises(ValueError, match="unknown grouping method"):
        group_records(records, method="vibes")


def test_threshold_must_be_positive(planted_root):
    records = index_mvtec(planted_root)
    with pytest.raises(ValueError, match="at least 1"):
        group_by_keypoints(records, threshold=0)


def test_calibration_respects_the_group_size_cap(planted_root):
    records = index_mvtec(planted_root)
    scores = inlier_distribution(records)
    # The largest planted component has three members, so a cap below that is
    # unsatisfiable by construction and is covered separately.
    for cap in (3, 5, 12):
        thresholds = calibrate_thresholds(records, scores, max_group_size=cap)
        grouping = group_by_keypoints(records, scores=scores, per_category=thresholds)
        assert grouping.largest <= cap


def test_calibration_picks_the_smallest_threshold_that_fits(planted_root):
    # Erring toward merging is the safe direction: a missed group leaks across
    # the split, an over-merged one only costs training data.
    records = index_mvtec(planted_root)
    scores = inlier_distribution(records)
    candidates = list(range(5, 200, 5))
    thresholds = calibrate_thresholds(
        records, scores, max_group_size=3, candidates=candidates
    )
    for category, chosen in thresholds.items():
        smaller = [c for c in candidates if c < chosen]
        for candidate in smaller:
            grouping = group_by_keypoints(
                records, scores=scores, per_category={**thresholds, category: candidate}
            )
            assert grouping.largest > 3, (
                f"{category} threshold {chosen} was not minimal; "
                f"{candidate} also satisfies the cap"
            )


def test_calibration_reports_a_category_it_cannot_separate(planted_root):
    records = index_mvtec(planted_root)
    scores = inlier_distribution(records)
    with pytest.raises(ValueError, match="not separating"):
        calibrate_thresholds(records, scores, max_group_size=1, candidates=[1, 2])


def test_per_category_thresholds_must_cover_every_category(planted_root):
    records = index_mvtec(planted_root)
    scores = inlier_distribution(records)
    with pytest.raises(ValueError, match="no threshold given"):
        group_by_keypoints(records, scores=scores, per_category={"widget": 20})


def test_score_cache_round_trips(planted_root, tmp_path):
    records = index_mvtec(planted_root)
    scores = inlier_distribution(records)
    path = save_scores(tmp_path / "scores.json", scores, min_score=5)

    reloaded, floor = load_scores(path)
    assert floor == 5
    # Grouping above the cache floor must be identical either way.
    from_live = group_by_keypoints(records, threshold=20, scores=scores)
    from_cache = group_by_keypoints(records, threshold=20, scores=reloaded)
    assert from_live.groups == from_cache.groups
