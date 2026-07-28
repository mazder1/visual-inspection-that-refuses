"""Tests for the split generator.

Grouping is stubbed rather than computed, so these exercise the split logic in
isolation: whether whole groups stay together, whether the scarce defective
images reach all three splits, and whether the artifact refuses to load once it
has been edited.
"""

from __future__ import annotations

import json

import pytest

from vinspect.data.grouping import GroupingResult
from vinspect.data.mvtec import index_mvtec
from vinspect.data.splits import (
    SPLITS,
    SplitError,
    assign_grouped,
    assign_random,
    build_artifact,
    digest,
    format_split_report,
    load_split,
    records_for_split,
    verify_no_leakage,
    write_split,
)


def _grouping(records, group_size=1):
    """Stub grouping: consecutive records within a category share a group."""
    groups, sizes, seen = {}, {}, {}
    for record in sorted(records, key=lambda r: r.key):
        index = seen.setdefault(record.category, 0)
        group_id = f"{record.category}:g{index // group_size:04d}"
        groups[record.key] = group_id
        sizes[group_id] = sizes.get(group_id, 0) + 1
        seen[record.category] = index + 1
    return GroupingResult(
        groups=groups, sizes=sizes, method="stub", threshold=1, params={}
    )


@pytest.fixture
def records(fake_mvtec_root):
    return index_mvtec(fake_mvtec_root)


def test_every_record_is_assigned(records):
    assignments = assign_grouped(records, _grouping(records, 2))
    assert set(assignments) == {r.key for r in records}
    assert set(assignments.values()) <= set(SPLITS)


@pytest.mark.parametrize("group_size", [1, 2, 3])
def test_grouped_split_never_straddles(records, group_size):
    grouping = _grouping(records, group_size)
    report = verify_no_leakage(assign_grouped(records, grouping), grouping)
    assert report["clean"], report["straddling"]
    assert report["n_straddling"] == 0


def test_random_split_does_straddle_when_groups_are_large(records):
    # The contrast the whole module exists to measure. If this ever stops
    # holding, the random baseline is not a baseline.
    grouping = _grouping(records, group_size=4)
    report = verify_no_leakage(assign_random(records, seed=0), grouping)
    assert report["n_straddling"] > 0


def test_defectives_reach_every_split(records):
    assignments = assign_grouped(records, _grouping(records, 1))
    by_key = {r.key: r for r in records}
    for split in SPLITS:
        defective = sum(
            by_key[k].label for k, s in assignments.items() if s == split
        )
        assert defective > 0, f"{split} got no defective images"


def test_ratios_are_approximately_honoured(records):
    assignments = assign_grouped(records, _grouping(records, 1))
    for split, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        share = sum(1 for s in assignments.values() if s == split) / len(records)
        assert abs(share - target) < 0.10, f"{split} got {share:.2f}, wanted {target}"


def test_grouped_assignment_is_deterministic(records):
    grouping = _grouping(records, 2)
    assert assign_grouped(records, grouping) == assign_grouped(records, grouping)


def test_random_assignment_depends_on_seed(records):
    assert assign_random(records, seed=0) != assign_random(records, seed=1)
    assert assign_random(records, seed=0) == assign_random(records, seed=0)


@pytest.mark.parametrize(
    "ratios, match",
    [
        ({"train": 0.6, "val": 0.2}, "exactly"),
        ({"train": 0.6, "val": 0.2, "test": 0.3}, "sum to 1.0"),
        ({"train": 1.0, "val": 0.0, "test": 0.0}, "positive"),
    ],
)
def test_bad_ratios_are_rejected(records, ratios, match):
    with pytest.raises(SplitError, match=match):
        assign_grouped(records, _grouping(records), ratios)


def test_artifact_round_trips(records, tmp_path):
    grouping = _grouping(records, 2)
    assignments = assign_grouped(records, grouping)
    payload = build_artifact(
        records, assignments, grouping, "grouped", dict(zip(SPLITS, (0.6, 0.2, 0.2))), 0, tmp_path
    )
    path = tmp_path / "grouped.json"
    checksum = write_split(path, payload)

    loaded = load_split(path)
    assert loaded == payload
    assert path.with_suffix(".sha256").read_text().startswith(checksum)

    recovered = {
        split: {r.key for r in records_for_split(records, loaded, split)}
        for split in SPLITS
    }
    for split in SPLITS:
        assert recovered[split] == {k for k, s in assignments.items() if s == split}


def test_digest_changes_when_membership_changes(records, tmp_path):
    grouping = _grouping(records, 2)
    ratios = dict(zip(SPLITS, (0.6, 0.2, 0.2)))
    first = build_artifact(
        records, assign_grouped(records, grouping), grouping, "grouped", ratios, 0, tmp_path
    )
    second = build_artifact(
        records, assign_random(records, seed=0), grouping, "random", ratios, 0, tmp_path
    )
    assert digest(first) != digest(second)


def test_digest_is_stable_across_calls(records, tmp_path):
    grouping = _grouping(records, 2)
    payload = build_artifact(
        records, assign_grouped(records, grouping), grouping, "grouped",
        dict(zip(SPLITS, (0.6, 0.2, 0.2))), 0, tmp_path,
    )
    assert digest(payload) == digest(json.loads(json.dumps(payload)))


def test_edited_artifact_is_refused(records, tmp_path):
    grouping = _grouping(records, 2)
    payload = build_artifact(
        records, assign_grouped(records, grouping), grouping, "grouped",
        dict(zip(SPLITS, (0.6, 0.2, 0.2))), 0, tmp_path,
    )
    path = tmp_path / "grouped.json"
    write_split(path, payload)

    document = json.loads(path.read_text())
    victim = sorted(document["payload"]["assignments"])[0]
    document["payload"]["assignments"][victim]["split"] = "test"
    path.write_text(json.dumps(document))

    with pytest.raises(SplitError, match="digest mismatch"):
        load_split(path)


def test_unassigned_record_is_an_error(records, tmp_path):
    grouping = _grouping(records, 2)
    assignments = assign_grouped(records, grouping)
    assignments.pop(sorted(assignments)[0])
    with pytest.raises(SplitError, match="never assigned"):
        build_artifact(
            records, assignments, grouping, "grouped",
            dict(zip(SPLITS, (0.6, 0.2, 0.2))), 0, tmp_path,
        )


def test_split_and_index_must_agree(records, tmp_path):
    grouping = _grouping(records, 2)
    payload = build_artifact(
        records, assign_grouped(records, grouping), grouping, "grouped",
        dict(zip(SPLITS, (0.6, 0.2, 0.2))), 0, tmp_path,
    )
    payload["assignments"].pop(sorted(payload["assignments"])[0])
    with pytest.raises(SplitError, match="absent from the split artifact"):
        records_for_split(records, payload, "train")


def test_unknown_kind_and_split_are_rejected(records, tmp_path):
    grouping = _grouping(records, 2)
    ratios = dict(zip(SPLITS, (0.6, 0.2, 0.2)))
    with pytest.raises(SplitError, match="unknown split kind"):
        build_artifact(
            records, assign_grouped(records, grouping), grouping, "sideways", ratios, 0, tmp_path
        )
    payload = build_artifact(
        records, assign_grouped(records, grouping), grouping, "grouped", ratios, 0, tmp_path
    )
    with pytest.raises(SplitError, match="unknown split"):
        records_for_split(records, payload, "holdout")


def test_report_renders(records):
    text = format_split_report(records, assign_grouped(records, _grouping(records)))
    assert "bottle" in text and "train" in text and "ALL" in text
