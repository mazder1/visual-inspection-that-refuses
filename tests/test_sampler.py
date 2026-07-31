"""Tests for batch composition.

The reason this exists at all: Tversky is computed only over images with a
non-empty mask, so a batch of entirely clean images contributes nothing through
the region term. At batch 8 with an 18.6% defective rate, 19% of unconstrained
batches would be in that state.
"""

from __future__ import annotations

import pytest

from vinspect.train.sampler import StratifiedBatchSampler


def _labels(n_defective=20, n_clean=100):
    return [1] * n_defective + [0] * n_clean


def test_every_batch_contains_defective_images():
    labels = _labels()
    sampler = StratifiedBatchSampler(labels, batch_size=8)
    for batch in sampler:
        defective = sum(labels[i] for i in batch)
        assert defective == sampler.defective_per_batch
        assert defective > 0
        assert len(batch) == 8


def test_defective_count_follows_the_natural_rate():
    # 20 of 120 is 16.7%; ceil(8 * 0.167) = 2.
    sampler = StratifiedBatchSampler(_labels(20, 100), batch_size=8)
    assert sampler.defective_per_batch == 2
    assert sampler.realised_rate == pytest.approx(0.25)
    assert sampler.natural_rate == pytest.approx(20 / 120)


def test_realised_rate_is_reported_for_the_log():
    """The oversampling is a real distortion of the prior module 03 calibrates
    against, so it has to be visible rather than implicit."""
    sampler = StratifiedBatchSampler(_labels(), batch_size=8)
    assert sampler.realised_rate != sampler.natural_rate


def test_every_clean_image_is_seen_once_per_epoch():
    labels = _labels(20, 96)
    sampler = StratifiedBatchSampler(labels, batch_size=8)
    seen = [i for batch in sampler for i in batch if labels[i] == 0]
    assert sorted(seen) == [i for i, label in enumerate(labels) if label == 0]


def test_defective_images_are_recycled_not_exhausted():
    # Far fewer defective than clean, so they must repeat within an epoch.
    labels = _labels(3, 100)
    sampler = StratifiedBatchSampler(labels, batch_size=8)
    drawn = [i for batch in sampler for i in batch if labels[i] == 1]
    assert len(drawn) > 3
    assert set(drawn) == {0, 1, 2}


def test_epochs_differ_but_are_reproducible():
    sampler = StratifiedBatchSampler(_labels(), batch_size=8, seed=0)
    sampler.set_epoch(0)
    first = [list(b) for b in sampler]
    sampler.set_epoch(0)
    again = [list(b) for b in sampler]
    sampler.set_epoch(1)
    second = [list(b) for b in sampler]

    assert first == again, "same seed and epoch must give the same batches"
    assert first != second, "a new epoch must reshuffle"


def test_seed_changes_the_order():
    a = [list(b) for b in StratifiedBatchSampler(_labels(), batch_size=8, seed=0)]
    b = [list(b) for b in StratifiedBatchSampler(_labels(), batch_size=8, seed=1)]
    assert a != b


def test_length_matches_what_is_yielded():
    sampler = StratifiedBatchSampler(_labels(20, 100), batch_size=8)
    assert len(list(sampler)) == len(sampler)


def test_a_split_with_no_defective_images_is_rejected():
    with pytest.raises(ValueError, match="no defective images"):
        StratifiedBatchSampler([0] * 50, batch_size=8)


def test_a_split_with_no_clean_images_is_rejected():
    with pytest.raises(ValueError, match="no clean images"):
        StratifiedBatchSampler([1] * 50, batch_size=8)


def test_defective_count_must_leave_room_for_clean():
    with pytest.raises(ValueError, match="must leave room"):
        StratifiedBatchSampler(_labels(), batch_size=4, defective_per_batch=4)
