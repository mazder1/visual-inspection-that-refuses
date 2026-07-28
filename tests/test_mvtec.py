"""Tests for the MVTec AD index and loader.

These run against the synthetic tree in conftest, so they check the loader's
contract rather than the real dataset's contents. The contract worth pinning is
the part a later bug would hide: every defective image has a mask, every clean
image has an empty one, masks stay binary through resizing, and the index order
is stable so a frozen split file stays valid.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from vinspect.data import (
    MVTecDataset,
    MVTecLayoutError,
    format_inventory,
    index_mvtec,
    summarise,
)
from vinspect.data.mvtec import discover_categories


def test_discovers_categories_by_structure(fake_mvtec_root):
    assert discover_categories(fake_mvtec_root) == ["bottle", "grid"]


def test_index_counts_match_the_tree(fake_mvtec_root, fake_mvtec_counts):
    records = index_mvtec(fake_mvtec_root)
    stats = summarise(records)
    expected_total = sum(c["total"] for c in fake_mvtec_counts.values())

    assert stats["n_images"] == expected_total
    assert stats["n_clean"] == sum(c["clean"] for c in fake_mvtec_counts.values())
    assert stats["n_defective"] == sum(
        c["defective"] for c in fake_mvtec_counts.values()
    )
    for category, expected in fake_mvtec_counts.items():
        row = stats["by_category"][category]
        assert row["total"] == expected["total"]
        assert row["n_defect_types"] == expected["n_defect_types"]


def test_every_defective_record_has_a_mask_and_clean_records_have_none(fake_mvtec_root):
    for record in index_mvtec(fake_mvtec_root):
        if record.label == 1:
            assert record.mask_path is not None and record.mask_path.is_file()
        else:
            assert record.mask_path is None


def test_shipped_train_directory_is_clean_only(fake_mvtec_root):
    # If this ever fails on the real data, the split generator's assumptions
    # about where defective images live are wrong.
    train = [r for r in index_mvtec(fake_mvtec_root) if r.origin == "train"]
    assert train and all(r.label == 0 for r in train)


def test_index_order_is_stable(fake_mvtec_root):
    first = [r.key for r in index_mvtec(fake_mvtec_root)]
    second = [r.key for r in index_mvtec(fake_mvtec_root)]
    assert first == second
    assert len(set(first)) == len(first), "record keys must be unique"


def test_category_filter_and_unknown_category(fake_mvtec_root):
    records = index_mvtec(fake_mvtec_root, categories=["bottle"])
    assert {r.category for r in records} == {"bottle"}
    with pytest.raises(MVTecLayoutError, match="not found"):
        index_mvtec(fake_mvtec_root, categories=["bottle", "screw"])


def test_missing_mask_is_an_error_not_a_skip(fake_mvtec_root, tmp_path):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(fake_mvtec_root, broken)
    (broken / "bottle" / "ground_truth" / "broken_large" / "000_mask.png").unlink()
    with pytest.raises(MVTecLayoutError, match="no mask"):
        index_mvtec(broken)


def test_missing_root_is_an_error(tmp_path):
    with pytest.raises(MVTecLayoutError, match="does not exist"):
        index_mvtec(tmp_path / "nope")


@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_sample_shapes_and_ranges(fake_mvtec_root, image_size):
    dataset = MVTecDataset(index_mvtec(fake_mvtec_root), image_size=image_size)
    sample = dataset[0]

    assert sample["image"].shape == (3, image_size, image_size)
    assert sample["mask"].shape == (1, image_size, image_size)
    assert sample["image"].dtype == torch.float32
    assert 0.0 <= float(sample["image"].min()) and float(sample["image"].max()) <= 1.0


def test_grayscale_category_is_converted_to_three_channels(fake_mvtec_root):
    records = index_mvtec(fake_mvtec_root, categories=["grid"])
    sample = MVTecDataset(records, image_size=32)[0]
    assert sample["image"].shape[0] == 3
    # A grey source must stay grey, not pick up a channel ordering bug.
    channels = sample["image"].mean(dim=(1, 2))
    assert torch.allclose(channels, channels[0].expand(3), atol=1e-6)


@pytest.mark.parametrize("image_size", [16, 32, 100])
def test_masks_stay_binary_when_resized(fake_mvtec_root, image_size):
    # Downsampling with interpolation would produce values strictly between 0
    # and 1 here, which is what nearest-neighbour plus thresholding prevents.
    records = [r for r in index_mvtec(fake_mvtec_root) if r.label == 1]
    dataset = MVTecDataset(records, image_size=image_size)
    for i in range(len(dataset)):
        mask = dataset[i]["mask"]
        assert torch.equal(mask, mask.round()), "mask picked up interpolated values"
        assert mask.any(), "a defective image lost its mask entirely"


def test_clean_images_have_an_empty_mask(fake_mvtec_root):
    records = [r for r in index_mvtec(fake_mvtec_root) if r.label == 0]
    dataset = MVTecDataset(records, image_size=32)
    for i in range(len(dataset)):
        assert not dataset[i]["mask"].any()


def test_labels_agree_with_mask_contents(fake_mvtec_root):
    dataset = MVTecDataset(index_mvtec(fake_mvtec_root), image_size=64)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert bool(sample["mask"].any()) == bool(int(sample["label"]) == 1)


def test_batches_through_a_dataloader(fake_mvtec_root):
    dataset = MVTecDataset(index_mvtec(fake_mvtec_root), image_size=32)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))

    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["mask"].shape == (4, 1, 32, 32)
    assert batch["label"].shape == (4,)
    assert len(batch["key"]) == 4 and isinstance(batch["key"][0], str)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="no records"):
        MVTecDataset([])


def test_inventory_renders(fake_mvtec_root):
    text = format_inventory(index_mvtec(fake_mvtec_root))
    assert "bottle" in text and "TOTAL" in text
