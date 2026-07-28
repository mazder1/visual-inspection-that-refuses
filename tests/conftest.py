"""A synthetic MVTec AD tree, so the loader can be tested without the 5 GB download.

The fixture reproduces the layout and the properties the loader depends on:
train/ is clean-only, defective images live under test/ with a matching mask in
ground_truth/, and one category is single-channel to exercise the RGB
conversion. Pixel content is arbitrary; geometry is not.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

# category -> (mode, size, n_train_good, n_test_good, {defect: n})
FAKE_LAYOUT = {
    "bottle": ("RGB", (64, 64), 4, 2, {"broken_large": 3, "contamination": 2}),
    "grid": ("L", (48, 48), 3, 1, {"bent": 2}),
}


def _write_image(path: Path, mode: str, size: tuple, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, value).save(path)


def _write_mask(path: Path, size: tuple, index: int) -> None:
    """A mask with a solid 0/255 rectangle, area varying with ``index``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", size, 0)
    side = 4 + index * 2
    for x in range(side):
        for y in range(side):
            mask.putpixel((x, y), 255)
    mask.save(path)


@pytest.fixture(scope="session")
def fake_mvtec_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("mvtec")
    for category, (mode, size, n_train, n_test_good, defects) in FAKE_LAYOUT.items():
        fill = 128 if mode == "L" else (128, 100, 60)
        for i in range(n_train):
            _write_image(root / category / "train" / "good" / f"{i:03d}.png", mode, size, fill)
        for i in range(n_test_good):
            _write_image(root / category / "test" / "good" / f"{i:03d}.png", mode, size, fill)
        for defect, count in defects.items():
            for i in range(count):
                _write_image(
                    root / category / "test" / defect / f"{i:03d}.png", mode, size, fill
                )
                _write_mask(
                    root / category / "ground_truth" / defect / f"{i:03d}_mask.png",
                    size,
                    i,
                )
    return root


# category -> list of components, each a list of rotations in degrees. Two
# images sharing a component are the same synthetic part re-placed on the rig.
PLANTED_COMPONENTS = {
    "widget": [[0.0, 4.0, -3.0], [0.0, 7.0], [0.0], [0.0, -5.0], [0.0]],
    "sprocket": [[0.0, 6.0], [0.0], [0.0, -8.0, 3.0]],
}
PART_SIZE = 480


def _draw_part(seed: int, size: int = PART_SIZE) -> Image.Image:
    """A synthetic part with enough distinctive structure for ORB to latch on.

    Blobs and bars rather than noise: keypoints on noise do not survive the
    interpolation of a rotation, so a noise-based fixture would test nothing.
    """
    from PIL import ImageDraw

    rng = random.Random(seed)
    canvas = Image.new("RGB", (size, size), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    for _ in range(28):
        x, y = rng.randint(40, size - 90), rng.randint(40, size - 90)
        w, h = rng.randint(18, 60), rng.randint(18, 60)
        colour = tuple(rng.randint(70, 255) for _ in range(3))
        if rng.random() < 0.5:
            draw.ellipse([x, y, x + w, y + h], fill=colour)
        else:
            draw.rectangle([x, y, x + w, y + h], fill=colour)
    return canvas


@pytest.fixture(scope="session")
def planted_root(tmp_path_factory) -> Path:
    """An MVTec-shaped tree where component identity is known by construction."""
    root = tmp_path_factory.mktemp("planted")
    seed = 0
    for category, components in PLANTED_COMPONENTS.items():
        for component_index, rotations in enumerate(components):
            seed += 1
            base = _draw_part(seed)
            for rotation_index, angle in enumerate(rotations):
                image = base.rotate(angle, resample=Image.BICUBIC, fillcolor=(18, 18, 22))
                name = f"{component_index:02d}{rotation_index}"
                # Half the components carry a defect, so grouping is exercised
                # across the clean/defective boundary as well as within it.
                if component_index % 2 == 0:
                    path = root / category / "train" / "good" / f"{name}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                else:
                    path = root / category / "test" / "scratch" / f"{name}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                    _write_mask(
                        root / category / "ground_truth" / "scratch" / f"{name}_mask.png",
                        (PART_SIZE, PART_SIZE),
                        rotation_index,
                    )
        # Every category needs a test/good directory to look like MVTec.
        _write_image(
            root / category / "test" / "good" / "900.png", "RGB", (PART_SIZE, PART_SIZE), (18, 18, 22)
        )
    return root


@pytest.fixture(scope="session")
def planted_groups(planted_root) -> dict:
    """record key -> planted component id, the ground truth for grouping."""
    truth = {}
    for category, components in PLANTED_COMPONENTS.items():
        for component_index, rotations in enumerate(components):
            for rotation_index, _ in enumerate(rotations):
                name = f"{component_index:02d}{rotation_index}"
                origin, defect = (
                    ("train", "good")
                    if component_index % 2 == 0
                    else ("test", "scratch")
                )
                key = f"{category}/{origin}/{defect}/{name}"
                truth[key] = f"{category}:c{component_index}"
    return truth


@pytest.fixture(scope="session")
def fake_mvtec_counts() -> dict:
    counts = {}
    for category, (_, _, n_train, n_test_good, defects) in FAKE_LAYOUT.items():
        n_defective = sum(defects.values())
        counts[category] = {
            "clean": n_train + n_test_good,
            "defective": n_defective,
            "total": n_train + n_test_good + n_defective,
            "n_defect_types": len(defects),
        }
    return counts
