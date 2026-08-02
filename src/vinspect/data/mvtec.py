"""Indexing and loading for MVTec AD.

The dataset ships in this layout::

    <root>/<category>/train/good/<id>.png
    <root>/<category>/test/good/<id>.png
    <root>/<category>/test/<defect>/<id>.png
    <root>/<category>/ground_truth/<defect>/<id>_mask.png

Two properties of that layout drive the code below.

First, the shipped ``train`` directory is defect-free only, and every defective
image lives under ``test``. The shipped division is therefore not an evaluation
split, and this module does not treat it as one. It records which directory an
image came from as ``origin`` and leaves splitting to the split generator,
which needs the whole pool to work from.

Second, only defective images carry a ground-truth mask. Clean images get an
all-zero mask synthesised at load time so every record has the same shape.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# MVTec AD ships two kinds of category. A texture image is a crop of a
# continuous surface; an object image is one manufactured item centred in the
# frame. The distinction matters to the split generator, because "same physical
# component" means something different for each.
TEXTURE_CATEGORIES: Tuple[str, ...] = ("carpet", "grid", "leather", "tile", "wood")
OBJECT_CATEGORIES: Tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
CATEGORIES: Tuple[str, ...] = tuple(sorted(TEXTURE_CATEGORIES + OBJECT_CATEGORIES))

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
GOOD = "good"


class MVTecLayoutError(RuntimeError):
    """The directory tree is not the MVTec AD layout this module expects."""


@dataclass(frozen=True)
class MVTecRecord:
    """One image, its mask if it has one, and where it came from.

    Frozen and hashable so a split file can be built from these directly.
    """

    category: str
    origin: str  # "train" or "test", as shipped -- not an evaluation split
    defect_type: str  # "good", or e.g. "broken_large"
    image_id: str
    image_path: Path
    mask_path: Optional[Path]

    @property
    def label(self) -> int:
        """1 if the image contains a defect, 0 if it is clean."""
        return 0 if self.defect_type == GOOD else 1

    @property
    def key(self) -> str:
        """Stable identifier, independent of where the dataset is mounted."""
        return f"{self.category}/{self.origin}/{self.defect_type}/{self.image_id}"


def _image_files(directory: Path) -> List[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _find_mask(ground_truth_dir: Path, defect_type: str, image_id: str) -> Path:
    defect_dir = ground_truth_dir / defect_type
    if not defect_dir.is_dir():
        raise MVTecLayoutError(
            f"no ground-truth directory for defect type {defect_type!r} "
            f"(expected {defect_dir})"
        )
    exact = defect_dir / f"{image_id}_mask.png"
    if exact.is_file():
        return exact
    # Tolerate a different extension, but not a missing mask.
    candidates = sorted(defect_dir.glob(f"{image_id}_mask.*"))
    if candidates:
        return candidates[0]
    raise MVTecLayoutError(f"no mask for {image_id!r} under {defect_dir}")


def discover_categories(root: Path) -> List[str]:
    """Category directories actually present under ``root``.

    Identified by structure rather than by name, so a partial download or an
    extra directory does not silently change what gets indexed.
    """
    root = Path(root)
    if not root.is_dir():
        raise MVTecLayoutError(f"dataset root does not exist: {root}")
    found = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "train").is_dir() and (p / "test").is_dir()
    )
    if not found:
        raise MVTecLayoutError(
            f"no category directories under {root}. Expected subdirectories "
            f"each containing train/ and test/, e.g. {root / 'bottle' / 'train'}"
        )
    return found


def index_mvtec(
    root: Path,
    categories: Optional[Sequence[str]] = None,
) -> List[MVTecRecord]:
    """Walk the dataset and return one record per image, in a stable order.

    Reads no pixels -- this is a directory walk, so it is cheap enough to call
    on every run rather than caching an index that can drift from the data.

    Raises :class:`MVTecLayoutError` rather than skipping anything: a silently
    short index is the kind of bug that shows up later as an inflated score.
    """
    root = Path(root)
    available = discover_categories(root)
    if categories is None:
        wanted = available
    else:
        wanted = list(categories)
        missing = [c for c in wanted if c not in available]
        if missing:
            raise MVTecLayoutError(
                f"requested categories not found under {root}: {missing}. "
                f"Available: {available}"
            )

    records: List[MVTecRecord] = []
    for category in wanted:
        category_dir = root / category
        ground_truth_dir = category_dir / "ground_truth"
        for origin in ("train", "test"):
            origin_dir = category_dir / origin
            if not origin_dir.is_dir():
                raise MVTecLayoutError(f"missing directory: {origin_dir}")
            defect_dirs = sorted(p for p in origin_dir.iterdir() if p.is_dir())
            if not defect_dirs:
                raise MVTecLayoutError(f"no subdirectories under {origin_dir}")
            for defect_dir in defect_dirs:
                defect_type = defect_dir.name
                images = _image_files(defect_dir)
                if not images:
                    raise MVTecLayoutError(f"no images under {defect_dir}")
                for image_path in images:
                    image_id = image_path.stem
                    mask_path = (
                        None
                        if defect_type == GOOD
                        else _find_mask(ground_truth_dir, defect_type, image_id)
                    )
                    records.append(
                        MVTecRecord(
                            category=category,
                            origin=origin,
                            defect_type=defect_type,
                            image_id=image_id,
                            image_path=image_path,
                            mask_path=mask_path,
                        )
                    )
    return records


def summarise(records: Sequence[MVTecRecord]) -> Dict[str, object]:
    """Counts needed to size a split, plus bytes on disk.

    ``bytes_on_disk`` stats each file; it does not decode them.
    """
    per_category: Dict[str, Dict[str, object]] = {}
    for category in sorted({r.category for r in records}):
        rows = [r for r in records if r.category == category]
        defect_types = sorted({r.defect_type for r in rows if r.label == 1})
        per_category[category] = {
            "total": len(rows),
            "clean": sum(1 for r in rows if r.label == 0),
            "defective": sum(1 for r in rows if r.label == 1),
            "defect_types": defect_types,
            "n_defect_types": len(defect_types),
            "per_defect_type": dict(
                Counter(r.defect_type for r in rows if r.label == 1)
            ),
            "shipped_train": sum(1 for r in rows if r.origin == "train"),
            "shipped_test": sum(1 for r in rows if r.origin == "test"),
            "bytes_on_disk": sum(r.image_path.stat().st_size for r in rows),
        }
    return {
        "n_images": len(records),
        "n_clean": sum(1 for r in records if r.label == 0),
        "n_defective": sum(1 for r in records if r.label == 1),
        "n_categories": len(per_category),
        "n_defect_types": sum(
            int(v["n_defect_types"]) for v in per_category.values()  # type: ignore[arg-type]
        ),
        "bytes_on_disk": sum(
            int(v["bytes_on_disk"]) for v in per_category.values()  # type: ignore[arg-type]
        ),
        "by_category": per_category,
    }


def format_inventory(records: Sequence[MVTecRecord]) -> str:
    """Render :func:`summarise` as a table. This is the sizing report."""
    stats = summarise(records)
    by_category: Dict[str, Dict[str, object]] = stats["by_category"]  # type: ignore[assignment]

    header = f"{'category':<12} {'kind':<8} {'total':>6} {'clean':>6} {'defect':>7} {'types':>6} {'MB':>7}"
    lines = [header, "-" * len(header)]
    for category, row in by_category.items():
        kind = "texture" if category in TEXTURE_CATEGORIES else "object"
        lines.append(
            f"{category:<12} {kind:<8} {row['total']:>6} {row['clean']:>6} "
            f"{row['defective']:>7} {row['n_defect_types']:>6} "
            f"{int(row['bytes_on_disk']) / 1e6:>7.1f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<12} {'':<8} {stats['n_images']:>6} {stats['n_clean']:>6} "
        f"{stats['n_defective']:>7} {stats['n_defect_types']:>6} "
        f"{int(stats['bytes_on_disk']) / 1e6:>7.1f}"  # type: ignore[arg-type]
    )
    return "\n".join(lines)


class MVTecDataset(Dataset):
    """Images and masks for a given list of records.

    Takes records rather than a root directory, so the split generator decides
    membership and this class only decides how pixels are loaded. Augmentation
    is deliberately not here: it belongs to the training policy, and baking it
    in would make the evaluation path silently non-deterministic.

    Masks resample with nearest-neighbour and are re-thresholded afterwards, so
    a resized mask stays binary. Interpolating a mask invents partial defects
    at every boundary and quietly inflates IoU.
    """

    def __init__(
        self,
        records: Sequence[MVTecRecord],
        image_size: int = 256,
        mask_threshold: int = 0,
    ) -> None:
        if not records:
            raise ValueError("MVTecDataset was given no records")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.records = list(records)
        self.image_size = int(image_size)
        self.mask_threshold = int(mask_threshold)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> torch.Tensor:
        size = (self.image_size, self.image_size)
        with Image.open(path) as handle:
            # Several categories ship as single-channel; force 3 so the batch
            # shape does not depend on which category it was drawn from.
            image = handle.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(self.image_size, self.image_size, 3)
        return tensor.permute(2, 0, 1).float().div_(255.0)

    def _load_mask(self, record: MVTecRecord) -> torch.Tensor:
        if record.mask_path is None:
            return torch.zeros(1, self.image_size, self.image_size)
        size = (self.image_size, self.image_size)
        with Image.open(record.mask_path) as handle:
            mask = handle.convert("L").resize(size, Image.Resampling.NEAREST)
        tensor = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(1, self.image_size, self.image_size)
        return (tensor > self.mask_threshold).float()

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        image = self._load_image(record.image_path)
        mask = self._load_mask(record)
        if record.label == 0 and mask.any():
            raise MVTecLayoutError(
                f"clean image {record.key} produced a non-empty mask"
            )
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(record.label, dtype=torch.long),
            "category": record.category,
            "defect_type": record.defect_type,
            "key": record.key,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory an MVTec AD tree.")
    parser.add_argument("--root", type=Path, required=True, help="dataset root")
    parser.add_argument(
        "--categories", nargs="*", default=None, help="defaults to all found"
    )
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="decode one image per category to verify the loader end to end",
    )
    args = parser.parse_args(argv)

    records = index_mvtec(args.root, args.categories)
    print(format_inventory(records))

    if args.check_load:
        print()
        for category in sorted({r.category for r in records}):
            rows = [r for r in records if r.category == category]
            defective = next((r for r in rows if r.label == 1), rows[0])
            sample = MVTecDataset([defective], image_size=256)[0]
            covered = float(sample["mask"].mean()) * 100
            print(
                f"{category:<12} {tuple(sample['image'].shape)} "
                f"mask {tuple(sample['mask'].shape)} "
                f"defect area {covered:5.2f}%  {sample['key']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
