"""Region-level scoring: a defect is a connected blob, not a set of pixels.

Every per-pixel summary -- max, top-k mean, area above threshold -- is spatially
blind. A hundred pixels at 0.9 scattered across the image and a hundred forming
a tight blob score identically under all of them. But a defect is produced by a
physical process and is *contiguous*: scattered specks cannot be one. Encoding
that is not a statistical convenience, it is the strongest prior available.

Measured on hazelnut, thresholding at 0.5 and requiring nothing else, 35 of 86
clean test images carried some prediction. Their largest connected region had a
median of 27 px and a 90th percentile of 232 px. Real defect regions on the same
model ran to a median of 2,100 px. **The model's mistakes are specks; its
correct answers are blobs**, separated by roughly two orders of magnitude, and
no per-pixel statistic can see the difference.

Uncertainty is treated the same way, because two physically different doubts
get conflated by a per-pixel average:

* **Does this region exist at all?** -- measured by ``persistence``, the share of
  MC passes in which the blob survives. A region that appears in 12 of 20 passes
  is a very different claim from one that appears in all 20.
* **Where exactly is its boundary?** -- measured by the difference between
  ``interior_std`` and whole-region wobble. A blob whose edge moves a pixel is
  not in doubt; only its outline is.

The first matters for a pass/fail decision. The second barely does. Averaging
per-pixel disagreement across a region mixes them into one uninformative number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

from vinspect.uncertainty.mc_dropout import MCPrediction

#: Pixels eroded from a region's edge before measuring its interior. Two is
#: enough to drop the boundary band where disagreement is expected and
#: uninteresting.
INTERIOR_EROSION = 2

#: A region counts as surviving a pass if this share of it is still above
#: threshold in that pass.
SURVIVAL_SHARE = 0.5


@dataclass(frozen=True)
class Region:
    """One connected blob and how it behaved across the MC passes."""

    area: int
    mass: float  # summed probability: area x mean_probability
    mean_probability: float
    #: Share of MC passes in which most of this region stayed above threshold.
    #: The existence signal.
    persistence: float
    #: Spread of the region's per-pass area, relative to its mean. Large means
    #: the blob grows and shrinks between passes.
    area_cv: float
    #: Mean probability and disagreement away from the boundary band.
    interior_probability: float
    interior_std: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageScore:
    """What one image reduces to, after regions are found and filtered."""

    defect_score: float  # total mass over surviving regions
    n_regions: int
    largest_area: int
    #: Taken from the region carrying the most mass, since that is the one the
    #: verdict rests on.
    persistence: float
    area_cv: float
    interior_probability: float
    interior_std: float
    regions: Tuple[Region, ...]

    @property
    def predicted_defective(self) -> bool:
        return self.n_regions > 0


def _region_stats(
    member: np.ndarray,
    mean_map: np.ndarray,
    std_map: np.ndarray,
    above: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> Region:
    area = int(member.sum())
    mass = float(mean_map[member].sum())

    if above is None or above.shape[0] == 0:
        persistence, area_cv = 1.0, 0.0
    else:
        per_pass = above[:, member].sum(axis=1).astype(np.float64)
        persistence = float((per_pass >= SURVIVAL_SHARE * area).mean())
        mean_area = per_pass.mean()
        area_cv = float(per_pass.std() / mean_area) if mean_area > 0 else 0.0

    interior = ndimage.binary_erosion(member, iterations=INTERIOR_EROSION)
    if not interior.any():
        # Regions thinner than the erosion have no interior; fall back to the
        # whole region rather than reporting nothing.
        interior = member

    return Region(
        area=area,
        mass=mass,
        mean_probability=mass / max(area, 1),
        persistence=persistence,
        area_cv=area_cv,
        interior_probability=float(mean_map[interior].mean()),
        interior_std=float(std_map[interior].mean()),
        bbox=bbox,
    )


def extract_regions(
    prediction: MCPrediction,
    threshold: float = 0.5,
    min_area: int = 0,
) -> List[Region]:
    """Connected regions of the mean map, above ``threshold`` and ``min_area``.

    Regions are found on the *mean* map rather than per pass, so there is one
    stable set of blobs to describe; the passes are then used to say how each
    one behaved.
    """
    mean_map = prediction.mean.numpy()
    std_map = prediction.std.numpy()
    binary = mean_map > threshold

    labelled, count = ndimage.label(binary)
    if count == 0:
        return []

    above = (
        (prediction.passes.numpy() > threshold)
        if prediction.passes.numel()
        else None
    )
    slices = ndimage.find_objects(labelled)

    regions = []
    for index, window in enumerate(slices, start=1):
        member = labelled == index
        if member.sum() < min_area:
            continue
        bbox = (
            window[0].start,
            window[1].start,
            window[0].stop,
            window[1].stop,
        )
        regions.append(_region_stats(member, mean_map, std_map, above, bbox))

    return sorted(regions, key=lambda r: -r.mass)


def score_image(
    prediction: MCPrediction,
    threshold: float = 0.5,
    min_area: int = 0,
) -> ImageScore:
    """Collapse one image to a score plus the evidence behind it.

    ``defect_score`` sums the mass of every surviving region rather than taking
    only the largest. Measured on the training masks, 84% of real defects are a
    single region -- but 31% of hazelnut defects are multi-part, so keeping only
    the largest would discard real evidence on a third of them.
    """
    regions = extract_regions(prediction, threshold, min_area)
    if not regions:
        return ImageScore(
            defect_score=0.0,
            n_regions=0,
            largest_area=0,
            persistence=0.0,
            area_cv=0.0,
            interior_probability=0.0,
            interior_std=0.0,
            regions=(),
        )

    dominant = regions[0]
    return ImageScore(
        defect_score=float(sum(r.mass for r in regions)),
        n_regions=len(regions),
        largest_area=max(r.area for r in regions),
        persistence=dominant.persistence,
        area_cv=dominant.area_cv,
        interior_probability=dominant.interior_probability,
        interior_std=dominant.interior_std,
        regions=tuple(regions),
    )


def ground_truth_region_areas(
    masks: Sequence[torch.Tensor], largest_only: bool = False
) -> np.ndarray:
    """Areas of connected regions across a set of ground-truth masks.

    With ``largest_only``, one area per mask: the biggest region in it.
    """
    areas: List[int] = []
    for mask in masks:
        binary = mask.squeeze().numpy() > 0.5
        labelled, count = ndimage.label(binary)
        if not count:
            continue
        sizes = [int(a) for a in ndimage.sum(binary, labelled, range(1, count + 1))]
        areas.append(max(sizes)) if largest_only else areas.extend(sizes)
    return np.asarray(areas, dtype=np.float64)


def calibrate_min_area(
    masks: Sequence[torch.Tensor], percentile: float = 5.0
) -> int:
    """Smallest region worth believing, from the training ground truth.

    Set from what real defects look like, not by sweeping test performance --
    choosing this on the test set would be exactly the leakage this project
    exists to avoid. The rule: refuse to call anything a defect if it is smaller
    than ``percentile``% of the real defects seen in training.

    Calibrated on the **largest** region per defective image, not on every
    region. MVTec's masks are hand-drawn and contain stray single pixels: taking
    every region, hazelnut's 5th percentile comes out at 1 px, because 87
    "regions" across 42 images are mostly annotation specks rather than defects.
    Every defective image contains at least one real defect, and its largest
    region is the sound estimate of that defect's size.

    Per category, because the categories are not comparable: carpet's median
    defect region is 2,663 px against bottle's 21,982.
    """
    areas = ground_truth_region_areas(masks, largest_only=True)
    if areas.size == 0:
        raise ValueError("no defective masks given, so nothing to calibrate against")
    return int(np.percentile(areas, percentile))
