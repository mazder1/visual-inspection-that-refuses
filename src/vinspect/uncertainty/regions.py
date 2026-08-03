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

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from vinspect.uncertainty.mc_dropout import MCPrediction

# torch is deliberately not imported at module level: the ONNX serving
# container has no torch, and everything in the scoring path is numpy/scipy.
# Only the ground-truth helpers below touch torch tensors, and they import it
# lazily.

#: Pixels eroded from a region's edge before measuring its interior. Two is
#: enough to drop the boundary band where disagreement is expected and
#: uninteresting.
INTERIOR_EROSION = 2

#: A region counts as surviving a pass if this share of it is still above
#: threshold in that pass.
SURVIVAL_SHARE = 0.5

#: Probabilities are clipped before taking log-odds, since a saturated sigmoid
#: returns exactly 1.0 in float32 and the logit would be infinite.
LOGIT_CLIP = 1e-6


@dataclass(frozen=True)
class Region:
    """One connected blob and how it behaved across the MC passes.

    Under hysteresis extraction, ``area`` and every evidence statistic refer to
    the **core** -- the pixels above the strong threshold -- while ``extent`` is
    the full weak-connected footprint. Weak pixels establish connectivity and
    nothing else: at p < 0.5 their log-odds are negative, so letting them into
    the sum would make a defect's evidence *shrink* as more of it comes into
    faint view.
    """

    area: int
    #: Full footprint including the weak bridges. Equals ``area`` when no weak
    #: threshold is used.
    extent: int
    mass: float  # summed probability: area x mean_probability
    #: Summed log-odds. Evidence accumulates additively in log-odds, not in
    #: probability: logit(0.99) is 11x logit(0.6), where 0.99 is only 1.65x
    #: 0.6. That is what lets a small, very confident region outweigh a large,
    #: uncertain one -- 500 px at 0.99 scores 2,300 against 2,027 for 5,000 px
    #: at 0.6, which plain mass gets backwards by a factor of six.
    logodds: float
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
    """What one image reduces to, after its regions are found.

    ``defect_score`` is the summed log-odds. Compared on validation at a 1%
    false-alarm budget -- the operating point a line would actually run -- a
    region score catches 92% of bottle defects where the old per-pixel maximum
    caught 69%. AUROC scored those two 0.995 and 0.992 and called it a tie,
    because it averages over every decision line and hides exactly the
    behaviour that matters.

    ``mass_score`` is kept alongside for comparison. The two are level on this
    data; they differ only on small, very confident regions, which these
    validation sets do not contain.
    """

    defect_score: float  # summed log-odds over regions
    mass_score: float  # summed probability, for contrast
    n_regions: int
    largest_area: int
    #: Weak-connected footprint of the biggest region; equals largest_area
    #: without hysteresis.
    largest_extent: int
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
    core: np.ndarray,
    mean_map: np.ndarray,
    logit_map: np.ndarray,
    std_map: np.ndarray,
    above: Optional[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> Region:
    area = int(core.sum())
    extent = int(member.sum())
    mass = float(mean_map[core].sum())
    logodds = float(logit_map[core].sum())

    if above is None or above.shape[0] == 0:
        persistence, area_cv = 1.0, 0.0
    else:
        per_pass = above[:, core].sum(axis=1).astype(np.float64)
        persistence = float((per_pass >= SURVIVAL_SHARE * area).mean())
        mean_area = per_pass.mean()
        area_cv = float(per_pass.std() / mean_area) if mean_area > 0 else 0.0

    interior = ndimage.binary_erosion(core, iterations=INTERIOR_EROSION)
    if not interior.any():
        # Regions thinner than the erosion have no interior; fall back to the
        # whole core rather than reporting nothing.
        interior = core

    return Region(
        area=area,
        extent=extent,
        mass=mass,
        logodds=logodds,
        mean_probability=mass / max(area, 1),
        persistence=persistence,
        area_cv=area_cv,
        interior_probability=float(mean_map[interior].mean()),
        interior_std=float(std_map[interior].mean()),
        bbox=bbox,
    )


def extract_regions(
    prediction: "MCPrediction",
    threshold: float = 0.5,
    min_area: int = 0,
    weak_threshold: Optional[float] = None,
) -> List[Region]:
    """Connected regions of the mean map, optionally with hysteresis.

    With ``weak_threshold`` set, regions are connected components of the *weak*
    mask that contain at least one *strong* pixel -- Canny's hysteresis rule.
    A weak pixel alone is nothing; a weak pixel adjacent to strong evidence is
    part of it. This exists because faint defects present as chains of strong
    fragments separated by weak valleys: one hard threshold shatters the chain
    into pieces individually indistinguishable from noise, while the weak mask
    reconnects them into a single region. Diffuse clean-part whisper has no
    strong seed to attach to and never activates.

    Evidence statistics stay on the strong core; weak pixels only connect.

    Regions are found on the *mean* map rather than per pass, so there is one
    stable set of blobs to describe; the passes are then used to say how each
    one behaved.
    """
    if weak_threshold is not None and not 0.0 < weak_threshold < threshold:
        raise ValueError(
            f"weak_threshold must sit below threshold, got "
            f"{weak_threshold} vs {threshold}"
        )
    mean_map = prediction.mean.numpy()
    std_map = prediction.std.numpy()
    clipped = np.clip(mean_map, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    logit_map = np.log(clipped / (1.0 - clipped))

    strong = mean_map > threshold
    support = strong if weak_threshold is None else mean_map > weak_threshold

    labelled, count = ndimage.label(support)
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
        core = member & strong
        if not core.any():
            # Weak whisper with no strong seed: not evidence, by design.
            continue
        if core.sum() < min_area:
            continue
        bbox = (
            window[0].start,
            window[1].start,
            window[0].stop,
            window[1].stop,
        )
        regions.append(
            _region_stats(member, core, mean_map, logit_map, std_map, above, bbox)
        )

    return sorted(regions, key=lambda r: -r.mass)


def score_image(
    prediction: "MCPrediction",
    threshold: float = 0.5,
    min_area: int = 0,
    weak_threshold: Optional[float] = None,
) -> ImageScore:
    """Collapse one image to a score plus the evidence behind it.

    ``min_area`` defaults to zero: **no gate**. A hard minimum area was measured
    to be unnecessary -- the region score alone reaches 92-100% recall at a 1%
    false-alarm budget without one -- and it was actively harmful, because
    hazelnut's calibrated threshold of 1,617 px sat above the smallest real
    defect on record at 609 px. Anything in that band was deleted and the part
    called clean, with nothing flagged. A borderline defect should become an
    abstention, not a silent pass.

    Scores take the **strongest single region**, not the sum over all of them.
    Summing is spatially blind and quietly undoes the whole point: 100 scattered
    single pixels sum to exactly the same total as one 100-pixel blob, because
    a sum over regions is just a sum over pixels wearing a disguise. Taking the
    maximum states the physical claim directly -- the case for this part being
    defective rests on its best piece of *connected* evidence.

    The cost is that a genuinely multi-part defect is scored by its largest
    part alone. That is acceptable here: the decision is whether the part is
    defective, not how many defects it has, and the largest part is enough to
    make that call.
    """
    regions = extract_regions(prediction, threshold, min_area, weak_threshold)
    if not regions:
        return ImageScore(
            defect_score=0.0,
            mass_score=0.0,
            n_regions=0,
            largest_area=0,
            largest_extent=0,
            persistence=0.0,
            area_cv=0.0,
            interior_probability=0.0,
            interior_std=0.0,
            regions=(),
        )

    dominant = regions[0]
    return ImageScore(
        defect_score=float(max(r.logodds for r in regions)),
        mass_score=float(max(r.mass for r in regions)),
        n_regions=len(regions),
        largest_area=max(r.area for r in regions),
        largest_extent=max(r.extent for r in regions),
        persistence=dominant.persistence,
        area_cv=dominant.area_cv,
        interior_probability=dominant.interior_probability,
        interior_std=dominant.interior_std,
        regions=tuple(regions),
    )


def ground_truth_region_areas(
    masks: "Sequence[torch.Tensor]", largest_only: bool = False
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
    masks: "Sequence[torch.Tensor]", percentile: float = 5.0
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
