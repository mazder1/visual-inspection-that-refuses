"""Turning raw region scores into probabilities that mean what they say.

A defect score of 2,300 ranks parts, but tells an operator nothing. Calibrated
means: of all the parts where the system says 90%, about 90% are actually
defective. The only way to know what a score is worth is to check it against
outcomes on held-out data -- the mapping is counted, not derived.

The estimator is isotonic regression (pool-adjacent-violators), implemented
here directly rather than imported, for the same reason as the U-Net: every
piece of it should be explainable.

* **Adaptive steps.** PAV merges neighbouring scores until each step has
  enough agreeing data to justify itself -- wide steps where data is thin,
  narrow where it is dense. Equal-width bins would rest some estimates on two
  images and others on eighty.
* **Monotone, and nothing more.** A higher score never maps to a lower
  probability. That constraint is exactly what we know to be true about the
  score, and nothing else is assumed. A smooth curve through the same points
  would carry no more information while inventing values between them, and a
  polynomial can dip downward between fit points, which is physically absurd
  here.
* **Laplace shrinkage at the steps.** The raw staircase claims 100% at its top
  step and 0% at its bottom, from a handful of images. Each step is shrunk
  toward the base rate by a pseudo-count, so a step resting on 30 images barely
  moves while one resting on 3 moves a lot. Certainty from small counts is the
  disease calibration exists to cure; the calibrator must not have it itself.

Fit on validation, report on test. The staircase is chosen to fit validation,
so measuring it there would be marking our own homework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared distance between claimed probability and outcome.

    Punishes confident wrongness hardest: claiming 0.95 on a clean part costs
    0.90, claiming 0.55 on the same mistake costs 0.30. Lower is better; 0.25
    is what always saying 0.5 scores.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probabilities - labels) ** 2))


@dataclass(frozen=True)
class CalibrationStep:
    """One step of the fitted staircase, with the evidence behind it."""

    lower: float  # scores from here (inclusive)...
    upper: float  # ...up to here map to this step
    probability: float
    count: int  # validation images the step rests on
    raw_rate: float  # observed defective rate before shrinkage


class IsotonicCalibrator:
    """Monotone score -> probability mapping, counted from held-out outcomes."""

    def __init__(self, pseudocount: float = 1.0) -> None:
        if pseudocount < 0:
            raise ValueError(f"pseudocount must be non-negative, got {pseudocount}")
        self.pseudocount = float(pseudocount)
        self.steps: List[CalibrationStep] = []
        self.base_rate: float = float("nan")

    # --- fitting ----------------------------------------------------------

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if scores.shape != labels.shape or scores.ndim != 1:
            raise ValueError(
                f"scores and labels must be equal-length 1-D, got "
                f"{scores.shape} and {labels.shape}"
            )
        if len(scores) < 2:
            raise ValueError("cannot calibrate on fewer than 2 outcomes")
        if not set(np.unique(labels)) <= {0.0, 1.0}:
            raise ValueError("labels must be 0 or 1")
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "calibration needs both outcomes present; a set that is all "
                "clean or all defective pins every score to one answer"
            )

        order = np.argsort(scores, kind="stable")
        sorted_scores, sorted_labels = scores[order], labels[order]
        self.base_rate = float(labels.mean())

        # Tied scores must share a block from the start: the mapping is a
        # function of the score, so identical scores cannot get different
        # probabilities.
        unique_scores, first_index, tie_counts = np.unique(
            sorted_scores, return_index=True, return_counts=True
        )
        block_sum = np.add.reduceat(sorted_labels, first_index)

        # Pool adjacent violators: merge neighbouring blocks until the mean
        # defective rate is non-decreasing in score. The surviving blocks ARE
        # the adaptive steps.
        blocks: List[List[float]] = []  # [sum, count, lower, upper]
        for value, total, count in zip(unique_scores, block_sum, tie_counts):
            blocks.append([float(total), float(count), float(value), float(value)])
            while len(blocks) > 1 and (
                blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
            ):
                last = blocks.pop()
                blocks[-1][0] += last[0]
                blocks[-1][1] += last[1]
                blocks[-1][3] = last[3]

        # Laplace shrinkage, each step toward the base rate by pseudocount
        # observations. A final running maximum re-enforces monotonicity:
        # shrinkage toward a single prior can, in edge cases, swap two steps
        # when the lower one rests on far fewer images.
        shrunk: List[float] = []
        for total, count, _, _ in blocks:
            shrunk.append(
                (total + self.pseudocount * self.base_rate)
                / (count + self.pseudocount)
            )
        shrunk = list(np.maximum.accumulate(shrunk))

        self.steps = [
            CalibrationStep(
                lower=lower,
                upper=upper,
                probability=float(probability),
                count=int(count),
                raw_rate=float(total / count),
            )
            for (total, count, lower, upper), probability in zip(blocks, shrunk)
        ]
        return self

    # --- predicting -------------------------------------------------------

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        """Map scores to probabilities via the fitted staircase.

        Scores beyond the fitted range take the end steps' values: the staircase
        says "I have no data past here, so the estimate stops changing", which
        is the honest answer.
        """
        if not self.steps:
            raise RuntimeError("fit() has not been called")
        scores = np.asarray(scores, dtype=np.float64)
        boundaries = np.array([step.lower for step in self.steps])
        values = np.array([step.probability for step in self.steps])
        index = np.clip(
            np.searchsorted(boundaries, scores, side="right") - 1, 0, len(values) - 1
        )
        return values[index]

    def supported(self, scores: Sequence[float]) -> np.ndarray:
        """Whether each score falls inside a fitted step's observed range.

        A score in the gap *between* steps was never seen at fit time: the
        probability returned for it is an extrapolation, not a measurement.
        On this data the gaps are exactly where the misses hid -- bottle's
        staircase jumps from 252 to 1,927, and two truly defective test images
        scored 449 and 679, inside the gap, where the step value of 0.5% was
        asserted on zero validation images. Scores beyond either end of the
        staircase are likewise unsupported.
        """
        if not self.steps:
            raise RuntimeError("fit() has not been called")
        scores = np.asarray(scores, dtype=np.float64)
        inside = np.zeros(scores.shape, dtype=bool)
        for step in self.steps:
            inside |= (scores >= step.lower) & (scores <= step.upper)
        return inside

    def summary(self) -> Dict[str, object]:
        return {
            "n_steps": len(self.steps),
            "base_rate": self.base_rate,
            "pseudocount": self.pseudocount,
            "steps": [
                {
                    "scores": [step.lower, step.upper],
                    "probability": step.probability,
                    "count": step.count,
                    "raw_rate": step.raw_rate,
                }
                for step in self.steps
            ],
        }


def reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 8,
    max_claim_spread: float = 0.15,
) -> List[Dict[str, float]]:
    """Claimed probability against observed rate, binned by claim level.

    Equal-count binning misleads on clumpy distributions, which calibrated
    staircase outputs always are: with 178 images claiming ~0% and 29 claiming
    ~95%, a forced middle bin sweeps together claims from 1% to 90% and prints
    their average as "42%", a level almost no image actually claimed. So bins
    are grown over sorted claims and cut wherever the claim value jumps by more
    than ``max_claim_spread`` -- images are only averaged with images making a
    similar claim. ``n_bins`` caps how fine the result gets; each row carries
    its count so a reader can see the evidence behind it.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probabilities) == 0:
        return []

    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]

    target = max(1, len(order) // max(n_bins, 1))
    rows: List[Dict[str, float]] = []
    start = 0
    for i in range(1, len(order) + 1):
        boundary = i == len(order)
        if not boundary:
            jump = sorted_probabilities[i] - sorted_probabilities[start]
            boundary = i - start >= target and jump > 0
            boundary = boundary or jump > max_claim_spread
        if boundary:
            chunk_p = sorted_probabilities[start:i]
            chunk_l = sorted_labels[start:i]
            rows.append(
                {
                    "claimed": float(chunk_p.mean()),
                    "observed": float(chunk_l.mean()),
                    "count": int(i - start),
                }
            )
            start = i
    return rows


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], n_bins: int = 8
) -> float:
    """Count-weighted mean gap between claimed and observed, over the bins."""
    rows = reliability_bins(probabilities, labels, n_bins)
    total = sum(row["count"] for row in rows)
    if not total:
        return 0.0
    return float(
        sum(abs(row["claimed"] - row["observed"]) * row["count"] for row in rows)
        / total
    )
