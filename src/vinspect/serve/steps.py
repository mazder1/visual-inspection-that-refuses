"""Engine-independent pieces of the serving chain: staircase and verdicts.

Separated from the torch predictor so the ONNX serving path can import them
without pulling torch into the container -- removing torch is most of the
image-size win that justified shipping INT8 at all.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class StepCalibrator:
    """The isotonic staircase, replayed from its exported summary."""

    def __init__(self, summary: Dict) -> None:
        self.steps = summary["steps"]
        self.boundaries = np.array([step["scores"][0] for step in self.steps])
        self.uppers = np.array([step["scores"][1] for step in self.steps])
        self.values = np.array([step["probability"] for step in self.steps])

    def predict(self, score: float) -> float:
        index = int(
            np.clip(
                np.searchsorted(self.boundaries, score, side="right") - 1,
                0,
                len(self.values) - 1,
            )
        )
        return float(self.values[index])

    def supported(self, score: float) -> bool:
        return bool(((score >= self.boundaries) & (score <= self.uppers)).any())


def decide(
    probability: float,
    supported: bool,
    weak_px: int,
    weak_floor: float,
    no_call_band: Tuple[float, float],
) -> str:
    """The three-layer verdict, identical for every engine."""
    if not supported:
        return "no-call"
    if no_call_band[0] <= probability <= no_call_band[1]:
        return "no-call"
    if probability >= 0.5:
        return "fail"
    if weak_px > weak_floor:
        return "no-call"
    return "pass"
