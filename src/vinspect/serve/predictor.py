"""The frozen chain as one object: image bytes in, verdict out.

Everything comes from the bundle written at export time. Nothing is fitted,
swept or tuned here; the calibrator steps are replayed from JSON, and the
verdict logic is the same three-layer rule the evaluation used: calibrated
verdict, support gap, weak-evidence floor.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from vinspect.models.unet import UNet
from vinspect.uncertainty.mc_dropout import mc_predict
from vinspect.uncertainty.regions import score_image


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
        return bool(
            ((score >= self.boundaries) & (score <= self.uppers)).any()
        )


class Predictor:
    """One category's model plus its frozen decision chain."""

    def __init__(self, bundle_dir: Path) -> None:
        bundle_dir = Path(bundle_dir)
        self.chain = json.loads((bundle_dir / "chain.json").read_text(encoding="utf-8"))
        model_config = self.chain["model"]

        self.model = UNet(
            base_channels=model_config["base_channels"],
            depth=model_config["depth"],
            dropout=model_config["dropout"],
        )
        checkpoint = torch.load(
            bundle_dir / "model.pt", map_location="cpu", weights_only=False
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.image_size = model_config["image_size"]
        settings = self.chain["chain"]
        self.passes = settings["mc_passes"]
        self.threshold = settings["threshold"]
        self.weak_threshold = settings["weak_threshold"]
        self.weak_floor = settings["weak_floor"]
        self.no_call_band = tuple(settings["no_call_band"])
        self.calibrator = StepCalibrator(settings["calibrator"])

    # --- pieces -----------------------------------------------------------

    def _prepare(self, payload: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(payload)) as handle:
            image = handle.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _verdict(self, probability: float, supported: bool, weak_px: int) -> str:
        if not supported:
            return "no-call"
        if self.no_call_band[0] <= probability <= self.no_call_band[1]:
            return "no-call"
        if probability >= 0.5:
            return "fail"
        if weak_px > self.weak_floor:
            return "no-call"
        return "pass"

    @staticmethod
    def _mask_png(mask: np.ndarray) -> str:
        image = Image.fromarray((mask * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    # --- the whole chain --------------------------------------------------

    def inspect(self, payload: bytes) -> Dict:
        started = time.perf_counter()
        tensor = self._prepare(payload)

        with torch.no_grad():
            prediction = mc_predict(
                self.model, tensor, passes=self.passes, keep_passes=True
            )
        image_score = score_image(
            prediction,
            threshold=self.threshold,
            weak_threshold=self.weak_threshold,
        )
        weak_px = int((prediction.mean.numpy() > self.weak_threshold).sum())

        probability = self.calibrator.predict(image_score.defect_score)
        supported = self.calibrator.supported(image_score.defect_score)
        verdict = self._verdict(probability, supported, weak_px)

        mask = (prediction.mean.numpy() > self.threshold).astype(np.uint8)
        regions: List[Dict] = [
            {
                "area_px": region.area,
                "extent_px": region.extent,
                "mean_probability": round(region.mean_probability, 4),
                "persistence": round(region.persistence, 3),
                "bbox": list(region.bbox),
            }
            for region in image_score.regions[:10]
        ]

        return {
            "category": self.chain["category"],
            "verdict": verdict,
            "probability_defective": round(probability, 4),
            "probability_is_supported": supported,
            "defect_score": round(image_score.defect_score, 2),
            "weak_evidence_px": weak_px,
            "weak_floor": self.weak_floor,
            "n_regions": image_score.n_regions,
            "regions": regions,
            "mask_png_base64": self._mask_png(mask),
            "mask_size": [self.image_size, self.image_size],
            "mc_passes": self.passes,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "explanation": (
                "The mask shows where the model looked, not what the defect "
                "is. A no-call means: route this part to a human."
            ),
            "provenance": self.chain["provenance"],
            "disclaimer": self.chain["disclaimer"],
        }
