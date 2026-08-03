"""The serving chain on ONNX Runtime, with no torch anywhere in the path.

Torch exists in the fp32 container only to run the model; ONNX Runtime is
~50 MB against torch's ~1.3 GB, so this module is what shrinks the image from
1.95 GB to roughly 400 MB. The region scoring is reused VERBATIM from the
evaluated chain -- ``regions.score_image`` is numpy/scipy inside and only ever
calls ``.numpy()`` on the prediction fields, so a thin view class satisfies it
without porting a line of the scoring logic. No port, no new place for the
numbers to diverge; a parity test holds both predictors to the same output.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
from PIL import Image

from vinspect.serve.steps import StepCalibrator, decide
from vinspect.uncertainty.regions import score_image


class _View:
    """Duck-typed stand-in for a torch tensor: just enough for score_image."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array

    def numel(self) -> int:
        return int(self._array.size)


class OnnxPredictor:
    """One category's INT8 (or fp32) ONNX model plus its frozen chain."""

    def __init__(self, bundle_dir: Path) -> None:
        import onnxruntime as ort

        bundle_dir = Path(bundle_dir)
        self.chain = json.loads(
            (bundle_dir / "chain.json").read_text(encoding="utf-8")
        )
        model_file = self.chain["model"].get("file", "model.onnx")
        options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(bundle_dir / model_file),
            options,
            providers=["CPUExecutionProvider"],
        )

        self.image_size = self.chain["model"]["image_size"]
        settings = self.chain["chain"]
        self.passes = settings["mc_passes"]
        self.threshold = settings["threshold"]
        self.weak_threshold = settings["weak_threshold"]
        self.weak_floor = settings["weak_floor"]
        self.no_call_band = tuple(settings["no_call_band"])
        self.calibrator = StepCalibrator(settings["calibrator"])

    def _prepare(self, payload: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(payload)) as handle:
            image = handle.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return array.transpose(2, 0, 1)[None]

    @staticmethod
    def _mask_png(mask: np.ndarray) -> str:
        image = Image.fromarray((mask * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def inspect(self, payload: bytes) -> Dict:
        started = time.perf_counter()
        array = self._prepare(payload)

        stack = np.stack(
            [
                1.0
                / (1.0 + np.exp(-self.session.run(None, {"image": array})[0][0, 0]))
                for _ in range(self.passes)
            ]
        ).astype(np.float32)
        prediction = SimpleNamespace(
            mean=_View(stack.mean(axis=0)),
            std=_View(stack.std(axis=0)),
            passes=_View(stack),
        )

        image_score = score_image(
            prediction, threshold=self.threshold, weak_threshold=self.weak_threshold
        )
        mean_map = stack.mean(axis=0)
        weak_px = int((mean_map > self.weak_threshold).sum())

        probability = self.calibrator.predict(image_score.defect_score)
        supported = self.calibrator.supported(image_score.defect_score)
        verdict = decide(
            probability, supported, weak_px, self.weak_floor, self.no_call_band
        )

        mask = (mean_map > self.threshold).astype(np.uint8)
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
            "engine": self.chain["model"].get("engine", "onnxruntime"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "explanation": (
                "The mask shows where the model looked, not what the defect "
                "is. A no-call means: route this part to a human."
            ),
            "provenance": self.chain["provenance"],
            "disclaimer": self.chain["disclaimer"],
        }
