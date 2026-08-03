"""Static INT8 quantisation of the exported serving graphs.

Weights need no data: their ranges are exact constants in the file, and each
conv gets per-channel rulers (the outlier defence). Activations are the reason
calibration exists -- their ranges are a property of data flowing through, so
they are estimated by observation and frozen into the graph.

Calibration policy, decided deliberately:

* **Training images, never validation.** Fitting activation rulers is a fit,
  and validation already has two jobs here -- checkpoint selection and the
  probability calibrator. Spending its independence on a third fit would
  muddy exactly the drift-vs-repair measurement this experiment exists for.
  Training shapes the model; validation fits the probability map; test is
  touched once.
* **Stratified: clean and defective.** Defects produce the extreme
  activations. Calibrate on clean parts only and those extremes fall off the
  rulers and are clipped precisely on the images that matter.
* **Through the stochastic serving graph.** Dropout stays live during
  calibration, so the recorded ranges include the values dropout's rescaling
  actually produces at serving time.

One caveat carried on record: the model has memorised the training images, and
activations on memorised data can run slightly narrower than on unseen data --
rulers fitted this way may clip marginally on novel inputs. Stage 3's drift
arms measure whether that matters rather than assuming either way.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from vinspect.serve.onnx_export import make_session

#: Calibration set size per category. Ranges converge quickly; the tail of a
#: larger set rarely widens what the first hundred images established.
N_CLEAN = 80
N_DEFECTIVE = 40


class ArrayCalibrationReader:
    """Feeds pre-loaded (1, 3, H, W) float32 arrays to the ORT quantiser."""

    def __init__(self, arrays: List[np.ndarray]) -> None:
        self._iterator = iter(arrays)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        batch = next(self._iterator, None)
        return None if batch is None else {"image": batch}


def calibration_arrays_from_training(
    root: Path,
    split_path: Path,
    category: str,
    image_size: int,
    n_clean: int = N_CLEAN,
    n_defective: int = N_DEFECTIVE,
    seed: int = 0,
) -> List[np.ndarray]:
    """A stratified, deterministic sample of TRAINING images as arrays."""
    from vinspect.data.mvtec import MVTecDataset, index_mvtec
    from vinspect.data.splits import load_split, records_for_split

    payload = load_split(split_path)
    records = index_mvtec(root, [category])
    train = records_for_split(records, payload, "train")

    rng = random.Random(seed)
    clean = sorted((r for r in train if r.label == 0), key=lambda r: r.key)
    defective = sorted((r for r in train if r.label == 1), key=lambda r: r.key)
    rng.shuffle(clean)
    rng.shuffle(defective)
    chosen = clean[:n_clean] + defective[:n_defective]

    dataset = MVTecDataset(chosen, image_size=image_size)
    return [
        dataset[i]["image"].numpy()[None].astype(np.float32)
        for i in range(len(dataset))
    ]


def op_histogram(onnx_path: Path) -> Dict[str, int]:
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    return dict(Counter(node.op_type for node in graph.node))


def quantize_bundle(
    bundle_dir: Path,
    calibration_arrays: List[np.ndarray],
    out_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Quantise ``model.onnx`` -> ``model.int8.onnx`` and report the census.

    Structural gate carried over from stage 1: the RandomUniform dropout ops
    must survive quantisation untouched, or the uncertainty machinery is dead
    and every downstream number is fiction.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    bundle_dir = Path(bundle_dir)
    fp32_path = bundle_dir / "model.onnx"
    if not fp32_path.is_file():
        raise FileNotFoundError(f"{fp32_path} missing; run the stage-1 export first")
    out_path = out_path or bundle_dir / "model.int8.onnx"

    before = op_histogram(fp32_path)
    # Conv only, deliberately. The convolutions carry essentially all the
    # FLOPs; Resize picked up a QDQ wrapping that fails ORT's runtime scale
    # validation, and GroupNorm has no int8 kernel regardless. Scoping to Conv
    # gives int8 where the compute is and keeps the fragile ops in float.
    quantize_static(
        str(fp32_path),
        str(out_path),
        ArrayCalibrationReader(calibration_arrays),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv"],
    )
    after = op_histogram(out_path)

    if after.get("RandomUniform", 0) != before.get("RandomUniform", 0):
        raise RuntimeError(
            f"dropout ops changed under quantisation: "
            f"{before.get('RandomUniform', 0)} -> {after.get('RandomUniform', 0)}"
        )

    # Sanity: the quantised graph must load and run.
    session = make_session(out_path)
    example = calibration_arrays[0]
    first = session.run(None, {"image": example})[0]
    second = session.run(None, {"image": example})[0]
    if np.allclose(first, second):
        raise RuntimeError("quantised graph runs deterministically; dropout dead")

    return {
        "fp32_bytes": fp32_path.stat().st_size,
        "int8_bytes": out_path.stat().st_size,
        "ops_before": before,
        "ops_after": after,
        "quantize_dequantize_pairs": after.get("DequantizeLinear", 0),
        "out_path": str(out_path),
    }
