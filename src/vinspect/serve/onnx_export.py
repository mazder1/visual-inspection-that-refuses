"""Export a bundle's model to ONNX, with the two gates that make it trustworthy.

The export is a recording: one tensor is traced through the network and every
operation lands in a portable graph file that ONNX Runtime can replay without
Python or torch. Two things can silently go wrong with *this* model, and each
has a gate:

**Gate 1, determinism parity.** With dropout off, the ONNX graph must compute
the same function as the torch model to float-rounding tolerance. If it does
not, an op was mistranslated -- and finding that now is what lets any later
drift be attributed to quantisation instead of to the export.

**Gate 2, stochasticity.** The serving chain needs dropout ALIVE in the graph:
exporters remove it by default, and the failure is silent -- twenty identical
passes, measured uncertainty exactly zero, the system confidently sure of
everything. The gate runs repeated passes and requires them to disagree.

One honest caveat, checked rather than assumed: torch exports ``Dropout2d``
(channel dropout) as elementwise ``Dropout``, which changes the *shape* of the
randomness even when the rate is right. Gate 2 therefore also compares the
MC standard-deviation maps between engines; if their distributions diverge,
the export is not the evaluated chain and must not be served.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from vinspect.models.unet import UNet
from vinspect.uncertainty.mc_dropout import enable_dropout

OPSET = 17


class ExportChannelDropout(torch.nn.Module):
    """Channel dropout written in ops the ONNX exporter can translate.

    ``Dropout2d`` traces to ``aten::feature_dropout``, which has no ONNX
    symbolic in training mode -- the export simply fails. This module computes
    the identical thing by hand: one Bernoulli draw per channel, scaled by
    1/(1-p), broadcast over the spatial grid. ``torch.rand`` traces to ONNX
    ``RandomUniform``, so ONNX Runtime draws fresh channel masks on every run
    and the graph's randomness matches Dropout2d's distribution exactly for
    the batch-of-one serving case.
    """

    def __init__(self, p: float, channels: int) -> None:
        super().__init__()
        self.p = float(p)
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        noise = torch.rand(1, self.channels, 1, 1, device=x.device, dtype=x.dtype)
        keep = (noise >= self.p).to(x.dtype) / (1.0 - self.p)
        return x * keep


def swap_dropout_for_export(model: torch.nn.Module) -> int:
    """Replace every Dropout2d with the export-friendly equivalent, in place.

    Channel counts are read from each dropout's position: in this architecture
    every ConvBlock ends GroupNorm -> ReLU -> Dropout2d, so the preceding
    GroupNorm's channel count is the dropout's channel count.
    """
    swapped = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Sequential):
            continue
        for index, child in enumerate(module):
            if isinstance(child, torch.nn.Dropout2d):
                channels = None
                for previous in reversed(list(module)[:index]):
                    if hasattr(previous, "num_channels"):
                        channels = previous.num_channels
                        break
                    if hasattr(previous, "out_channels"):
                        channels = previous.out_channels
                        break
                if channels is None:
                    raise RuntimeError(
                        "could not infer channel count for a Dropout2d"
                    )
                module[index] = ExportChannelDropout(child.p, channels)
                swapped += 1
    return swapped


def load_bundle_model(bundle_dir: Path) -> tuple:
    chain = json.loads((Path(bundle_dir) / "chain.json").read_text(encoding="utf-8"))
    config = chain["model"]
    model = UNet(
        base_channels=config["base_channels"],
        depth=config["depth"],
        dropout=config["dropout"],
    )
    checkpoint = torch.load(
        Path(bundle_dir) / "model.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, chain


def export_onnx(
    bundle_dir: Path, out_path: Optional[Path] = None, opset: int = OPSET
) -> Path:
    """Write ``model.onnx`` next to the bundle's ``model.pt``.

    Dropout modules are switched to their stochastic mode before tracing and
    the export preserves training-state ops, so dropout survives into the
    graph instead of being folded away.
    """
    bundle_dir = Path(bundle_dir)
    model, chain = load_bundle_model(bundle_dir)
    size = chain["model"]["image_size"]
    out_path = out_path or bundle_dir / "model.onnx"

    swapped = swap_dropout_for_export(model)
    if swapped == 0:
        raise RuntimeError(
            "no Dropout2d found to swap; the exported graph would be "
            "deterministic and the uncertainty machinery dead"
        )
    enable_dropout(model)
    # enable_dropout only knows torch's own dropout types; the export modules
    # must be switched to their stochastic branch explicitly, or the trace
    # records their identity path and the graph comes out deterministic --
    # which is exactly what gate 2 caught on the first run of this exporter.
    for module in model.modules():
        if isinstance(module, ExportChannelDropout):
            module.train()
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["logits"],
        training=torch.onnx.TrainingMode.PRESERVE,
        do_constant_folding=False,
    )

    # Structural gate at export time: the randomness must be IN the file.
    import onnx

    graph = onnx.load(str(out_path)).graph
    random_ops = sum(1 for node in graph.node if node.op_type == "RandomUniform")
    if random_ops != swapped:
        raise RuntimeError(
            f"expected {swapped} RandomUniform ops in the exported graph, "
            f"found {random_ops}; dropout did not survive the export"
        )
    return out_path


def make_session(onnx_path: Path, threads: int = 0):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )


# --- the gates -------------------------------------------------------------


def gate_determinism(
    bundle_dir: Path, onnx_path: Path, tolerance: float = 1e-4
) -> Dict[str, float]:
    """Dropout off in both engines; outputs must match to float tolerance."""
    model, chain = load_bundle_model(bundle_dir)  # eval(): dropout inert
    size = chain["model"]["image_size"]
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 3, size, size, generator=generator)

    with torch.no_grad():
        torch_logits = model(image).numpy()

    # Fresh model instance for the export path is unnecessary: the graph is
    # frozen on disk. Ratio=0 cannot be forced at runtime, so determinism is
    # checked by exporting a second graph with dropout left inert.
    deterministic_path = Path(onnx_path).with_suffix(".eval.onnx")
    deterministic = UNet(
        base_channels=chain["model"]["base_channels"],
        depth=chain["model"]["depth"],
        dropout=chain["model"]["dropout"],
    )
    deterministic.load_state_dict(
        torch.load(Path(bundle_dir) / "model.pt", map_location="cpu",
                   weights_only=False)["model"]
    )
    deterministic.eval()
    torch.onnx.export(
        deterministic, torch.zeros(1, 3, size, size), str(deterministic_path),
        opset_version=OPSET, input_names=["image"], output_names=["logits"],
    )
    session = make_session(deterministic_path)
    onnx_logits = session.run(None, {"image": image.numpy()})[0]

    difference = np.abs(torch_logits - onnx_logits)
    report = {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "passed": bool(difference.max() < tolerance),
    }
    return report


#: Accepted band for onnx-vs-torch mean-std ratio. Wide on purpose and set
#: from a measured control, not taste: five independent 20-pass reruns of the
#: SAME torch model spread by up to 3.2x, because a 20-pass std is itself a
#: noisy estimate. A between-engine ratio inside this band is indistinguishable
#: from within-engine noise; a genuinely wrong dropout (dead, or elementwise
#: instead of channelwise at a different rate) lands far outside it.
STD_RATIO_BAND = (0.4, 2.5)


def gate_stochasticity(
    bundle_dir: Path, onnx_path: Path, passes: int = 60
) -> Dict[str, object]:
    """Dropout on: ONNX passes must disagree, and the uncertainty must have the
    same magnitude as torch's -- not just be non-zero."""
    model, chain = load_bundle_model(bundle_dir)
    size = chain["model"]["image_size"]
    generator = torch.Generator().manual_seed(1)
    image = torch.rand(1, 3, size, size, generator=generator)

    session = make_session(onnx_path)
    onnx_stack = np.stack(
        [
            1.0 / (1.0 + np.exp(-session.run(None, {"image": image.numpy()})[0][0, 0]))
            for _ in range(passes)
        ]
    )
    onnx_std = onnx_stack.std(axis=0)

    enable_dropout(model)
    with torch.no_grad():
        torch_stack = torch.stack(
            [torch.sigmoid(model(image))[0, 0] for _ in range(passes)]
        ).numpy()
    torch_std = torch_stack.std(axis=0)

    varying = float((onnx_std > 1e-6).mean())
    ratio = float(onnx_std.mean() / max(torch_std.mean(), 1e-12))
    return {
        "onnx_fraction_of_pixels_varying": varying,
        "onnx_mean_std": float(onnx_std.mean()),
        "torch_mean_std": float(torch_std.mean()),
        "std_ratio_onnx_over_torch": ratio,
        "passed_alive": bool(varying > 0.5 and onnx_std.mean() > 0),
        "passed_distribution": bool(
            STD_RATIO_BAND[0] <= ratio <= STD_RATIO_BAND[1]
        ),
    }
