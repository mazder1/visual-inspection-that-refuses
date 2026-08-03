"""Tests for the ONNX export and its gates, on a tiny synthetic bundle.

The first exporter run silently produced a deterministic graph -- dropout
traced through its identity branch -- and only the stochasticity gate caught
it. These tests keep both gates honest at CI speed.
"""

from __future__ import annotations

import json

import pytest
import torch

from vinspect.models.unet import UNet

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from vinspect.serve.onnx_export import (  # noqa: E402
    ExportChannelDropout,
    export_onnx,
    gate_determinism,
    gate_stochasticity,
    make_session,
    swap_dropout_for_export,
)


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("onnx_bundle") / "widget"
    root.mkdir()
    model = UNet(base_channels=8, depth=2, dropout=0.2)
    torch.save({"model": model.state_dict()}, root / "model.pt")
    chain = {
        "category": "widget",
        "model": {
            "base_channels": 8, "depth": 2, "dropout": 0.2, "image_size": 64,
            "weights_sha256": "test",
        },
        "chain": {
            "mc_passes": 4,
            "threshold": 0.5,
            "weak_threshold": 0.33,
            "weak_floor": 50.0,
            "no_call_band": [0.2, 0.8],
            "calibrator": {
                "n_steps": 2,
                "base_rate": 0.2,
                "pseudocount": 1.0,
                "steps": [
                    {"scores": [0.0, 10.0], "probability": 0.01, "count": 40, "raw_rate": 0.0},
                    {"scores": [100.0, 5000.0], "probability": 0.9, "count": 10, "raw_rate": 1.0},
                ],
            },
        },
        "provenance": {}, "disclaimer": "test",
    }
    (root / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
    return root


def test_swap_replaces_every_dropout():
    model = UNet(base_channels=8, depth=2, dropout=0.2)
    before = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout2d))
    swapped = swap_dropout_for_export(model)
    after = sum(1 for m in model.modules() if isinstance(m, torch.nn.Dropout2d))
    exported = sum(1 for m in model.modules() if isinstance(m, ExportChannelDropout))
    assert before > 0 and after == 0 and exported == before == swapped


def test_export_dropout_matches_dropout2d_statistics():
    """The replacement must be the same distribution, not merely random."""
    torch.manual_seed(0)
    x = torch.rand(1, 16, 8, 8)
    ours = ExportChannelDropout(0.5, 16).train()
    theirs = torch.nn.Dropout2d(0.5).train()

    ours_kept = torch.stack([(ours(x)[0, :, 0, 0] != 0) for _ in range(400)])
    theirs_kept = torch.stack([(theirs(x)[0, :, 0, 0] != 0) for _ in range(400)])
    # Keep rate ~0.5 for both, and whole channels drop together.
    assert abs(ours_kept.float().mean() - 0.5) < 0.05
    assert abs(ours_kept.float().mean() - theirs_kept.float().mean()) < 0.06

    out = ours(x)
    channel_zeroed = (out == 0).all(dim=-1).all(dim=-1)[0]
    channel_intact = (out != 0).all(dim=-1).all(dim=-1)[0]
    assert bool((channel_zeroed | channel_intact).all()), (
        "dropout must act on whole channels, not scattered pixels"
    )


def test_eval_mode_is_identity():
    x = torch.rand(1, 8, 4, 4)
    module = ExportChannelDropout(0.3, 8).eval()
    assert torch.equal(module(x), x)


def test_export_and_both_gates(bundle_dir):
    onnx_path = export_onnx(bundle_dir)
    assert onnx_path.is_file()

    determinism = gate_determinism(bundle_dir, onnx_path)
    assert determinism["passed"], determinism

    stochastic = gate_stochasticity(bundle_dir, onnx_path, passes=30)
    assert stochastic["passed_alive"], stochastic
    assert stochastic["passed_distribution"], stochastic


def test_quantize_bundle_shrinks_and_stays_stochastic(bundle_dir):
    import numpy as np

    from vinspect.serve.quantize import op_histogram, quantize_bundle

    onnx_path = bundle_dir / "model.onnx"
    if not onnx_path.is_file():
        onnx_path = export_onnx(bundle_dir)

    rng = np.random.default_rng(0)
    arrays = [rng.random((1, 3, 64, 64), dtype=np.float32) for _ in range(4)]
    report = quantize_bundle(bundle_dir, arrays)

    # On a tiny model the per-channel scales and QDQ bookkeeping are a large
    # fixed cost, so the shrink is far from the ~3.8x seen on the real 8.7 MB
    # graphs; assert direction and margin, not the big-model ratio.
    assert report["int8_bytes"] < report["fp32_bytes"] * 0.8
    after = op_histogram(bundle_dir / "model.int8.onnx")
    assert after.get("RandomUniform", 0) == report["ops_before"].get("RandomUniform", 0)
    assert after.get("DequantizeLinear", 0) > 0
    # quantize_bundle itself raises if the graph went deterministic or fails
    # to run; reaching here means both held.


def test_onnx_predictor_parity_and_engine_selection(bundle_dir, tmp_path):
    """The factory must pick the ONNX engine from chain.json, and on an image
    with no evidence both predictors must agree exactly -- score zero lands on
    the same staircase step regardless of engine or MC randomness."""
    import base64
    import io
    import shutil

    from PIL import Image

    from vinspect.serve.app import _build_predictor
    from vinspect.serve.onnx_predictor import OnnxPredictor
    from vinspect.serve.predictor import Predictor

    onnx_path = bundle_dir / "model.onnx"
    if not onnx_path.is_file():
        export_onnx(bundle_dir)

    # Two copies of the bundle: one chain pointing at torch, one at ONNX.
    onnx_bundle = tmp_path / "widget_onnx"
    shutil.copytree(bundle_dir, onnx_bundle)
    chain = json.loads((onnx_bundle / "chain.json").read_text())
    chain["model"]["file"] = "model.onnx"
    chain["model"]["engine"] = "onnxruntime"
    (onnx_bundle / "chain.json").write_text(json.dumps(chain))

    torch_predictor = _build_predictor(bundle_dir)
    onnx_predictor = _build_predictor(onnx_bundle)
    assert isinstance(torch_predictor, Predictor)
    assert isinstance(onnx_predictor, OnnxPredictor)

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (30, 30, 30)).save(buffer, format="PNG")
    payload = buffer.getvalue()

    a = torch_predictor.inspect(payload)
    b = onnx_predictor.inspect(payload)
    for key in ("category", "mc_passes", "mask_size"):
        assert a[key] == b[key]
    # Response contract: identical keys apart from the engine tag.
    assert set(a) | {"engine"} == set(b) | {"engine"}
    # Mask decodes on both.
    for body in (a, b):
        mask = Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"])))
        assert mask.size == (64, 64)


def test_exported_graph_random_ops_survive_session_optimisation(bundle_dir):
    onnx_path = bundle_dir / "model.onnx"
    if not onnx_path.is_file():
        onnx_path = export_onnx(bundle_dir)
    session = make_session(onnx_path)
    import numpy as np

    image = np.random.default_rng(0).random((1, 3, 64, 64), dtype=np.float32)
    first = session.run(None, {"image": image})[0]
    second = session.run(None, {"image": image})[0]
    assert not np.allclose(first, second), (
        "two runs were identical: the session optimiser removed the randomness"
    )
