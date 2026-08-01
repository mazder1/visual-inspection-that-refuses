"""Tests for Monte Carlo dropout.

The failure mode worth guarding is silent: if dropout is not actually active,
every pass is identical, the measured uncertainty is exactly zero, and the
system reports perfect confidence about everything without anything crashing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vinspect.models.unet import UNet
from vinspect.uncertainty.mc_dropout import (
    MCPrediction,
    enable_dropout,
    mc_predict,
    mc_predict_batch,
)


@pytest.fixture
def model():
    return UNet(base_channels=8, depth=2, dropout=0.2)


def _image(size=32):
    return torch.randn(3, size, size)


def test_enable_dropout_switches_only_dropout(model):
    count = enable_dropout(model)
    assert count > 0

    for module in model.modules():
        if isinstance(module, nn.Dropout2d):
            assert module.training, "dropout should be stochastic"
        elif isinstance(module, (nn.Conv2d, nn.GroupNorm)):
            assert not module.training, (
                "only dropout should be switched; leaving other layers in "
                "training mode is how a BatchNorm added later would silently "
                "corrupt its running statistics during inference"
            )


def test_a_model_without_dropout_is_an_error():
    # Zero dropout means UNet builds no dropout layers at all. Returning
    # confident, identical passes would be far worse than refusing.
    with pytest.raises(ValueError, match="no dropout layers"):
        mc_predict(UNet(base_channels=8, depth=2, dropout=0.0), _image())


def test_too_few_passes_is_an_error(model):
    with pytest.raises(ValueError, match="at least 2 passes"):
        mc_predict(model, _image(), passes=1)


def test_shapes_and_ranges(model):
    prediction = mc_predict(model, _image(32), passes=6)
    assert isinstance(prediction, MCPrediction)
    assert prediction.mean.shape == (32, 32)
    assert prediction.std.shape == (32, 32)
    assert prediction.passes.shape == (6, 32, 32)
    assert prediction.n_passes == 6
    assert 0.0 <= float(prediction.mean.min())
    assert float(prediction.mean.max()) <= 1.0
    assert float(prediction.std.min()) >= 0.0


def test_passes_actually_disagree(model):
    prediction = mc_predict(model, _image(32), passes=10)
    assert float(prediction.std.mean()) > 0.0
    assert not torch.allclose(prediction.passes[0], prediction.passes[1])


def test_the_mean_is_near_the_deterministic_prediction(model):
    """Dropout scales survivors to compensate, so averaging many stochastic
    passes lands close to the ordinary eval output rather than somewhere else."""
    image = _image(32)
    model.eval()
    with torch.no_grad():
        deterministic = torch.sigmoid(model(image.unsqueeze(0))[0, 0])

    prediction = mc_predict(model, image, passes=40)
    assert float((prediction.mean - deterministic).abs().mean()) < 0.15


def test_no_gradients_are_taken(model):
    image = _image(32).requires_grad_(True)
    prediction = mc_predict(model, image, passes=4)
    assert not prediction.mean.requires_grad
    assert image.grad is None
    assert all(p.grad is None for p in model.parameters())


def test_dropping_the_passes_saves_memory(model):
    prediction = mc_predict(model, _image(32), passes=4, keep_passes=False)
    assert prediction.passes.numel() == 0
    assert prediction.mean.shape == (32, 32)


def test_batch_form_matches_the_single_form(model):
    images = torch.stack([_image(32), _image(32)])
    predictions = mc_predict_batch(model, images, passes=5)
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.mean.shape == (32, 32)
        assert prediction.passes.shape == (5, 32, 32)


def test_a_batch_of_more_than_one_is_rejected_by_the_single_form(model):
    with pytest.raises(ValueError, match="single image"):
        mc_predict(model, torch.randn(2, 3, 32, 32))
