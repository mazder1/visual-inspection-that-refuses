"""The contract ``vinspect.models.unet.UNet`` has to satisfy.

Written before the model, so there is something concrete to implement against.
Every test here goes through the public interface only::

    UNet(in_channels=3, out_channels=1, base_channels=16, depth=4, dropout=0.1)
    forward(x: (B, in_channels, H, W)) -> (B, out_channels, H, W) raw logits

Nothing inspects layer names or internal structure, because that would specify
an implementation rather than verify one. Organise the inside however you like;
these only care that the outside behaves.

The tests fall into four groups, and the last one is the only one that can
catch a missing skip connection. Shape tests cannot: a U-Net with the skips
forgotten still constructs, still has valid shapes, still trains, and is simply
worse.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vinspect.train.loss import FocalTverskyLoss

# Skipped rather than failed at import, so the rest of the suite still runs
# while this is the outstanding work.
UNet = pytest.importorskip(
    "vinspect.models.unet",
    reason="src/vinspect/models/unet.py is not written yet -- it is module 02",
).UNet

# The agreed configuration: bilinear upsampling, concatenated skips, padded
# convolutions, GroupNorm with 8 groups, base 16 and depth 4, dropout in the
# decoder for module 03's MC dropout.
DEFAULTS = dict(in_channels=3, out_channels=1, base_channels=16, depth=4, dropout=0.1)
GROUPS = 8


def _build(**overrides):
    return UNet(**{**DEFAULTS, **overrides})


def _count(model):
    return sum(p.numel() for p in model.parameters())


# --- shape ----------------------------------------------------------------


@pytest.mark.parametrize("batch, size", [(1, 256), (2, 256), (1, 512)])
def test_output_shape_matches_input(batch, size):
    model = _build().eval()
    with torch.no_grad():
        output = model(torch.randn(batch, 3, size, size))
    assert output.shape == (batch, 1, size, size)


def test_channel_counts_are_respected():
    model = _build(in_channels=1, out_channels=4).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 64, 64))
    assert output.shape == (1, 4, 64, 64)


def test_indivisible_input_does_not_silently_change_shape():
    """At depth 4, 100 pools down to 6 and comes back as 96.

    Raising is a fine answer. Returning a differently-sized tensor is not:
    the mismatch would surface much later, as a confusing loss error.
    """
    model = _build().eval()
    image = torch.randn(1, 3, 100, 100)
    try:
        with torch.no_grad():
            output = model(image)
    except (RuntimeError, ValueError):
        return
    assert output.shape[-2:] == image.shape[-2:], (
        "an input size not divisible by 2**depth came back a different size "
        "instead of failing loudly"
    )


# --- the design decisions, enforced ---------------------------------------


def test_no_batch_normalisation():
    """GroupNorm was chosen because its statistics do not depend on batch size
    and behave identically in train and eval. Batch norm would reintroduce both
    problems, and the service handles one image at a time.
    """
    batch_norms = [
        type(module).__name__
        for module in _build().modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    assert not batch_norms, f"found batch normalisation: {batch_norms}"


def test_no_transposed_convolution():
    """Bilinear upsampling plus a convolution was chosen to avoid the
    checkerboard artifacts transposed convolution produces, which land directly
    on defect mask boundaries.
    """
    offenders = [
        type(module).__name__
        for module in _build().modules()
        if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d))
    ]
    assert not offenders, f"found transposed convolution: {offenders}"


def test_group_norm_is_present_and_divides_every_width():
    model = _build()
    norms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert norms, "no GroupNorm found"
    for norm in norms:
        assert norm.num_groups == GROUPS, (
            f"expected {GROUPS} groups, found {norm.num_groups} on a layer with "
            f"{norm.num_channels} channels"
        )
        assert norm.num_channels % norm.num_groups == 0


def test_dropout_layers_exist():
    """Module 03 estimates uncertainty with MC dropout, which needs dropout in
    the trained weights. Discovering this after six training runs is expensive.
    """
    dropouts = [m for m in _build().modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]
    assert dropouts, "no dropout layers found"


def test_parameter_count_is_in_the_expected_range():
    """Base 16 at depth 4 lands near 2.2M. Base 64 would be about 34M, which
    would overfit 133 defective images without ever announcing itself.
    """
    count = _count(_build())
    assert 1.0e6 < count < 4.0e6, f"{count / 1e6:.2f}M parameters"


def test_width_scales_parameters_quadratically():
    ratio = _count(_build(base_channels=16)) / _count(_build(base_channels=8))
    assert 3.4 < ratio < 4.6, f"doubling width changed parameters {ratio:.2f}x"


def test_depth_changes_the_model():
    assert _count(_build(depth=4)) > _count(_build(depth=3))


# --- behaviour ------------------------------------------------------------


def test_gradient_reaches_every_parameter():
    """Catches a block that was built but never called, which is invisible from
    the output shape.
    """
    model = _build().eval()
    model(torch.randn(2, 3, 64, 64)).sum().backward()

    unreached = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    assert not unreached, f"no gradient reached: {unreached}"


def test_output_is_logits_not_probabilities():
    model = _build().eval()
    with torch.no_grad():
        output = model(torch.randn(4, 3, 64, 64))
    assert float(output.min()) < 0.0, (
        "output never goes negative, which suggests a sigmoid on the end. The "
        "loss applies it internally for numerical stability."
    )


def test_dropout_is_active_in_train_mode():
    model = _build(dropout=0.2).train()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(model(image), model(image)), (
            "two training-mode passes were identical, so dropout is not active"
        )


def test_dropout_is_inert_in_eval_mode():
    model = _build(dropout=0.2).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(image), model(image))


def test_zero_dropout_is_deterministic_in_train_mode():
    model = _build(dropout=0.0).train()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(image), model(image))


def test_mc_dropout_produces_per_pixel_variance():
    """Module 03's prerequisite: repeated passes must disagree, per pixel."""
    model = _build(dropout=0.2).train()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        passes = torch.stack([model(image) for _ in range(8)])

    variance = passes.var(dim=0)
    assert float(variance.mean()) > 0.0
    assert float((variance > 0).float().mean()) > 0.5, (
        "fewer than half the pixels varied across passes; dropout may sit in "
        "only one place, or too late in the decoder to affect the output"
    )


# --- the one that proves the skip connections exist -----------------------


@pytest.mark.slow
def test_recovers_thin_lines_it_has_to_localise():
    """Fit six images, each with a 2-pixel line at a different column.

    This is the only test here that can detect a missing skip connection.
    Shapes cannot, and the parameter count barely moves. But an encoder-decoder
    without skips downsamples 16x, so a 2-pixel line's position survives only to
    the nearest 16 pixels and comes back as a smear -- Dice around 0.25.

    The line positions differ per image on purpose. With a single fixed example
    the decoder could memorise the answer and ignore the input entirely, which
    would pass without any skip connection at all.
    """
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size, count, steps = 96, 6, 300

    target = torch.zeros(count, 1, size, size)
    for i, column in enumerate(torch.linspace(8, size - 10, count).long()):
        target[i, 0, :, column : column + 2] = 1.0
    image = 0.25 + 0.5 * target.repeat(1, 3, 1, 1) + 0.15 * torch.rand(count, 3, size, size)

    image, target = image.to(device), target.to(device)
    model = _build(dropout=0.0).to(device).train()
    criterion = FocalTverskyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, _ = criterion(model(image), target)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        prediction = (torch.sigmoid(model(image)) > 0.5).float()
    dice = 2 * (prediction * target).sum() / (prediction.sum() + target.sum() + 1e-8)

    assert float(dice) > 0.75, (
        f"Dice {float(dice):.3f} after {steps} steps. The model cannot reproduce "
        f"a 2-pixel line it can see in its own input, which is what happens when "
        f"the skip connections are missing or wired to the wrong levels."
    )
