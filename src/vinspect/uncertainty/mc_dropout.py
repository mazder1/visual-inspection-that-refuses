"""Monte Carlo dropout: ask the same question repeatedly and watch it wobble.

Dropout normally does nothing at inference. Left switched on, it kills a
different random handful of channels on every forward pass, so the same image
gives a slightly different answer each time. Where the model's evidence is
spread across many channels, losing one or two changes nothing and the passes
agree. Where the answer hangs on a single channel, it flips. **That
disagreement is the uncertainty.**

No retraining is needed -- the dropout layers are already in the trained
weights, which is why they were required in the architecture from the start.

One detail that looks like a bug and is not. :func:`enable_dropout` calls
``model.eval()`` and then puts *only* the dropout layers back into their
stochastic mode. On this model that is identical to ``model.train()``, because
GroupNorm ignores the flag and there is no batch normalisation for
``train()`` to disturb. It is written explicitly anyway, for two reasons:

* It relies on an absence. Add a BatchNorm layer later and ``model.train()``
  would start normalising each image against whichever others share its batch,
  *and* would update the running statistics -- silently corrupting the trained
  model every time uncertainty was measured. Nothing would crash.
* ``model.train()`` inside a function that only predicts reads like a mistake.
  Someone deletes it, MC dropout quietly becomes N identical passes, the
  measured uncertainty becomes exactly zero, and the system reports perfect
  confidence about everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn

#: Passes per image. Twenty is the usual choice and the variance of the mean
#: has largely settled by then; the cost is linear, so raise it if the
#: uncertainty estimates look noisy.
DEFAULT_PASSES = 20

DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def dropout_layers(model: nn.Module) -> Iterator[nn.Module]:
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            yield module


def enable_dropout(model: nn.Module) -> int:
    """Everything deterministic, except dropout. Returns how many were found.

    A return of zero means the model has no dropout and MC dropout would
    silently produce identical passes, so callers should treat it as an error
    rather than as perfect confidence.
    """
    model.eval()
    count = 0
    for layer in dropout_layers(model):
        layer.train()
        count += 1
    return count


@dataclass(frozen=True)
class MCPrediction:
    """Result of N stochastic passes over one image.

    ``passes`` is kept rather than reduced away because region-level
    uncertainty needs it: whether a blob *survives* each pass is a different
    question from how much its individual pixels wobble, and only the first one
    tells you whether the defect is real.
    """

    mean: Tensor  # (H, W) averaged probability
    std: Tensor  # (H, W) per-pixel disagreement
    passes: Tensor  # (N, H, W) every pass, for region-level analysis

    @property
    def n_passes(self) -> int:
        return int(self.passes.shape[0])


@torch.no_grad()
def mc_predict(
    model: nn.Module,
    image: Tensor,
    passes: int = DEFAULT_PASSES,
    keep_passes: bool = True,
) -> MCPrediction:
    """Run one image ``passes`` times with dropout live.

    ``image`` is ``(3, H, W)`` or ``(1, 3, H, W)``. No gradients are taken and
    no weights change: this is inference, and the only thing borrowed from
    training is dropout's randomness.
    """
    if enable_dropout(model) == 0:
        raise ValueError(
            "the model has no dropout layers, so every MC pass would be "
            "identical and the measured uncertainty would be exactly zero"
        )
    if passes < 2:
        raise ValueError(f"need at least 2 passes to measure disagreement, got {passes}")

    batched = image if image.dim() == 4 else image.unsqueeze(0)
    if batched.shape[0] != 1:
        raise ValueError(f"expected a single image, got batch {batched.shape[0]}")

    device = next(model.parameters()).device
    batched = batched.to(device)
    stack = torch.stack(
        [torch.sigmoid(model(batched).float())[0, 0] for _ in range(passes)]
    )

    return MCPrediction(
        mean=stack.mean(dim=0).cpu(),
        std=stack.std(dim=0).cpu(),
        passes=stack.cpu() if keep_passes else stack[:0].cpu(),
    )


@torch.no_grad()
def mc_predict_batch(
    model: nn.Module, images: Tensor, passes: int = DEFAULT_PASSES
) -> list:
    """MC prediction for a batch, one entry per image.

    The whole batch goes through together on each pass, so dropout masks are
    shared within a pass. That is harmless here: GroupNorm means no image's
    prediction depends on any other, so sharing a mask does not couple them.
    """
    if enable_dropout(model) == 0:
        raise ValueError("the model has no dropout layers")

    device = next(model.parameters()).device
    images = images.to(device)
    stack = torch.stack(
        [torch.sigmoid(model(images).float())[:, 0] for _ in range(passes)]
    )  # (N, B, H, W)

    return [
        MCPrediction(
            mean=stack[:, i].mean(dim=0).cpu(),
            std=stack[:, i].std(dim=0).cpu(),
            passes=stack[:, i].cpu(),
        )
        for i in range(images.shape[0])
    ]
