"""U-Net, implemented from the 2015 paper, with the deviations documented.

The shape of the network at a 512x512 input, base 16 and depth 4::

                 ENCODER                              DECODER
    level 1   3 -> 16    512x512  ---- skip ---->   32 -> 16   512x512 -> 1x512x512
                 v pool                                ^ upsample
    level 2  16 -> 32    256x256  ---- skip ---->   64 -> 32   256x256
                 v pool                                ^ upsample
    level 3  32 -> 64    128x128  ---- skip ---->  128 -> 64   128x128
                 v pool                                ^ upsample
    level 4  64 -> 128    64x64   ---- skip ---->  256 -> 128   64x64
                 v pool                                ^ upsample
    bottom  128 -> 256    32x32

Going down trades *where* for *what*: the bottleneck understands that there is
a crack somewhere but only to within 16 pixels of where. The skip connections
are what carry position back, which is why this can outline a two-pixel thread
and a plain encoder-decoder cannot. Removing them is not a shape error and does
not change the parameter count much -- the network still trains, just badly --
so `tests/test_unet.py` proves they are connected by fitting thin lines whose
position varies per image.

Four choices deviate from the paper or from the common default. Each is
enforced by a test so it cannot drift:

**Padded convolutions.** The paper uses unpadded ones, so a 572x572 input gives
a 388x388 output and full images need the overlap-tile strategy. That was a
workaround for 2012-era GPU memory. Padding keeps output size equal to input
size, so the predicted mask aligns with ground truth with no cropping step for
an off-by-one to hide in.

**Bilinear upsampling followed by a convolution**, not a transposed
convolution. A transposed convolution's kernel and stride overlap unevenly, so
some output pixels accumulate contributions from more kernel positions than
others -- the checkerboard artifact. On a defect mask that lands directly on
the boundary being scored.

**GroupNorm rather than BatchNorm.** Its statistics come from within one sample,
so they do not depend on batch size and are identical in training and
inference. The service in module 04 handles one image at a time, where
BatchNorm's train/eval discrepancy is a known source of silent breakage.

**Base width 16, not the paper's 64.** Convolution parameters scale with the
square of width, so this is about 2.2M parameters against 34M. The training
split holds 133 defective images.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: GroupNorm groups. Every width in the default configuration (16, 32, 64, 128,
#: 256) is divisible by 8, so one value works at every level. The GroupNorm
#: paper's usual default of 32 cannot divide a 16-channel layer.
GROUPS = 8


class ConvBlock(nn.Module):
    """Two padded 3x3 convolutions, each with GroupNorm and ReLU.

    Convolution bias is omitted because GroupNorm immediately re-centres the
    activations, which makes it redundant.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if out_channels % GROUPS:
            raise ValueError(
                f"{out_channels} channels is not divisible by {GROUPS} GroupNorm "
                f"groups; pick a base width that is a multiple of {GROUPS}"
            )
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(GROUPS, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            # Dropout2d drops whole feature maps rather than scattered
            # activations. That produces genuinely different predictions across
            # MC dropout passes, which is what module 03 needs to measure.
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, halve the channels, concatenate the skip, then convolve.

    Concatenation rather than addition: the skip carries high-frequency spatial
    detail and the decoder path carries coarse semantic context. They are
    different kinds of information, so the following convolution is left to
    learn how to mix them rather than being handed a 1:1 sum. ResNet adds
    because there the branch is a residual correction to its own input -- same
    semantics, same scale, a different job.
    """

    def __init__(self, in_channels: int, skip_channels: int, dropout: float) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, skip_channels, 3, padding=1, bias=False)
        self.block = ConvBlock(skip_channels * 2, skip_channels, dropout=dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Defect segmentation network. Returns raw logits, never probabilities.

    Logits, because the loss applies the sigmoid internally via
    ``binary_cross_entropy_with_logits``, which stays finite at logits of +-60
    where sigmoid-then-log underflows.

    Args:
        in_channels: image channels, 3 after the loader's RGB conversion.
        out_channels: 1 for a binary defect mask.
        base_channels: width of the first level; doubles at each level down.
        depth: number of downsampling steps. Input size must be divisible by
            ``2 ** depth``.
        dropout: applied in the bottleneck and every decoder block. Must be
            non-zero at training time or module 03's MC dropout has nothing to
            sample from, and the models have to be retrained.
        prior: optional base rate for initialising the output bias, so an
            untrained network predicts the base rate rather than 0.5. Measured
            at 0.0086 on the training split, giving a bias of about -4.75. This
            is the RetinaNet initialisation, and focal loss can start with
            vanishing gradients without it.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        dropout: float = 0.1,
        prior: Optional[float] = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if prior is not None and not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")

        self.depth = depth
        self.stride = 2 ** depth
        widths = [base_channels * 2 ** level for level in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths[:-1]:
            self.encoders.append(ConvBlock(previous, width))
            previous = width

        self.bottleneck = ConvBlock(previous, widths[-1], dropout=dropout)

        # Built deepest-first so decoders[i] pairs with the skip from
        # encoders[-1 - i].
        self.decoders = nn.ModuleList(
            UpBlock(widths[level + 1], widths[level], dropout=dropout)
            for level in reversed(range(depth))
        )

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)
        if prior is not None:
            nn.init.constant_(self.head.bias, -math.log((1.0 - prior) / prior))

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if height % self.stride or width % self.stride:
            raise ValueError(
                f"input {(height, width)} must be divisible by {self.stride} "
                f"(2 ** depth={self.depth}); pooling would return a different "
                f"size than it received"
            )

        skips: List[Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)
