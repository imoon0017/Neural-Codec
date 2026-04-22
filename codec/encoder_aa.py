"""Anti-aliased encoder for CurveCodecV2.

Identical architecture to ``Encoder`` except that each stride-2 down-block is
split into two sequential convolutions:

1. A stride-1 3×3 conv (feature extraction — no aliasing at stride=1)
2. A learnable depthwise 3×3 conv with stride=2 (downsampling filter)

The depthwise strided conv acts as a per-channel downsampling filter whose
kernel is learned jointly with the rest of the network, allowing it to adapt
to the cSDF signal distribution rather than relying on a fixed (e.g. binomial)
kernel.

Architecture:

    Conv(1 → C, 3×3) + GroupNorm + GELU                   — stem
    [Conv(C → C, 3×3, s=1, bias=False) + GN + GELU
     + DWConv(C, 3×3, s=2, groups=C, bias=False)] × N     — N = log2(c) down-blocks
    Conv(C → D, 1×1)                                       — channel projection
"""

from __future__ import annotations

import math
from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.encoder import _gn  # reuse GN helper


class _DownBlockAA(nn.Module):
    """Stride-1 conv + GroupNorm + GELU + learnable depthwise stride-2 conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        # Depthwise strided conv: one learnable 3×3 filter per channel.
        # Operates on already-activated features — no norm/activation needed.
        self.downsample = nn.Conv2d(
            channels, channels, 3, stride=2, padding=1, groups=channels, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample(self.conv(x))


class EncoderAA(nn.Module):
    """Anti-aliased convolutional encoder: ``[B, 1, S, S]`` → ``[B, D, S/c, S/c]``.

    Drop-in replacement for ``Encoder``.  Each down-block uses a learnable
    depthwise stride-2 conv for downsampling instead of a single stride-2
    conv, decoupling feature extraction (stride-1) from spatial compression
    (stride-2) and allowing the network to learn an appropriate anti-aliasing
    filter.

    Args:
        config: Parsed YAML config dict.  Reads ``model.latent_dim`` (D),
            ``model.compaction_ratio`` (c, power of 2), and
            ``model.encoder_channels`` (C, default 64).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        D: int = int(config["model"]["latent_dim"])
        c: int = int(config["model"]["compaction_ratio"])
        C: int = int(config["model"].get("encoder_channels", 64))
        n_stages: int = int(round(math.log2(c)))

        if 2**n_stages != c:
            raise ValueError(f"compaction_ratio must be a power of 2, got {c}")

        layers: list[nn.Module] = [
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            _gn(C),
            nn.GELU(),
        ]
        for _ in range(n_stages):
            layers.append(_DownBlockAA(C))
        layers.append(nn.Conv2d(C, D, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: cSDF patch ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Continuous latent map ``[B, D, S/c, S/c]`` in ``float32``.
        """
        return self.net(x)
