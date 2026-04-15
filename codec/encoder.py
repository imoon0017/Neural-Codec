"""Encoder network for the CurveCodec autoencoder.

Accepts a batch of cSDF patches of shape ``[B, 1, S, S]`` and produces a
continuous latent map of shape ``[B, D, S/c, S/c]``, where ``D`` is
``model.latent_dim`` and ``c`` is ``model.compaction_ratio`` from config.

Architecture (fully convolutional, shift-invariant):

    Conv(1 → C, 3×3) + GroupNorm + GELU          — stem
    [Conv(C → C, 3×3, stride=2) + GN + GELU] × N  — N = log2(c) down-blocks
    Conv(C → D, 1×1)                               — channel projection
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

_GN_GROUPS: int = 8  # GroupNorm group count (channels per group = C / 8)


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with at most ``_GN_GROUPS`` groups."""
    groups = min(_GN_GROUPS, channels)
    # Ensure channels is divisible by groups
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _DownBlock(nn.Module):
    """Stride-2 3×3 conv + GroupNorm + GELU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            _gn(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Convolutional encoder: ``[B, 1, S, S]`` → ``[B, D, S/c, S/c]``.

    Args:
        config: Parsed YAML config dict.  Reads ``model.latent_dim`` (D),
            ``model.compaction_ratio`` (c, power of 2), and
            ``model.encoder_channels`` (C, hidden width, default 64).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        D: int = int(config["model"]["latent_dim"])
        c: int = int(config["model"]["compaction_ratio"])
        C: int = int(config["model"].get("encoder_channels", 64))
        n_stages: int = int(round(math.log2(c)))

        if 2 ** n_stages != c:
            raise ValueError(f"compaction_ratio must be a power of 2, got {c}")

        layers: list[nn.Module] = [
            # Stem
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            _gn(C),
            nn.GELU(),
        ]
        for _ in range(n_stages):
            layers.append(_DownBlock(C))
        # Channel projection to latent dim
        layers.append(nn.Conv2d(C, D, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: cSDF patch ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Continuous latent map ``[B, D, S/c, S/c]`` in ``float32``.
        """
        return self.net(x)
