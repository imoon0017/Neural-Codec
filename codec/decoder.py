"""Decoder network for the CurveCodec autoencoder.

Accepts a latent map of shape ``[B, D, S/c, S/c]`` and reconstructs a
cSDF patch of shape ``[B, 1, S, S]``.  ``D`` is ``model.latent_dim`` and
``c`` is ``model.compaction_ratio`` from config.

Architecture (fully convolutional, shift-invariant):

    Conv(D → C, 1×1)                                   — channel expansion
    [Conv(C → 4C, 3×3) + PixelShuffle(2) + GN + GELU] × N  — N up-blocks
    Conv(C → 1, 3×3) + Sigmoid                         — output head

Pixel-shuffle upsampling avoids checkerboard artefacts and is
shift-invariant (spec §Shift-Invariance).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from codec.encoder import _gn  # reuse GN helper


class _UpBlock(nn.Module):
    """×2 upsampling via pixel-shuffle: C → 4C (3×3) → shuffle → C + GN + GELU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),        # [B, 4C, H, W] → [B, C, 2H, 2W]
            _gn(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """Convolutional decoder: ``[B, D, S/c, S/c]`` → ``[B, 1, S, S]``.

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
            nn.Conv2d(D, C, 1),
        ]
        for _ in range(n_stages):
            layers.append(_UpBlock(C))
        layers.extend([
            nn.Conv2d(C, 1, 3, padding=1),
            nn.Sigmoid(),  # clamp output to [0, 1] — prevents spurious contours
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, z_q: Tensor) -> Tensor:
        """Args:
            z_q: Quantized latent map ``[B, D, S/c, S/c]`` in ``float32``.

        Returns:
            Reconstructed cSDF patch ``[B, 1, S, S]`` in ``[0, 1]``.
        """
        return self.net(z_q)
