"""Transposed-conv decoder for CurveCodecV2.

Replaces the ``PixelShuffle``-based up-blocks in ``Decoder`` with
``ConvTranspose2d`` up-blocks.  Pixel-shuffle rearranges a fixed spatial
block into channels; the shuffle pattern introduces high-frequency
chessboard-like artefacts when the channel-to-pixel mapping is not perfectly
learned.  A 4×4 ``ConvTranspose2d(stride=2, padding=1)`` upsamples spatially
with a proper learnable filter, avoiding this.

Architecture:

    Conv(D → C, 1×1)                                      — channel expansion
    [ConvTranspose2d(C → C, 4×4, stride=2, padding=1)
     + GroupNorm + GELU] × N                               — N up-blocks
    Conv(C → 1, 3×3) + Sigmoid                             — output head

The 4×4 transposed conv with stride=2, padding=1 produces an exact ×2 output:
    H_out = (H_in − 1) × 2 − 2×1 + 4 = 2 × H_in
"""

from __future__ import annotations

import math
from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.encoder import _gn  # reuse GN helper


class _UpBlockTC(nn.Module):
    """×2 upsampling via transposed conv: ConvTranspose2d(C→C, 4×4, s=2) + GN + GELU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 4×4 kernel, stride=2, padding=1 → exact ×2 spatial upsample
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False),
            _gn(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderTC(nn.Module):
    """Transposed-conv decoder: ``[B, D, S/c, S/c]`` → ``[B, 1, S, S]``.

    Drop-in replacement for ``Decoder`` without pixel-shuffle artefacts.
    Parameter count is lower than ``Decoder``: each up-block uses
    ``C×C×16`` weights instead of ``C×4C×9``.

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
            nn.Conv2d(D, C, 1),
        ]
        for _ in range(n_stages):
            layers.append(_UpBlockTC(C))
        layers.extend([
            nn.Conv2d(C, 1, 3, padding=1),
            nn.Sigmoid(),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, z_q: Tensor) -> Tensor:
        """Args:
            z_q: Quantized latent map ``[B, D, S/c, S/c]`` in ``float32``.

        Returns:
            Reconstructed cSDF patch ``[B, 1, S, S]`` in ``[0, 1]``.
        """
        return self.net(z_q)
