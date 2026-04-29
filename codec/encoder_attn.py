"""EncoderAttn: convolutional encoder with a windowed self-attention bottleneck.

Identical to ``Encoder`` up to and including the 1×1 channel projection, then
applies one ``WindowedSelfAttentionBlock`` at the compressed spatial resolution
``S/c × S/c``.  This gives the latent global spatial context before
quantization, at negligible parameter cost (~13 K extra for D=32, heads=4).

Architecture::

    Conv(1 → C, 3×3, pad=1) + GN + GELU          — stem
    [Conv(C → C, 3×3, stride=2) + GN + GELU] × N  — N = log2(c) down-blocks
    Conv(C → D, 1×1)                               — channel projection
    WindowedSelfAttentionBlock(D, heads, w)        — bottleneck attention
"""

from __future__ import annotations

import math
from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.attention import build_attention_block
from codec.encoder import _DownBlock, _gn


class EncoderAttn(nn.Module):
    """cSDF ``[B, 1, S, S]`` → latent ``[B, D, S/c, S/c]`` with bottleneck attention.

    Args:
        config: Parsed YAML config dict.  Reads ``model.latent_dim`` (D),
            ``model.compaction_ratio`` (c), ``model.encoder_channels`` (C),
            ``model.attn_heads`` (default 4), and
            ``model.attn_window_size`` (default 7).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        D: int = int(config["model"]["latent_dim"])
        c: int = int(config["model"]["compaction_ratio"])
        C: int = int(config["model"].get("encoder_channels", 64))
        n_stages: int = int(round(math.log2(c)))

        if 2 ** n_stages != c:
            raise ValueError(f"compaction_ratio must be a power of 2, got {c}")

        conv_layers: list[nn.Module] = [
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            _gn(C),
            nn.GELU(),
        ]
        for _ in range(n_stages):
            conv_layers.append(_DownBlock(C))
        conv_layers.append(nn.Conv2d(C, D, 1))

        self.conv_net = nn.Sequential(*conv_layers)
        self.attn = build_attention_block(config)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: cSDF patch ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Latent map ``[B, D, S/c, S/c]`` in ``float32``.
        """
        return self.attn(self.conv_net(x))
