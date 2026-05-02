"""EncoderMultiAttn: convolutional encoder with windowed self-attention after every stage.

Architecture (for compaction_ratio c = 2^N, N down-blocks)::

    Conv(1 → C, 3×3) + GN + GELU                    — stem
    WindowedAttn(C, heads, w=window_sizes[0])         — stem attention       (S × S)
    [Conv(C → C, 3×3, stride=2) + GN + GELU]         — down-block 0
    WindowedAttn(C, heads, w=window_sizes[1])         — mid attention 0      (S/2 × S/2)
    ...
    [Conv(C → C, 3×3, stride=2) + GN + GELU]         — down-block N-1
    Conv(C → D, 1×1)                                  — channel projection
    WindowedAttn(D, heads, w=window_sizes[N])         — bottleneck attention  (S/c × S/c)

``N+1`` attention blocks total: one after the stem, one after each of the first
``N-1`` down-blocks (on ``C`` channels), and one at the bottleneck (on ``D``
channels after the 1×1 projection).

Config keys::

    model.latent_dim          — D
    model.compaction_ratio    — c (power of 2)
    model.encoder_channels    — C (default 64)
    model.attn_heads          — attention heads (default 4, must divide both C and D)
    model.attn_window_sizes   — list of N+1 ints: [w_stem, w_mid0, …, w_bottleneck]
                                default: [7, 5, 3, 3, …] (stem=7, first mid=5, rest=3)
"""

from __future__ import annotations

import math
from typing import Any

import torch.nn as nn
from torch import Tensor

from codec.attention import WindowedSelfAttentionBlock
from codec.encoder import _DownBlock, _gn


class EncoderMultiAttn(nn.Module):
    """cSDF ``[B, 1, S, S]`` → latent ``[B, D, S/c, S/c]`` with per-stage attention.

    Args:
        config: Parsed YAML config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        D: int = int(config["model"]["latent_dim"])
        c: int = int(config["model"]["compaction_ratio"])
        C: int = int(config["model"].get("encoder_channels", 64))
        heads: int = int(config["model"].get("attn_heads", 4))
        n_stages: int = int(round(math.log2(c)))

        if 2**n_stages != c:
            raise ValueError(f"compaction_ratio must be a power of 2, got {c}")
        if C % heads != 0:
            raise ValueError(f"attn_heads ({heads}) must divide encoder_channels ({C})")
        if D % heads != 0:
            raise ValueError(f"attn_heads ({heads}) must divide latent_dim ({D})")

        # Window sizes: n_stages+1 values [w_stem, w_mid0, …, w_bottleneck]
        default_windows: list[int] = [7, 5] + [3] * (n_stages - 1)
        raw = config["model"].get("attn_window_sizes", default_windows)
        window_sizes: list[int] = [int(w) for w in raw]
        if len(window_sizes) != n_stages + 1:
            raise ValueError(
                f"attn_window_sizes must have {n_stages + 1} entries for "
                f"compaction_ratio={c} (N={n_stages} stages), got {len(window_sizes)}"
            )

        self._c = c

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            _gn(C),
            nn.GELU(),
        )
        self.stem_attn = WindowedSelfAttentionBlock(C, heads, window_sizes[0])

        # ── Down-blocks + mid attentions (on C channels) ──────────────────────
        # N down-blocks; attention after each except the last (which feeds proj).
        self.down_blocks = nn.ModuleList([_DownBlock(C) for _ in range(n_stages)])
        self.mid_attns = nn.ModuleList([
            WindowedSelfAttentionBlock(C, heads, window_sizes[i + 1])
            for i in range(n_stages - 1)
        ])

        # ── 1×1 projection + bottleneck attention (on D channels) ─────────────
        self.proj = nn.Conv2d(C, D, 1)
        self.bottleneck_attn = WindowedSelfAttentionBlock(D, heads, window_sizes[-1])

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: cSDF patch ``[B, 1, S, S]`` in ``float32``.

        Returns:
            Latent map ``[B, D, S/c, S/c]`` in ``float32``.
        """
        h, w = x.shape[-2], x.shape[-1]
        if h % self._c != 0 or w % self._c != 0:
            raise ValueError(
                f"Input spatial size ({h}×{w}) must be divisible by "
                f"compaction_ratio={self._c}."
            )

        x = self.stem_attn(self.stem(x))

        for i, down in enumerate(self.down_blocks):
            x = down(x)
            if i < len(self.down_blocks) - 1:
                x = self.mid_attns[i](x)

        return self.bottleneck_attn(self.proj(x))
