"""Windowed self-attention block for the CurveCodec bottleneck.

Applies multi-head self-attention within non-overlapping w×w spatial windows
of a latent feature map ``[B, D, H, W]``.  No positional encodings — fully
shift-invariant by design (per spec §Shift-Invariance).

Block structure (one pass)::

    x  →  LayerNorm  →  Window Partition  →  MHSA  →  Window Unpartition  →  + x
       →  LayerNorm  →  FFN (D → 4D → D)                                   →  + x

The spatial dims are zero-padded to the next multiple of ``window_size``
before partitioning and the padding is removed afterwards, so arbitrary
spatial sizes are supported.

Configured via ``model.attn_heads`` (default 4) and
``model.attn_window_size`` (default 7).  At ``S/c = 105`` with ``w=7``,
the map tiles into 15×15 = 225 windows of 49 tokens each.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WindowedSelfAttentionBlock(nn.Module):
    """Windowed MHSA + FFN at the latent bottleneck.

    Args:
        dim: Feature channels ``D``.
        num_heads: Number of attention heads.  Must divide ``dim``.
        window_size: Spatial window side-length ``w`` (tokens per side).
        ffn_expansion: Hidden-dim multiplier for the FFN (default 4 → D→4D→D).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"attn_heads ({num_heads}) must divide latent_dim ({dim})"
            )
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

    # ── Window helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _partition(x: Tensor, w: int) -> tuple[Tensor, int, int]:
        """``[B, H, W, D]`` → ``[B·nH·nW, w², D]``."""
        B, H, W, D = x.shape
        nH, nW = H // w, W // w
        # [B, nH, w, nW, w, D] → [B, nH, nW, w, w, D] → [B·nH·nW, w², D]
        x = x.view(B, nH, w, nW, w, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B * nH * nW, w * w, D), nH, nW

    @staticmethod
    def _unpartition(x: Tensor, nH: int, nW: int, B: int, w: int) -> Tensor:
        """``[B·nH·nW, w², D]`` → ``[B, nH·w, nW·w, D]``."""
        D = x.shape[-1]
        x = x.view(B, nH, nW, w, w, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, nH * w, nW * w, D)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Apply windowed self-attention + FFN.

        Args:
            x: Latent map ``[B, D, H, W]`` in ``float32``.

        Returns:
            Same shape ``[B, D, H, W]``.
        """
        B, D, H, W = x.shape
        w = self.window_size

        # Pad to multiple of w
        pH = (w - H % w) % w
        pW = (w - W % w) % w
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))  # pad right and bottom
        Hp, Wp = H + pH, W + pW

        # Channel-first → channel-last for attention ops
        x = x.permute(0, 2, 3, 1)  # [B, Hp, Wp, D]

        # ── Windowed MHSA + residual ──────────────────────────────────────────
        wins, nH, nW = self._partition(x, w)          # [B·nH·nW, w², D]
        attn_out, _ = self.attn(
            self.norm1(wins), self.norm1(wins), self.norm1(wins),
            need_weights=False,
        )
        wins = wins + attn_out
        x = self._unpartition(wins, nH, nW, B, w)     # [B, Hp, Wp, D]

        # ── FFN + residual ────────────────────────────────────────────────────
        x = x + self.ffn(self.norm2(x))

        # Channel-last → channel-first, remove padding
        x = x.permute(0, 3, 1, 2)                     # [B, D, Hp, Wp]
        return x[:, :, :H, :W]


def build_attention_block(config: dict[str, Any]) -> WindowedSelfAttentionBlock:
    """Construct a ``WindowedSelfAttentionBlock`` from a YAML config dict.

    Reads:
        ``model.latent_dim``       — feature channels D
        ``model.attn_heads``       — number of heads (default 4)
        ``model.attn_window_size`` — window side length (default 7)
    """
    D: int = int(config["model"]["latent_dim"])
    heads: int = int(config["model"].get("attn_heads", 4))
    window: int = int(config["model"].get("attn_window_size", 7))
    return WindowedSelfAttentionBlock(dim=D, num_heads=heads, window_size=window)
