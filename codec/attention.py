"""Attention blocks for the CurveCodec bottleneck.

Two variants are available:

``WindowedSelfAttentionBlock``
    Non-overlapping w×w tile attention (original).  Fast but tokens at tile
    boundaries cannot attend across the boundary.

``SlidingWindowAttentionBlock``
    True local attention: every position attends to its w×w neighbourhood via
    ``F.unfold``.  No boundary artefacts; ~(w²/old_w²) training-time cost.
    Q is shaped ``[B·H·W, 1, D]`` and K/V ``[B·H·W, w², D]`` so the whole
    operation is a single batched matmul inside ``nn.MultiheadAttention``.

Both blocks share the same external interface::

    x: [B, D, H, W]  →  [B, D, H, W]

and the same config keys::

    model.attn_heads        — attention heads (default 4)
    model.attn_window_size  — window side length w (default 3)
    model.attn_type         — "sliding" (default) | "windowed"
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Windowed (original) ───────────────────────────────────────────────────────

class WindowedSelfAttentionBlock(nn.Module):
    """Non-overlapping windowed MHSA + FFN.

    Args:
        dim: Feature channels ``D``.
        num_heads: Number of attention heads.  Must divide ``dim``.
        window_size: Spatial window side-length ``w`` (tokens per side).
        ffn_expansion: Hidden-dim multiplier for the FFN (default 4).
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

    @staticmethod
    def _partition(x: Tensor, w: int) -> tuple[Tensor, int, int]:
        """``[B, H, W, D]`` → ``[B·nH·nW, w², D]``."""
        B, H, W, D = x.shape
        nH, nW = H // w, W // w
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

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``[B, D, H, W]``  Returns: same shape.
        """
        B, D, H, W = x.shape
        w = self.window_size

        pH = (w - H % w) % w
        pW = (w - W % w) % w
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH))
        Hp, Wp = H + pH, W + pW

        x = x.permute(0, 2, 3, 1)  # [B, Hp, Wp, D]

        wins, nH, nW = self._partition(x, w)
        attn_out, _ = self.attn(
            self.norm1(wins), self.norm1(wins), self.norm1(wins),
            need_weights=False,
        )
        wins = wins + attn_out
        x = self._unpartition(wins, nH, nW, B, w)

        x = x + self.ffn(self.norm2(x))

        x = x.permute(0, 3, 1, 2)
        return x[:, :, :H, :W]


# ── Sliding window (true local attention) ────────────────────────────────────

class SlidingWindowAttentionBlock(nn.Module):
    """Local sliding-window MHSA + FFN.

    Every spatial position attends to its ``w×w`` neighbourhood (``w`` must be
    odd).  Neighbours are gathered with ``F.unfold``; border positions use
    reflect padding so no artificial zero tokens appear at the edges.

    Memory footprint of the unfolded K/V tensor: ``B·D·(H·W)·w²`` elements
    (≈51 MB at B=4, D=32, H=W=105, w=3).

    Args:
        dim: Feature channels ``D``.
        num_heads: Number of attention heads.  Must divide ``dim``.
        window_size: Neighbourhood side-length ``w`` (must be odd, ≥ 1).
        ffn_expansion: Hidden-dim multiplier for the FFN (default 4).
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
        if window_size % 2 == 0:
            raise ValueError(
                f"attn_window_size must be odd for sliding-window attention, got {window_size}"
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

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: ``[B, D, H, W]``  Returns: same shape.
        """
        B, D, H, W = x.shape
        k = self.window_size
        r = k // 2

        # ── MHSA ─────────────────────────────────────────────────────────────
        # Pre-norm in channel-last space
        x_ln = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, D, H, W]

        # Q: every position as a single query token → [B·H·W, 1, D]
        q = x_ln.permute(0, 2, 3, 1).reshape(B * H * W, 1, D)

        # K/V: w×w neighbourhood for every position via unfold.
        # Reflect padding keeps border statistics consistent with interior.
        x_pad = F.pad(x_ln, (r, r, r, r), mode="reflect")          # [B, D, H+2r, W+2r]
        # unfold dims 2 and 3 with step=1 to get [B, D, H, W, k, k]
        kv = x_pad.unfold(2, k, 1).unfold(3, k, 1)                  # [B, D, H, W, k, k]
        kv = kv.reshape(B, D, H * W, k * k)                         # [B, D, H·W, k²]
        kv = kv.permute(0, 2, 3, 1).contiguous()                    # [B, H·W, k², D]
        kv = kv.reshape(B * H * W, k * k, D)                        # [B·H·W, k², D]

        attn_out, _ = self.attn(q, kv, kv, need_weights=False)      # [B·H·W, 1, D]
        attn_out = attn_out.reshape(B, H, W, D).permute(0, 3, 1, 2) # [B, D, H, W]
        x = x + attn_out

        # ── FFN ──────────────────────────────────────────────────────────────
        x_cl = x.permute(0, 2, 3, 1)                    # [B, H, W, D]
        x_cl = x_cl + self.ffn(self.norm2(x_cl))
        return x_cl.permute(0, 3, 1, 2)                 # [B, D, H, W]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_attention_block(config: dict[str, Any]) -> nn.Module:
    """Construct an attention block from a YAML config dict.

    Config keys:
        ``model.latent_dim``       — feature channels D
        ``model.attn_heads``       — number of heads (default 4)
        ``model.attn_window_size`` — window / neighbourhood side length (default 3)
        ``model.attn_type``        — ``"sliding"`` (default) | ``"windowed"``
    """
    D: int = int(config["model"]["latent_dim"])
    heads: int = int(config["model"].get("attn_heads", 4))
    window: int = int(config["model"].get("attn_window_size", 3))
    attn_type: str = str(config["model"].get("attn_type", "sliding"))

    if attn_type == "sliding":
        return SlidingWindowAttentionBlock(dim=D, num_heads=heads, window_size=window)
    elif attn_type == "windowed":
        return WindowedSelfAttentionBlock(dim=D, num_heads=heads, window_size=window)
    else:
        raise ValueError(f"model.attn_type must be 'sliding' or 'windowed', got '{attn_type}'")
